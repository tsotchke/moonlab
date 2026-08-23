#!/usr/bin/env python3
"""Focused contracts for the deterministic release corpus generators."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_circuit_corpus import (  # noqa: E402
    CLASSES,
    DEPTHS,
    QUBIT_COUNTS,
    build_corpus,
    emit_json,
)
from gen_diff_corpus import (  # noqa: E402
    CONVENTION,
    CORPUS_VERSION,
    generate,
    write_txt,
)


def parse_flat_mirror(path: Path) -> dict:
    """Parse the C-facing mirror strictly enough to compare every payload."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[-1] != "END":
        raise AssertionError("flat corpus must terminate with END")
    header = lines[0].split()
    if len(header) != 4 or header[0] != "CORPUS":
        raise AssertionError(f"invalid corpus header: {lines[0]}")
    version, seed, count = map(int, header[1:])
    cases = []
    index = 1
    while index < len(lines) - 1:
        fields = lines[index].split()
        if len(fields) != 8 or fields[0] != "CASE":
            raise AssertionError(f"invalid CASE row: {lines[index]}")
        _, case_id, case_class, n, depth, clifford_only, case_seed, gate_count = fields
        case = {
            "id": case_id,
            "class": case_class,
            "num_qubits": int(n),
            "depth": int(depth),
            "seed": int(case_seed),
            "clifford_only": bool(int(clifford_only)),
            "gates": [],
            "reference": {},
        }
        expected_gate_count = int(gate_count)
        index += 1
        while index < len(lines) and lines[index].startswith("G "):
            fields = lines[index].split()
            if len(fields) != 6:
                raise AssertionError(f"invalid G row: {lines[index]}")
            _, name, q0, q1, q2, angle = fields
            case["gates"].append(
                {
                    "name": name,
                    "qubits": [int(q) for q in (q0, q1, q2) if int(q) >= 0],
                    "angle": float(angle),
                }
            )
            index += 1
        if len(case["gates"]) != expected_gate_count or index >= len(lines):
            raise AssertionError("flat gate count does not match CASE")
        fields = lines[index].split()
        if len(fields) != 2 or fields[0] != "PROB":
            raise AssertionError(f"missing PROB row for {case_id}")
        probability_count = int(fields[1])
        probabilities = [float(value) for value in lines[index + 1].split()]
        if len(probabilities) != probability_count:
            raise AssertionError(f"PROB count does not match payload for {case_id}")
        index += 2
        fields = lines[index].split()
        if len(fields) != 2 or fields[0] != "EXPZ":
            raise AssertionError(f"missing EXPZ row for {case_id}")
        exp_z_count = int(fields[1])
        exp_z = [float(value) for value in lines[index + 1].split()]
        if len(exp_z) != exp_z_count:
            raise AssertionError(f"EXPZ count does not match payload for {case_id}")
        index += 2
        fields = lines[index].split()
        if len(fields) != 2 or fields[0] != "EXPZZ":
            raise AssertionError(f"missing EXPZZ row for {case_id}")
        exp_zz_count = int(fields[1])
        exp_zz = []
        index += 1
        for _ in range(exp_zz_count):
            fields = lines[index].split()
            if len(fields) != 3:
                raise AssertionError(f"invalid EXPZZ row: {lines[index]}")
            exp_zz.append([int(fields[0]), int(fields[1]), float(fields[2])])
            index += 1
        if index >= len(lines) or lines[index] != "ENDCASE":
            raise AssertionError(f"missing ENDCASE for {case_id}")
        case["reference"] = {
            "probabilities": probabilities,
            "exp_z": exp_z,
            "exp_zz": exp_zz,
        }
        cases.append(case)
        index += 1
    if index != len(lines) - 1:
        raise AssertionError("flat corpus contains trailing rows")
    if len(cases) != count:
        raise AssertionError("flat corpus count does not match CASE rows")
    return {"version": version, "seed": seed, "num_cases": count, "cases": cases}


def assert_mirror_matches(testcase: unittest.TestCase, mirror: dict, structured: dict) -> None:
    testcase.assertEqual(mirror["version"], structured["version"])
    testcase.assertEqual(mirror["seed"], structured["seed"])
    testcase.assertEqual(mirror["num_cases"], structured["num_cases"])
    testcase.assertEqual(len(mirror["cases"]), len(structured["cases"]))
    for flat, expected in zip(mirror["cases"], structured["cases"]):
        for field in ("id", "class", "num_qubits", "depth", "seed", "clifford_only"):
            testcase.assertEqual(flat[field], expected[field])
        testcase.assertEqual(len(flat["gates"]), len(expected["gates"]))
        for flat_gate, expected_gate in zip(flat["gates"], expected["gates"]):
            testcase.assertEqual(flat_gate["name"], expected_gate["name"])
            testcase.assertEqual(flat_gate["qubits"], expected_gate["qubits"])
            testcase.assertEqual(flat_gate["angle"], expected_gate.get("angle", 0.0))
        testcase.assertEqual(flat["reference"], expected["reference"])


class CircuitCorpusGeneratorTests(unittest.TestCase):
    def run_cli(self, output: Path, seed: int = 20260823) -> None:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/gen_circuit_corpus.py"),
                "--seed",
                str(seed),
                "--out-dir",
                str(output),
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

    def test_cli_is_deterministic_and_emits_schema_valid_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            self.run_cli(first)
            self.run_cli(second)
            for name in ("circuit_corpus.json", "circuit_corpus.h"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            document = json.loads((first / "circuit_corpus.json").read_text(encoding="utf-8"))
            expected_count = len(CLASSES) * len(QUBIT_COUNTS) * len(DEPTHS)
            self.assertEqual(document.keys(), {"circuits", "count", "seed"})
            self.assertEqual(document["count"], expected_count)
            self.assertEqual(len(document["circuits"]), expected_count)
            self.assertEqual({item["class"] for item in document["circuits"]}, set(CLASSES))
            self.assertEqual(
                {item["num_qubits"] for item in document["circuits"]},
                set(QUBIT_COUNTS),
            )
            self.assertEqual({item["depth"] for item in document["circuits"]}, set(DEPTHS))
            self.assertEqual(
                len({item["id"] for item in document["circuits"]}), expected_count
            )
            for circuit in document["circuits"]:
                self.assertEqual(circuit["num_gates"], len(circuit["gates"]))
                for gate in circuit["gates"]:
                    self.assertIn(gate["g"], {"h", "x", "y", "z", "s", "sdg", "cnot", "cz", "swap", "rx", "ry", "rz", "t", "tdg"})
                    self.assertTrue(1 <= len(gate["q"]) <= 2)
                    self.assertTrue(all(0 <= q < circuit["num_qubits"] for q in gate["q"]))
                    self.assertTrue(math.isfinite(gate["p"]))
            header = (first / "circuit_corpus.h").read_text(encoding="utf-8")
            self.assertIn("#ifndef MOONLAB_ORACLE_CIRCUIT_CORPUS_H", header)
            self.assertIn("oracle_corpus_count = (int)(sizeof(oracle_corpus)", header)

    def test_function_output_round_trips_through_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "circuit_corpus.json"
            corpus = build_corpus(17)
            emit_json(corpus, path)
            decoded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(decoded, corpus)


class DifferentialCorpusGeneratorTests(unittest.TestCase):
    def run_cli(self, output: Path, seed: int = 17) -> None:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/gen_diff_corpus.py"),
                "--out-dir",
                str(output),
                "--seed",
                str(seed),
                "--qubits",
                "2,3",
                "--depths",
                "2,4",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

    def test_cli_is_deterministic_and_json_text_mirrors_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            self.run_cli(first)
            self.run_cli(second)
            for name in ("corpus.json", "corpus.txt"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            document = json.loads((first / "corpus.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(document), {"version", "seed", "convention", "num_cases", "cases"}
            )
            self.assertEqual(document["version"], CORPUS_VERSION)
            self.assertEqual(document["convention"], CONVENTION)
            self.assertEqual(document["num_cases"], len(document["cases"]))
            self.assertEqual(
                {case["class"] for case in document["cases"]},
                {"clifford", "rot_cnot_ladder", "random_universal", "param_layer", "ghz", "qft"},
            )
            for case in document["cases"]:
                self.assertEqual(len(case["reference"]["probabilities"]), 1 << case["num_qubits"])
                self.assertAlmostEqual(sum(case["reference"]["probabilities"]), 1.0, places=10)
                self.assertTrue(all(probability >= -1e-12 for probability in case["reference"]["probabilities"]))
                for gate in case["gates"]:
                    self.assertIn(gate["name"], {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p", "cx", "cz", "cp", "swap", "ccx"})
                    self.assertTrue(all(0 <= q < case["num_qubits"] for q in gate["qubits"]))

            mirror = parse_flat_mirror(first / "corpus.txt")
            assert_mirror_matches(self, mirror, document)

    def test_function_output_round_trips_through_flat_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = generate(23, [2], [2])
            json_path = root / "corpus.json"
            text_path = root / "corpus.txt"
            json_path.write_text(json.dumps(corpus, indent=1), encoding="utf-8")
            write_txt(corpus, text_path)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), corpus)
            assert_mirror_matches(self, parse_flat_mirror(text_path), corpus)

    def test_self_test_cli_catches_reference_invariants(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/gen_diff_corpus.py"), "--self-test", "--seed", "23"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertIn("self-test PASS", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
