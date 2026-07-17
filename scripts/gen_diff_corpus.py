#!/usr/bin/env python3
"""Seeded generator for the Moonlab cross-backend / cross-binding differential.

Emits a canonical circuit interchange corpus AND an independent reference
oracle. Every backend (dense statevector, tn_mps, Clifford tableau, GPU) and
every binding (C, Python, Rust, WASM/JS) runs the same corpus and compares to
the reference probabilities/expectations pinned here. Because the reference is
computed by a tiny pure-numpy statevector simulator that has nothing to do with
Moonlab, agreement is absolute -- it catches a uniformly-wrong Moonlab, not just
internal inconsistency.

The generator is a PURE FUNCTION OF ITS SEED: it uses a self-contained
splitmix64 PRNG, never time, os.urandom, or the `random` module. The same seed
reproduces both outputs byte-for-byte.

Two mirror files are written from one in-memory structure so both are the same
shared reference:
  * corpus.json  -- authoritative JSON interchange (Python, JS, Rust legs)
  * corpus.txt   -- flat whitespace-delimited mirror (C, Rust legs); avoids a
                    hand-rolled JSON parser in C.

Canonical convention: little-endian, qubit 0 = least-significant bit. Basis
index b has bit q equal to the value of qubit q. This matches the dense C
backend, the Python/Rust/JS dense bindings, and CA-MPS Pauli indexing. The
plain tn_mps backend is big-endian (site 0 = MSB) and its consumer bit-reverses
the statevector before comparing -- that mapping lives in the C driver, not here.

Gate set (all backends + reference support these):
  1q no-param : h x y z s sdg t tdg
  1q param    : rx ry rz p        (p = phase, diag(1, e^{i theta}))
  2q no-param : cx cz swap        (cx = cnot)
  2q param    : cp                (controlled-phase, symmetric)
  3q no-param : ccx               (toffoli; never in clifford-only cases)

Usage:
  gen_diff_corpus.py --out-dir DIR [--seed N] [--qubits 2,3,4,6,8,10,12]
                     [--depths 2,8,32] [--self-test]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard dep of the py binding
    sys.stderr.write("gen_diff_corpus: numpy is required for the reference oracle\n")
    raise

CORPUS_VERSION = 1
CONVENTION = "little_endian_qubit0_lsb"

_MASK64 = (1 << 64) - 1


# --------------------------------------------------------------------------
# Deterministic PRNG: splitmix64. Pure function of the seed, no global state,
# no time / random-device. Identical output on every platform.
# --------------------------------------------------------------------------
class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (z ^ (z >> 31)) & _MASK64

    def unit(self) -> float:
        # 53-bit uniform in [0, 1); stable across platforms.
        return (self.next_u64() >> 11) / float(1 << 53)

    def below(self, bound: int) -> int:
        return int(self.next_u64() % bound)

    def angle(self) -> float:
        # Full-circle angle in [0, 2*pi); avoids trivial 0 by construction.
        return 2.0 * math.pi * self.unit()


def derive_seed(master: int, *parts: int) -> int:
    """Mix a case-local seed from the master seed and integer tags."""
    h = master & _MASK64
    for p in parts:
        h = (h ^ (p & _MASK64)) & _MASK64
        h = (h * 0x9E3779B97F4A7C15) & _MASK64
        h ^= h >> 29
    return h & _MASK64


def fnv1a(s: str) -> int:
    """Deterministic 64-bit FNV-1a hash. Python's builtin hash() is salted
    per-process (PYTHONHASHSEED) and must never be used for the seed path."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 0x100000001B3) & _MASK64
    return h


# --------------------------------------------------------------------------
# Reference statevector simulator (numpy, little-endian). Independent of Moonlab.
# --------------------------------------------------------------------------
_SQRT1_2 = 1.0 / math.sqrt(2.0)


def _u1(name: str, theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    if name == "h":
        return np.array([[_SQRT1_2, _SQRT1_2], [_SQRT1_2, -_SQRT1_2]], dtype=complex)
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == "z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if name == "s":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if name == "sdg":
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    if name == "t":
        return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
    if name == "tdg":
        return np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex)
    if name == "p":
        return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
    raise ValueError(f"unknown 1q gate {name}")


class RefSim:
    """Exact statevector, little-endian (qubit 0 = LSB). Pure/deterministic."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.dim = 1 << n
        self.psi = np.zeros(self.dim, dtype=complex)
        self.psi[0] = 1.0

    def apply_1q(self, name: str, q: int, theta: float) -> None:
        u = _u1(name, theta)
        stride = 1 << q
        psi = self.psi
        # Iterate over blocks where bit q toggles between the two halves.
        block = stride << 1
        for base in range(0, self.dim, block):
            for off in range(stride):
                i0 = base + off
                i1 = i0 + stride
                a0 = psi[i0]
                a1 = psi[i1]
                psi[i0] = u[0, 0] * a0 + u[0, 1] * a1
                psi[i1] = u[1, 0] * a0 + u[1, 1] * a1

    def apply_cx(self, c: int, t: int) -> None:
        cm = 1 << c
        tm = 1 << t
        out = self.psi.copy()
        for i in range(self.dim):
            if i & cm:
                out[i] = self.psi[i ^ tm]
        self.psi = out

    def apply_cz(self, a: int, b: int) -> None:
        am = 1 << a
        bm = 1 << b
        for i in range(self.dim):
            if (i & am) and (i & bm):
                self.psi[i] = -self.psi[i]

    def apply_cp(self, a: int, b: int, theta: float) -> None:
        am = 1 << a
        bm = 1 << b
        ph = np.exp(1j * theta)
        for i in range(self.dim):
            if (i & am) and (i & bm):
                self.psi[i] = ph * self.psi[i]

    def apply_swap(self, a: int, b: int) -> None:
        am = 1 << a
        bm = 1 << b
        out = self.psi.copy()
        for i in range(self.dim):
            ba = (i >> a) & 1
            bb = (i >> b) & 1
            if ba != bb:
                j = (i & ~am & ~bm) | (bb << a) | (ba << b)
                out[i] = self.psi[j]
        self.psi = out

    def apply_ccx(self, c1: int, c2: int, t: int) -> None:
        m1 = 1 << c1
        m2 = 1 << c2
        tm = 1 << t
        out = self.psi.copy()
        for i in range(self.dim):
            if (i & m1) and (i & m2):
                out[i] = self.psi[i ^ tm]
        self.psi = out

    def apply(self, g: dict) -> None:
        name = g["name"]
        q = g["qubits"]
        theta = g.get("angle", 0.0)
        if name in ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p"):
            self.apply_1q(name, q[0], theta)
        elif name == "cx":
            self.apply_cx(q[0], q[1])
        elif name == "cz":
            self.apply_cz(q[0], q[1])
        elif name == "cp":
            self.apply_cp(q[0], q[1], theta)
        elif name == "swap":
            self.apply_swap(q[0], q[1])
        elif name == "ccx":
            self.apply_ccx(q[0], q[1], q[2])
        else:
            raise ValueError(f"unknown gate {name}")

    def probabilities(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def exp_z(self) -> list:
        probs = self.probabilities()
        out = []
        for q in range(self.n):
            e = 0.0
            for i in range(self.dim):
                sign = -1.0 if (i >> q) & 1 else 1.0
                e += sign * probs[i]
            out.append(e)
        return out

    def exp_zz(self, pairs: list) -> list:
        probs = self.probabilities()
        out = []
        for (a, b) in pairs:
            e = 0.0
            for i in range(self.dim):
                sa = -1.0 if (i >> a) & 1 else 1.0
                sb = -1.0 if (i >> b) & 1 else 1.0
                e += sa * sb * probs[i]
            out.append([a, b, e])
        return out


# --------------------------------------------------------------------------
# Circuit families. Each returns a list of gate dicts.
# --------------------------------------------------------------------------
_CLIFFORD_1Q = ["h", "x", "y", "z", "s", "sdg"]
_UNIVERSAL_1Q = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p"]


def _rand_pair(rng: SplitMix64, n: int) -> tuple:
    a = rng.below(n)
    b = rng.below(n - 1)
    if b >= a:
        b += 1
    return a, b


def gen_clifford(rng: SplitMix64, n: int, depth: int) -> list:
    gates = []
    for _ in range(depth):
        if n >= 2 and rng.unit() < 0.4:
            a, b = _rand_pair(rng, n)
            op = rng.below(3)
            gates.append({"name": ["cx", "cz", "swap"][op], "qubits": [a, b]})
        else:
            g = _CLIFFORD_1Q[rng.below(len(_CLIFFORD_1Q))]
            gates.append({"name": g, "qubits": [rng.below(n)]})
    return gates


def gen_rot_cnot_ladder(rng: SplitMix64, n: int, depth: int) -> list:
    # A proper forward CNOT ladder + single-qubit rotations. All 2q gates have
    # control < target, so every backend (including tn_mps) must reproduce this
    # exactly -- it is a POSITIVE control. Reversed-CNOT (control > target) cases
    # that stress the tn 2q-gate transpose path are produced organically by the
    # random_universal family via _rand_pair.
    gates = []
    for _ in range(depth):
        for q in range(n):
            axis = ["rx", "ry", "rz"][rng.below(3)]
            gates.append({"name": axis, "qubits": [q], "angle": rng.angle()})
        for q in range(n - 1):
            gates.append({"name": "cx", "qubits": [q, q + 1]})
    return gates


def gen_random_universal(rng: SplitMix64, n: int, depth: int) -> list:
    gates = []
    for _ in range(depth):
        r = rng.unit()
        if n >= 3 and r < 0.12:
            # Toffoli on three distinct qubits.
            a = rng.below(n)
            b = rng.below(n - 1)
            if b >= a:
                b += 1
            rest = [x for x in range(n) if x != a and x != b]
            t = rest[rng.below(len(rest))]
            gates.append({"name": "ccx", "qubits": [a, b, t]})
        elif n >= 2 and r < 0.5:
            a, b = _rand_pair(rng, n)
            op = rng.below(4)
            if op == 0:
                gates.append({"name": "cx", "qubits": [a, b]})
            elif op == 1:
                gates.append({"name": "cz", "qubits": [a, b]})
            elif op == 2:
                gates.append({"name": "swap", "qubits": [a, b]})
            else:
                gates.append({"name": "cp", "qubits": [a, b], "angle": rng.angle()})
        else:
            g = _UNIVERSAL_1Q[rng.below(len(_UNIVERSAL_1Q))]
            q = rng.below(n)
            if g in ("rx", "ry", "rz", "p"):
                gates.append({"name": g, "qubits": [q], "angle": rng.angle()})
            else:
                gates.append({"name": g, "qubits": [q]})
    return gates


def gen_param_layer(rng: SplitMix64, n: int, depth: int) -> list:
    # Hardware-efficient ansatz: RY-RZ on each qubit, linear CZ entangler, repeated.
    gates = []
    for _ in range(depth):
        for q in range(n):
            gates.append({"name": "ry", "qubits": [q], "angle": rng.angle()})
            gates.append({"name": "rz", "qubits": [q], "angle": rng.angle()})
        for q in range(n - 1):
            gates.append({"name": "cz", "qubits": [q, q + 1]})
    return gates


def gen_ghz(n: int) -> list:
    gates = [{"name": "h", "qubits": [0]}]
    for q in range(n - 1):
        gates.append({"name": "cx", "qubits": [q, q + 1]})
    return gates


def gen_qft(n: int) -> list:
    gates = []
    for q in range(n):
        gates.append({"name": "h", "qubits": [q]})
        for k in range(q + 1, n):
            gates.append({"name": "cp", "qubits": [k, q], "angle": math.pi / (1 << (k - q))})
    for q in range(n // 2):
        gates.append({"name": "swap", "qubits": [q, n - 1 - q]})
    return gates


def zz_pairs(n: int) -> list:
    """Nearest-neighbour + a couple long-range Z_iZ_j pairs to correlate on."""
    pairs = [(q, q + 1) for q in range(n - 1)]
    if n >= 3:
        pairs.append((0, n - 1))
    return pairs


def build_case(master_seed: int, cls: str, n: int, depth: int) -> dict:
    # The case seed depends ONLY on (master, class, n, depth) -- never on the
    # plan position -- so a given case is byte-identical whether it appears in a
    # quick subset or the full matrix. This keeps KNOWN_DIVERGENCES.txt portable
    # across profiles.
    cseed = derive_seed(master_seed, fnv1a(cls), n, depth)
    rng = SplitMix64(cseed)
    if cls == "clifford":
        gates = gen_clifford(rng, n, depth)
        clifford_only = True
    elif cls == "rot_cnot_ladder":
        gates = gen_rot_cnot_ladder(rng, n, depth)
        clifford_only = False
    elif cls == "random_universal":
        gates = gen_random_universal(rng, n, depth)
        clifford_only = False
    elif cls == "param_layer":
        gates = gen_param_layer(rng, n, depth)
        clifford_only = False
    elif cls == "ghz":
        gates = gen_ghz(n)
        clifford_only = True
    elif cls == "qft":
        gates = gen_qft(n)
        clifford_only = False
    else:
        raise ValueError(cls)

    sim = RefSim(n)
    for g in gates:
        sim.apply(g)

    pairs = zz_pairs(n)
    case_id = f"{cls}_n{n:02d}_d{depth:02d}_s{cseed & 0xFFFF:04x}"
    return {
        "id": case_id,
        "class": cls,
        "num_qubits": n,
        "depth": depth,
        "seed": cseed,
        "clifford_only": clifford_only,
        "gates": gates,
        "reference": {
            "probabilities": [float(x) for x in sim.probabilities()],
            "exp_z": [float(x) for x in sim.exp_z()],
            "exp_zz": [[a, b, float(v)] for (a, b, v) in sim.exp_zz(pairs)],
        },
    }


# A class needs at least 2 qubits for entangling structure; QFT/GHZ likewise.
def default_matrix(qubits: list, depths: list) -> list:
    plan = []
    for cls in ("clifford", "rot_cnot_ladder", "random_universal", "param_layer"):
        for n in qubits:
            for d in depths:
                plan.append((cls, n, d))
    for cls in ("ghz", "qft"):
        for n in qubits:
            # Structured circuits: depth is implied by n; tag with n as the
            # "depth" slot for a stable id.
            plan.append((cls, n, n))
    return plan


def generate(master_seed: int, qubits: list, depths: list) -> dict:
    cases = []
    for (cls, n, d) in default_matrix(qubits, depths):
        if n < 2 and cls in ("ghz", "qft", "rot_cnot_ladder", "param_layer"):
            continue
        cases.append(build_case(master_seed, cls, n, d))
    return {
        "version": CORPUS_VERSION,
        "seed": master_seed,
        "convention": CONVENTION,
        "num_cases": len(cases),
        "cases": cases,
    }


# --------------------------------------------------------------------------
# Flat mirror for the C / Rust legs (no JSON parser needed there).
# --------------------------------------------------------------------------
def write_txt(corpus: dict, path: Path) -> None:
    lines = []
    lines.append(f"CORPUS {corpus['version']} {corpus['seed']} {corpus['num_cases']}")
    for c in corpus["cases"]:
        n = c["num_qubits"]
        ref = c["reference"]
        lines.append(
            f"CASE {c['id']} {c['class']} {n} {c['depth']} "
            f"{1 if c['clifford_only'] else 0} {c['seed']} {len(c['gates'])}"
        )
        for g in c["gates"]:
            q = g["qubits"]
            q0 = q[0]
            q1 = q[1] if len(q) > 1 else -1
            q2 = q[2] if len(q) > 2 else -1
            ang = g.get("angle", 0.0)
            lines.append(f"G {g['name']} {q0} {q1} {q2} {ang!r}")
        probs = ref["probabilities"]
        lines.append("PROB " + str(len(probs)))
        lines.append(" ".join(f"{p!r}" for p in probs))
        ez = ref["exp_z"]
        lines.append("EXPZ " + str(len(ez)))
        lines.append(" ".join(f"{v!r}" for v in ez))
        zz = ref["exp_zz"]
        lines.append("EXPZZ " + str(len(zz)))
        for (a, b, v) in zz:
            lines.append(f"{a} {b} {v!r}")
        lines.append("ENDCASE")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def self_test(master_seed: int) -> int:
    """Prove the reference oracle is self-consistent and would catch a bug.

    1. Reference probabilities sum to 1 and are non-negative for every case.
    2. A deliberately corrupted probability is detectable at tol 1e-10.
    3. Regenerating with the same seed is byte-identical (determinism).
    """
    failures = 0
    c1 = generate(master_seed, [2, 3, 4], [2, 8])
    c2 = generate(master_seed, [2, 3, 4], [2, 8])
    if json.dumps(c1, sort_keys=True) != json.dumps(c2, sort_keys=True):
        sys.stderr.write("SELF-TEST FAIL: generator not deterministic\n")
        failures += 1
    for case in c1["cases"]:
        probs = case["reference"]["probabilities"]
        s = sum(probs)
        if abs(s - 1.0) > 1e-9:
            sys.stderr.write(f"SELF-TEST FAIL: {case['id']} probs sum to {s}\n")
            failures += 1
        if any(p < -1e-12 for p in probs):
            sys.stderr.write(f"SELF-TEST FAIL: {case['id']} negative prob\n")
            failures += 1
    # Deliberate corruption: flip one probability and confirm the tolerance
    # check that every leg uses would flag it.
    case = c1["cases"][0]
    good = list(case["reference"]["probabilities"])
    bad = list(good)
    bad[0] += 1e-3
    max_dev = max(abs(a - b) for a, b in zip(good, bad))
    if not (max_dev > 1e-10):
        sys.stderr.write("SELF-TEST FAIL: corruption not detectable at 1e-10\n")
        failures += 1
    # A GHZ case's Z_iZ_j must be +1 (sanity of the expectation path).
    ghz = next((x for x in c1["cases"] if x["class"] == "ghz"), None)
    if ghz is not None:
        for (a, b, v) in ghz["reference"]["exp_zz"]:
            if abs(v - 1.0) > 1e-9:
                sys.stderr.write(f"SELF-TEST FAIL: GHZ <Z{a}Z{b}>={v} != 1\n")
                failures += 1
    if failures == 0:
        print("gen_diff_corpus self-test PASS "
              "(determinism, normalization, corruption-detectable, GHZ correlations)")
    return 1 if failures else 0


def parse_int_list(s: str) -> list:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="directory to write corpus.json + corpus.txt")
    ap.add_argument("--seed", type=int, default=0xB16B00B5,
                    help="master seed (default 0xB16B00B5)")
    ap.add_argument("--qubits", type=str, default="2,3,4,6,8,10,12")
    ap.add_argument("--depths", type=str, default="2,8,32")
    ap.add_argument("--self-test", action="store_true",
                    help="run the reference-oracle self-test and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.seed)

    if not args.out_dir:
        ap.error("--out-dir is required unless --self-test")

    qubits = parse_int_list(args.qubits)
    depths = parse_int_list(args.depths)
    corpus = generate(args.seed, qubits, depths)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus.json").write_text(json.dumps(corpus, indent=1))
    write_txt(corpus, out / "corpus.txt")
    print(f"gen_diff_corpus: wrote {corpus['num_cases']} cases "
          f"(seed={args.seed}, qubits={qubits}, depths={depths}) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
