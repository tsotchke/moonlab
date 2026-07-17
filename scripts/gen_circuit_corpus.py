#!/usr/bin/env python3
"""Seed-deterministic quantum circuit corpus generator (adversarial pillar P1).

Emits JSON circuit families and a C header the oracle binaries include.
Generation is a PURE FUNCTION OF THE SEED: the only entropy is a splitmix64
stream seeded by --seed. No wall-clock, no os.urandom, no dict-ordering
dependence. Two runs with the same seed are byte-identical.

Circuit classes (each tagged in its `class` field):

  clifford         pure-Clifford chains (h,x,y,z,s,sdg,cnot,cz,swap), forward
  rot_cnot_ladder  1q rotations + forward CNOT brickwork ladders
  random_universal random universal gates over forward adjacent pairs
  param_layers     hardware-efficient parameterized layers (ry,rz + CNOT chain)
  reversed_2q      1q rotations + reversed adjacent CNOTs (control > target);
                   the dedicated 2q-transpose bug catcher (see _adjacent_forward)

Qubit counts {2,4,6,8,10} x depths {4,16,64}. `depth` counts gate layers.

Usage:
  scripts/gen_circuit_corpus.py --seed 20260717 --out-dir tests/oracle/corpus
"""

import argparse
import json
import math
import os

MASK64 = (1 << 64) - 1

# Qubit counts and depths swept by the corpus.
QUBIT_COUNTS = (2, 4, 6, 8, 10)
DEPTHS = (4, 16, 64)
CLASSES = ("clifford", "rot_cnot_ladder", "random_universal", "param_layers",
           "reversed_2q")

# Single-qubit Clifford generators (no angle).
CLIFFORD_1Q = ("h", "x", "y", "z", "s", "sdg")
# Two-qubit Clifford generators.
CLIFFORD_2Q = ("cnot", "cz", "swap")
# Non-Clifford single-qubit set for the universal class.
UNIVERSAL_1Q = ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz")
PARAM_1Q = ("rx", "ry", "rz")


class SplitMix64:
    """Deterministic splitmix64 PRNG. Pure integer arithmetic."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def below(self, bound):
        """Uniform integer in [0, bound). Rejection-free modulo is fine here:
        the corpus is a fixture, not a cryptographic sampler."""
        return self.next_u64() % bound

    def unit(self):
        """Uniform double in [0, 1)."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def angle(self):
        """Uniform double in [0, 2*pi)."""
        return self.unit() * 2.0 * math.pi


def _adjacent_forward(rng, n):
    """A forward adjacent qubit pair (i, i+1), control < target.

    Adjacency keeps the MPS two-qubit apply to a single local SVD (no swap
    network), so the corpus stays fast enough for the < 2 min oracle budget;
    adjacent brickwork is still universal, so full entanglement is reached at
    depth. Reversed-direction (control > target) gates -- the shape that
    exposes a two-qubit-gate transpose bug -- are isolated in the dedicated
    `reversed_2q` class, so the correct-path classes stay a clean green signal
    and the known bug is caught by one identifiable, quarantinable family."""
    i = rng.below(n - 1)
    return i, i + 1


def _g1(name, q, p=0.0):
    return {"g": name, "q": [q], "p": p}


def _g2(name, a, b, p=0.0):
    return {"g": name, "q": [a, b], "p": p}


def gen_clifford(rng, n, depth):
    gates = []
    for _ in range(depth):
        for q in range(n):
            if rng.unit() < 0.7:
                gates.append(_g1(CLIFFORD_1Q[rng.below(len(CLIFFORD_1Q))], q))
        for _ in range(max(1, n // 2)):
            a, b = _adjacent_forward(rng, n)
            gates.append(_g2(CLIFFORD_2Q[rng.below(len(CLIFFORD_2Q))], a, b))
    return gates


def gen_rot_cnot_ladder(rng, n, depth):
    gates = []
    for _ in range(depth):
        for q in range(n):
            axis = ("rx", "ry", "rz")[rng.below(3)]
            gates.append(_g1(axis, q, rng.angle()))
        # Forward brickwork ladder (control < target).
        for i in range(0, n - 1, 2):
            gates.append(_g2("cnot", i, i + 1))
        for i in range(1, n - 1, 2):
            gates.append(_g2("cnot", i, i + 1))
    return gates


def gen_random_universal(rng, n, depth):
    gates = []
    for _ in range(depth):
        for q in range(n):
            name = UNIVERSAL_1Q[rng.below(len(UNIVERSAL_1Q))]
            if name in ("rx", "ry", "rz"):
                gates.append(_g1(name, q, rng.angle()))
            else:
                gates.append(_g1(name, q))
        for _ in range(max(1, n // 2)):
            a, b = _adjacent_forward(rng, n)
            gates.append(_g2(CLIFFORD_2Q[rng.below(len(CLIFFORD_2Q))], a, b))
    return gates


def gen_param_layers(rng, n, depth):
    gates = []
    for _ in range(depth):
        for q in range(n):
            gates.append(_g1("ry", q, rng.angle()))
            gates.append(_g1("rz", q, rng.angle()))
        for i in range(n - 1):
            gates.append(_g2("cnot", i, i + 1))
    return gates


def gen_reversed_2q(rng, n, depth):
    """Dedicated reversed-direction (control > target) adjacent-CNOT ladder,
    interleaved with single-qubit rotations. Isolates the two-qubit-gate
    transpose bug class into one identifiable family so the correct-path
    classes stay green and the known bug is caught and quarantined here."""
    gates = []
    for _ in range(depth):
        for q in range(n):
            axis = ("rx", "ry", "rz")[rng.below(3)]
            gates.append(_g1(axis, q, rng.angle()))
        for i in range(0, n - 1, 2):
            gates.append(_g2("cnot", i + 1, i))   # reversed: control > target
        for i in range(1, n - 1, 2):
            gates.append(_g2("cnot", i + 1, i))
    return gates


GENERATORS = {
    "clifford": gen_clifford,
    "rot_cnot_ladder": gen_rot_cnot_ladder,
    "random_universal": gen_random_universal,
    "param_layers": gen_param_layers,
    "reversed_2q": gen_reversed_2q,
}


def build_corpus(seed):
    circuits = []
    for cls in CLASSES:
        for n in QUBIT_COUNTS:
            for depth in DEPTHS:
                # Per-circuit stream is derived from the master seed and the
                # circuit coordinates so the whole corpus stays a pure function
                # of `seed` while each circuit is independently reproducible.
                stream_seed = (seed
                               ^ (hash_coord(cls) * 0x100000001B3)
                               ^ (n * 0x9E3779B1)
                               ^ (depth * 0x85EBCA77)) & MASK64
                rng = SplitMix64(stream_seed)
                gates = GENERATORS[cls](rng, n, depth)
                circuits.append({
                    "id": f"{cls}_n{n}_d{depth}_s0",
                    "class": cls,
                    "num_qubits": n,
                    "depth": depth,
                    "num_gates": len(gates),
                    "gates": gates,
                })
    return {"seed": seed, "count": len(circuits), "circuits": circuits}


def hash_coord(s):
    """Deterministic FNV-1a over a class name (avoids Python's salted hash())."""
    h = 0xCBF29CE484222325
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * 0x100000001B3) & MASK64
    return h


def emit_json(corpus, path):
    with open(path, "w") as f:
        json.dump(corpus, f, indent=1, sort_keys=True)
        f.write("\n")


def emit_header(corpus, path):
    seed = corpus["seed"]
    lines = []
    lines.append("/* AUTO-GENERATED by scripts/gen_circuit_corpus.py -- DO NOT EDIT.")
    lines.append(f" * seed={seed}  circuits={corpus['count']}")
    lines.append(" * Regenerate: scripts/gen_circuit_corpus.py --seed <S> "
                 "--out-dir tests/oracle/corpus")
    lines.append(" */")
    lines.append("#ifndef MOONLAB_ORACLE_CIRCUIT_CORPUS_H")
    lines.append("#define MOONLAB_ORACLE_CIRCUIT_CORPUS_H")
    lines.append('#include "../oracle_common.h"')
    lines.append("")
    for c in corpus["circuits"]:
        arr = "oracle_gates_" + c["id"]
        parts = []
        for gt in c["gates"]:
            q = gt["q"]
            q0 = q[0]
            q1 = q[1] if len(q) > 1 else -1
            parts.append('{"%s",%d,%d,%.17g}' % (gt["g"], q0, q1, gt["p"]))
        lines.append("static const oracle_gate_t %s[] = {" % arr)
        # Chunk the initializer so no single source line is pathologically long.
        for i in range(0, len(parts), 6):
            lines.append("  " + ", ".join(parts[i:i + 6]) + ",")
        if not parts:
            lines.append("  {\"i\",0,-1,0.0} /* empty circuit guard */")
        lines.append("};")
    lines.append("")
    lines.append("static const oracle_circuit_t oracle_corpus[] = {")
    for c in corpus["circuits"]:
        arr = "oracle_gates_" + c["id"]
        lines.append('  {"%s","%s",%d,%d,%d,%s},' % (
            c["id"], c["class"], c["num_qubits"], c["depth"],
            c["num_gates"], arr))
    lines.append("};")
    lines.append("static const int oracle_corpus_count = "
                 f"(int)(sizeof(oracle_corpus)/sizeof(oracle_corpus[0]));")
    lines.append(f"static const unsigned long long oracle_corpus_seed = {seed}ULL;")
    lines.append("")
    lines.append("#endif /* MOONLAB_ORACLE_CIRCUIT_CORPUS_H */")
    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260717,
                    help="master seed (default 20260717)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: tests/oracle/corpus "
                         "relative to the repo root)")
    args = ap.parse_args()

    if args.out_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(repo_root, "tests", "oracle", "corpus")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    corpus = build_corpus(args.seed)
    emit_json(corpus, os.path.join(out_dir, "circuit_corpus.json"))
    emit_header(corpus, os.path.join(out_dir, "circuit_corpus.h"))
    total_gates = sum(c["num_gates"] for c in corpus["circuits"])
    print(f"gen_circuit_corpus: seed={args.seed} circuits={corpus['count']} "
          f"gates={total_gates} -> {out_dir}")


if __name__ == "__main__":
    main()
