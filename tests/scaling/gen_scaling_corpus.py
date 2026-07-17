#!/usr/bin/env python3
"""Seeded corpus generator for the Moonlab SCALING differential.

This is a sibling of scripts/gen_diff_corpus.py, specialised for pushing scale
and circuit diversity to hunt for scale-dependent cross-backend divergences.
The main differences from the cross-diff corpus:

  * a much wider family set aimed at the SWAP-network / long-range 2q path
    (non-adjacent controlled gates), QFT, GHZ/W, brickwork MPS-friendly local
    circuits, and forward random-universal circuits;
  * every family here is FORWARD-only (control < target on every cx) and
    contains NO Toffoli, so a tn_mps divergence is guaranteed NOT to be the
    known reversed-CNOT transpose bug -- it isolates other causes;
  * the sweep spans n in {10,12,14,16} x depth {8,32,128} plus a set of small-n
    (n=6,8) long-range cases that sit BELOW the known n>=10 forward-circuit
    tn bug, so any tn divergence there is a genuinely NEW finding.

Like the cross-diff generator this is a PURE FUNCTION OF ITS SEED: a
self-contained splitmix64 PRNG, never time / os.urandom / random. The same seed
reproduces the corpus and the pinned numpy reference byte-for-byte.

The reference oracle is a tiny pure-numpy statevector simulator with nothing to
do with Moonlab, so agreement is absolute. It is affordable up to n=16 (65536
amplitudes). For the large-n Clifford structure sweep (n=50,100) there is no
statevector reference; that is handled analytically by scaling_stab.c.

Convention: little-endian, qubit 0 = least-significant bit. Basis index b has
bit q equal to the value of qubit q. Matches the dense C backend and the
Clifford tableau. The plain tn_mps backend is big-endian (site 0 = MSB); its
consumer in scaling_diff.c bit-reverses the statevector before comparing.

Corpus interchange format is the flat whitespace-delimited CASE/G/PROB/EXPZ/
EXPZZ/ENDCASE mirror shared with scripts/gen_diff_corpus.py so the C driver
needs no JSON parser.

Usage:
  gen_scaling_corpus.py --out-dir DIR [--seed N]
                        [--qubits 10,12,14,16] [--depths 8,32,128]
                        [--small-longrange] [--self-test]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.stderr.write("gen_scaling_corpus: numpy is required for the reference oracle\n")
    raise

CORPUS_VERSION = 1
DEFAULT_SEED = 0x5CA11B16  # "SCAL1B16"
_MASK64 = (1 << 64) - 1


# --------------------------------------------------------------------------
# Deterministic PRNG: splitmix64. Pure function of the seed.
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
        return (self.next_u64() >> 11) / float(1 << 53)

    def below(self, bound: int) -> int:
        return int(self.next_u64() % bound)

    def angle(self) -> float:
        # A "generic" angle away from Clifford multiples of pi/2.
        return (self.unit() * 2.0 - 1.0) * math.pi


def stable_hash(s: str) -> int:
    """Deterministic FNV-1a 64-bit hash of a string (NOT Python's randomized
    str hash, which varies with PYTHONHASHSEED)."""
    h = 0xCBF29CE484222325
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * 0x100000001B3) & _MASK64
    return h


def derive_seed(master: int, *tags: int) -> int:
    s = master & _MASK64
    for t in tags:
        s = (s ^ ((t + 0x9E3779B97F4A7C15) & _MASK64)) & _MASK64
        s = SplitMix64(s).next_u64()
    return s


# --------------------------------------------------------------------------
# Pure-numpy statevector reference (little-endian, qubit0 = LSB).
# Independent of Moonlab.
# --------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
_TDG = np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex)


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def _p(t):
    return np.array([[1, 0], [0, np.exp(1j * t)]], dtype=complex)


_ONEQ = {"h": _H, "x": _X, "y": _Y, "z": _Z, "s": _S, "sdg": _SDG, "t": _T, "tdg": _TDG}
_ONEQ_PARAM = {"rx": _rx, "ry": _ry, "rz": _rz, "p": _p}


class StateVec:
    """Little-endian statevector: axis for qubit q is (n-1-q) after reshape."""

    def __init__(self, n: int):
        self.n = n
        self.psi = np.zeros(1 << n, dtype=complex)
        self.psi[0] = 1.0

    def _axis(self, q: int) -> int:
        return self.n - 1 - q

    def apply_1q(self, q: int, G: np.ndarray) -> None:
        t = self.psi.reshape([2] * self.n)
        a = self._axis(q)
        t = np.tensordot(G, t, axes=([1], [a]))
        t = np.moveaxis(t, 0, a)
        self.psi = t.reshape(1 << self.n)

    def apply_2q(self, qa: int, qb: int, G4: np.ndarray) -> None:
        # G4 is 4x4 in basis index 2*bit_qa + bit_qb (qa is the high bit).
        G = G4.reshape(2, 2, 2, 2)  # [out_a, out_b, in_a, in_b]
        t = self.psi.reshape([2] * self.n)
        aa, ab = self._axis(qa), self._axis(qb)
        t = np.tensordot(G, t, axes=([2, 3], [aa, ab]))
        # result leading axes: out_a (0), out_b (1); move them back
        t = np.moveaxis(t, [0, 1], [aa, ab])
        self.psi = t.reshape(1 << self.n)

    # named 2q gates
    def cx(self, c, tq):
        G = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        self.apply_2q(c, tq, G)

    def cz(self, a, b):
        G = np.diag([1, 1, 1, -1]).astype(complex)
        self.apply_2q(a, b, G)

    def swap(self, a, b):
        G = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        self.apply_2q(a, b, G)

    def cp(self, a, b, theta):
        G = np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)
        self.apply_2q(a, b, G)

    def apply_gate(self, name, q, angle):
        if name in _ONEQ:
            self.apply_1q(q[0], _ONEQ[name])
        elif name in _ONEQ_PARAM:
            self.apply_1q(q[0], _ONEQ_PARAM[name](angle))
        elif name == "cx":
            self.cx(q[0], q[1])
        elif name == "cz":
            self.cz(q[0], q[1])
        elif name == "swap":
            self.swap(q[0], q[1])
        elif name == "cp":
            self.cp(q[0], q[1], angle)
        else:
            raise ValueError(f"unknown gate {name}")

    def probabilities(self):
        return np.abs(self.psi) ** 2

    def exp_z(self, q):
        p = self.probabilities()
        idx = np.arange(1 << self.n)
        sign = np.where((idx >> q) & 1, -1.0, 1.0)
        return float(np.sum(sign * p))

    def exp_zz(self, a, b):
        p = self.probabilities()
        idx = np.arange(1 << self.n)
        sa = np.where((idx >> a) & 1, -1.0, 1.0)
        sb = np.where((idx >> b) & 1, -1.0, 1.0)
        return float(np.sum(sa * sb * p))


# --------------------------------------------------------------------------
# Gate emit helpers (build the interchange gate list).
# --------------------------------------------------------------------------
def g1(name, q, angle=0.0):
    return {"name": name, "qubits": [q], "angle": float(angle)}


def g2(name, a, b, angle=0.0):
    return {"name": name, "qubits": [a, b], "angle": float(angle)}


# --------------------------------------------------------------------------
# Circuit families. Each returns (gates, clifford_only).
# All FORWARD (control < target), NO Toffoli.
# --------------------------------------------------------------------------
def fam_clifford_local(rng, n, depth):
    """Adjacent-only forward Clifford (h, s, cx, cz on |i-j|=1)."""
    gates = []
    for _ in range(depth):
        for q in range(n):
            r = rng.below(3)
            if r == 0:
                gates.append(g1("h", q))
            elif r == 1:
                gates.append(g1("s", q))
        for q in range(0, n - 1, 2):
            gates.append(g2("cx" if rng.below(2) else "cz", q, q + 1))
        for q in range(1, n - 1, 2):
            gates.append(g2("cx" if rng.below(2) else "cz", q, q + 1))
    return gates, True


def fam_clifford_longrange(rng, n, depth):
    """Forward Clifford with NON-adjacent cx/cz (|i-j|>=2): SWAP-network path."""
    gates = []
    for _ in range(depth):
        for q in range(n):
            if rng.below(2):
                gates.append(g1("h", q))
        # non-adjacent forward pairs
        for _ in range(max(1, n // 3)):
            a = rng.below(n)
            span = 2 + rng.below(max(1, n - 2))
            b = a + span
            if b >= n:
                continue
            gates.append(g2("cx" if rng.below(2) else "cz", a, b))
    return gates, True


def fam_rot_cz_ladder(rng, n, depth):
    """ry/rz + forward ADJACENT cz ladder (the known n>=10 forward family)."""
    gates = []
    for _ in range(depth):
        for q in range(n):
            gates.append(g1("ry", q, rng.angle()))
            gates.append(g1("rz", q, rng.angle()))
        for q in range(n - 1):
            gates.append(g2("cz", q, q + 1))
    return gates, False


def fam_rot_longrange(rng, n, depth):
    """ry/rz + NON-adjacent forward cx/cz. SWAP-network under rotations."""
    gates = []
    for _ in range(depth):
        for q in range(n):
            gates.append(g1("ry", q, rng.angle()))
        for _ in range(max(1, n // 3)):
            a = rng.below(n - 2)
            b = a + 2 + rng.below(max(1, n - a - 2))
            if b >= n:
                b = n - 1
            if b <= a:
                continue
            gates.append(g2("cx" if rng.below(2) else "cz", a, b))
    return gates, False


def fam_random_universal(rng, n, depth):
    """Forward random universal: rx/ry/rz + forward cx (adjacent+non-adjacent)."""
    gates = []
    for _ in range(depth):
        for q in range(n):
            r = rng.below(3)
            gates.append(g1(("rx", "ry", "rz")[r], q, rng.angle()))
        for _ in range(max(1, n // 2)):
            a = rng.below(n - 1)
            b = a + 1 + rng.below(n - a - 1)  # forward, possibly non-adjacent
            if b <= a or b >= n:
                continue
            if rng.below(2):
                gates.append(g2("cx", a, b))
            else:
                gates.append(g2("cp", a, b, rng.angle()))
    return gates, False


def fam_qft(rng, n, depth):
    """QFT: H + controlled-phase over ALL pairs (heavily non-adjacent)."""
    gates = []
    for i in range(n):
        gates.append(g1("h", i))
        for j in range(i + 1, n):
            theta = math.pi / (1 << (j - i))
            gates.append(g2("cp", i, j, theta))
    # bit-reversal swaps (part of canonical QFT)
    for i in range(n // 2):
        gates.append(g2("swap", i, n - 1 - i))
    return gates, False


def fam_ghz_chain(rng, n, depth):
    """GHZ via H(0) + adjacent CNOT chain."""
    gates = [g1("h", 0)]
    for q in range(n - 1):
        gates.append(g2("cx", q, q + 1))
    return gates, True


def fam_ghz_star(rng, n, depth):
    """GHZ via H(0) + CNOT(0, k) for all k -- every control-target is LONG-range."""
    gates = [g1("h", 0)]
    for k in range(1, n):
        gates.append(g2("cx", 0, k))
    return gates, True


def fam_w_local(rng, n, depth):
    """W-like state via a cascade of ry + adjacent cx (not exact W, but a
    fixed structured superposition with graded amplitudes)."""
    gates = [g1("x", 0)]
    for q in range(n - 1):
        theta = 2.0 * math.acos(math.sqrt((n - 1 - q) / (n - q)))
        gates.append(g1("ry", q + 1, theta))
        gates.append(g2("cz", q, q + 1))
        gates.append(g1("ry", q + 1, -theta))
        gates.append(g2("cx", q, q + 1))
    return gates, False


def fam_mps_brick(rng, n, depth):
    """MPS-friendly brickwork of LOCAL 2q blocks (rz-cx-rz on adjacent pairs)."""
    gates = []
    for d in range(depth):
        start = d % 2
        for q in range(start, n - 1, 2):
            gates.append(g1("ry", q, rng.angle()))
            gates.append(g1("ry", q + 1, rng.angle()))
            gates.append(g2("cx", q, q + 1))
            gates.append(g1("rz", q + 1, rng.angle()))
            gates.append(g2("cx", q, q + 1))
    return gates, False


FAMILIES = {
    "clifford_local": fam_clifford_local,
    "clifford_longrange": fam_clifford_longrange,
    "rot_cz_ladder": fam_rot_cz_ladder,
    "rot_longrange": fam_rot_longrange,
    "random_universal": fam_random_universal,
    "qft": fam_qft,
    "ghz_chain": fam_ghz_chain,
    "ghz_star": fam_ghz_star,
    "w_local": fam_w_local,
    "mps_brick": fam_mps_brick,
}


def zz_pairs(n):
    """A handful of Z_iZ_j observable pairs: adjacent + long-range + endpoints."""
    pairs = []
    if n >= 2:
        pairs.append((0, 1))
        pairs.append((0, n - 1))
        pairs.append((n // 2 - 1 if n >= 4 else 0, n // 2))
        if n >= 4:
            pairs.append((1, n - 2))
    # dedup
    seen, out = set(), []
    for (a, b) in pairs:
        a, b = min(a, b), max(a, b)
        if a != b and (a, b) not in seen:
            seen.add((a, b))
            out.append((a, b))
    return out


def build_case(master_seed, cls, n, depth):
    case_seed = derive_seed(master_seed, stable_hash(cls), n, depth)
    rng = SplitMix64(case_seed)
    gates, clifford_only = FAMILIES[cls](rng, n, depth)
    sim = StateVec(n)
    for g in gates:
        sim.apply_gate(g["name"], g["qubits"], g.get("angle", 0.0))
    probs = sim.probabilities()
    exp_z = [sim.exp_z(q) for q in range(n)]
    pairs = zz_pairs(n)
    exp_zz = [(a, b, sim.exp_zz(a, b)) for (a, b) in pairs]
    short = f"{case_seed & 0xffff:04x}"
    return {
        "id": f"{cls}_n{n:02d}_d{depth:03d}_s{short}",
        "class": cls,
        "num_qubits": n,
        "depth": depth,
        "clifford_only": clifford_only,
        "seed": case_seed,
        "gates": gates,
        "reference": {
            "probabilities": [float(p) for p in probs],
            "exp_z": exp_z,
            "exp_zz": [[a, b, float(v)] for (a, b, v) in exp_zz],
        },
    }


def build_corpus(master_seed, qubits, depths, small_longrange):
    cases = []
    for n in qubits:
        for depth in depths:
            for cls in FAMILIES:
                # QFT/GHZ/W ignore depth structurally; emit once at the first depth.
                if cls in ("qft", "ghz_chain", "ghz_star", "w_local") and depth != depths[0]:
                    continue
                cases.append(build_case(master_seed, cls, n, depth))
    if small_longrange:
        # Small-n (below the known n>=10 forward tn bug) long-range / SWAP-network
        # cases: any tn divergence here is a genuinely NEW finding.
        for n in (4, 6, 8):
            for depth in (4, 16):
                for cls in ("clifford_longrange", "rot_longrange", "ghz_star", "qft"):
                    if cls == "qft" and depth != 4:
                        continue
                    cases.append(build_case(master_seed, cls, n, depth))
    return {"version": CORPUS_VERSION, "seed": master_seed, "num_cases": len(cases), "cases": cases}


def write_txt(corpus, path):
    lines = [f"CORPUS {corpus['version']} {corpus['seed']} {corpus['num_cases']}"]
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
            lines.append(f"G {g['name']} {q0} {q1} {q2} {g.get('angle', 0.0)!r}")
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


def self_test(master_seed):
    # 1. numpy engine sanity: GHZ parity, Bell, QFT unitarity.
    ghz = build_case(master_seed, "ghz_chain", 6, 8)
    for (a, b, v) in ghz["reference"]["exp_zz"]:
        if abs(v - 1.0) > 1e-9:
            sys.stderr.write(f"SELF-TEST FAIL: GHZ <Z{a}Z{b}>={v} != 1\n")
            return 1
    star = build_case(master_seed, "ghz_star", 6, 8)
    for (a, b, v) in star["reference"]["exp_zz"]:
        if abs(v - 1.0) > 1e-9:
            sys.stderr.write(f"SELF-TEST FAIL: GHZ-star <Z{a}Z{b}>={v} != 1\n")
            return 1
    # 2. probabilities normalised + non-negative for every family.
    corpus = build_corpus(master_seed, [8], [8], True)
    for c in corpus["cases"]:
        p = c["reference"]["probabilities"]
        s = sum(p)
        if abs(s - 1.0) > 1e-9:
            sys.stderr.write(f"SELF-TEST FAIL: {c['id']} probs sum to {s}\n")
            return 1
        if min(p) < -1e-15:
            sys.stderr.write(f"SELF-TEST FAIL: {c['id']} negative prob\n")
            return 1
    # 3. determinism: same seed -> identical corpus.
    a = build_corpus(master_seed, [8], [8], False)
    b = build_corpus(master_seed, [8], [8], False)
    if [c["reference"]["probabilities"] for c in a["cases"]] != \
       [c["reference"]["probabilities"] for c in b["cases"]]:
        sys.stderr.write("SELF-TEST FAIL: generator not deterministic\n")
        return 1
    # 4. QFT of |0..0> is uniform superposition (all probs equal).
    q = build_case(master_seed, "qft", 5, 4)
    p = q["reference"]["probabilities"]
    if max(abs(pi - 1.0 / len(p)) for pi in p) > 1e-9:
        sys.stderr.write("SELF-TEST FAIL: QFT|0> not uniform\n")
        return 1
    sys.stderr.write("gen_scaling_corpus self-test: PASS\n")
    return 0


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=DEFAULT_SEED)
    ap.add_argument("--qubits", type=parse_int_list, default=[10, 12, 14, 16])
    ap.add_argument("--depths", type=parse_int_list, default=[8, 32, 128])
    ap.add_argument("--small-longrange", action="store_true",
                    help="add small-n (4,6,8) long-range cases below the known tn bug")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args.seed)

    if not args.out_dir:
        ap.error("--out-dir is required (unless --self-test)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(args.seed, args.qubits, args.depths, args.small_longrange)
    write_txt(corpus, args.out_dir / "corpus.txt")
    sys.stderr.write(
        f"gen_scaling_corpus: {corpus['num_cases']} cases "
        f"(seed=0x{args.seed:X}, qubits={args.qubits}, depths={args.depths}, "
        f"small_longrange={args.small_longrange}) -> {args.out_dir/'corpus.txt'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
