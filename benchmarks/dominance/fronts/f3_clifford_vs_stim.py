"""Front F3 -- Clifford / stabilizer throughput: MoonLab vs Stim.

Head-to-head between MoonLab's Aaronson-Gottesman tableau
(`src/backends/clifford/`, exposed as `moonlab.clifford.Clifford`) and
Stim's `stim.TableauSimulator` (Gidney 2021, the SOTA stabilizer
simulator).  Both engines drive the SAME reproducible random Clifford
circuit one gate at a time through their interactive tableau, which is
the apples-to-apples primitive: a single Clifford gate updating an
n-qubit tableau, called per gate from Python.

Primary metric: gate applications per second (higher is better).  Only
the gate-application loop is timed; tableau construction and terminal
measurement are excluded, symmetrically, for both engines.

Correctness gate (cross-engine, exact -- no loosened tolerance): a
stabilizer state's computational-basis measurement distribution is
uniform over an affine subspace of GF(2)^n.  Two engines implement the
same circuit semantics iff their sampled measurement outcomes span the
SAME affine subspace.  We sample many shots from each engine on a shared
circuit and verify, over GF(2), that
  dim(affine hull of Stim samples)
    == dim(affine hull of MoonLab samples)
    == dim(affine hull of the two sample sets stacked together).
Equal individual dims plus an equal combined dim means neither engine's
samples fall outside the other's hull: identical support, hence -- for
stabilizer states -- identical distribution.  A single wrong gate rule
(e.g. a swapped CNOT control/target, a bad S sign) moves the affine
subspace and trips the gate.  Mismatch => correctness_ok=False =>
verdict 'incorrect'.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness  # noqa: E402

INCUMBENT = "stim"

# Throughput sizes.  Depth per size is chosen so every run applies a
# comparable (>= ~8000) number of gates, keeping best-of timing stable
# without tuning the ratio (gates/s is depth-invariant).
_SIZES = (20, 50, 100)
_TARGET_GATES = 8000
_THROUGHPUT_SEED = 0x5DEECE66D
_REPEATS = 5
_WARMUP = 1

# Correctness circuit: small enough to sample its full affine support
# many times over, large enough that a semantic bug is very unlikely to
# leave the hull dimension accidentally unchanged.
_CORR_QUBITS = 18
_CORR_DEPTH = 36
_CORR_SHOTS = 1000
_CORR_SEED = 0xC0FFEE


# --------------------------------------------------------------------------
# Shared, reproducible workload
# --------------------------------------------------------------------------

def _build_circuit(n: int, depth: int, seed: int):
    """Return (ops, gate_count) for a random Clifford circuit.

    ops is a list of tuples driving both engines identically:
      ('h', q) | ('s', q) | ('cx', c, t)
    Each layer applies one random single-qubit gate (H or S) to every
    qubit, then a round of CNOTs over a random disjoint pairing with a
    random direction.  Fully determined by `seed`.
    """
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(depth):
        # Single-qubit sublayer: H or S on each qubit.
        picks = rng.integers(0, 2, size=n)
        for q in range(n):
            ops.append(("h", q) if picks[q] == 0 else ("s", q))
        # Two-qubit sublayer: random disjoint CNOT pairing.
        perm = rng.permutation(n)
        for i in range(0, n - 1, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            if rng.integers(0, 2) == 0:
                ops.append(("cx", a, b))
            else:
                ops.append(("cx", b, a))
    return ops, len(ops)


# --------------------------------------------------------------------------
# Engine drivers (identical loop structure for a fair comparison)
# --------------------------------------------------------------------------

def _apply_moonlab(sim, ops) -> None:
    for op in ops:
        k = op[0]
        if k == "h":
            sim.h(op[1])
        elif k == "s":
            sim.s(op[1])
        else:
            sim.cnot(op[1], op[2])


def _apply_stim(sim, ops) -> None:
    for op in ops:
        k = op[0]
        if k == "h":
            sim.h(op[1])
        elif k == "s":
            sim.s(op[1])
        else:
            sim.cnot(op[1], op[2])


# --------------------------------------------------------------------------
# GF(2) linear algebra for the correctness gate
# --------------------------------------------------------------------------

def _gf2_affine_dim(rows: np.ndarray) -> int:
    """Affine-hull dimension of a set of GF(2) points.

    Equal to the GF(2) row rank of (rows - rows[0]).  `rows` is an
    (S, n) uint8 array of 0/1.
    """
    if rows.shape[0] <= 1:
        return 0
    diff = (rows[1:] ^ rows[0]).astype(np.uint8).copy()
    return _gf2_rank(diff)


def _gf2_rank(mat: np.ndarray) -> int:
    """GF(2) rank via Gaussian elimination on a copy of `mat` (uint8)."""
    m = mat.copy()
    n_rows, n_cols = m.shape
    rank = 0
    for col in range(n_cols):
        pivot = -1
        for r in range(rank, n_rows):
            if m[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        mask = m[:, col].astype(bool).copy()
        mask[rank] = False
        m[mask] ^= m[rank]
        rank += 1
        if rank == n_rows:
            break
    return rank


# --------------------------------------------------------------------------
# Sampling for the correctness gate
# --------------------------------------------------------------------------

def _sample_moonlab(CliffordCls, ops, n: int, shots: int, seed: int) -> np.ndarray:
    out = np.zeros((shots, n), dtype=np.uint8)
    for s in range(shots):
        sim = CliffordCls(n, seed=(seed ^ (s * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF)
        _apply_moonlab(sim, ops)
        for q in range(n):
            out[s, q] = sim.measure(q)[0]
    return out


def _sample_stim(stim, ops, n: int, shots: int, seed: int) -> np.ndarray:
    out = np.zeros((shots, n), dtype=np.uint8)
    for s in range(shots):
        sim = stim.TableauSimulator(seed=(seed + s) & 0x7FFFFFFF)
        _apply_stim(sim, ops)
        for q in range(n):
            out[s, q] = 1 if sim.measure(q) else 0
    return out


def _correctness_gate(CliffordCls, stim):
    """Cross-engine affine-hull agreement. Returns (ok, detail, extra)."""
    ops, _ = _build_circuit(_CORR_QUBITS, _CORR_DEPTH, _CORR_SEED)
    m_samples = _sample_moonlab(CliffordCls, ops, _CORR_QUBITS, _CORR_SHOTS, _CORR_SEED)
    s_samples = _sample_stim(stim, ops, _CORR_QUBITS, _CORR_SHOTS, _CORR_SEED)

    d_moon = _gf2_affine_dim(m_samples)
    d_stim = _gf2_affine_dim(s_samples)
    combined = np.vstack([s_samples, m_samples])
    d_comb = _gf2_affine_dim(combined)

    ok = (d_moon == d_stim == d_comb)
    detail = (
        f"cross-engine affine-hull agreement on a {_CORR_QUBITS}-qubit, "
        f"depth-{_CORR_DEPTH} random Clifford circuit, {_CORR_SHOTS} shots/engine. "
        f"GF(2) affine dim: stim={d_stim}, moonlab={d_moon}, "
        f"stacked={d_comb}. Equal all three => identical measurement support "
        f"=> identical stabilizer distribution."
    )
    extra = {
        "corr_qubits": _CORR_QUBITS,
        "corr_depth": _CORR_DEPTH,
        "corr_shots": _CORR_SHOTS,
        "affine_dim_stim": d_stim,
        "affine_dim_moonlab": d_moon,
        "affine_dim_stacked": d_comb,
    }
    return ok, detail, extra


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

def benchmark():
    """Run F3 for every size; return a list of finalized BenchmarkResults."""
    from moonlab.clifford import Clifford
    import stim

    ok, corr_detail, corr_extra = _correctness_gate(Clifford, stim)

    results = []
    for n in _SIZES:
        depth = max(_CORR_DEPTH, -(-_TARGET_GATES // (max(1, (3 * n) // 2))))
        ops, gate_count = _build_circuit(n, depth, _THROUGHPUT_SEED + n)

        # MoonLab: construct once (excluded), time only the gate loop.
        m_sim = Clifford(n, seed=1)
        m_best, _ = harness.timed(lambda: _apply_moonlab(m_sim, ops),
                                  repeats=_REPEATS, warmup=_WARMUP)
        m_rate = gate_count / m_best

        # Stim: identical protocol.
        s_sim = stim.TableauSimulator()
        s_best, _ = harness.timed(lambda: _apply_stim(s_sim, ops),
                                  repeats=_REPEATS, warmup=_WARMUP)
        s_rate = gate_count / s_best

        r = harness.BenchmarkResult(
            front="F3",
            name=f"clifford_tableau_throughput_n{n}",
            incumbent=INCUMBENT,
            metric="gate_applications_per_second",
            higher_is_better=True,
            moonlab_value=m_rate,
            incumbent_value=s_rate,
            correctness_ok=ok,
            correctness_detail=corr_detail,
            workload=(
                f"random Clifford circuit, n={n} qubits, depth={depth} layers "
                f"of per-qubit {{H,S}} + random disjoint CNOT round, "
                f"{gate_count} total gate applications (seeded, shared by both "
                f"engines); interactive tableau, one gate per call"
            ),
            repeats=_REPEATS,
            notes=(
                "Both engines drive the identical seeded gate list through their "
                "interactive stabilizer tableau (moonlab.clifford.Clifford vs "
                "stim.TableauSimulator), one Clifford gate per Python call. "
                "Construction and terminal measurement excluded from timing. "
                "Stim additionally offers a batched stim.Circuit sampler path "
                "not exercised by this per-gate tableau metric."
            ),
            extra={
                "num_qubits": n,
                "depth_layers": depth,
                "gate_count": gate_count,
                "moonlab_best_seconds": m_best,
                "stim_best_seconds": s_best,
                **corr_extra,
            },
        ).finalize()
        results.append(r)

    return results


if __name__ == "__main__":
    for res in benchmark():
        print(res.to_json())
