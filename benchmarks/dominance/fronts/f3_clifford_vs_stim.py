"""F3 front: MoonLab bit-packed Clifford tableau vs Stim.

Two gates, both mandatory for a passing front:

  1. Cross-engine correctness (correctness_ok): after a random Clifford
     circuit, both engines sample all qubits and must agree on the canonical
     GF(2) affine hull of the samples (the stabilizer support). A faster wrong
     simulator fails here.

  2. Gate-application throughput: the same logical circuit is driven per-gate
     through each engine's public single-gate entry point (ctypes for MoonLab,
     stim.TableauSimulator methods for the incumbent), reporting gates/second
     and the MoonLab/Stim ratio. verdict is 'lead' at ratio > 1.1, 'parity'
     above 0.9, else 'behind'.

Run via ``benchmarks/dominance/run_dominance.py`` with MOONLAB_LIB_DIR and
PYTHONPATH pointing at a built quantumsim + the python binding.
"""
import ctypes
import random
import time

import stim
from moonlab.core import _lib
from moonlab.clifford import Clifford

SIZES = [20, 50, 100, 200, 500]


# --------------------------------------------------------------------------
# Correctness: canonical GF(2) affine hull agreement.
# --------------------------------------------------------------------------
def _affine_hull(samples):
    base = samples[0]
    basis = []
    for s in samples:
        cur = s ^ base
        for b in basis:
            piv = b.bit_length() - 1
            if (cur >> piv) & 1:
                cur ^= b
        if cur:
            basis.append(cur)
            basis.sort(key=lambda v: v.bit_length(), reverse=True)
    for i in range(len(basis)):
        piv = basis[i].bit_length() - 1
        for j in range(len(basis)):
            if i != j and (basis[j] >> piv) & 1:
                basis[j] ^= basis[i]
    cbase = base
    for b in basis:
        piv = b.bit_length() - 1
        if (cbase >> piv) & 1:
            cbase ^= b
    return cbase, frozenset(basis)


def _gen_circuit(n, depth, rng):
    prog = []
    for _ in range(depth):
        if rng.random() < 0.5:
            a, b = rng.sample(range(n), 2)
            prog.append((rng.choice(["cnot", "cz", "swap"]), a, b))
        else:
            prog.append((rng.choice(["h", "s", "s_dag", "x", "y", "z"]),
                         rng.randrange(n)))
    return prog


def _moonlab_hull(n, gates, shots, seed):
    S = []
    for s in range(shots):
        c = Clifford(n, seed=seed + s * 2654435761)
        for g in gates:
            getattr(c, g[0])(*g[1:])
        bits = 0
        for q in range(n):
            if c.measure(q)[0]:
                bits |= 1 << q
        S.append(bits)
    return _affine_hull(S)


_STIM_GATE = {"h": "H", "s": "S", "s_dag": "S_DAG", "x": "X", "y": "Y",
              "z": "Z", "cnot": "CNOT", "cz": "CZ", "swap": "SWAP"}


def _stim_hull(n, gates, shots, seed):
    circ = stim.Circuit()
    for g in gates:
        circ.append(_STIM_GATE[g[0]], list(g[1:]))
    for q in range(n):
        circ.append("M", [q])
    res = circ.compile_sampler(seed=seed).sample(shots)
    S = []
    for row in res:
        bits = 0
        for q in range(n):
            if row[q]:
                bits |= 1 << q
        S.append(bits)
    return _affine_hull(S)


def check_correctness(rng):
    ok = True
    per_size = {}
    for n in SIZES:
        shots = max(4 * n, 600)
        gates = _gen_circuit(n, 3 * n, rng)
        ml = _moonlab_hull(n, gates, shots, seed=rng.randrange(1 << 60))
        st = _stim_hull(n, gates, shots, seed=rng.randrange(1 << 60))
        agree = ml == st
        per_size[n] = agree
        ok = ok and agree
    return ok, per_size


# --------------------------------------------------------------------------
# Throughput: per-gate driven gates/second.
# --------------------------------------------------------------------------
def _bind_gate(name, two):
    f = getattr(_lib, f"clifford_{name}")
    f.restype = ctypes.c_int
    f.argtypes = ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t] if two
                  else [ctypes.c_void_p, ctypes.c_size_t])
    return f


def _throughput_moonlab(n, prog, reps):
    _lib.clifford_tableau_create.restype = ctypes.c_void_p
    _lib.clifford_tableau_create.argtypes = [ctypes.c_size_t]
    _lib.clifford_tableau_free.argtypes = [ctypes.c_void_p]
    h = _bind_gate("h", False)
    s = _bind_gate("s", False)
    cn = _bind_gate("cnot", True)
    best = 0.0
    for _ in range(reps):
        handle = _lib.clifford_tableau_create(n)
        calls = []
        for g in prog:
            if g[0] == "cnot":
                calls.append((cn, handle, g[1], g[2]))
            elif g[0] == "h":
                calls.append((h, handle, g[1]))
            else:
                calls.append((s, handle, g[1]))
        t0 = time.perf_counter()
        for c in calls:
            c[0](*c[1:])
        dt = time.perf_counter() - t0
        _lib.clifford_tableau_free(handle)
        best = max(best, len(prog) / dt)
    return best


def _throughput_stim(n, prog, reps):
    best = 0.0
    for _ in range(reps):
        sim = stim.TableauSimulator()
        sim.set_num_qubits(n)
        h, s, cn = sim.h, sim.s, sim.cnot
        calls = []
        for g in prog:
            if g[0] == "cnot":
                calls.append((cn, g[1], g[2]))
            elif g[0] == "h":
                calls.append((h, g[1]))
            else:
                calls.append((s, g[1]))
        t0 = time.perf_counter()
        for c in calls:
            c[0](*c[1:])
        dt = time.perf_counter() - t0
        best = max(best, len(prog) / dt)
    return best


def _gate_stream(n, ngates, seed):
    rng = random.Random(seed)
    prog = []
    for _ in range(ngates):
        r = rng.random()
        if r < 0.5:
            a, b = rng.sample(range(n), 2)
            prog.append(("cnot", a, b))
        elif r < 0.75:
            prog.append(("h", rng.randrange(n)))
        else:
            prog.append(("s", rng.randrange(n)))
    return prog


def measure_throughput(ngates=300000, reps=3):
    rows = []
    for n in SIZES:
        prog = _gate_stream(n, ngates, seed=1000 + n)
        ml = _throughput_moonlab(n, prog, reps)
        st = _throughput_stim(n, prog, reps)
        ratio = ml / st
        verdict = "lead" if ratio > 1.1 else ("parity" if ratio > 0.9 else "behind")
        rows.append({"n": n, "moonlab_gps": ml, "stim_gps": st,
                     "ratio": ratio, "verdict": verdict})
    return rows


def run():
    rng = random.Random(0xF3)
    correctness_ok, per_size = check_correctness(rng)
    throughput = measure_throughput()
    return {"correctness_ok": correctness_ok,
            "correctness_per_size": per_size,
            "throughput": throughput}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str))
