"""
Python-bindings smoke test.

Verifies that the ctypes bridge loads the built libquantumsim, creates
a 2-qubit quantum state, applies H then CNOT to build |Phi+>, and
reads back the expected amplitudes [1/sqrt(2), 0, 0, 1/sqrt(2)].

The test is fast (sub-second) and only exercises `moonlab.core` --
`moonlab.algorithms` is import-resilient in __init__.py because several
of its C symbols are not yet exported; that gap is tracked as a Phase
1G item.
"""

import math
import sys


def main() -> int:
    from moonlab import QuantumState  # noqa: E402

    s = QuantumState(num_qubits=2)
    s.h(0)
    s.cnot(0, 1)

    sv = s.get_statevector()
    if len(sv) != 4:
        print(f"  [FAIL] expected 4 amplitudes, got {len(sv)}")
        return 1

    expected = [1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)]
    tol = 1e-12
    ok = True
    for i, (got, want) in enumerate(zip(sv, expected)):
        diff = abs(complex(got) - complex(want))
        marker = "OK  " if diff < tol else "FAIL"
        print(f"  [{marker}] amp |{i:02b}> got={got} want={want} diff={diff:.3e}")
        if diff >= tol:
            ok = False

    probs = s.probabilities()
    if len(probs) != 4:
        print(f"  [FAIL] probabilities length {len(probs)} != 4")
        return 1

    p00, p11 = probs[0], probs[3]
    if (math.isclose(p00, 0.5, abs_tol=1e-12)
            and math.isclose(p11, 0.5, abs_tol=1e-12)):
        print("  [OK  ] P(|00>)=P(|11>)=0.5, off-diagonals are zero")
    else:
        print(f"  [FAIL] P(|00>)={p00} P(|11>)={p11}, expected 0.5 each")
        ok = False

    # Quantum Volume binding: width=3 with 20 trials passes on a
    # noiseless simulator (mean HOP ~ 0.85).
    from moonlab import quantum_volume
    qv = quantum_volume(width=3, num_trials=20, seed=0xC0FFEE)
    if qv.mean_hop > 0.75 and qv.passed:
        print(f"  [OK  ] QV w=3 mean_hop={qv.mean_hop:.3f} passed={qv.passed}")
    else:
        print(f"  [FAIL] QV w=3 mean_hop={qv.mean_hop:.3f} passed={qv.passed}")
        ok = False

    # Clifford binding: 100-qubit GHZ, measure qubit 0 then verify all
    # remaining qubits collapse to the same outcome. Impossible on the
    # dense simulator (2^100 states); trivial on the tableau.
    from moonlab import Clifford
    c = Clifford(num_qubits=100, seed=0xBADDCAFE)
    c.h(0)
    for q in range(1, 100):
        c.cnot(0, q)
    first, kind0 = c.measure(0)
    consistent = all(c.measure(q)[0] == first for q in range(1, 100))
    if consistent and kind0 == 1:
        print(f"  [OK  ] Clifford GHZ-100 first_outcome={first} "
              "all-or-nothing consistent")
    else:
        print(f"  [FAIL] Clifford GHZ-100 first={first} kind0={kind0} "
              f"consistent={consistent}")
        ok = False

    # moonlab.algorithms should be importable on a full build. If it
    # falls back to False here, the user's libquantumsim is partial.
    import moonlab
    if getattr(moonlab, "_ALGO_AVAILABLE", False):
        print("  [OK  ] moonlab.algorithms imported (VQE/QAOA/Grover/BellTest)")
    else:
        print("  [FAIL] moonlab.algorithms did not import -- partial build?")
        ok = False

    # Chern KPM binding: topological QWZ on a 12 x 12 lattice. Bulk
    # mean should be ~ +1.0 for m=-1 in the topological phase.
    from moonlab import ChernKPM
    sys_ = ChernKPM(L=12, m=-1.0, n_cheby=100)
    mp = sys_.bulk_map(4, 8)
    mean = float(mp.mean())
    if abs(mean - 1.0) < 0.25:
        print(f"  [OK  ] ChernKPM L=12 bulk mean={mean:.3f} (~+1)")
    else:
        print(f"  [FAIL] ChernKPM L=12 bulk mean={mean:.3f}, expected +1")
        ok = False

    # Momentum-space Chern via the stable ABI surface.
    from moonlab import qwz_chern
    c_top = qwz_chern(m=1.0, N=32)   # expect -1
    c_triv = qwz_chern(m=3.0, N=32)  # expect 0
    if c_top == -1 and c_triv == 0:
        print(f"  [OK  ] qwz_chern m=1 -> {c_top}, m=3 -> {c_triv}")
    else:
        print(f"  [FAIL] qwz_chern m=1 -> {c_top} (expected -1), "
              f"m=3 -> {c_triv} (expected 0)")
        ok = False

    # SSH winding number.
    from moonlab import ssh_winding
    w_top = ssh_winding(1.0, 2.0)   # expect +1
    w_triv = ssh_winding(2.0, 1.0)  # expect 0
    if w_top == 1 and w_triv == 0:
        print(f"  [OK  ] ssh_winding topological={w_top} trivial={w_triv}")
    else:
        print(f"  [FAIL] ssh_winding topological={w_top} (expected 1) "
              f"trivial={w_triv} (expected 0)")
        ok = False

    # Berry curvature grid: shape is (N, N), sum should match Chern * 2pi.
    import numpy as np
    from moonlab import berry_grid_qwz
    bg = berry_grid_qwz(m=-1.0, N=32)
    chern_integrated = bg.sum() / (2 * np.pi)
    if bg.shape == (32, 32) and abs(chern_integrated - 1.0) < 1e-3:
        print(f"  [OK  ] berry_grid_qwz shape={bg.shape} "
              f"sum/2pi={chern_integrated:.4f}")
    else:
        print(f"  [FAIL] berry_grid_qwz shape={bg.shape} "
              f"sum/2pi={chern_integrated} (expected ~1)")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
