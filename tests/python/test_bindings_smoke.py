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

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
