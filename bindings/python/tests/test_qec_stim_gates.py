"""Exhaustive proof of the Stim gate table moonlab lowers.

Every Clifford gate name ``src/qec/stim_circuit.c`` accepts is driven
through the SHIPPED path -- parse a one-instruction circuit, lower it to
the Pauli-frame op list, replay that op list on moonlab's
Aaronson-Gottesman tableau -- and the resulting tableau is compared
element by element, signs included, against
``stim.Tableau.from_named_gate(name)``.  Nothing here reimplements the
decomposition in Python: a wrong table entry fails the test.

Tableau equality is the right notion of equality here.  A tableau
records the conjugation action ``P -> G P G^dagger``, which fixes the
Clifford up to a global phase, and a global phase is unobservable, so
"equal tableaus" is exactly "the same gate".
"""

import ctypes

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from moonlab.clifford import Clifford            # noqa: E402
from moonlab.core import _lib                    # noqa: E402
from moonlab.qec import PF_OP, StimCircuit       # noqa: E402


# ---------------------------------------------------------------------------
# Tableau readout.
#
# clifford_row_pauli exposes the rows moonlab's tableau already stores:
# row i (i < n) is D X_i D^dagger, row n+i is D Z_i D^dagger, which is
# precisely stim's Tableau.x_output(i) / z_output(i).  Pauli bytes use
# the same 0=I, 1=X, 2=Y, 3=Z encoding stim's PauliString indexing does;
# the phase code is 0 for +1 and 2 for -1.
# ---------------------------------------------------------------------------

_lib.clifford_row_pauli.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int),
]
_lib.clifford_row_pauli.restype = ctypes.c_int


def _row(tableau: Clifford, row: int):
    n = tableau.num_qubits
    buf = (ctypes.c_uint8 * n)()
    phase = ctypes.c_int(0)
    rc = _lib.clifford_row_pauli(tableau._handle, row, buf, ctypes.byref(phase))
    assert rc == 0, f"clifford_row_pauli(row={row}) failed with rc={rc}"
    return list(buf), phase.value


def _stim_row(pauli_string):
    """(pauli bytes, phase code) for a stim.PauliString, matching moonlab."""
    sign = complex(pauli_string.sign)
    assert sign in (1 + 0j, -1 + 0j), f"unexpected sign {sign}"
    return ([pauli_string[i] for i in range(len(pauli_string))],
            0 if sign == 1 + 0j else 2)


# ---------------------------------------------------------------------------
# Replay a lowered op list on the tableau.
# ---------------------------------------------------------------------------

_UNARY = {
    PF_OP.H: "h", PF_OP.S: "s", PF_OP.S_DAG: "s_dag",
    PF_OP.X: "x", PF_OP.Y: "y", PF_OP.Z: "z",
}
_BINARY = {PF_OP.CNOT: "cnot", PF_OP.CZ: "cz", PF_OP.SWAP: "swap"}


def _replay(ops, num_qubits: int) -> Clifford:
    tableau = Clifford(num_qubits)
    for op in ops:
        kind = int(op["kind"])
        q0, q1 = int(op["q0"]), int(op["q1"])
        if kind in _UNARY:
            getattr(tableau, _UNARY[kind])(q0)
        elif kind in _BINARY:
            getattr(tableau, _BINARY[kind])(q0, q1)
        else:
            raise AssertionError(
                f"unitary lowering produced non-Clifford op kind {kind} "
                f"({PF_OP(kind).name if kind in set(PF_OP) else '?'})")
    return tableau


def _lowered_tableau(text: str, num_qubits: int) -> Clifford:
    circuit = StimCircuit.from_text(text)
    assert circuit.num_qubits == num_qubits, (
        f"{text!r}: num_qubits={circuit.num_qubits}, expected {num_qubits}")
    ops, chan = circuit.lower()
    assert chan.size == 0, f"{text!r}: unitary circuit produced channel args"
    return _replay(ops, num_qubits)


def _assert_matches(tableau: Clifford, expected, label: str) -> None:
    n = tableau.num_qubits
    assert len(expected) == n, (
        f"{label}: stim tableau spans {len(expected)} qubits, ours {n}")
    for k in range(n):
        for kind, row, want in (("X", k, expected.x_output(k)),
                                ("Z", n + k, expected.z_output(k))):
            got = _row(tableau, row)
            exp = _stim_row(want)
            assert got == exp, (
                f"{label}: {kind}_output({k}) mismatch -- "
                f"moonlab pauli={got[0]} phase={got[1]}, "
                f"stim pauli={exp[0]} phase={exp[1]}")


# ---------------------------------------------------------------------------
# The gate table.  Every name in the frozen spec, aliases included.
# ---------------------------------------------------------------------------

ONE_QUBIT_GATES = [
    "I",
    "X", "Y", "Z",
    "H", "H_XZ", "H_XY", "H_YZ",
    "S", "SQRT_Z", "S_DAG", "SQRT_Z_DAG",
    "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG",
    "C_XYZ", "C_ZYX",
]

TWO_QUBIT_GATES = [
    "CX", "CNOT", "ZCX",
    "CY", "ZCY",
    "CZ", "ZCZ",
    "XCX", "XCY", "XCZ",
    "YCX", "YCY", "YCZ",
    "SWAP", "ISWAP", "ISWAP_DAG",
    "CXSWAP", "SWAPCX", "CZSWAP", "SWAPCZ",
    "SQRT_XX", "SQRT_XX_DAG",
    "SQRT_YY", "SQRT_YY_DAG",
    "SQRT_ZZ", "SQRT_ZZ_DAG",
]

ALL_GATES = ONE_QUBIT_GATES + TWO_QUBIT_GATES


@pytest.mark.parametrize("name", ONE_QUBIT_GATES)
def test_one_qubit_gate_matches_stim_tableau(name):
    """Lowered `NAME 0` conjugates Paulis exactly as stim's named gate."""
    tableau = _lowered_tableau(f"{name} 0", 1)
    _assert_matches(tableau, stim.Tableau.from_named_gate(name), name)


@pytest.mark.parametrize("name", TWO_QUBIT_GATES)
def test_two_qubit_gate_matches_stim_tableau(name):
    """Lowered `NAME 0 1` conjugates Paulis exactly as stim's named gate."""
    tableau = _lowered_tableau(f"{name} 0 1", 2)
    _assert_matches(tableau, stim.Tableau.from_named_gate(name), name)


@pytest.mark.parametrize("name", TWO_QUBIT_GATES)
def test_two_qubit_gate_is_not_symmetric_by_accident(name):
    """`NAME 1 0` matches stim's gate with its targets swapped.

    Catches a decomposition that happens to be right on (0, 1) only
    because it silently normalised the target order.
    """
    tableau = _lowered_tableau(f"{name} 1 0", 2)
    reference = stim.Circuit(f"{name} 1 0").to_tableau()
    _assert_matches(tableau, reference, f"{name} 1 0")


@pytest.mark.parametrize("name", ONE_QUBIT_GATES)
def test_one_qubit_gate_applies_to_every_target(name):
    """`NAME 0 1 2` is three independent gates, as stim defines it."""
    tableau = _lowered_tableau(f"{name} 0 1 2", 3)
    _assert_matches(tableau, stim.Circuit(f"{name} 0 1 2").to_tableau(),
                    f"{name} 0 1 2")


@pytest.mark.parametrize("name", TWO_QUBIT_GATES)
def test_two_qubit_gate_consumes_targets_in_pairs(name):
    """`NAME 0 1 2 3` is two gates, on (0, 1) and (2, 3)."""
    tableau = _lowered_tableau(f"{name} 0 1 2 3", 4)
    _assert_matches(tableau, stim.Circuit(f"{name} 0 1 2 3").to_tableau(),
                    f"{name} 0 1 2 3")


def test_two_qubit_gate_rejects_odd_target_count():
    """A dangling target is a parse error, not a dropped gate."""
    from moonlab.qec import StimFormatError

    with pytest.raises(StimFormatError) as excinfo:
        StimCircuit.from_text("CX 0 1 2")
    assert excinfo.value.line == 1


def test_random_clifford_circuit_matches_stim():
    """End-to-end: random circuits over the whole table match stim.

    Composition is where a per-gate-correct table can still go wrong
    (target order, op ordering inside a decomposition), so this stacks
    200 random instructions and compares the composed tableau.
    """
    rng = np.random.default_rng(0xC11FF0)
    n = 5
    for trial in range(8):
        lines = []
        for _ in range(200):
            if rng.random() < 0.4:
                name = ONE_QUBIT_GATES[rng.integers(len(ONE_QUBIT_GATES))]
                q = int(rng.integers(n))
                lines.append(f"{name} {q}")
            else:
                name = TWO_QUBIT_GATES[rng.integers(len(TWO_QUBIT_GATES))]
                a, b = rng.choice(n, size=2, replace=False)
                lines.append(f"{name} {int(a)} {int(b)}")
        # Pin the qubit count so num_qubits is deterministic.
        text = "\n".join(lines) + f"\nI {n - 1}\n"
        tableau = _lowered_tableau(text, n)
        _assert_matches(tableau, stim.Circuit(text).to_tableau(),
                        f"random trial {trial}")


def test_gate_names_are_case_insensitive():
    """Stim instruction names are case-insensitive; ours too."""
    for name in ("h", "H", "sqrt_x", "SqRt_X", "cx", "Cx"):
        arity = 2 if name.lower() in ("cx",) else 1
        targets = "0 1" if arity == 2 else "0"
        tableau = _lowered_tableau(f"{name} {targets}", arity)
        _assert_matches(tableau,
                        stim.Tableau.from_named_gate(name.upper()),
                        name)


def test_serializer_uses_canonical_upper_case_spelling():
    """to_text() round-trips through the same tableau in canonical case."""
    text = "h 0\nsqrt_x 1\ncx 0 1\niswap_dag 1 0\n"
    circuit = StimCircuit.from_text(text)
    serialised = circuit.to_text()
    assert "H 0" in serialised
    assert "SQRT_X 1" in serialised
    reparsed = StimCircuit.from_text(serialised)
    ops_a, _ = circuit.lower()
    ops_b, _ = reparsed.lower()
    # Field-wise: pf_circuit_op_t has interior padding that carries no
    # meaning and is not guaranteed to survive a C struct copy.
    for field in ("kind", "q0", "q1", "p"):
        assert ops_a[field].tobytes() == ops_b[field].tobytes(), field
    _assert_matches(_replay(ops_b, 2), stim.Circuit(text).to_tableau(),
                    "round-tripped mixed-case circuit")


def test_every_spec_gate_is_covered():
    """Guard against a gate quietly dropping out of the table above."""
    assert len(ALL_GATES) == 44
    assert len(set(ALL_GATES)) == 44
    for name in ALL_GATES:
        stim.Tableau.from_named_gate(name)
