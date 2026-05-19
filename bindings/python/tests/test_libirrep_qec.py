"""Python integration tests for moonlab.libirrep_qec.

Mirrors the C test_libirrep_css unit test: verify each of the eight
CSS-code factories returns a code with the published [[n, k, d]]
parameters.  Tests skip cleanly when moonlab was compiled without
libirrep (the no-bridge case).
"""

from __future__ import annotations

import pytest

pytest.importorskip("moonlab.libirrep_qec",
                    reason="libirrep_qec binding not built")

from moonlab.libirrep_qec import (
    LibirrepQecCode,
    LibirrepError,
    LibirrepNotBuiltError,
    is_available,
)


pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="moonlab was compiled without -DQSIM_ENABLE_LIBIRREP=ON",
)


def test_surface_d3_shape():
    code = LibirrepQecCode.surface(3)
    assert code.n_qubits == 9
    assert code.n_x_stabs == 4
    assert code.n_z_stabs == 4
    assert code.logical_qubits == 1
    assert code.distance == 3  # brute-force enumeration is tractable at n=9


def test_surface_d5_shape():
    code = LibirrepQecCode.surface(5)
    assert code.n_qubits == 25
    assert code.n_x_stabs == 12
    assert code.n_z_stabs == 12
    assert code.logical_qubits == 1


def test_toric_L3_shape():
    code = LibirrepQecCode.toric(3, 3)
    assert code.n_qubits == 18
    assert code.n_x_stabs == 9
    assert code.n_z_stabs == 9
    assert code.logical_qubits == 2  # two homology classes of the torus


def test_toric_L4_shape():
    code = LibirrepQecCode.toric(4, 4)
    assert code.n_qubits == 32
    assert code.logical_qubits == 2


def test_steane_shape_and_distance():
    code = LibirrepQecCode.steane()
    assert code.n_qubits == 7
    assert code.n_x_stabs == 3
    assert code.n_z_stabs == 3
    assert code.logical_qubits == 1
    assert code.distance == 3


def test_hamming_15_7_3_shape():
    code = LibirrepQecCode.hamming_15_7_3()
    assert code.n_qubits == 15
    assert code.logical_qubits == 7


def test_bb_gross_shapes():
    """Bravyi et al. 2024 Nature 627, 778 Table 3 instances."""
    g72 = LibirrepQecCode.bb_72_12_6()
    assert g72.n_qubits == 72 and g72.logical_qubits == 12

    g144 = LibirrepQecCode.bb_144_12_12()
    assert g144.n_qubits == 144 and g144.logical_qubits == 12

    g288 = LibirrepQecCode.bb_288_12_18()
    assert g288.n_qubits == 288 and g288.logical_qubits == 12


def test_hgp_repetition_ladder():
    """Tillich-Zemor 2009 hypergraph product over repetition codes."""
    expected = [(3, 13), (4, 25), (5, 41)]
    for d, n in expected:
        code = LibirrepQecCode.hgp_repetition(d)
        assert code.n_qubits == n
        assert code.logical_qubits == 1


def test_hgp_repetition_rejects_out_of_range():
    with pytest.raises(LibirrepError):
        LibirrepQecCode.hgp_repetition(6)


def test_check_row_accessor():
    """Each Steane stabiliser is weight-4 (the standard [[7,1,3]] support
    pattern from Steane 1996)."""
    code = LibirrepQecCode.steane()
    for row in range(code.n_x_stabs):
        support = code.x_check_row(row)
        assert len(support) == code.n_qubits
        weight = sum(support)
        assert weight == 4, f"Steane X-row {row} weight = {weight}"


def test_surface_rejects_distance_below_2():
    with pytest.raises(LibirrepError):
        LibirrepQecCode.surface(1)


def test_repr_smoke():
    code = LibirrepQecCode.steane()
    r = repr(code)
    assert "LibirrepQecCode" in r
    assert "7" in r  # n_qubits
