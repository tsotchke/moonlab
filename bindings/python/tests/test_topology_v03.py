"""Pytest suite for the v0.3 n-band QGT Python bindings.

Validates Kane-Mele, BHZ, Kitaev p-wave, Hofstadter, and the
projector-trace + parallel-transport gauge-invariant Chern integrators
against the analytical phase boundaries in the literature.
"""

from __future__ import annotations

import pytest

import moonlab
from moonlab.topology import (
    chern_qwz_proj, chern_qwz_parallel_transport,
    kane_mele_z2, bhz_z2, kitaev_chain_z2, hofstadter_chern,
    qwz_chern, ssh_winding,
)


def test_qwz_chern_three_integrators_agree():
    """FHS link variable, projector trace, and parallel transport must
    return identical integers on every gapped phase point."""
    for m in (-1.5, -0.5, 0.5, 1.5, 2.5, -2.5):
        c_fhs = qwz_chern(m, N=32)
        c_proj = chern_qwz_proj(m, N=32)
        c_pt = chern_qwz_parallel_transport(m, N=32)
        assert c_fhs == c_proj == c_pt, (
            f"integrator disagreement at m={m}: "
            f"fhs={c_fhs}, proj={c_proj}, pt={c_pt}"
        )


def test_qwz_chern_proj_signs():
    """QWZ phase diagram with the projector-trace integrator."""
    assert chern_qwz_proj(-1.5, N=48) == +1
    assert chern_qwz_proj(+1.5, N=48) == -1
    assert chern_qwz_proj(+3.0, N=48) == 0
    assert chern_qwz_proj(-3.0, N=48) == 0


def test_kane_mele_z2_phases():
    """Z_2 = 1 (QSH) for |lambda_v| < 3 sqrt(3) |lambda_so|, else 0."""
    # 3 sqrt(3) * 0.06 ~= 0.3118.
    assert kane_mele_z2(t=1.0, lambda_so=0.06, lambda_v=0.10, N=24) == 1
    assert kane_mele_z2(t=1.0, lambda_so=0.06, lambda_v=0.40, N=24) == 0


def test_bhz_z2_window():
    """BHZ lattice regularisation: QSH for 0 < M/B < 8."""
    A, B = 1.0, 1.0
    # M = 3 -> topological.
    assert bhz_z2(A=A, B=B, M=3.0, N=24) == 1
    # M = -1 -> trivial (M/B < 0).
    assert bhz_z2(A=A, B=B, M=-1.0, N=24) == 0
    # M = 9 -> trivial (M/B > 8).
    assert bhz_z2(A=A, B=B, M=9.0, N=24) == 0


def test_kitaev_chain_z2_topological_window():
    """Z_2 = 1 for |mu| < 2|t|, else 0."""
    assert kitaev_chain_z2(t=1.0, mu=0.5, delta=1.0) == 1
    assert kitaev_chain_z2(t=1.0, mu=-1.5, delta=1.0) == 1
    assert kitaev_chain_z2(t=1.0, mu=2.5, delta=1.0) == 0
    assert kitaev_chain_z2(t=1.0, mu=-3.0, delta=1.0) == 0


def test_hofstadter_lowest_band_chern():
    """For phi = 1/q, the lowest magnetic sub-band has Chern = +1."""
    for q in (3, 5):
        c = hofstadter_chern(p=1, q=q, n_occupied=1, t=1.0, N=24)
        assert c == 1, f"q={q}: expected +1, got {c}"


def test_hofstadter_q3_two_band_sum():
    """Lowest two bands of q=3 sum to +1 + (-2) = -1."""
    c = hofstadter_chern(p=1, q=3, n_occupied=2, t=1.0, N=24)
    assert c == -1


def test_ssh_winding_unchanged_in_v03():
    """SSH winding is from v0.2; ensure v0.3 didn't regress it."""
    assert ssh_winding(t1=1.0, t2=2.0, N=64) == 1
    assert ssh_winding(t1=2.0, t2=1.0, N=64) == 0


def test_input_validation():
    with pytest.raises(ValueError):
        chern_qwz_proj(0.0, N=2)
    with pytest.raises(ValueError):
        kane_mele_z2(N=7)            # odd N rejected
    with pytest.raises(ValueError):
        hofstadter_chern(q=1)
    with pytest.raises(ValueError):
        hofstadter_chern(q=3, n_occupied=3)
