#!/usr/bin/env python3
"""
Quantum geometric tensor: topological phase diagrams (Python).

This example reproduces the principal phase boundaries of six
canonical lattice models using the Moonlab Python bindings to the
v0.3 quantum-geometric-tensor module.  Each call below returns the
integer topological invariant on a Brillouin-zone grid; the
analytical phase boundary is identified inline against the primary
literature.

Models covered:
    SSH                 Su, Schrieffer, Heeger, PRL 42, 1698 (1979)
    Qi-Wu-Zhang         Qi, Wu, Zhang,        PRB 74, 085308 (2006)
    Haldane             Haldane,              PRL 61, 2015 (1988)
    Kane-Mele           Kane, Mele,           PRL 95, 146802 (2005)
    Bernevig-Hughes-Zhang Bernevig, Hughes, Zhang,
                                              Science 314, 1757 (2006)
    Kitaev p-wave chain Kitaev,               Phys.-Uspekhi 44, 131 (2001)
    Hofstadter          Hofstadter,           PRB 14, 2239 (1976)

Run::

    python3 examples/topological/qgt_phase_diagrams.py
"""

from __future__ import annotations

import numpy as np

from moonlab.topology import (
    bhz_z2,
    chern_qwz_parallel_transport,
    chern_qwz_proj,
    hofstadter_chern,
    kane_mele_z2,
    kitaev_chain_z2,
    qwz_chern,
    ssh_winding,
)


def banner(title: str) -> None:
    bar = "=" * (len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")


def section_ssh() -> None:
    banner("SSH chain (1D winding)")
    print("Topological for |t2| > |t1|; trivial otherwise.")
    print(f"  t1=0.5, t2=1.0  ->  W = {ssh_winding(0.5, 1.0, 64):+d}")
    print(f"  t1=1.0, t2=0.5  ->  W = {ssh_winding(1.0, 0.5, 64):+d}")
    print(f"  t1=t2=1.0       ->  W = {ssh_winding(1.0, 1.0, 64):+d} (gap closed)")


def section_qwz() -> None:
    banner("Qi-Wu-Zhang model: three Chern integrators must agree")
    print("Phase boundaries at m in {-2, 0, +2}; gap closures otherwise.")
    print(f"  {'m':>5} {'FHS':>5} {'proj':>5} {'p.t.':>5}")
    for m in np.linspace(-3.0, 3.0, 7):
        fhs = qwz_chern(m, N=32)
        proj = chern_qwz_proj(m, N=32)
        pt = chern_qwz_parallel_transport(m, N=32)
        print(f"  {m:+5.1f} {fhs:+5d} {proj:+5d} {pt:+5d}")


def section_kane_mele() -> None:
    banner("Kane-Mele Z_2 (S_z conserving)")
    print("QSH (Z_2 = 1) when |lambda_v| < 3 sqrt(3) |lambda_so|.")
    lambda_so = 0.06
    boundary = 3.0 * np.sqrt(3.0) * lambda_so
    print(f"  Boundary at |lambda_v| = 3 sqrt(3) * {lambda_so} = {boundary:.5f}")
    for lambda_v in (0.05, 0.10, 0.15, 0.20, 0.25, 0.40):
        z2 = kane_mele_z2(t=1.0, lambda_so=lambda_so, lambda_v=lambda_v, N=24)
        marker = "QSH" if z2 == 1 else "trivial"
        print(f"  lambda_v = {lambda_v:.2f} ->  Z_2 = {z2}  ({marker})")


def section_bhz() -> None:
    banner("Bernevig-Hughes-Zhang (HgTe quantum well)")
    print("Lattice regularisation gives QSH for 0 < M / B < 8.")
    A, B = 1.0, 1.0
    for M in (-1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
        z2 = bhz_z2(A=A, B=B, M=M, N=24)
        marker = "QSH" if z2 == 1 else "trivial"
        print(f"  M = {M:+5.1f}  ->  Z_2 = {z2}  ({marker})")


def section_kitaev_chain() -> None:
    banner("Kitaev p-wave chain (Pfaffian-sign Z_2)")
    print("Topological with Majorana edges when |mu| < 2 |t|.")
    t = 1.0
    delta = 1.0
    for mu in (-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0):
        z2 = kitaev_chain_z2(t=t, mu=mu, delta=delta)
        marker = "Majorana" if z2 == 1 else "trivial"
        print(f"  mu = {mu:+4.1f}  ->  Z_2 = {z2}  ({marker})")


def section_hofstadter() -> None:
    banner("Hofstadter magnetic sub-band Chern numbers (phi = 1/q)")
    print("For phi = 1/q the lowest sub-band carries Chern = +1 (TKNN 1982).")
    for q in (3, 4, 5, 6, 7):
        c = hofstadter_chern(p=1, q=q, n_occupied=1, t=1.0, N=24)
        print(f"  q = {q}: lowest band Chern = {c:+d}")


def main() -> None:
    section_ssh()
    section_qwz()
    section_kane_mele()
    section_bhz()
    section_kitaev_chain()
    section_hofstadter()
    print()


if __name__ == "__main__":
    main()
