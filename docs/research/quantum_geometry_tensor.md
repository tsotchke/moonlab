# QGT module — design notes

`src/algorithms/quantum_geometry/qgt.{c,h}` is moonlab's momentum-space
topology calculator.  This document captures the design choices behind
the public API + the three Berry-grid integrators we ship in v0.3.

## Scope

What the module computes:

- Berry curvature `Omega_{xy}(k) = -2 Im Q_{xy}(k)` and Fubini-Study
  metric `g_{mu nu}(k) = Re Q_{mu nu}(k)` of any user-supplied 1D or
  2D Bloch Hamiltonian.
- Integer Chern numbers via the Fukui-Hatsugai-Suzuki (2005) discrete
  link-variable construction, generalised to non-Abelian U(M)
  occupied subspaces in the n-band path.
- Z_2 invariants for 4-band TR-symmetric systems (Sz-conserving fast
  path; full Pfaffian Fukui-Hatsugai 2007 deferred to v0.3.x) and 1D
  particle-hole symmetric BdG systems.
- Discrete winding number for 1D chiral systems (SSH).
- Phase diagrams via 1D and 2D parameter sweeps.

What it does NOT compute (and why):

- *Continuous* Berry-curvature density beyond a finite plaquette sum.
  The FHS construction is exactly integer-valued at any finite N when
  the gap is open, so the discrete sum *is* the topological invariant;
  there's no need for a continuum limit.
- Full Pfaffian Fukui-Hatsugai 2007 Z_2 (deferred): the Sz-conserving
  spin-Chern shortcut covers the canonical Kane-Mele and BHZ
  (Rashba-free) regimes that the v0.3 plan calls for.

## Three integrators

We ship three independent Berry-grid integrators, each handling the
gauge problem differently.  All return the same integer Chern on a
gapped Hamiltonian.

### 1. `qgt_berry_grid` — eigvec FHS (legacy)

Computes the lower-band eigenvector via a closed-form 2x2 Bloch-sphere
parametrisation; picks the larger-norm of two complementary formulae
to avoid the formula's south-pole singularity.  Fast and mature.

The closed-form has a known issue: the formula switching boundary is
the equator `h_z = 0`, and switching introduces a relative phase
between the two formulas that can affect plaquettes straddling the
boundary.  In practice this works correctly for QWZ, post-fix Haldane,
Kane-Mele, BHZ, Kitaev, and Hofstadter — the half-step grid offset
keeps grid points off the formula's exact zero locus.

### 2. `qgt_berry_grid_pt` — parallel-transport gauge

Phase-fixes each eigenvector u(k) so that `<u(k_prev) | u(k)> > 0` (real
positive) along a kx-major spanning tree of the BZ.  Removes the
LAPACK-eigvec-gauge randomness that would otherwise break the FHS
plaquette continuity for n-band paths going through
`hermitian_eigen_decomposition`.

The plaquette product on the kx=N-1 column captures the BZ-wrap
holonomy; this is the integer Chern.

### 3. `qgt_berry_grid_proj` — projector trace (gauge-free)

Uses the rank-1 projector `P_-(k) = (1 - h.σ/|h|)/2` as the gauge-
invariant primitive.  The plaquette holonomy

```
F_xy(k) = -arg Tr[P_-(k) P_-(k+x) P_-(k+x+y) P_-(k+y)]
```

is gauge-free without any phase-fix scaffolding.  This is the
canonical Wilson-loop-style construction.  Recommended for any 2-band
Bloch model where gauge sensitivity is suspect.

## n-band non-Abelian path

`qgt_berry_grid_nband` generalises FHS to a multi-band occupied subspace
by replacing the Abelian U(1) link variable

```
U_mu(k) = <u(k) | u(k + dk_mu)>
```

with the determinant SU(M) link

```
U_mu(k) = det <u_occ(k) | u_occ(k + dk_mu)>
```

where `u_occ` is the M-by-n_bands stack of the lowest M occupied
eigenvectors.  The plaquette holonomy is the same FHS formula on
these det-link variables; it gives the total Chern of the occupied
subspace.

For TR-symmetric systems this total Chern is zero by construction; in
that case use `qgt_z2_invariant` for the spin-Chern Z_2.

## Z_2 invariant via spin-Chern shortcut

`qgt_z2_invariant` requires `n_bands == 4` and `n_occupied == 2` and
assumes Sz conservation.  It extracts the upper-left 2x2 block of the
4x4 Bloch Hamiltonian (basis-order: spin-up sector first), computes
its Chern via `qgt_berry_grid` (the original 2-band FHS path), and
returns `|C_up| mod 2`.

This works exactly for canonical Kane-Mele (`lambda_r = 0`) and BHZ.
Adding Rashba (`lambda_r != 0`) breaks Sz conservation; that requires
the full Pfaffian Fukui-Hatsugai 2007 Z_2 path on the half-BZ + TRIM
loop integrals, which is on the v0.3.x roadmap.

## Z_2 invariant for 1D BdG

`qgt_z2_invariant_1d_bdg` works on any 2x2 BdG system with vanishing
off-diagonal pairing at the TR-invariant momenta `k = 0, pi`.  At
those points the Pfaffian of the 2x2 BdG matrix reduces to the
diagonal coefficient, so the Kitaev formula

```
nu = (1 - sgn(M(0)) sgn(M(pi))) / 2
```

is implemented directly.  For Kitaev p-wave: `M(k) = -2t cos(k) - mu`
at the TRIM points, giving `nu = 1` for `|mu| < 2|t|`.

## Hamiltonian convention

All built-in models use a **primitive-reciprocal-coordinate** parametrisation:
`kx, ky` ∈ `[-pi, pi]` are components of `k` along primitive
reciprocal vectors, NOT physical Cartesian momenta.  Integration over
`[-pi, pi]^2` covers exactly one Brillouin zone for the primitive
unit cell.

For honeycomb lattices (Haldane, Kane-Mele) the actual Dirac points
in this convention are at `(0, +/-2*pi/3)` — NOT the textbook
`(±2*pi/3, 0)` points that arise from a different (Cartesian)
convention.  The antisymmetric NNN sum

```
c2(k) = sin(ky) * (1 + 2 cos(kx)) = sin(ky) + sin(ky+kx) + sin(ky-kx)
```

evaluates to `±3*sqrt(3)/2` at these primitive Dirac points and gives
the canonical Haldane phase boundary `|M| < 3*sqrt(3)*|t2*sin(phi)|`.

This was a real bug in v0.2.x: the original `c2 = sin(kx-ky) - sin(kx)
+ sin(ky)` form vanished at the actual Dirac points, leaving the
Haldane SOC mass term unable to gap them.  The bug was diagnosed in
v0.3.0 by building two independent gauge-free Chern integrators
(parallel-transport, projector-trace) that all agreed on the same
"wrong" answer — pointing upstream of the integrator at the
Hamiltonian itself.

## Cross-check infrastructure

- `tests/unit/test_qgt_integrators.c` runs all three Berry-grid
  integrators on QWZ + Haldane phase diagrams; they must all agree.
- `tests/unit/test_qgt_vs_chern_marker.c` cross-checks the
  momentum-space `qgt_berry_grid_proj` against the real-space
  Bianco-Resta local Chern marker (`chern_marker.h`) on QWZ; two
  completely independent topology calculations land on the same
  integer at every test point.
- `tests/performance/bench_topology_phase_diagrams.c` emits a
  136-record JSON archive sweeping every QGT model's phase parameter
  for downstream consumption (paper figures, QGTL submodule pinning).

## References

- Provost & Vallee, *Riemannian structure on manifolds of quantum
  states*, Commun. Math. Phys. **76**, 289 (1980).  Origin of the QGT.
- Berry, *Quantal phase factors accompanying adiabatic changes*,
  Proc. R. Soc. Lond. A **392**, 45 (1984).  Berry connection and phase.
- Fukui, Hatsugai & Suzuki, *Chern numbers in discretized Brillouin
  zone*, J. Phys. Soc. Jpn. **74**, 1674 (2005), arXiv:cond-mat/0503172.
  Link-variable quantisation we implement.
- Fukui & Hatsugai, *Quantum spin Hall effect in three-dimensional
  materials: Lattice computation of Z_2 topological invariants*, J.
  Phys. Soc. Jpn. **76**, 053702 (2007).  The full Pfaffian Z_2 path
  we'll implement in v0.3.x.
- Bernevig & Hughes, *Topological Insulators and Topological
  Superconductors*, Princeton (2013).  Textbook reference for all
  built-in models.
- Thouless, Kohmoto, Nightingale & den Nijs, *Quantized Hall
  conductance in a two-dimensional periodic potential*, Phys. Rev.
  Lett. **49**, 405 (1982).  Hofstadter Diophantine equation.
