# Archived Moonlab Documentation: QGT module — design notes

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# QGT module — design notes

`src/algorithms/quantum_geometry/qgt.{c,h}` is moonlab's momentum-space
topology calculator.  This document captures the design choices behind
the public API + the three Berry-grid integrators we ship in v0.3.

## Scope

What the module computes:

- Berry curvature `Omega_{xy}(k) = -2 Im Q_{xy}(k)` and Fubini-Study
  metric `g_{mu nu}(k) = Re Q_{mu nu}(k)` of any user-supplied 1D or
  2D Bloch Hamiltonian.
- Integer Chern numbers via the Fukui-Hatsugai-Suzuki [3] discrete
  link-variable construction, generalised to non-Abelian U(M)
  occupied subspaces in the n-band path.
- Z_2 invariants for 4-band time-reversal-symmetric systems via the
  S_z-conserving spin-Chern formulation (Kane and Mele [5]) and for
  one-dimensional particle-hole-symmetric BdG systems via the
  Pfaffian-sign product at the TR-invariant momenta (Kitaev [7]).
- Discrete winding number for 1D chiral systems (SSH [4]) via the
  Zak phase [10].
- Phase diagrams obtained by sweeping a model's parameter space over
  one or two dimensions.

What the module does *not* compute, and why:

- A *continuum* Berry-curvature density beyond the finite plaquette
  sum.  The FHS construction is exactly integer-valued at any finite
  `N` once the gap is open [3]; the discrete sum *is* the
  topological invariant and no continuum limit is needed.
- The full Pfaffian variant of Fukui and Hatsugai [11] for systems
  that break S_z symmetry (e.g. Kane-Mele with Rashba coupling).
  This is scheduled for a future release; the S_z-conserving
  shortcut covers the canonical Kane-Mele and BHZ regimes.

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

[archived fence delimiter: ```]
F_xy(k) = -arg Tr[P_-(k) P_-(k+x) P_-(k+x+y) P_-(k+y)]
[archived fence delimiter: ```]

is gauge-free without any phase-fix scaffolding.  This is the
canonical Wilson-loop-style construction.  Recommended for any 2-band
Bloch model where gauge sensitivity is suspect.

## n-band non-Abelian path

`qgt_berry_grid_nband` generalises FHS to a multi-band occupied subspace
by replacing the Abelian U(1) link variable

[archived fence delimiter: ```]
U_mu(k) = <u(k) | u(k + dk_mu)>
[archived fence delimiter: ```]

with the determinant SU(M) link

[archived fence delimiter: ```]
U_mu(k) = det <u_occ(k) | u_occ(k + dk_mu)>
[archived fence delimiter: ```]

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

This is exact for the canonical Kane-Mele model (`lambda_r = 0`) [5]
and for the BHZ model [6].  Adding Rashba coupling
(`lambda_r != 0`) breaks S_z conservation, in which case the full
Pfaffian construction of Fukui and Hatsugai [11] on the half
Brillouin zone with TRIM line integrals is required; that path is
scheduled for a future release.

## Z_2 invariant for 1D BdG

`qgt_z2_invariant_1d_bdg` works on any 2x2 BdG system with vanishing
off-diagonal pairing at the TR-invariant momenta `k = 0, pi`.  At
those points the Pfaffian of the 2x2 BdG matrix reduces to the
diagonal coefficient, so the Kitaev formula

[archived fence delimiter: ```]
nu = (1 - sgn(M(0)) sgn(M(pi))) / 2
[archived fence delimiter: ```]

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

[archived fence delimiter: ```]
c2(k) = sin(ky) * (1 + 2 cos(kx)) = sin(ky) + sin(ky+kx) + sin(ky-kx)
[archived fence delimiter: ```]

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

[1] J. P. Provost and G. Vallee, "Riemannian structure on manifolds
    of quantum states", *Commun. Math. Phys.* **76**, 289 (1980).
    Origin of the quantum geometric tensor.

[2] M. V. Berry, "Quantal phase factors accompanying adiabatic
    changes", *Proc. R. Soc. Lond. A* **392**, 45 (1984).  Berry
    connection and the geometric phase.

[3] T. Fukui, Y. Hatsugai, and H. Suzuki, "Chern numbers in
    discretized Brillouin zone: Efficient method of computing
    (spin) Hall conductances", *J. Phys. Soc. Jpn.* **74**, 1674
    (2005); arXiv:cond-mat/0503172.  Link-variable quantisation
    used in `qgt_berry_grid` and `qgt_berry_grid_nband`.

[4] W. P. Su, J. R. Schrieffer, and A. J. Heeger, "Solitons in
    polyacetylene", *Phys. Rev. Lett.* **42**, 1698 (1979).  SSH
    chain.

[5] C. L. Kane and E. J. Mele, "Z_2 topological order and the
    quantum spin Hall effect", *Phys. Rev. Lett.* **95**, 146802
    (2005).  Kane-Mele model and Z_2 invariant.

[6] B. A. Bernevig, T. L. Hughes, and S.-C. Zhang, "Quantum spin
    Hall effect and topological phase transition in HgTe quantum
    wells", *Science* **314**, 1757 (2006).  BHZ model.

[7] A. Y. Kitaev, "Unpaired Majorana fermions in quantum wires",
    *Physics-Uspekhi* **44**, 131 (2001).  1D BdG Z_2 from the
    Pfaffian-sign product at TR-invariant momenta.

[8] D. R. Hofstadter, "Energy levels and wave functions of Bloch
    electrons in rational and irrational magnetic fields",
    *Phys. Rev. B* **14**, 2239 (1976).  Hofstadter butterfly.

[9] D. J. Thouless, M. Kohmoto, M. P. Nightingale, and M. den Nijs,
    "Quantized Hall conductance in a two-dimensional periodic
    potential", *Phys. Rev. Lett.* **49**, 405 (1982).  TKNN
    Diophantine equation for magnetic sub-band Cherns.

[10] J. Zak, "Berry's phase for energy bands in solids",
    *Phys. Rev. Lett.* **62**, 2747 (1989).  Discrete Zak phase
    used in `qgt_winding_1d`.

[11] T. Fukui and Y. Hatsugai, "Quantum spin Hall effect in
    three-dimensional materials: Lattice computation of Z_2
    topological invariants", *J. Phys. Soc. Jpn.* **76**, 053702
    (2007).  Pfaffian Z_2 construction for Rashba-coupled
    Hamiltonians.

[12] R. Bianco and R. Resta, "Mapping topological order in
    coordinate space", *Phys. Rev. B* **84**, 241106(R) (2011).
    Real-space local Chern marker used by the cross-check against
    `qgt_berry_grid_proj`.

[13] B. A. Bernevig and T. L. Hughes, *Topological Insulators and
    Topological Superconductors*, Princeton University Press, 2013.
    Standard textbook reference for the built-in models.
```
