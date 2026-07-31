# QGT module — design notes

`src/algorithms/quantum_geometry/qgt.{c,h}` is moonlab's momentum-space
topology calculator.  This document captures the design choices behind
the public API + the three Berry-grid integrators we ship in v0.3.

## Scope

What the module computes:

- Berry curvature `Omega_{xy}(k) = -2 Im Q_{xy}(k)` and Fubini-Study
  metric `g_{mu nu}(k) = Re Q_{mu nu}(k)` of any user-supplied 1D or
  2D Bloch Hamiltonian.
- The *pointwise continuum* `Omega_{xy}(k)` and `g_{mu nu}(k)` of a
  two-band system, as a curvature density rather than a plaquette
  sum: in closed form from an analytic `d(k)` and its momentum
  gradients (`qgt_dsigma_metric_curvature`, machine-precision, no
  finite difference anywhere), or from any Bloch callback via the
  projector trace with a central difference of `H` itself
  (`qgt_curvature_at`, accurate to `O(dk^2)`).  See the fourth
  integrator section below.
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

- A continuum curvature density for systems with more than two bands.
  The pointwise path above is two-band only: it rests on the
  `H = d.sigma` decomposition and the rank-1 lower-band projector.
  The n-band path stays discrete (FHS link variables over a U(M)
  occupied subspace), where the plaquette sum is exactly integer at
  any finite `N` once the gap is open [3] and the topological answer
  needs no continuum limit.
- The full Pfaffian variant of Fukui and Hatsugai [11] for systems
  that break S_z symmetry (e.g. Kane-Mele with Rashba coupling).
  This is scheduled for a future release; the S_z-conserving
  shortcut covers the canonical Kane-Mele and BHZ regimes.

An earlier revision of this section stated that the module computes
no continuum curvature density at all, on the grounds that the
invariant does not need one.  That is right about the *invariant* and
wrong about the module: the density is what a caller wants for
quantum-geometry work that is not a Chern number -- metric
anisotropy, the trace/determinant bounds relating `g` to `Omega`,
quantum-metric contributions to transport -- and the two-band
pointwise path now supplies it.

## Integrators

We ship three independent Berry-grid integrators, each handling the
gauge problem differently.  All return the same integer Chern on a
gapped Hamiltonian.  A fourth path, added in v1.2.1, is not a grid
integrator at all: it evaluates the curvature and metric *at a point*,
and is described in the section after them.

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

## Pointwise band geometry (two-band)

The three integrators above answer "what is the Chern number".  The
pointwise path answers "what is the curvature and the metric *here*",
which is what quantum-geometry work outside topology needs.  Two
entry points, differing in what the caller can supply.

### `qgt_dsigma_metric_curvature` — closed form from analytic `d(k)`

For `H(k) = d(k).sigma` the lower-band QGT is exactly

```
Q_munu = (1/4) [ d_mu dhat . d_nu dhat
                 - i dhat . (d_mu dhat x d_nu dhat) ],   dhat = d/|d|
```

so

```
g_munu   = (1/4) d_mu dhat . d_nu dhat
Omega_xy = -2 Im Q_xy = (1/2) dhat . (d_x dhat x d_y dhat)
```

the skyrmion density of the `dhat` texture.  Both are U(1)-gauge
invariant and depend only on `d` and `grad d`: no eigenvector is ever
formed, so the gauge problem the three integrators each work around
does not arise, and there is no finite difference of any kind.  The
result is machine-precision.

The caller supplies `d`, `d_x d` and `d_y d` directly.  The two
built-in `d.sigma` models carry theirs: `qgt_model_qwz` and
`qgt_model_haldane` populate an analytic d-vector at construction, so
`qgt_exact_curvature_at(sys, k, g, &omega)` reaches the closed form
from a model handle without the caller re-deriving anything, and
`qgt_dsigma_at` exposes the vector itself.  A system built through
`qgt_create` attaches its own with `qgt_set_dsigma`, or leaves it
unset and uses the finite-difference path below; the unset case
returns `-3` rather than guessing.

Haldane carries an identity component (`2 t2 cos(phi) c1`) on top of
its `d.sigma` part.  The d-vector reports only the traceless part:
an identity term shifts both bands equally, drops out of the band
projectors, and therefore does not enter the geometry at all.

### `qgt_curvature_at` — projector trace from any Bloch callback

For a system known only through its `qgt_bloch_fn`, the same tensor
comes from the projector form

```
Q_munu(k) = Tr[ P_- d_mu H P_+ d_nu H ] / (DeltaE)^2,   DeltaE = 2|h(k)|
```

with `P_±` the exact 2x2 band projectors.  This differentiates the
*Hamiltonian matrix entries*, never the eigenvector, so it is still
gauge-free and needs no phase-fixing — but `d_mu H` comes from a
central difference in `k`, so the result is `O(dk^2)`-accurate rather
than machine-exact.  Use it when there is no analytic `d(k)`; use the
closed form when there is.

### Band touchings

Both entry points return `-2` at a band touching, where the geometry
is genuinely undefined, and both test for it *relative to the scale of
their inputs* rather than against an exact zero — `1e-12` times the
Hamiltonian's trace scale for `qgt_curvature_at`, matching
`lower_eigvec_2x2`, and `1e-12` times the L1 scale of `d` and its
gradients for the closed form.  `g` and `Omega` both diverge as the
gap closes (`Q` carries a `1/(DeltaE)^2`), so an absolute-zero test
would let a near-gapless `k` return an enormous finite number that a
caller cannot tell from a measurement.

### Cross-checks

`tests/unit/test_qgt_exact_curvature.c` pins the pointwise path
against three independent references: the projector-form QGT built
from an explicit 2x2 diagonalisation (agreement ~1e-16 on QWZ and
Haldane); the FHS Chern number, by integrating the pointwise
`Omega_xy` over the BZ (`|diff| < 5e-15` across QWZ `C = 0, ±1` and
the Haldane transition at `phi = -pi/2`); and the finite-difference
path against the closed form (~1e-9).

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
