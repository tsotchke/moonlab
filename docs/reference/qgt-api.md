# Archived Moonlab Documentation: Quantum Geometric Tensor (QGT) — API reference

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Quantum Geometric Tensor (QGT) — API reference

`src/algorithms/quantum_geometry/qgt.{c,h}` — momentum-space topology
calculator for parameterised band Hamiltonians.  Computes Berry
curvature [1], Chern numbers [2,3], Z_2 invariants [4,5], and the
Fubini-Study metric [1] on user-supplied 1D and 2D Bloch
Hamiltonians.

Status: stable.  v0.3.0 expanded the module from a 2-band core to a
full n-band path with 4-band TI primitives and three independent
gauge-aware Chern integrators.

For the full bibliography (Provost-Vallee 1980, Berry 1984,
Fukui-Hatsugai-Suzuki 2005, Kane-Mele 2005, Bernevig-Hughes-Zhang
2006, Kitaev 2001, Hofstadter 1976, TKNN 1982, Zak 1989,
Fukui-Hatsugai 2007, Bianco-Resta 2011, Bernevig-Hughes 2013) and
the derivations behind the three integrators see
[`../research/quantum_geometry_tensor.md`](../research/quantum_geometry_tensor.md).
For a hands-on walkthrough across six canonical models see
[`../tutorials/topological_band_structure.md`](../tutorials/topological_band_structure.md).

## Concepts

For a parameterised pure state `|u(k)>`, the Provost-Vallee
quantum geometric tensor is

[archived fence delimiter: ```]
Q_{mu nu}(k) = <∂_mu u | (1 - |u><u|) | ∂_nu u>
[archived fence delimiter: ```]

with real symmetric part = Fubini-Study metric `g_{mu nu}` and
imaginary antisymmetric part = -1/2 Berry curvature
`Omega_{mu nu} = -2 Im Q`.  In 2D the Chern number is

[archived fence delimiter: ```]
C = (1/2 pi) ∫_BZ Omega_{xy} d^2 k
[archived fence delimiter: ```]

Moonlab implements the Fukui-Hatsugai-Suzuki (2005) discretised
plaquette construction, the parallel-transport gauge variant, and
the projector-trace formulation that's gauge-free by construction.

## 2-band 2D systems

[archived fence delimiter: ```c]
typedef void (*qgt_bloch_fn)(const double k[2], void* user,
                             qgt_complex_t h[4]);

qgt_system_t* qgt_create(qgt_bloch_fn f, void* user);
void          qgt_free(qgt_system_t* sys);
[archived fence delimiter: ```]

Custom Bloch Hamiltonian with row-major 2x2 output `h[4]`.  Built-in
models:

[archived fence delimiter: ```c]
qgt_system_t* qgt_model_qwz(double m);
qgt_system_t* qgt_model_haldane(double t1, double t2,
                                 double phi, double M_stagger);
[archived fence delimiter: ```]

QWZ: `C = +1` for `-2 < m < 0`, `C = -1` for `0 < m < 2`, trivial
otherwise.  Haldane: topological for `|M| < 3*sqrt(3) * |t2 sin(phi)|`.

### Berry-grid integrators

Three integrators, all return the same integer Chern on a gapped
Hamiltonian — pick by gauge-stability needs.

[archived fence delimiter: ```c]
typedef struct {
    size_t  N;
    double* berry;     /* row-major N*N plaquette flux array */
    double  chern;     /* (1/2pi) sum F_xy over BZ -- exact integer */
} qgt_berry_grid_t;

int  qgt_berry_grid     (const qgt_system_t* sys, size_t N,
                         qgt_berry_grid_t* out);
int  qgt_berry_grid_pt  (const qgt_system_t* sys, size_t N,
                         qgt_berry_grid_t* out);
int  qgt_berry_grid_proj(const qgt_system_t* sys, size_t N,
                         qgt_berry_grid_t* out);
void qgt_berry_grid_free(qgt_berry_grid_t* g);
[archived fence delimiter: ```]

- `qgt_berry_grid` — original eigvec FHS path; fast, mature,
  picks "larger-norm" branch of two closed-form formulae for the
  lower-band eigvec.
- `qgt_berry_grid_pt` — phase-fixes each eigvec against its
  predecessor along a kx-major spanning tree; removes LAPACK-gauge
  randomness for n-band paths.
- `qgt_berry_grid_proj` — uses the rank-1 projector
  `P = (1 - h.sigma/|h|)/2`, gauge-free by construction.  Best for
  problems where any gauge sensitivity is suspect.

Recommended `N`: 32 for QWZ-like, 48-64 for Haldane / Kane-Mele
(narrower gap regions need finer grids).

### Other 2-band tools

[archived fence delimiter: ```c]
int qgt_metric_at(const qgt_system_t* sys, const double k[2],
                  double dk, double g[4]);
int qgt_wilson_loop(const qgt_system_t* sys,
                    const double* path_k, size_t num_points,
                    double* out_phase);
[archived fence delimiter: ```]

Phase-diagram sweeps over a single tunable parameter:

[archived fence delimiter: ```c]
typedef qgt_system_t* (*qgt_param_system_fn)(void* user, double param);

int qgt_phase_diagram_chern(qgt_param_system_fn factory, void* user,
                             double param_min, double param_max,
                             size_t K, size_t N, int* chern_out);
int qgt_phase_diagram_chern_2d(...);
[archived fence delimiter: ```]

## 1D 2-band systems

[archived fence delimiter: ```c]
typedef void (*qgt_bloch_1d_fn)(double k, void* user, qgt_complex_t h[4]);

qgt_system_1d_t* qgt_create_1d(qgt_bloch_1d_fn f, void* user);
void             qgt_free_1d(qgt_system_1d_t* sys);

qgt_system_1d_t* qgt_model_ssh(double t1, double t2);
qgt_system_1d_t* qgt_model_kitaev_chain(double t, double mu, double delta);

int qgt_winding_1d(const qgt_system_1d_t* sys, size_t N, double* out_raw);
int qgt_z2_invariant_1d_bdg(const qgt_system_1d_t* sys, int* z2);
[archived fence delimiter: ```]

SSH winding: `W = +1` in topological phase `|t2| > |t1|`.  Kitaev
Z_2: `1` (Majorana edge modes) for `|mu| < 2|t|`.

## n-band 2D systems

[archived fence delimiter: ```c]
typedef void (*qgt_bloch_n_fn)(const double k[2], void* user,
                               qgt_complex_t* h);

qgt_system_n_t* qgt_create_nband(qgt_bloch_n_fn f, void* user,
                                  size_t n_bands, size_t n_occupied);
void            qgt_free_nband(qgt_system_n_t* sys);

int qgt_berry_grid_nband(const qgt_system_n_t* sys, size_t N,
                          qgt_berry_grid_t* out);

int qgt_z2_invariant(const qgt_system_n_t* sys, size_t N, int* z2);
[archived fence delimiter: ```]

`qgt_berry_grid_nband` uses non-Abelian U(M) link variables (`M = n_occupied`):
`U_mu(k) = det <u_occ(k) | u_occ(k + dk_mu)>`.  Returns the total
Chern of the occupied subspace.  For TR-symmetric systems this is
zero; use `qgt_z2_invariant` instead.

`qgt_z2_invariant` requires `n_bands == 4 && n_occupied == 2` and
assumes S_z conservation (Rashba SOC = 0).  It extracts the
upper-left 2 x 2 block (spin-up sector) and returns `|C_up| mod 2`.
A full Rashba-compatible Pfaffian variant of Fukui and Hatsugai
(2007) is scheduled for a future release.

### 4-band TI models

[archived fence delimiter: ```c]
qgt_system_n_t* qgt_model_kane_mele(double t, double lambda_so,
                                     double lambda_r, double lambda_v);
qgt_system_n_t* qgt_model_bhz(double A, double B, double M);
qgt_system_n_t* qgt_model_hofstadter(double t,
                                      size_t p, size_t q,
                                      size_t n_occupied);
[archived fence delimiter: ```]

Kane-Mele: 4-band honeycomb in basis (A_up, B_up, A_down, B_down).
QSH for `|lambda_v| < 3*sqrt(3) * |lambda_so|`.  Set `lambda_r = 0`
for the canonical KM (Rashba-free).

BHZ: 4-band square-lattice in basis (s+, p+, s-, p-).  Lattice
regularisation `mass(k) = M - 2B(2 - cos kx - cos ky)`.  QSH for
`0 < M / B < 8` (X-corner closings cancel; M-corner closing at 8B
re-trivialises the system).

Hofstadter: q-band magnetic Bloch system with flux `phi = p/q`.
Use `n_occupied = 1` for the lowest band.  TKNN Chern: `+1` for the
lowest band of the `phi = 1/q` model.

## Cross-checks

`tests/unit/test_qgt_integrators.c` exercises QWZ and Haldane on all
three 2-band integrators.  `tests/unit/test_qgt_vs_chern_marker.c`
cross-checks `qgt_berry_grid_proj` (momentum-space) against
`chern_marker.h` (real-space Bianco-Resta) on QWZ.  Both must give
the same integer for any gapped Hamiltonian.

`tests/performance/bench_topology_phase_diagrams.c` emits a 136-record
JSON archive sweeping every model's phase parameter.  Use this for
regression-pinned reference data.

## Related modules

- `src/algorithms/topology_realspace/chern_marker.{c,h}` — Bianco-Resta
  local Chern marker on dense real-space lattices (currently QWZ-specific).
- `src/algorithms/topology_realspace/chern_kpm.{c,h}` — Chebyshev-KPM
  local Chern marker via tensor networks (Antão-Sun-Fumega-Lado 2026).
- `src/algorithms/topology_realspace/chern_fhs.{c,h}` — momentum-space
  FHS for direct two-band lattice models (alternative entry point).

## Language bindings

- **Python** (`moonlab.topology`): `chern_qwz_proj`,
  `chern_qwz_parallel_transport`, `kane_mele_z2`, `bhz_z2`,
  `kitaev_chain_z2`, `hofstadter_chern`, plus the v0.2
  `qwz_chern`, `berry_grid_qwz`, `berry_grid_haldane`, and
  `ssh_winding`.  Curvature-grid variants `berry_grid_qwz_proj`
  and `berry_grid_qwz_pt` (v0.3.2) return NumPy `float64` arrays
  for plotting and integration cross-checks.  Validated by
  `bindings/python/tests/test_topology.py`.
- **Rust** (`moonlab::topology`): `qwz_chern`, `ssh_winding`,
  `ChernKpm`, plus the v0.3.x n-band surface
  (`chern_qwz_proj`, `chern_qwz_parallel_transport`,
  `kane_mele_z2`, `bhz_z2`, `kitaev_chain_z2`,
  `hofstadter_chern`).  See `bindings/rust/moonlab/examples/topology_demo.rs`.

## See also

- [`../research/quantum_geometry_tensor.md`](../research/quantum_geometry_tensor.md):
  module theory, derivations of the three integrators, and the
  full reference bibliography.
- [`../tutorials/topological_band_structure.md`](../tutorials/topological_band_structure.md):
  full hands-on walkthrough across SSH, QWZ, Haldane, Kane-Mele,
  BHZ, and Hofstadter with primary-source citations.
- `tests/unit/test_qgt_*.c` (10 cases, all passing): minimal
  example callers covering every integrator and model.
- `examples/topological/qgt_phase_diagrams.py` and
  `bindings/rust/moonlab/examples/topology_demo.rs`: Python and
  Rust worked examples that reproduce every phase boundary in the
  tutorial.

## References

The bracketed numbers in the body correspond to:

[1] J. P. Provost and G. Vallee, "Riemannian structure on manifolds
    of quantum states", *Commun. Math. Phys.* **76**, 289 (1980).
    Berry-curvature / Fubini-Study metric definitions.

[2] T. Fukui, Y. Hatsugai, and H. Suzuki, "Chern numbers in
    discretized Brillouin zone", *J. Phys. Soc. Jpn.* **74**, 1674
    (2005); arXiv:cond-mat/0503172.  Link-variable plaquette
    construction used by `qgt_berry_grid` and `qgt_berry_grid_nband`.

[3] D. J. Thouless, M. Kohmoto, M. P. Nightingale, and M. den Nijs,
    "Quantized Hall conductance in a two-dimensional periodic
    potential", *Phys. Rev. Lett.* **49**, 405 (1982).  TKNN
    Diophantine equation for the Hofstadter Chern numbers.

[4] C. L. Kane and E. J. Mele, "Z_2 topological order and the
    quantum spin Hall effect", *Phys. Rev. Lett.* **95**, 146802
    (2005).  Kane-Mele model and `qgt_z2_invariant` for 4-band TR
    systems.

[5] A. Y. Kitaev, "Unpaired Majorana fermions in quantum wires",
    *Physics-Uspekhi* **44**, 131 (2001).
    `qgt_z2_invariant_1d_bdg` Pfaffian construction.

Additional sources used throughout the module documentation are
listed in [`../research/quantum_geometry_tensor.md`](../research/quantum_geometry_tensor.md).
```
