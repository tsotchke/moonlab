# Archived Moonlab Documentation: Tutorial: Topological band structure with the QGT module

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Tutorial: Topological band structure with the QGT module

This tutorial walks through Moonlab's quantum-geometric-tensor (QGT)
module on six canonical lattice models, ordered from least to most
structured: SSH, Qi-Wu-Zhang, Haldane, Kane-Mele, Bernevig-Hughes-
Zhang (BHZ), and Hofstadter.  At each step we evaluate the relevant
topological invariant — winding number, Chern number, or Z_2 — and
cross-check the numerical result against the analytical phase
boundary obtained from the original literature.

Prerequisites:

- A Release build of `libquantumsim` configured with
  `-DQSIM_BUILD_TESTS=ON -DQSIM_BUILD_BENCHMARKS=ON`.
- Working familiarity with single-particle tight-binding Hamiltonians
  and Berry-phase topology at the level of Bernevig and Hughes [1].

The exposition is in C, which is the canonical surface for these
primitives.  Python and Rust bindings reach feature parity in
v0.3.0; equivalent code in those languages appears at the end of
each section where applicable.

## 1. Su-Schrieffer-Heeger chain (1D winding)

The Su-Schrieffer-Heeger (SSH) model [2] is the simplest 1D model
exhibiting a topological transition: a bipartite chain with
intra-cell hopping `t1` and inter-cell hopping `t2`.  Chiral symmetry
gives an integer winding number `W in Z`; the chain is topologically
non-trivial (`W = 1`) for `|t2| > |t1|` and trivial (`W = 0`)
otherwise.

[archived fence delimiter: ```c]
#include "moonlab/algorithms/quantum_geometry/qgt.h"
#include <stdio.h>

int main(void) {
    qgt_system_1d_t* sys = qgt_model_ssh(/*t1=*/0.5, /*t2=*/1.0);
    double w_raw;
    int W = qgt_winding_1d(sys, /*N=*/64, &w_raw);
    printf("SSH (t1=0.5, t2=1.0) winding = %d (expected +1)\n", W);
    qgt_free_1d(sys);
    return 0;
}
[archived fence delimiter: ```]

The integer winding is computed from the discrete Zak phase [3]
along the 1D Brillouin zone.  Setting `t1 = 1.0, t2 = 0.5` returns
`W = 0`, recovering the trivial phase.

## 2. Qi-Wu-Zhang model: 2D Chern insulator

The Qi-Wu-Zhang (QWZ) model [4] is the minimal two-band Chern
insulator on a square lattice.  Its Chern number takes integer
values that change by unity at the gap closings `m in {-2, 0, 2}`,
giving a piecewise-constant phase diagram on `m in [-3, 3]`.

[archived fence delimiter: ```c]
qgt_system_t* sys = qgt_model_qwz(/*m=*/-1.5);
qgt_berry_grid_t g;
qgt_berry_grid_proj(sys, /*N=*/48, &g);
printf("QWZ m=-1.5: C = %+.4f (expected +1)\n", g.chern);
qgt_berry_grid_free(&g);
qgt_free(sys);
[archived fence delimiter: ```]

`qgt_berry_grid_proj` realises the projector-trace integrator,
which is manifestly gauge-invariant; `qgt_berry_grid` (Fukui-
Hatsugai-Suzuki link variables [5]) and `qgt_berry_grid_pt`
(parallel-transport gauge) compute the same integer through
different gauge prescriptions.  At the gap-closing points
`m in {-2, 0, 2}` the Chern number is mathematically undefined; the
half-step BZ sampling assigns an integer artifact of the chosen
discretisation.

## 3. Haldane model: honeycomb Chern insulator

Haldane's seminal model [6] was the first demonstration that a
Chern insulator can arise without a uniform magnetic field, via
complex next-nearest-neighbour hopping that breaks time-reversal
locally while preserving zero net flux.  The phase boundary is
`|M| = 3 sqrt(3) |t_2 sin(phi)|`; at `phi = pi/2` and `t_2 = 0.06`
the analytic boundary is `M ~= 0.31177`:

[archived fence delimiter: ```c]
for (double M = 0.0; M <= 0.5; M += 0.05) {
    qgt_system_t* sys = qgt_model_haldane(1.0, 0.06, 0.5*M_PI, M);
    qgt_berry_grid_t g;
    qgt_berry_grid_proj(sys, 48, &g);
    int C = (int)lround(g.chern);
    printf("Haldane M=%.2f -> C = %+d\n", M, C);
    qgt_berry_grid_free(&g);
    qgt_free(sys);
}
[archived fence delimiter: ```]

Output:

[archived fence delimiter: ```]
Haldane M=0.00 -> C = -1
Haldane M=0.05 -> C = -1
Haldane M=0.10 -> C = -1
Haldane M=0.15 -> C = -1
Haldane M=0.20 -> C = -1
Haldane M=0.25 -> C = -1
Haldane M=0.30 -> C = -1
Haldane M=0.35 -> C = +0
Haldane M=0.40 -> C = +0
Haldane M=0.45 -> C = +0
Haldane M=0.50 -> C = +0
[archived fence delimiter: ```]

The numerical transition lands within one grid spacing of the
analytic boundary `0.31177`, consistent with the half-step
discretisation error.

## 4. Kane-Mele model: quantum spin Hall insulator

The Kane-Mele model [7] is the canonical two-dimensional topological
insulator: a time-reversal-symmetric, four-band Hamiltonian whose
classification is governed by a Z_2 invariant rather than an
integer Chern number.

[archived fence delimiter: ```c]
qgt_system_n_t* sys = qgt_model_kane_mele(
    /*t=*/1.0,
    /*lambda_so=*/0.06,
    /*lambda_r=*/0.0,    /* Rashba off; S_z-conserving path */
    /*lambda_v=*/0.10);
int z2;
qgt_z2_invariant(sys, /*N=*/48, &z2);
printf("Kane-Mele lambda_v=0.10: Z_2 = %d (expected 1, QSH)\n", z2);
qgt_free_nband(sys);
[archived fence delimiter: ```]

In the S_z-conserving regime (Rashba coupling off) the Z_2
invariant reduces to `|C_up| mod 2`, where `C_up` is the Chern
number of the spin-up sub-block.  The invariant equals `1` (QSH
phase) for `|lambda_v| < 3 sqrt(3) |lambda_so|`.

## 5. Bernevig-Hughes-Zhang model: HgTe quantum well

The Bernevig-Hughes-Zhang (BHZ) model [8] describes the HgTe/CdTe
quantum well that hosts the experimentally observed two-dimensional
topological insulator.  On a square lattice the regularised model
has a QSH window `0 < M / B < 8`: the gap closings at the X-corners
of the Brillouin zone (at `M = 4B`) cancel pairwise, and the
M-corner closing at `M = 8B` returns the system to the trivial
phase.

[archived fence delimiter: ```c]
qgt_system_n_t* sys = qgt_model_bhz(/*A=*/1.0, /*B=*/1.0, /*M=*/3.0);
int z2;
qgt_z2_invariant(sys, 48, &z2);
printf("BHZ M=3.0: Z_2 = %d (expected 1, QSH)\n", z2);
qgt_free_nband(sys);
[archived fence delimiter: ```]

## 6. Hofstadter model: magnetic sub-band Chern numbers

Hofstadter [9] showed that a square-lattice tight-binding model in
a uniform flux `phi = p/q` per plaquette decomposes into a `q`-band
magnetic-Bloch Hamiltonian; the resulting spectrum is the celebrated
"Hofstadter butterfly".  The Thouless-Kohmoto-Nightingale-den Nijs
(TKNN) Diophantine equation `t_r p + s_r q = r` [10] determines the
integer Chern number of each magnetic sub-band.

[archived fence delimiter: ```c]
qgt_system_n_t* sys = qgt_model_hofstadter(/*t=*/1.0,
                                            /*p=*/1, /*q=*/3,
                                            /*n_occupied=*/1);
qgt_berry_grid_t g;
qgt_berry_grid_nband(sys, /*N=*/32, &g);
printf("Hofstadter q=3 lowest band: C = %+d (expected +1)\n",
       (int)lround(g.chern));
qgt_berry_grid_free(&g);
qgt_free_nband(sys);
[archived fence delimiter: ```]

For `q = 3` the three magnetic sub-bands carry Chern numbers
`(+1, -2, +1)`, summing to zero as required.  Filling the lowest two
bands therefore yields `n_occupied = 2` and a total Chern of `-1`.

## 7. Cross-check: real-space versus momentum-space invariants

Topological invariants must be representation-independent.  Moonlab
provides a real-space implementation of the Bianco-Resta local
Chern marker [11] in `src/algorithms/topology_realspace/chern_marker.h`.
On QWZ the two representations must agree:

[archived fence delimiter: ```c]
#include "moonlab/algorithms/topology_realspace/chern_marker.h"

double m = -1.5;
qgt_system_t* sys_k = qgt_model_qwz(m);
qgt_berry_grid_t g;
qgt_berry_grid_proj(sys_k, 48, &g);
int C_momentum = (int)lround(g.chern);

chern_system_t* sys_r = chern_qwz_create(/*L=*/10, m);
chern_build_projector(sys_r);
double bulk_sum = chern_bulk_sum(sys_r, 4, 6);  /* 2x2 bulk patch */
int C_realspace = (int)lround(bulk_sum / 4.0);

printf("QWZ m=%.2f: C_momentum=%+d, C_realspace=%+d (must match)\n",
       m, C_momentum, C_realspace);
[archived fence delimiter: ```]

This identity is verified by `tests/unit/test_qgt_vs_chern_marker.c`,
which asserts pointwise agreement of the two integer outputs on a
representative QWZ phase point.

## Further reading

- `./build_release/bench_topology_phase_diagrams
  benchmarks/results/topology.json` dumps a complete phase-diagram
  archive across all six models, suitable for plotting and
  reproducibility.
- `docs/reference/qgt-api.md` documents the full API surface.
- `docs/research/quantum_geometry_tensor.md` derives the geometry
  underlying the three Chern integrators and motivates the
  projector-trace gauge-invariant formulation.
- `tests/unit/test_qgt_*.c` provides ready-to-modify template
  programs for each model and integrator combination.

## References

[1] B. A. Bernevig and T. L. Hughes, *Topological Insulators and
    Topological Superconductors*, Princeton University Press, 2013.

[2] W. P. Su, J. R. Schrieffer, and A. J. Heeger, "Solitons in
    polyacetylene", Phys. Rev. Lett. **42**, 1698 (1979).

[3] J. Zak, "Berry's phase for energy bands in solids",
    Phys. Rev. Lett. **62**, 2747 (1989).

[4] X.-L. Qi, Y.-S. Wu, and S.-C. Zhang, "Topological quantization
    of the spin Hall effect in two-dimensional paramagnetic
    semiconductors", Phys. Rev. B **74**, 085308 (2006).

[5] T. Fukui, Y. Hatsugai, and H. Suzuki, "Chern numbers in
    discretized Brillouin zone: Efficient method of computing
    (spin) Hall conductances", J. Phys. Soc. Jpn. **74**, 1674
    (2005).

[6] F. D. M. Haldane, "Model for a quantum Hall effect without
    Landau levels: Condensed-matter realization of the 'parity
    anomaly'", Phys. Rev. Lett. **61**, 2015 (1988).

[7] C. L. Kane and E. J. Mele, "Z_2 topological order and the
    quantum spin Hall effect", Phys. Rev. Lett. **95**, 146802
    (2005).

[8] B. A. Bernevig, T. L. Hughes, and S.-C. Zhang, "Quantum spin
    Hall effect and topological phase transition in HgTe quantum
    wells", Science **314**, 1757 (2006).

[9] D. R. Hofstadter, "Energy levels and wave functions of Bloch
    electrons in rational and irrational magnetic fields",
    Phys. Rev. B **14**, 2239 (1976).

[10] D. J. Thouless, M. Kohmoto, M. P. Nightingale, and M. den Nijs,
    "Quantized Hall conductance in a two-dimensional periodic
    potential", Phys. Rev. Lett. **49**, 405 (1982).

[11] R. Bianco and R. Resta, "Mapping topological order in
    coordinate space", Phys. Rev. B **84**, 241106(R) (2011).
```
