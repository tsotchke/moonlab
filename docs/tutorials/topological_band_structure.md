# Tutorial: Topological band structure with the QGT module

This tutorial walks through Moonlab's quantum-geometric-tensor module
on five canonical models, ordered from simplest to richest.  By the
end you will have computed Chern numbers and Z_2 invariants from
momentum-space Bloch Hamiltonians and verified each against its
analytical phase boundary.

Prerequisites:

- Built `libquantumsim` with `-DQSIM_BUILD_TESTS=ON
  -DQSIM_BUILD_BENCHMARKS=ON`.
- Familiarity with second-quantised tight-binding Hamiltonians and
  Berry-phase topology at the level of Bernevig & Hughes
  *Topological Insulators and Topological Superconductors*.

We work entirely in C; Python and Rust wrappers for the same
primitives land in v0.3.x.

## 1. SSH chain — the warm-up

Su-Schrieffer-Heeger (1979) is the simplest 1D topological model:
two-site unit cell with intra-cell hopping `t1` and inter-cell `t2`.
Topological for `|t2| > |t1|`.

```c
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
```

The integer winding number is computed via the discrete Zak phase
along the 1D BZ.  Try `t1=1.0, t2=0.5` — you should get `W = 0`
(trivial phase).

## 2. QWZ — minimal 2D Chern insulator

Qi-Wu-Zhang (2006) is the 2-band Chern insulator that lives in every
topology textbook.  Two integer Chern jumps as `m` sweeps `[-3, 3]`.

```c
qgt_system_t* sys = qgt_model_qwz(/*m=*/-1.5);
qgt_berry_grid_t g;
qgt_berry_grid_proj(sys, /*N=*/48, &g);
printf("QWZ m=-1.5: C = %+.4f (expected +1)\n", g.chern);
qgt_berry_grid_free(&g);
qgt_free(sys);
```

`qgt_berry_grid_proj` is the projector-trace integrator (gauge-free
by construction); `qgt_berry_grid` and `qgt_berry_grid_pt` give the
same answer via different gauge prescriptions.  At the gap-closing
points `m = 0, ±2` the Chern is technically undefined; on the
half-step grid the integrator reports an integer that's an artifact
of the discretisation.

## 3. Haldane — the parent honeycomb Chern insulator

Haldane (1988) is the first model that shows a Chern insulator
without a uniform magnetic field.  The phase boundary is at
`|M| = 3*sqrt(3)*|t2*sin(phi)|`.  At fixed `phi = pi/2` and
`t2 = 0.06`, the boundary is `M ≈ 0.31177`:

```c
for (double M = 0.0; M <= 0.5; M += 0.05) {
    qgt_system_t* sys = qgt_model_haldane(1.0, 0.06, 0.5*M_PI, M);
    qgt_berry_grid_t g;
    qgt_berry_grid_proj(sys, 48, &g);
    int C = (int)lround(g.chern);
    printf("Haldane M=%.2f -> C = %+d\n", M, C);
    qgt_berry_grid_free(&g);
    qgt_free(sys);
}
```

Output:

```
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
```

The transition lands within one M-grid spacing of the analytical
boundary `0.31177`.

## 4. Kane-Mele — quantum spin Hall

Kane-Mele (2005) is the canonical 2D TI: time-reversal-symmetric,
4-band, has a Z_2 invariant rather than an integer Chern.

```c
qgt_system_n_t* sys = qgt_model_kane_mele(
    /*t=*/1.0,
    /*lambda_so=*/0.06,
    /*lambda_r=*/0.0,    /* Rashba off; full Pfaffian path lands in v0.3.x */
    /*lambda_v=*/0.10);
int z2;
qgt_z2_invariant(sys, /*N=*/48, &z2);
printf("Kane-Mele lambda_v=0.10: Z_2 = %d (expected 1, QSH)\n", z2);
qgt_free_nband(sys);
```

In the Sz-conserving regime (Rashba off) the Z_2 invariant equals
`|C_up| mod 2`, where `C_up` is the spin-up sub-block's Chern.  The
invariant is `1` (QSH) for `|lambda_v| < 3*sqrt(3) * |lambda_so|`.

## 5. BHZ — square-lattice TI (HgTe quantum well)

BHZ has a similar Z_2 invariant on a square lattice with a tunable
mass.  Lattice regularisation gives the QSH window
`0 < M / B < 8` (the X-corner closings at `M = 4B` cancel; the
M-corner closing at `M = 8B` re-trivialises).

```c
qgt_system_n_t* sys = qgt_model_bhz(/*A=*/1.0, /*B=*/1.0, /*M=*/3.0);
int z2;
qgt_z2_invariant(sys, 48, &z2);
printf("BHZ M=3.0: Z_2 = %d (expected 1, QSH)\n", z2);
qgt_free_nband(sys);
```

## 6. Hofstadter — fractal sub-band Cherns

Hofstadter (1976) on a square lattice in flux `phi = p/q` per
plaquette gives a `q`-band magnetic-Bloch Hamiltonian.  The famous
TKNN Diophantine equation `t_r p + s_r q = r` determines each
sub-band's integer Chern.

```c
qgt_system_n_t* sys = qgt_model_hofstadter(/*t=*/1.0,
                                            /*p=*/1, /*q=*/3,
                                            /*n_occupied=*/1);
qgt_berry_grid_t g;
qgt_berry_grid_nband(sys, /*N=*/32, &g);
printf("Hofstadter q=3 lowest band: C = %+d (expected +1)\n",
       (int)lround(g.chern));
qgt_berry_grid_free(&g);
qgt_free_nband(sys);
```

For `q = 3` the three bands have Chern `(+1, -2, +1)` — the lowest
plus middle gives `n_occupied = 2` Chern `-1`.

## 7. Cross-check: real-space ↔ momentum-space

A topological invariant should not depend on which representation we
use.  Moonlab ships a real-space Bianco-Resta local Chern marker
implementation in `src/algorithms/topology_realspace/chern_marker.h`.
On QWZ we can verify the two paths agree:

```c
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
```

This is the test in `tests/unit/test_qgt_vs_chern_marker.c`.  Both
paths must return the same integer for any gapped 2-band Hamiltonian.

## Where to go from here

- Run `./build_release/bench_topology_phase_diagrams
  benchmarks/results/topology.json` to dump a full phase-diagram
  archive across all six models.  Useful for plotting.
- See `docs/reference/qgt-api.md` for the full API.
- See `docs/research/quantum_geometry_tensor.md` for the theory
  underlying the three integrators and the design rationale.
- Check the `tests/unit/test_qgt_*.c` files for ready-to-modify
  template programs.
