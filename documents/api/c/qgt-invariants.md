# QGT Topology Invariants C API

Scalar-in / scalar-out band-topology invariants exposed on the stable ABI surface
(`src/applications/moonlab_export.h`). Each function bundles model construction, invariant
calculation, and teardown into a single call, so a binding consumer never marshals an opaque
model handle across the FFI boundary. The heavier opaque-handle QGT surface lives in the
internal `src/algorithms/quantum_geometry/qgt.h`; these one-shots are the frozen convenience
layer.

Every function returns the **integer invariant**, or `INT_MIN` on allocation or argument
error. Intended consumers: QGTL, libirrep, SbNN, and the JS/WASM TypeScript layer.

## Header

```c
#include "applications/moonlab_export.h"
#include <limits.h>   /* INT_MIN sentinel */
```

## Invariants

### `moonlab_qwz_chern` (since 0.2.0)

```c
int moonlab_qwz_chern(double m, size_t N, double* out_chern);
```

Chern number of the Qi-Wu-Zhang (QWZ) lower band on an `N x N` Brillouin-zone grid
(`N >= 4`; 32 is plenty for a clean integer). `m` is the QWZ mass parameter. `out_chern` is
optional (may be `NULL`); when non-NULL it receives the raw pre-rounding Chern value.

### `moonlab_ssh_winding` (since 0.6.0)

```c
int moonlab_ssh_winding(double t1, double t2, size_t N);
```

Winding number of the Su-Schrieffer-Heeger chain (`t1` intra-cell hopping, `t2` inter-cell)
on an `N`-point Brillouin-zone grid (`N >= 4`).

### `moonlab_kitaev_chain_z2` (since 0.6.0)

```c
int moonlab_kitaev_chain_z2(double t, double mu, double delta);
```

Z2 (0/1) invariant of the Kitaev p-wave chain in its Bogoliubov-de Gennes form for hopping
`t`, chemical potential `mu`, and pairing `delta`.

### `moonlab_chern_qwz_proj` (since 0.6.0)

```c
int moonlab_chern_qwz_proj(double m, size_t N);
```

Chern number of the QWZ lower band via the projector `(P dP dP)` Berry-curvature grid on an
`N x N` BZ mesh (`N >= 4`).

### `moonlab_chern_qwz_pt` (since 0.6.0)

```c
int moonlab_chern_qwz_pt(double m, size_t N);
```

Chern number of the QWZ lower band via the parallel-transport (Wilson-loop) Berry-curvature
grid on an `N x N` BZ mesh.

### `moonlab_kane_mele_z2` (since 0.6.0)

```c
int moonlab_kane_mele_z2(double t, double lambda_so,
                         double lambda_r, double lambda_v, size_t N);
```

Z2 invariant of the Kane-Mele model (`t` hopping, `lambda_so` spin-orbit, `lambda_r` Rashba,
`lambda_v` sublattice potential) on an `N x N` mesh (`N >= 8`, even).

### `moonlab_bhz_z2` (since 0.6.0)

```c
int moonlab_bhz_z2(double A, double B, double M, size_t N);
```

Z2 invariant of the Bernevig-Hughes-Zhang model (`A`, `B`, `M`) on an `N x N` mesh
(`N >= 8`, even).

### `moonlab_hofstadter_chern` (since 0.6.0)

```c
int moonlab_hofstadter_chern(double t, size_t p, size_t q,
                             size_t n_occupied, size_t N);
```

Chern number of the Hofstadter model at flux `p / q` with `n_occupied` filled sub-bands on
an `N x N` mesh (`N >= 4`, `q >= 2`, `1 <= n_occupied < q`).

## Example

```c
#include "applications/moonlab_export.h"
#include <limits.h>
#include <stdio.h>

int main(void) {
    // QWZ Chern number sweeps from 0 to -1 to +1 as m crosses the band-touchings.
    for (double m = -3.0; m <= 3.0; m += 1.0) {
        int c = moonlab_chern_qwz_proj(m, 32);
        if (c == INT_MIN) { fprintf(stderr, "error at m=%.1f\n", m); continue; }
        printf("m = %+.1f  Chern = %d\n", m, c);
    }

    // SSH is topological (winding 1) when the inter-cell hopping dominates.
    printf("SSH winding (t1=0.5, t2=1.0): %d\n", moonlab_ssh_winding(0.5, 1.0, 64));
    return 0;
}
```

## See Also

- [Topological Computing API](topological.md) - anyon models, braiding, surface codes
- [C API Index](index.md)
