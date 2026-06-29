# Archived Moonlab Documentation: libirrep + SbNN integration -- v1.0 commitment

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# libirrep + SbNN integration -- v1.0 commitment

Moonlab pairs with two sibling libraries for the hardware-design
workbench story:

- [**libirrep**](https://github.com/tsotchke/libirrep) -- production-grade
  QEC + rep-theory substrate.  18-module QEC code zoo, sector ED,
  Clebsch-Gordan tables for SO(3) / SU(2) / O(3) / SE(3).
- [**SbNN**](https://github.com/tsotchke/spin-based-neural-network) -- spin-based
  neural networks, equivariant ML / neural-net decoders for QEC.

Moonlab provides the substrate; the siblings consume it via the
opt-in bridges documented here.

## What v1.0 ships

| Bridge      | Build flag                 | Default | Without flag                       |
|-------------|----------------------------|---------|------------------------------------|
| libirrep    | `QSIM_ENABLE_LIBIRREP=ON`  | OFF     | every call returns `MOONLAB_LIBIRREP_NOT_BUILT` (-201) |
| SBNN decoder| `QSIM_ENABLE_SBNN=ON`      | OFF     | `decoder_bench` dispatcher returns "not built" for the SBNN slot |

Both bridges are **always linkable** from C, Python, Rust, JS.  The
opt-in flag controls whether the bridge actually does work or returns
the explicit "not built" status.  This is a deliberate design choice:
moonlab does not silently degrade.  Callers `if (rc == -201) ...` to
detect a build-time opt-out and fall back as they choose.

## libirrep coverage

When built with `-DQSIM_ENABLE_LIBIRREP=ON`, the following APIs in
`src/integration/libirrep_bridge.h` are live:

### Sector ED

[archived fence delimiter: ```c]
int moonlab_libirrep_kagome12_e0(double *out_energy);
int moonlab_libirrep_heisenberg_sector_e0(
        int N, const int *lattice_topology, ...);
[archived fence delimiter: ```]

The 12-site kagome reference energy E0 = -5.44487522 used as ground
truth in the moonlab CA-MPS validation suite is computed live, not
hardcoded.  General sector ED runs at N up to 24 (limited by
libirrep's sparse-LANCZOS budget, not moonlab).

### QEC code zoo

| Family                | Factory                                       |
|-----------------------|-----------------------------------------------|
| Surface (rotated)     | `moonlab_libirrep_surface_code_new(d, out)`   |
| Toric (Lx × Ly)       | `moonlab_libirrep_toric_code_new(Lx, Ly, out)`|
| 2D color (Steane)     | `moonlab_libirrep_color_steane_new(out)`      |
| 2D color (Hamming)    | `moonlab_libirrep_color_hamming_15_7_3_new(out)` |
| Bivariate bicycle     | `moonlab_libirrep_bb_72_12_6_new(out)`        |
|                       | `moonlab_libirrep_bb_144_12_12_new(out)`      |
|                       | `moonlab_libirrep_bb_288_12_18_new(out)`      |
| Hypergraph product    | `moonlab_libirrep_hgp_repetition_new(d, out)` |

Each opaque `moonlab_libirrep_qec_t` exposes:

- `n_qubits` -- physical qubit count
- `n_x_stabs`, `n_z_stabs` -- stabilizer counts
- `logical_qubits` -- k
- (no `distance` getter yet -- callers compute via decoder shoot-out)

The factories validate against the published `[[n, k, d]]` shape via
`tests/integration/test_libirrep_qec_*`.  Failure to load libirrep at
runtime is reported with an explicit error, not a silent fallback.

### Cross-language coverage

| Capability                   | C   | Python              | Rust                | JS                  |
|------------------------------|-----|---------------------|---------------------|---------------------|
| libirrep bridge availability check | ✅ | `libirrep_qec.py`  | `libirrep_qec.rs`  | `libirrep-qec.ts`  |
| 6 QEC code factories         | ✅  | `libirrep_qec.py`   | `libirrep_qec.rs`   | `libirrep-qec.ts`   |
| Sector ED                    | ✅  | `algorithms.py` ¹  | -                   | -                   |

¹ Python wraps the `moonlab_libirrep_kagome12_e0` and Heisenberg
sector ED calls; Rust / JS do not -- the use case is research-grade,
not cloud-runtime.

## SBNN coverage

The decoder zoo in `src/applications/decoder_bench.{c,h}` dispatches
between five decoders:

| Slot                 | C tag                      | Built by default? |
|----------------------|----------------------------|-------------------|
| GREEDY (in-tree)     | `MOONLAB_DECODER_GREEDY`   | yes               |
| MWPM (exact)         | `MOONLAB_DECODER_MWPM_EXACT` | yes             |
| MWPM (approx)        | `MOONLAB_DECODER_MWPM_APPROX`| yes             |
| SBNN (learned)       | `MOONLAB_DECODER_SBNN`     | `-DQSIM_ENABLE_SBNN=ON` |
| LIBIRREP_SS (single-shot) | `MOONLAB_DECODER_LIBIRREP_SS` | `-DQSIM_ENABLE_LIBIRREP=ON` |
| PYMATCHING (subprocess) | `MOONLAB_DECODER_PYMATCHING` | yes (no compile-time linkage) |

When `QSIM_ENABLE_SBNN=ON`, moonlab links against SbNN's
`qec_decoder_*` API.  The SBNN slot then handles syndrome decoding
in the moonlab harness as just-another-decoder.

Without `QSIM_ENABLE_SBNN`, `moonlab_decoder_decode(..., MOONLAB_DECODER_SBNN, ...)`
returns `MOONLAB_DECODER_NOT_BUILT = -601`.  Same explicit-status
convention as the libirrep bridge.

## Build matrix

| Configuration                | Capabilities                                          |
|------------------------------|-------------------------------------------------------|
| default (no flags)           | core simulator + control plane + 3 decoder slots      |
| `+libirrep`                  | + 6 QEC code factories + sector ED + LIBIRREP_SS slot |
| `+SBNN`                      | + SBNN learned-decoder slot                           |
| `+libirrep + SBNN`           | full sibling-library coverage                         |
| `+MPI`                       | distributed state-vector (orthogonal axis)            |
| `+TLS`                       | control-plane TLS + mTLS (orthogonal axis)            |

## Sample integration code

C consumer that opts in cleanly:

[archived fence delimiter: ```c]
#include "moonlab/integration/libirrep_bridge.h"
#include <stdio.h>

int main(void) {
    int avail = moonlab_libirrep_available();
    if (avail != 1) {
        fprintf(stderr,
                "rebuild with -DQSIM_ENABLE_LIBIRREP=ON to use this demo\n");
        return 1;
    }

    moonlab_libirrep_qec_t *code = NULL;
    int rc = moonlab_libirrep_surface_code_new(7, &code);
    if (rc != 0) {
        fprintf(stderr, "factory rc=%d\n", rc);
        return 1;
    }
    printf("d=7 rotated surface code: n=%d, k=%d\n",
           moonlab_libirrep_qec_n_qubits(code),
           moonlab_libirrep_qec_logical_qubits(code));
    moonlab_libirrep_qec_free(code);
    return 0;
}
[archived fence delimiter: ```]

## Deferred to v1.1

The following integration depth did NOT ship in v1.0:

- **SBNN training-time bridge.**  v1.0 only consumes a *pre-trained*
  SBNN model at inference; you cannot train an SBNN network from the
  moonlab side (you train via SbNN, then load).
- **libirrep irrep-resolved sector ED beyond N=24.**  Requires a
  scratch-disk-backed sparse solver inside libirrep that's still on
  its v2 roadmap.
- **Hardware-aware QEC code factories.**  The current factories take
  pure code parameters (d, Lx, Ly, ...); hardware-noise-tailored
  variants are deferred until QGTL ships its noise-model bridge.
- **JS topological-LGT.**  `bindings/javascript/.../topology.ts` does
  not yet expose the Z2 1+1D lattice gauge theory routines.  This is
  not a hard constraint -- the bridge is C-side -- but the use case
  is research, not browser-runtime.

## See also

- `docs/PARITY_MATRIX.md` -- which capability is wired through which binding
- `docs/STABLE_ABI.md` -- the v1.0 contract
- libirrep repo: <https://github.com/tsotchke/libirrep>
- SbNN repo: <https://github.com/tsotchke/spin-based-neural-network>
```
