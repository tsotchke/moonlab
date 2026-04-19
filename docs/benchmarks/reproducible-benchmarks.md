# Reproducible benchmarks

Every benchmark that matters for a v0.2+ release writes a
reproducibility manifest alongside its human-readable output.  The
manifest captures the git SHA, build flags, compiler, host info, run
timestamps, and per-benchmark metrics in a single JSON file that a
third party can diff against their own run.

The machinery is defined in `src/utils/manifest.{c,h}` and the
CMake-generated `build/generated/moonlab_build_info.h`.  Any bench
that links `quantumsim` can opt in by capturing a
`moonlab_manifest_t`, attaching a metrics JSON fragment, and calling
`moonlab_manifest_write_json_pretty` when `MOONLAB_MANIFEST_OUT` is
set in the environment.

Benches currently wired in:

| Bench                              | What it measures                                           | Env vars consumed |
|------------------------------------|------------------------------------------------------------|-------------------|
| `bench_tensor_matmul_eshkol`       | CPU-forced vs Eshkol-dispatched tensor_matmul             | `MOONLAB_MANIFEST_OUT` |
| `bench_dmrg_workspace`             | Legacy calloc-per-apply vs persistent-workspace H_eff     | `MOONLAB_MANIFEST_OUT` |
| `bench_chern_kpm`                  | Matrix-free local Chern marker scaling in L               | `MOONLAB_MANIFEST_OUT` |
| `bench_chern_mosaic_hq`            | Full bulk Chern mosaic for a modulated QWZ lattice        | `MOONLAB_MANIFEST_OUT`, `MOONLAB_CHERN_OUT_CSV`, `MOONLAB_CHERN_OUT_PPM` |

## Reproducing a mosaic render

```
mkdir -p /tmp/moonlab-chern
MOONLAB_CHERN_OUT_CSV=/tmp/moonlab-chern/c4.csv \
MOONLAB_CHERN_OUT_PPM=/tmp/moonlab-chern/c4.ppm \
MOONLAB_MANIFEST_OUT=/tmp/moonlab-chern/c4_manifest.json \
./build/bench_chern_mosaic_hq \
  --L 64 --n 4 --V0 0.3 --Q 0.8976 --n-cheby 200
```

The PPM output is a 56x56 raw-pixmap; open it with any image viewer
that handles netpbm (Preview.app, ImageMagick, Pillow).  The
accompanying manifest pins the full provenance:

```
{
  "run_label": "bench_chern_mosaic_hq",
  "git_sha": "3a6c4b7...",
  "git_branch": "master",
  "compiler_id": "AppleClang",
  "platform": "macOS",
  "cpu_brand": "Apple M2 Ultra",
  ...
  "metrics": { "L": 64, "n_sym": 4, "V0": 0.3, "Q": 0.8976,
               "sites": 3136, "mean": 0.9807, "wall_s": 4.06, ... }
}
```

In the deep-topological regime (`V0 <= 0.3` for C_4) the per-site
marker averages very close to +1 across the bulk, with the outer few
rings contaminated by the open boundary.  Increasing `V0` into the
transition regime produces the "mosaic" structure -- spatial islands
of +1 and 0 markers -- that is the Antao-Sun-Fumega-Lado signature
in quasicrystals (PRL 136, 156601 (2026)).  The current
sparse-stencil backend caps out around `L = 300`; the MPO/QTCI
milestones (P5.08) lift that to the 10^6-10^8-site regime the paper
actually probes.

## Diffing two runs

Because every manifest is self-describing JSON, a simple `jq` pass
suffices to confirm that two runs on different hosts match within
numerical tolerance:

```
jq '.metrics.mean' run-a.json run-b.json
jq '.metrics.wall_s / .metrics.sites' run-*.json
```

The manifest format is designed so `git_dirty: 1` is a loud warning:
anyone who reproduces a number against a dirty tree has an honest
audit trail showing so.  Don't report headline numbers from dirty
trees for any release.
