# Reproducibility manifest — Moonlab v0.2.1 paper

Each row maps a paper claim to: the harness binary that produces it, the
exact command, the archived JSON path, and the tolerance the paper
asserts.  Re-run on your own host to confirm.

Build:
```sh
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release \
  -DQSIM_BUILD_TESTS=ON -DQSIM_BUILD_EXAMPLES=ON -DQSIM_BUILD_BENCHMARKS=ON
cmake --build build_release -j8
```

All paper-grade benches must run from a Release build.  The
`build_type` field in each manifest-style JSON is asserted to equal
`Release` (one prior-cycle Debug-build JSON,
`chern_mosaic_L96_V0_0p2_M2Ultra_2026-05-01.json`, has been
superseded by the 2026-05-02 Release-build archive).

| Paper section | Claim | Harness | Command | JSON archive | Asserted tolerance |
| --- | --- | --- | --- | --- | --- |
| §3.2 Tab. 2 | State-vector gate throughput on M2 Ultra at $n \in [16, 26]$, mean$\pm$stddev over $k=5$ reps | `bench_state_throughput` | `./bench_state_throughput state_throughput_v2.json 5` | `state_throughput_v2_M2Ultra_2026-05-02.json` | mean within $\pm$3 stddev of archive on cold cache, host-dependent |
| §4.1 | Kagome AFM Heisenberg cluster 18b $E_0 = -8.048270773\,J$ matched to $5.4\times 10^{-10}\,J$ | `test_kagome_ed_large` (label `long`) | `ctest -L long -R kagome_ed_large` | `kagome_ed_18b_M2Ultra_2026-05-01.json` | $|\Delta E| \le 10^{-9}\,J$ (machine-precision Lanczos) |
| §4.2 | CHSH violation $\widetilde{\mathrm{CHSH}} = 2.81 \pm 0.025$ over 10 runs | `example_bell_chsh_aggregate` | `./example_bell_chsh_aggregate bell_chsh.json 10 10000` | `bell_chsh_aggregate_2026-05-01.json` | mean $\in [2.7, 2.83]$, all 10 runs $> 2$ |
| §4.2 (variants) | Mermin-3 + Mermin–Klyshko-{4,5} on GHZ states | `example_bell_variants_aggregate` | `./example_bell_variants_aggregate bell_v.json 10 10000` | `bell_variants_aggregate_M2Ultra_2026-05-04.json` | every variant: 10/10 runs violate classical bound; saturate quantum bound |
| §4.3 | QWZ marker at $L=12$, $m=-1$: bulk-patch mean $0.9998$ | `test_chern_marker`, `test_chern_kpm` | `ctest -R chern` | (regression assertion in test) | $|c - 1| < 5 \times 10^{-3}$ at gapped point |
| §4.4 Tab. 6 | Toric-code threshold sweep, $d \in \{3,5,7,9\}$, MWPM | `example_surface_code_threshold` | `./example_surface_code_threshold sc_threshold.json 5000` | `surface_code_threshold_v2_2026-05-02.json` | $p_{\rm log}(d=9, p=0.030) \le p_{\rm log}(d=3, p=0.030)$, threshold crossing visible |
| §4.5 Tab. 4 | CA-MPS performance vs plain MPS on six circuit classes | `bench_ca_mps` | `./bench_ca_mps ca_mps_bench.json` | `ca_mps_v3_2026-04-28.json` | sign-of-effect matches table per circuit class |
| §4.6 Tab. 5 | var-D vs DMRG TFIM at $n=8$: rel err $\in [0.1\%, 0.5\%]$, $S_{|\phi\rangle}/S_{|\psi\rangle}$ sweep | `example_ca_mps_var_d_vs_plain_dmrg` | `./example_ca_mps_var_d_vs_plain_dmrg vd_vs_dmrg.json` | `ca_mps_var_d_vs_plain_dmrg_2026-04-29.json` | $|\Delta E|/|E_{\rm DMRG}| \le 5 \times 10^{-3}$ |
| §4.7 Tab. 7 | CA-PEPS row-major imag-time vs dense ED, 2D TFIM at $L_x \times L_y \in \{2\!\times\!2, 3\!\times\!2, 3\!\times\!3, 4\!\times\!3\}$ | `example_ca_peps_2d_tfim_vs_ed` | `./example_ca_peps_2d_tfim_vs_ed peps_vs_ed.json` | `ca_peps_2d_tfim_vs_ed_2026-05-01.json` | rel err $\le 3 \times 10^{-3}$ at $n \le 12$ |
| §4.7 (caption) | CA-PEPS var-D companion at $n \in \{4, 6\}$ | `example_ca_peps_2d_tfim_var_d_vs_ed` | `./example_ca_peps_2d_tfim_var_d_vs_ed peps_vd.json` | `ca_peps_2d_tfim_var_d_vs_ed_2026-05-01.json` | rel err $\le 5 \times 10^{-4}$ at $n \le 6$ |
| §4.8 Tab. 8 | Pauli-frame batched throughput at $d \in \{5,9,15,23,31\}$ + Stim head-to-head | `bench_pauli_frame`, `tests/performance/stim_vs_moonlab.py` | `./bench_pauli_frame pf.json 10000 10 0.005`; `python3 tests/performance/stim_vs_moonlab.py` | `pauli_frame_M2Ultra_2026-05-01.json`, `stim_vs_moonlab_M2Ultra_2026-05-02.json` | Moonlab frame-ops/s $> 10^9$; Moonlab/Stim ratio $> 10$ on M2 Ultra |
| §4.9 (HOP) | IBM QV protocol noiseless, mean HOP $\sim 0.85$ across $w \le 10$ | `bench_quantum_volume` | `./bench_quantum_volume qv.json` | `quantum_volume_M2Ultra_2026-05-01.json` | every width: lower 97.5% CI on HOP $> 2/3$ |
| §4.9 (HWEA iters/s) | HWEA circuits-per-second at $L=5$, $n \in \{8,12,16,20\}$ | `bench_clops` | `./bench_clops clops.json` | `clops_M2Ultra_2026-05-01.json` | within $\pm 30\%$ of archive on M2 Ultra |
| §4.9 (PT distance) | $F_{\rm xeb}$ on noiseless $L=10$ HWEA | `bench_xeb` | `./bench_xeb xeb.json` | `xeb_M2Ultra_2026-05-01.json` | exact and sampled estimators agree within shot noise |
| §4.9 (fusion) | Gate-fusion DAG speedup vs unfused | `bench_fusion` | `./bench_fusion fusion.json` | `fusion_M2Ultra_2026-05-01.json` | speedup $> 4 \times$ at $n \ge 12$, $L = 5$ |
| §4.10 Tab. (cross-bk TFIM) | TFIM $n=8$ ground-state energy via ED, DMRG, var-D | `bench_cross_backend_tfim` | `./bench_cross_backend_tfim cross_bk.json 8` | `cross_backend_tfim_n8_M2Ultra_2026-05-04.json` | DMRG: $|\Delta E|/|E_{\rm ED}| \le 10^{-6}$; var-D: $\le 10^{-3}$ |
| §4.10 Tab. (cross-bk XXZ) | Heisenberg XXZ $n=6$, sweep $\Delta \in \{0, 0.5, 1, 1.5, 2\}$ | `bench_cross_backend_xxz` | `./bench_cross_backend_xxz cross_xxz.json 6` | `cross_backend_xxz_n6_M2Ultra_2026-05-04.json` | DMRG: $|\Delta E|/|E_{\rm ED}| \le 10^{-9}$; var-D underperforms (17%-24%) on SU(2)-symmetric points |
| §4.11 | Chern marker $L=96$, $V_0=0.2$, $\langle c \rangle = 0.9998$ | `bench_chern_mosaic_hq` | `MOONLAB_MANIFEST_OUT=chern.json ./bench_chern_mosaic_hq --L 96 --V0 0.2` | `chern_mosaic_L96_V0_0p2_M2Ultra_2026-05-02_release.json` | $|\langle c \rangle - 1| \le 5 \times 10^{-4}$, build_type=Release |
| §4.12 | Test-suite ctest wall on M2 Ultra | `ctest --label-exclude '(long|aarch64_flake|exhaustive)' -j8` | `ctest --label-exclude '(long\|aarch64_flake\|exhaustive)' -j8` | `test_suite_walltime_M2Ultra_2026-05-01.json` | all_passed=true |

## Superseded archives (do not cite in v0.2.1+ paper text)

These earlier-cycle JSONs remain in the directory for diff history but
are NOT referenced in the current paper.  Citing them in v0.2.1+ work
is a regression.

| Superseded JSON | Reason | Replacement |
| --- | --- | --- |
| `state_throughput_M2Ultra_2026-05-01.json` | single-run, omitted $n \in \{16, 18, 20\}$ | `state_throughput_v2_M2Ultra_2026-05-02.json` |
| `surface_code_threshold_2026-05-01.json` | structurally broken (anti-threshold), planar-code geometry mismatch | `surface_code_threshold_v2_2026-05-02.json` |
| `chern_mosaic_L96_V0_0p2_M2Ultra_2026-05-01.json` | Debug build, 254s wall | `chern_mosaic_L96_V0_0p2_M2Ultra_2026-05-02_release.json` (25.4s) |
| `ca_mps_v2_2026-04-28.json` | superseded by v3 schema bump | `ca_mps_v3_2026-04-28.json` |
| `ca_mps_disentangler_2026-04-29.json` | exploratory, not paper-cited | (none) |
| `ca_mps_oracle_proof_2026-04-28.json` | exploratory, not paper-cited | (none) |
| `ca_mps_var_d_v4_2026-04-29.json` | exploratory, not paper-cited | (none) |
| `ca_mps_var_d_chi_scan_2026-04-29.json` | exploratory, not paper-cited | (none) |
| `ca_mps_var_d_vs_plain_2026-04-28.json` | superseded by `_vs_plain_dmrg_2026-04-29.json` (DMRG bond-dim cleaner) | `ca_mps_var_d_vs_plain_dmrg_2026-04-29.json` |

## Host figures are single-host

Every per-host throughput / wall-time number in the paper was measured
on a single Apple M2 Ultra (Atlas.local, 24 cores, 192 GB RAM,
macOS 24.1.0, AppleClang 16.0).  No cross-platform multipliers are
asserted.  Re-run the harnesses listed above on your own hardware to
get host-specific figures.

## How to verify the manifest itself

```sh
# 1. fresh release build
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release \
  -DQSIM_BUILD_TESTS=ON -DQSIM_BUILD_EXAMPLES=ON -DQSIM_BUILD_BENCHMARKS=ON
cmake --build build_release -j8

# 2. ctest gate (excludes long / aarch64_flake / exhaustive)
ctest --test-dir build_release --label-exclude '(long|aarch64_flake|exhaustive)' -j8

# 3. paper benches (writes to /tmp/repro/, compare to benchmarks/results/)
mkdir -p /tmp/repro
cd build_release
./bench_state_throughput      /tmp/repro/state_throughput.json 5
./bench_chern_mosaic_hq       --L 96 --V0 0.2 \
  ; mv ${MOONLAB_MANIFEST_OUT:-/dev/null} /tmp/repro/chern_mosaic_L96.json 2>/dev/null
./bench_cross_backend_tfim    /tmp/repro/cross_backend.json 8
./example_surface_code_threshold /tmp/repro/sc_threshold.json 5000
./bench_pauli_frame           /tmp/repro/pauli_frame.json 10000 10 0.005
./bench_quantum_volume        /tmp/repro/qv.json
./bench_clops                 /tmp/repro/clops.json
./bench_xeb                   /tmp/repro/xeb.json
./bench_fusion                /tmp/repro/fusion.json
./example_bell_chsh_aggregate /tmp/repro/bell_chsh.json 10 10000
python3 ../tests/performance/stim_vs_moonlab.py --out /tmp/repro/stim_vs_moonlab.json
```
