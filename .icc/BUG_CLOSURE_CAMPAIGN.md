# Moonlab v1.2.0 bug-closure campaign (ICC-gated)

The 2026-07-17 adversarial hunt (oracle P1-P6 + fuzz + differential + statistical
+ deep lanes: TSan, numerical, scaling) surfaced **10 real defects**. This is the
ICC-gated campaign to close every one before the v1.2.0 tag. ICC gates the tag:
`icc readiness --repo moonlab --target moonlab-release-readiness --trace-dir scripts/icc_traces`
must be `ready`, which requires the `all_discovered_bugs_closed` criterion, which
requires every quarantine allowlist below to be **empty** and every deep-hunt lane
event to be **clean**.

## The rule
A bug is CLOSED only when: (1) a regression test that fails-before / passes-after is
committed, (2) its quarantine entry is removed from the allowlist, (3) the owning
lane's harness re-runs green, (4) validated on the mesh where hardware-relevant.
Never loosen a tolerance, delete a probe, or narrow a corpus to make HEAD pass.

## Ledger

| # | bug | severity | file | regression / gate | owner | status |
|---|-----|----------|------|-------------------|-------|--------|
| 1 | reversed-CNOT 2q transpose (also OOB) | HIGH | tn_gates.c | oracle backend_differential (reversed_2q) | tn | CLOSED |
| 2 | control-plane resource-amplification DoS (64 GB alloc) | MED-HIGH | control_plane.c + moonlab_qgtl_backend.c | fuzz control_plane_protocol; crashes-pending/ empty | fuzz/integrator | OPEN (task 23) |
| 3 | n>=10 tn_mps Metal read-out norm | HIGH | tn_gates.c (Metal) | unit_tn_mps_deep_forward; oracle | tn | CLOSED |
| 4 | SVD-fallback non-orthonormal U (no-LAPACK/Windows) | HIGH | tensor.c:1356 | t_svd_fallback; numerical_edge_clean | tn/integrator | OPEN (task 28) |
| 5 | qsim_config_global() singleton race | HIGH | config.c:127/148 | conc_core_init; tsan_clean | integrator | OPEN (task 29) |
| 6 | SIMD dispatch vtable race (from dispatch rewrite) | HIGH | simd_ops.c:762/120 | conc_core_init; tsan_clean | tn/integrator | OPEN (task 29) |
| 7 | control-plane setter races (doc says thread-safe) | MED | control_plane.c:1296+ | conc_control_plane; tsan_clean | integrator | OPEN (task 29) |
| 8 | audit-buffer destroy deadlock | HIGH | audit_buffer.c:76-87 | conc_audit_buffer; tsan_clean | integrator | OPEN (task 29) |
| 9 | tn_mps bulk-bond 2q-gate norm (wider than #3, CPU, n=4) | HIGH | tn_gates.c:701 | scaling S1 (ghz_chain n<=4); oracle | tn | OPEN (task 30) |
| 10 | real-time TDVP wrong for chains > 2 sites | HIGH | tdvp.c | unit_tdvp_bulk_site (n>=3 vs dense); scaling S2 | tn | OPEN (task 30) |

Adjacent (not from the hunt, tracked): QRNG multi-MB draw robustness (task 24, nightly
battery), wheel Accelerate -Werror (task 22), consumer OpenMP linkage (task 27).

## ICC gate wiring
- `moonlab-adversarial-matrix` consumes: moonlab_oracle (5 pillars), moonlab_fuzz,
  moonlab_differential, moonlab_statistical, moonlab_tsan (tsan_clean),
  moonlab_numerical (numerical_edge_clean, uninit_clean), moonlab_scaling
  (scaling_differential_clean).
- `moonlab-release-readiness` requires `all_discovered_bugs_closed`
  (moonlab_smoke quarantines_empty PASS) + `adversarial_matrix_green`.
- Evidence producers: run_moonlab_oracles.sh, run_fuzz.sh replay, run_cross_diff.sh,
  run_statistical.sh, run_tsan.sh, run_numerical.sh, run_scaling.sh,
  run_moonlab_release_smoke.sh (emits quarantines_empty by scanning every allowlist).
- Mesh: run_mesh_release_smoke.sh must be 5/5 on the fleet (real Metal/CUDA) for any
  hardware-relevant fix; GitHub CI covers Windows.

## Invariants added (.icc/architecture-model.yaml)
thread-safety (no data race / lazy-init-without-sync in shared state); mps-norm-preservation
(a unitary gate on a normalized MPS keeps norm 1 in every gauge/bond position);
tdvp-correctness-at-scale (real-time TDVP matches dense exp(-iHt) for n>2, not only n<=2);
svd-isometry (every SVD path, incl the no-LAPACK fallback, returns an orthonormal U);
mesh-hardware-validation (fixes validate on the real fleet, not only CI).
