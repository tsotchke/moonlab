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
| 1 | reversed-CNOT 2q transpose (also OOB) | HIGH | tn_gates.c | oracle backend_differential (reversed_2q); differential re-run | tn | CLOSED (confirmed: all 26 reversed-CNOT/Toffoli differential cases pass on re-run) |
| 2 | control-plane resource-amplification DoS (64 GB alloc) | MED-HIGH | control_plane.c | control_max_qubits() cap (default 24, env override, 30 ceil); test_control_plane_hardening path 3; crashes-pending/ empty | integrator | CLOSED (188323c; oversized frame -> ERR -401, 0 alloc; fuzz seed moved to corpus) |
| 3 | n>=10 tn_mps Metal read-out norm | HIGH | tn_gates.c (Metal) | unit_tn_mps_deep_forward; oracle | tn | CLOSED |
| 4 | SVD-fallback non-orthonormal U (no-LAPACK/Windows) | HIGH | tensor.c:1356 | t_svd_fallback; numerical_edge_clean | tn/integrator | OPEN (task 28) |
| 5 | qsim_config_global() singleton race | HIGH | config.c:127/148 | conc_core_init; tsan_clean | integrator | OPEN (task 29) |
| 6 | SIMD dispatch vtable race (from dispatch rewrite) | HIGH | simd_ops.c:762/120 | conc_core_init; tsan_clean | tn/integrator | OPEN (task 29) |
| 7 | control-plane setter races (doc says thread-safe) | MED | control_plane.c:1296+ | conc_control_plane; tsan_clean | integrator | OPEN (task 29) |
| 8 | audit-buffer destroy deadlock | HIGH | audit_buffer.c:76-87 | conc_audit_buffer; tsan_clean | integrator | OPEN (task 29) |
| 9 | tn_mps bulk-bond 2q-gate norm (interior bond, CPU, n=4) | HIGH | tn_gates.c (mixed_canonicalize before 2q SVD) | unit_tn_mps_interior_gate | tn | CLOSED (merged 7c124a8; norm 1.0, verified on master) |
| 10 | real-time TDVP wrong for chains > 2 sites | HIGH | tdvp.c | unit_tdvp_bulk_site (n>=3 vs dense) | tn | NOT-A-BUG (TDVP matches dense exp(-iHt) to ~1e-13 for n=3..12 on master; the coordinator report was a STALE scaling-harness branch predating projector-splitting merge 234433e; unit_tdvp_bulk_site guards it -- passes on master, fails on the forward-only integrator) |
| 11 | tn_mps deep non-Clifford fp floor (~1e-9) at n>=10 | MED | tn_gates.c | differential @1e-10 tn_max_n 12; unit_tn_mps_deep_forward | tn | CLOSED (4933997) -- ROOT CAUSE was the float32 Metal GPU SVD (no fp64 on Apple GPUs) at bond>=GPU_BOND_THRESHOLD, NOT re-canon; exact 2q gates now stay on CPU LAPACK double, lossy Metal kernel opt-in via MOONLAB_TN_GPU_LOSSY. Differential 0 failed/0 quarantined at n=12; mesh 5/5 incl atlas+enki Metal |

Adjacent (not from the hunt, tracked): QRNG multi-MB draw robustness (task 24, nightly
battery), wheel Accelerate -Werror (task 22), consumer OpenMP linkage (task 27).

### Update 2026-07-17 (post-verification)
The adversarial re-run split the original TN cluster cleanly: bugs #1, #3, #9 are
genuinely CLOSED and verified on master; bug #10 was never real (stale harness);
and a NEW, narrower item (#11) surfaced -- a floating-point accumulation floor in
deep non-Clifford MPS circuits at n>=10, provably not a correctness bug
(bond-independent, cutoff-independent, norm-preserving) and under active
reducibility investigation before any tolerance/scoping decision.

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
