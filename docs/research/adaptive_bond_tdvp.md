# Archived Moonlab Documentation: Adaptive-bond two-site TDVP (design note + retrospective)

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Adaptive-bond two-site TDVP (design note + retrospective)

This document captures the entropy-feedback adaptive-bond two-site
TDVP integrator that shipped in v0.4.0.  It serves two purposes:

1. *Specification* of the algorithm, the API surface, and the
   four physics-validation criteria the implementation must pass.
2. *Retrospective* recording the measured numerical results from
   the v0.4.0 validation suite, so the design note stays accurate
   as the codebase evolves.

The implementation lives in
`src/algorithms/tensor_network/tdvp.{c,h}`.  Python and Rust
bindings live at `bindings/python/moonlab/tdvp.py` and
`bindings/rust/moonlab/src/tdvp.rs`.  The user-facing tutorial is
[`../tutorials/adaptive_bond_tdvp.md`](../tutorials/adaptive_bond_tdvp.md);
the API contract is
[`../reference/tdvp-api.md`](../reference/tdvp-api.md).

## Motivation

The current two-site TDVP integrator
(`src/algorithms/tensor_network/tdvp.{c,h}`) takes a fixed bond
dimension `max_bond_dim` (default 128) and an `svd_cutoff` (default
`1e-10`).  Both are passed verbatim into the SVD compression that
follows each two-site update.  This works well when the user knows
the maximum entanglement the simulation will reach, but is wasteful
or inaccurate when entanglement grows or shrinks across the system:

- **Wasteful**: a system that only reaches `chi ~ 16` in its
  high-entanglement region carries 128-wide tensors everywhere,
  paying `O(chi^3)` cost on bonds that need a fraction of it.
- **Inaccurate**: a system whose entanglement exceeds the cap is
  silently truncated; the user sees only the integrated
  `truncation_error` reported in `tdvp_result_t` and has no
  feedback channel to widen the cap mid-simulation.

The adaptive-bond approach replaces the fixed cap with an
*entropy-feedback PID controller* that monitors the von Neumann
entropy of the half-cut singular-value spectrum at each bond after
each two-site update, compares it to a target accuracy budget, and
adjusts the per-bond truncation threshold so the bond grows or
contracts to meet the budget.

Reference: ArXiv:2604.03960 (entropy-feedback bond control for
2TDVP).  The algorithm has direct precedent in the dynamical-bond
DMRG / TDVP literature; the v0.4 implementation adopts the explicit
PID parametrisation rather than the older "double-and-trim"
heuristic, because the PID has better behaviour when the entropy
crosses gap-closing transitions.

## Algorithm

Inputs:

- `target_entropy_error` epsilon_S: target absolute error on the
  block entropy after truncation.  Typical value 1e-3.
- `kp`, `ki`, `kd`: PID gains.  Defaults `(0.5, 0.05, 0.1)` per
  reference, validated by sweep.
- `chi_floor`, `chi_ceiling`: hard bounds on the per-bond
  dimension.  Defaults `(4, 4096)`.

At each two-site update on bond `b`:

1. Compute the SVD `M = U Sigma V^dagger` with no truncation.
2. Form the spectrum `lambda_i = sigma_i^2 / sum_j sigma_j^2`.
3. Compute the von Neumann entropy
   `S = -sum_i lambda_i ln lambda_i`.
4. Compute the truncation entropy of keeping the first `chi` values:
   `S_chi = -sum_{i <= chi} lambda_i ln lambda_i / Z_chi`
   where `Z_chi = sum_{i <= chi} lambda_i`.
5. Define the error signal `e_b = S - S_chi(chi_b)` at the current
   bond's working dimension `chi_b`.
6. Update the PID state across calls (per bond, integrating across
   TDVP sweeps):
   - `integral_b <- integral_b + e_b dt`
   - `derivative_b <- (e_b - e_b^{prev}) / dt`
   - `delta_chi = kp * e_b + ki * integral_b + kd * derivative_b`
7. Pick `chi_b^{new} = clamp(round(chi_b + alpha * delta_chi /
   epsilon_S), chi_floor, chi_ceiling)` where `alpha` translates an
   entropy excess into a bond-dimension increment.  Typical
   `alpha = 8`.
8. Truncate the spectrum to `chi_b^{new}` via
   `svd_compress_config_fixed(chi_b^{new})` (existing API).

The PID state is per-bond; the integrator carries one
`adaptive_state_t` per inter-site bond, persisting across TDVP
sweeps.

## API surface

New configuration block in `tdvp_config_t`:

[archived fence delimiter: ```c]
typedef struct {
    bool enabled;                /* false -> use legacy fixed cap */
    double target_entropy_error; /* eps_S */
    double kp, ki, kd;           /* PID gains */
    uint32_t chi_floor;
    uint32_t chi_ceiling;
    double alpha;                /* entropy -> bond-dim scaling */
} tdvp_adaptive_bond_config_t;
[archived fence delimiter: ```]

Added field:

[archived fence delimiter: ```c]
typedef struct {
    /* ... existing fields ... */
    tdvp_adaptive_bond_config_t adaptive_bond;
} tdvp_config_t;
[archived fence delimiter: ```]

Default constructor leaves `adaptive_bond.enabled = false` so
existing callers behave identically.  A new helper:

[archived fence delimiter: ```c]
tdvp_config_t tdvp_config_adaptive(double target_entropy_error);
[archived fence delimiter: ```]

returns a configuration with `adaptive_bond.enabled = true` and the
recommended PID gains.

Result reporting gains a per-bond chi histogram:

[archived fence delimiter: ```c]
typedef struct {
    /* ... existing tdvp_result_t fields ... */
    uint32_t *bond_chi_distribution; /* length n_bonds */
    uint32_t  n_bonds;
} tdvp_result_t;
[archived fence delimiter: ```]

Caller owns the `bond_chi_distribution` buffer; free it with the
new `tdvp_result_clear()` helper before the result struct goes out
of scope.

## Validation criteria and measured results

Four physics-validation criteria.  Each ships with a ctest-scope
unit covering an 8-site model (calibrated to run cheaply under
`ctest`); the design-note 24-site targets live in the planned
`benchmarks/` harness rather than the unit-test surface.

| # | Criterion | Test | Threshold | Observed |
|---|---|---|---|---|
| 1 | Legacy callers unchanged with `adaptive_bond.enabled = false` | `tests/unit/test_tdvp_adaptive_config.c` (5 cases) | bit-identical config | PASS |
| 2 | Real-time TDVP conserves energy under the controller | `tests/unit/test_tdvp_adaptive_energy_conservation.c` | `|E - E_0| / |E_0| < 5e-3` over 5 steps | **2.4 x 10^-5** |
| 3 | Imag-time TDVP converges to the DMRG ground state | `tests/unit/test_tdvp_adaptive_tfim_ground.c` | `|E - E_DMRG| / |E_DMRG| < 3%` after 30 steps on critical TFIM | **1.98%** at `tau = 1.5` |
| 4 | PID stable across `(kp, ki, kd)` sweep | `tests/unit/test_tdvp_adaptive_pid_stability.c` | >= 80% of `3 x 3 x 3` grid keeps `max_osc <= 4` | **27 / 27** (100%) |

Criterion (1) is mathematical: the legacy fixed-bond path must
remain bit-identical when the controller is disabled.  Criterion
(2) follows from the symplectic structure of two-site TDVP under a
time-independent Hamiltonian -- the PID changes which singular
values are kept after each two-site update but does not modify
the Hamiltonian or the integration step, so energy drift must
remain inside the integrator's per-step `O(dt^3)` error envelope.
Criterion (3) tests the substantive use case for imag-time TDVP:
the controller must not break the projection onto the ground
state.  Criterion (4) certifies the controller is well-behaved
across a meaningful gain window, not just at the reference-paper
defaults.

The validation suite cost an additional pre-existing fix.  Earlier
imag-time TDVP failed with "Failed at site X (R->L)" after roughly
five steps on Heisenberg and immediately on TFIM, on both the
legacy and adaptive paths.  Root cause: each two-site `exp(-H * dt)
@ theta` update shrinks the local tensor by `~exp(-E_max * dt)`,
and the end-of-step renormalisation fired too late to keep the
inner SVD / Lanczos well-conditioned.  Fixed in commit `1c7b100`
by renormalising `theta_evolved` to unit Frobenius norm after each
`lanczos_expm` call -- mathematically equivalent (ground state
invariant under rescaling) and a defensive no-op on the
norm-preserving real-time path.

## Implementation roadmap

1. Add the `tdvp_adaptive_bond_config_t` struct + the
   `tdvp_config_adaptive` helper (header only).
2. Carve the SVD compression call in `tdvp.c` into a small inner
   helper `tdvp_truncate_bond(svd_t *svd, bond_idx_t b,
   const tdvp_adaptive_bond_config_t *cfg, adaptive_state_t *state)`.
3. Implement the PID update inside `tdvp_truncate_bond`.
4. Allocate / persist the per-bond `adaptive_state_t` array in
   `tdvp_engine_t`.
5. Wire `bond_chi_distribution` into `tdvp_result_t` and
   `tdvp_history_t`.
6. Add the four validation tests under `tests/unit/`:
   - `test_tdvp_adaptive_backcompat.c` — fixed-cfg parity
   - `test_tdvp_adaptive_energy_conservation.c` — Heisenberg
   - `test_tdvp_adaptive_tfim_ground.c` — imag-time
   - `test_tdvp_adaptive_pid_stability.c` — gain sweep
7. Update `docs/reference/tdvp-api.md` and add a Python wrapper
   in `bindings/python/moonlab/tdvp.py` (currently absent — TDVP
   is accessed via the C ABI only).

## References

[1] *Adaptive bond TDVP via entropy-feedback PID*, arXiv:2604.03960.
[2] J. Haegeman et al., "Unifying time evolution and optimization
    with matrix product states", Phys. Rev. B **94**, 165116 (2016).
    Two-site TDVP foundation.
[3] U. Schollwöck, "The density-matrix renormalization group in the
    age of matrix product states", Ann. Phys. **326**, 96 (2011).
    Standard reference for the entanglement entropy / truncation
    error duality used in the PID error signal.
```
