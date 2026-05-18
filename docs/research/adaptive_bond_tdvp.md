# Adaptive-bond two-site TDVP (design note, v0.4 prep)

This is a scoping document for the entropy-feedback adaptive-bond
two-site TDVP integrator scheduled for v0.4 (Phase 3B of the
v0.x release plan).  No code is shipped with this note; it captures
the algorithm, the API surface, and the validation criterion so the
implementation work can begin from a clear specification.

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

```c
typedef struct {
    bool enabled;                /* false -> use legacy fixed cap */
    double target_entropy_error; /* eps_S */
    double kp, ki, kd;           /* PID gains */
    uint32_t chi_floor;
    uint32_t chi_ceiling;
    double alpha;                /* entropy -> bond-dim scaling */
} tdvp_adaptive_bond_config_t;
```

Added field:

```c
typedef struct {
    /* ... existing fields ... */
    tdvp_adaptive_bond_config_t adaptive_bond;
} tdvp_config_t;
```

Default constructor leaves `adaptive_bond.enabled = false` so
existing callers behave identically.  A new helper:

```c
tdvp_config_t tdvp_config_adaptive(double target_entropy_error);
```

returns a configuration with `adaptive_bond.enabled = true` and the
recommended PID gains.

Result reporting gains a per-bond chi histogram:

```c
typedef struct {
    /* ... existing tdvp_result_t fields ... */
    uint32_t *bond_chi_distribution; /* length n_bonds */
    uint32_t  n_bonds;
} tdvp_result_t;
```

Caller owns the `bond_chi_distribution` buffer when the result is
freed via the new `tdvp_result_free()` helper.

## Validation criteria

The implementation is accepted when all four hold:

1. **Backwards compatibility**: with `adaptive_bond.enabled = false`,
   every existing test in `tests/unit/test_tdvp*.c` passes
   bit-identically against the v0.3.1 baseline.
2. **Energy conservation on real-time evolution**: a 24-site
   Heisenberg chain evolved for `t = 10` with
   `target_entropy_error = 1e-4` shows `|E(t) - E(0)| / |E(0)| <
   1e-5`.  The fixed-chi reference at `chi = 256` achieves the same
   tolerance; the adaptive run is expected to use median `chi ~ 32`
   in the bulk with `chi ~ 100` near the chain centre.
3. **Imaginary-time convergence**: ground-state imaginary-time
   evolution on the 24-site TFIM at `g = 1` (critical point) reaches
   the same DMRG-converged energy at half the wall time of the
   fixed-`chi = 128` baseline.
4. **PID stability**: a sweep over `(kp, ki, kd)` near the defaults
   shows the bond dimensions converge (do not oscillate by more
   than `+/- 4` between successive sweeps) for at least 80% of the
   parameter grid; instability outside that range is documented in
   a calibration table.

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
