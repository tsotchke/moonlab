# Time-Dependent Variational Principle (TDVP) — API reference

`src/algorithms/tensor_network/tdvp.{c,h}` — two-site TDVP integrator
for matrix-product states.  Real-time evolution is symplectic on
the MPS manifold and conserves `<H>` for time-independent
Hamiltonians; imaginary-time evolution projects toward the ground
state and is the variational ground-state algorithm equivalent to
DMRG in the long-`tau` limit.

Since v0.4 the engine supports an entropy-feedback PID controller
(adaptive bond dimension) on top of the legacy fixed-`max_bond_dim`
path.  The adaptive controller selects each bond's truncation
threshold individually to meet a target accuracy budget without
oversizing the tensors elsewhere.

References:

- [1] J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, and
      F. Verstraete, "Unifying time evolution and optimization
      with matrix product states", *Phys. Rev. B* **94**, 165116
      (2016).  Two-site / projector variant.
- [2] arXiv:2604.03960 — entropy-feedback bond control for two-site
      TDVP.  Algorithm implemented by the adaptive path.
- [3] U. Schollwoeck, "The density-matrix renormalization group in
      the age of matrix product states", *Ann. Phys.* **326**, 96
      (2011).  Background for MPS / MPO conventions.

## Concepts

The engine state is a `tn_mps_state_t` (the variational MPS) plus a
`mpo_t` (the Hamiltonian).  At each step the integrator sweeps L → R
then R → L, locally evolving every adjacent pair of sites under
`exp(±i H_eff dt)` (real time) or `exp(-H_eff dt)` (imaginary time)
and then truncating the resulting two-site tensor via an SVD.

For the adaptive path each truncation runs through a PID controller
that watches the von Neumann entropy of the post-SVD spectrum and
adjusts per-bond χ to keep `|S - S_chi|` near the configured
`target_entropy_error`.  The controller state is one
`tdvp_bond_pid_state_t` slot per inter-site bond, owned by the
engine and persisted across sweeps.

## Configuration

```c
typedef enum {
    TDVP_REAL_TIME,
    TDVP_IMAGINARY_TIME
} tdvp_evolution_type_t;

typedef enum {
    TDVP_ONE_SITE,
    TDVP_TWO_SITE
} tdvp_variant_t;

typedef enum {
    INTEGRATOR_LANCZOS,
    INTEGRATOR_RUNGE_KUTTA,
    INTEGRATOR_EXPOKIT
} integrator_type_t;
```

### Adaptive-bond controller (since v0.4)

```c
typedef struct {
    bool enabled;
    double target_entropy_error;
    double kp;
    double ki;
    double kd;
    uint32_t chi_floor;
    uint32_t chi_ceiling;
    double alpha;
} tdvp_adaptive_bond_config_t;

tdvp_adaptive_bond_config_t
tdvp_adaptive_bond_config_default(double target_entropy_error);
tdvp_adaptive_bond_config_t
tdvp_adaptive_bond_config_disabled(void);
```

`tdvp_adaptive_bond_config_default(eps_S)` returns the
reference-paper gains: `kp = 0.5`, `ki = 0.05`, `kd = 0.1`,
`chi_floor = 4`, `chi_ceiling = 4096`, `alpha = 8`.  Use it directly
when wiring adaptive TDVP into a new pipeline; sweep the gains
afterwards if your model has a substantially different entanglement
profile.

### Top-level config

```c
typedef struct {
    tdvp_evolution_type_t evolution_type;
    tdvp_variant_t variant;
    integrator_type_t integrator;

    double dt;
    uint32_t max_bond_dim;
    double svd_cutoff;
    uint32_t lanczos_max_iter;
    double lanczos_tol;
    bool normalize;
    bool verbose;

    tdvp_adaptive_bond_config_t adaptive_bond;  /* since v0.4 */
} tdvp_config_t;

tdvp_config_t tdvp_config_default(void);
tdvp_config_t tdvp_config_adaptive(double target_entropy_error);
```

`tdvp_config_default()` leaves `adaptive_bond.enabled = false`, so
v0.3 callers see bit-identical behaviour to the legacy fixed-bond
path.  `tdvp_config_adaptive(eps_S)` turns on the PID controller at
the reference gains and raises `max_bond_dim` to the controller's
`chi_ceiling` so the outer-bound safety still applies.

## Result and history

```c
typedef struct {
    double time;
    double energy;
    double norm;
    double truncation_error;
    uint32_t max_bond_dim;
    double step_time;

    /* Since v0.4: per-bond chi snapshot from the adaptive
     * controller.  Heap-owned; length n_bonds = num_qubits - 1
     * when adaptive_bond.enabled, NULL otherwise. */
    uint32_t *bond_chi_distribution;
    uint32_t  n_bonds;
} tdvp_result_t;

void tdvp_result_clear(tdvp_result_t *result);
```

`tdvp_result_clear` frees `bond_chi_distribution` and zeroes the
struct in place.  Idempotent and safe on a zero-initialised result.
Always call it before letting a stack-allocated `tdvp_result_t` go
out of scope (otherwise the bond-chi buffer leaks).

```c
typedef struct {
    double *times;
    double *energies;
    double *norms;
    double *observables;
    uint32_t num_steps;
    uint32_t capacity;

    /* Since v0.4: row-major capacity * n_bonds buffer of per-bond
     * chi snapshots.  Entry (step, bond) lives at
     * step * n_bonds + bond.  NULL on the legacy path. */
    uint32_t *bond_chi_history;
    uint32_t  n_bonds;
} tdvp_history_t;

tdvp_history_t *tdvp_history_create(uint32_t initial_capacity);
void            tdvp_history_free(tdvp_history_t *hist);
void            tdvp_history_add(tdvp_history_t *hist,
                                 const tdvp_result_t *result);
```

`tdvp_history_add` lazy-allocates the chi buffer on the first added
result that carries a distribution; legacy histories pay no extra
memory.

## Engine

```c
typedef struct tdvp_engine_t {
    tn_mps_state_t *mps;
    mpo_t *mpo;
    dmrg_environments_t *env;
    tdvp_config_t config;
    double current_time;
    tdvp_bond_pid_state_t *bond_states;   /* since v0.4 */
    uint32_t num_bond_states;             /* since v0.4 */
} tdvp_engine_t;

tdvp_engine_t *tdvp_engine_create(tn_mps_state_t *mps,
                                   mpo_t *mpo,
                                   const tdvp_config_t *config);
void           tdvp_engine_free(tdvp_engine_t *engine);

int tdvp_step(tdvp_engine_t *engine, tdvp_result_t *result);
int tdvp_evolve_to(tdvp_engine_t *engine, double target_time,
                   tdvp_history_t *history);

uint32_t tdvp_bond_chi(const tdvp_engine_t *engine, uint32_t bond);
void     tdvp_set_dt(tdvp_engine_t *engine, double dt);
double   tdvp_get_time(const tdvp_engine_t *engine);
```

`tdvp_step` advances the state by one `dt`.  When `adaptive_bond.
enabled` is true the result's `bond_chi_distribution` is populated
from `engine->bond_states`; on the legacy path it is NULL and
`n_bonds == 0`.

`tdvp_bond_chi(engine, b)` reads the current PID-selected chi for
inter-site bond `b`.  Returns 0 when the controller is disabled or
the bond index is out of range.

## Language bindings

### Python (`moonlab.tdvp`, since v0.4)

```python
from moonlab.tdvp import (
    TdvpConfig, TdvpEngine, EvolutionType,
    mpo_heisenberg, mpo_tfim, random_mps,
)

mpo = mpo_heisenberg(num_sites=8, J=1.0, Delta=1.0)
mps = random_mps(num_sites=8, chi_init=8, max_bond_dim=32)

config = TdvpConfig.adaptive(target_entropy_error=1e-3)
config.evolution_type = EvolutionType.IMAGINARY_TIME
config.dt = 0.05

engine = TdvpEngine(mps, mpo, config)
for step in range(30):
    result = engine.step()
    print(step, result.energy, result.bond_chi_distribution)
```

Surface (see `bindings/python/moonlab/tdvp.py` for the full
docstrings):

- Enums: `EvolutionType`, `Variant`, `IntegratorType`.
- Config: `TdvpAdaptiveBondConfig`, `TdvpConfig` with class methods
  `default()` and `adaptive(target_entropy_error)`.
- Result: `TdvpResult` dataclass; `bond_chi_distribution` is
  exposed as a `numpy.uint32` array.
- Engine: `TdvpEngine.step()` (returns `TdvpResult`),
  `TdvpEngine.bond_chi(b)`.
- Hamiltonian factories: `mpo_heisenberg`, `mpo_tfim`.
- Initial state: `random_mps`.

The Python bindings are exercised by
`bindings/python/tests/test_tdvp.py` (9 cases): config defaults,
adaptive gains, lifecycle, legacy / adaptive engine paths, real-
time energy conservation, imag-time convergence toward the ground
state, and module-export contract.

### Rust

A Rust wrapper is on the v0.4.x roadmap; today the v0.4 surface is
reachable through `moonlab_sys` for callers who need it.

## Acceptance tests

Mirrors the four criteria in
`docs/research/adaptive_bond_tdvp.md`:

| # | Criterion | Test |
|---|---|---|
| 1 | Legacy callers unchanged with `adaptive_bond.enabled = false` | `tests/unit/test_tdvp_adaptive_config.c` |
| 2 | Real-time TDVP conserves energy under the controller | `tests/unit/test_tdvp_adaptive_energy_conservation.c` |
| 3 | Imag-time TDVP converges to the DMRG ground state | `tests/unit/test_tdvp_adaptive_tfim_ground.c` |
| 4 | PID stable across (kp, ki, kd) sweep | `tests/unit/test_tdvp_adaptive_pid_stability.c` |

Plus the Python end-to-end coverage at
`bindings/python/tests/test_tdvp.py`.

## See also

- `docs/research/adaptive_bond_tdvp.md` — design note and
  acceptance criteria for the v0.4 PID controller.
- `src/algorithms/tensor_network/dmrg.h` — DMRG and the MPS / MPO
  factories (`mpo_heisenberg_create`, `mpo_tfim_create`,
  `dmrg_init_random_mps`).
- `src/algorithms/tensor_network/svd_compress.h` — SVD compression
  primitive used by the truncation helper.
