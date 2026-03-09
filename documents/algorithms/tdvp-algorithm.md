# TDVP Algorithm

Time-Dependent Variational Principle for quantum dynamics simulation.

## Overview

The Time-Dependent Variational Principle (TDVP) is a powerful algorithm for simulating real-time dynamics of quantum many-body systems using Matrix Product States (MPS). Unlike exact methods that scale exponentially with system size, TDVP efficiently evolves MPS states while respecting the variational manifold structure.

Moonlab provides a complete TDVP implementation including:
- **One-site and two-site variants**: Trade-off between speed and accuracy
- **Real and imaginary time evolution**: Dynamics or ground state preparation
- **Lanczos matrix exponential**: Efficient time stepping
- **Spin-transfer torque dynamics**: Skyrmion motion under current drive

## Theoretical Background

### The Variational Principle

Standard time evolution follows the Schrödinger equation:

$$i\frac{\partial}{\partial t}|\psi(t)\rangle = H|\psi(t)\rangle$$

The exact solution $|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle$ requires exponentially large resources for many-body systems.

TDVP restricts the evolution to the MPS manifold $\mathcal{M}$ by projecting:

$$i\frac{\partial}{\partial t}|\psi(t)\rangle = P_{\mathcal{M}}H|\psi(t)\rangle$$

where $P_{\mathcal{M}}$ projects onto the tangent space of $\mathcal{M}$ at $|\psi(t)\rangle$.

### Two-Site TDVP

The two-site algorithm evolves $|\psi(t)\rangle \to |\psi(t+dt)\rangle$ by:

**Left-to-right sweep**:
1. Contract sites $i$ and $i+1$ into a two-site tensor $\theta$
2. Build effective Hamiltonian $H_{\text{eff}}$ for these sites
3. Evolve: $\theta \to e^{-iH_{\text{eff}}dt/2}\theta$
4. SVD to split back into two tensors
5. Update environments and move right

**Right-to-left sweep**:
- Same procedure with remaining $dt/2$

This is a second-order integrator with error $O(dt^3)$.

### One-Site TDVP

One-site TDVP is faster but keeps bond dimension fixed. It alternates:
1. Evolve single-site tensor
2. Evolve "bond tensor" between sites

This preserves the norm exactly and is preferred when bond dimension is sufficient.

## Basic Usage

### Configuration

```c
#include "algorithms/tensor_network/tdvp.h"

// Get default configuration
tdvp_config_t config = tdvp_config_default();

// Customize
config.evolution_type = TDVP_REAL_TIME;  // Real time dynamics
config.variant = TDVP_TWO_SITE;          // Allow bond dimension growth
config.integrator = INTEGRATOR_LANCZOS;  // Lanczos matrix exponential
config.dt = 0.01;                        // Time step
config.max_bond_dim = 128;               // Maximum bond dimension
config.svd_cutoff = 1e-10;               // Truncation threshold
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evolution_type` | `TDVP_REAL_TIME` | Real or imaginary time |
| `variant` | `TDVP_TWO_SITE` | One-site or two-site |
| `integrator` | `INTEGRATOR_LANCZOS` | Time integration method |
| `dt` | 0.01 | Time step |
| `max_bond_dim` | 128 | Maximum bond dimension |
| `svd_cutoff` | 1e-10 | SVD truncation threshold |
| `lanczos_max_iter` | 50 | Maximum Lanczos iterations |
| `lanczos_tol` | 1e-12 | Lanczos convergence tolerance |
| `normalize` | true | Normalize after each step |
| `verbose` | false | Print progress |

### Evolution Types

```c
typedef enum {
    TDVP_REAL_TIME,     // exp(-iHt): unitary dynamics
    TDVP_IMAGINARY_TIME // exp(-Ht): ground state preparation
} tdvp_evolution_type_t;
```

### Algorithm Variants

```c
typedef enum {
    TDVP_ONE_SITE,      // Fixed bond dim, faster
    TDVP_TWO_SITE       // Adaptive bond dim, more accurate
} tdvp_variant_t;
```

### Time Integrators

```c
typedef enum {
    INTEGRATOR_LANCZOS,     // Lanczos matrix exponential (recommended)
    INTEGRATOR_RUNGE_KUTTA, // 4th order Runge-Kutta
    INTEGRATOR_EXPOKIT      // Krylov subspace method
} integrator_type_t;
```

## Engine-Based Evolution

For multi-step evolution, use the TDVP engine:

```c
// Create MPS and Hamiltonian
tn_mps_state_t *mps = tn_mps_create(num_sites, phys_dim, bond_dim);
mpo_t *mpo = mpo_heisenberg_chain(num_sites, J, 0.0);

// Create TDVP engine
tdvp_config_t config = tdvp_config_default();
config.dt = 0.01;
config.max_bond_dim = 64;

tdvp_engine_t *engine = tdvp_engine_create(mps, mpo, &config);

// Evolve to target time
tdvp_history_t *history = tdvp_history_create(1000);
tdvp_evolve_to(engine, 10.0, history);  // Evolve to t=10

// Check results
printf("Final time: %.4f\n", tdvp_get_time(engine));
printf("Final energy: %.8f\n", history->energies[history->num_steps - 1]);

// Cleanup
tdvp_history_free(history);
tdvp_engine_free(engine);
mpo_free(mpo);
tn_mps_free(mps);
```

### Engine Functions

#### `tdvp_engine_create`

Create TDVP engine.

```c
tdvp_engine_t *tdvp_engine_create(tn_mps_state_t *mps,
                                  mpo_t *mpo,
                                  const tdvp_config_t *config);
```

#### `tdvp_engine_free`

Free TDVP engine.

```c
void tdvp_engine_free(tdvp_engine_t *engine);
```

#### `tdvp_step`

Perform one TDVP time step.

```c
int tdvp_step(tdvp_engine_t *engine, tdvp_result_t *result);
```

#### `tdvp_evolve_to`

Evolve to target time.

```c
int tdvp_evolve_to(tdvp_engine_t *engine,
                   double target_time,
                   tdvp_history_t *history);
```

#### `tdvp_set_dt`

Change time step.

```c
void tdvp_set_dt(tdvp_engine_t *engine, double dt);
```

#### `tdvp_get_time`

Get current time.

```c
double tdvp_get_time(const tdvp_engine_t *engine);
```

## Single-Step Evolution

For simple use cases:

```c
double energy;
int err = tdvp_single_step(mps, mpo, dt, &config, &energy);
if (err == 0) {
    printf("Energy after step: %.8f\n", energy);
}
```

## Recording History

Track evolution with `tdvp_history_t`:

```c
// Create history with capacity for 1000 steps
tdvp_history_t *history = tdvp_history_create(1000);

// Evolve
tdvp_evolve_to(engine, total_time, history);

// Access recorded data
for (uint32_t i = 0; i < history->num_steps; i++) {
    printf("t=%.4f: E=%.8f, norm=%.8f\n",
           history->times[i],
           history->energies[i],
           history->norms[i]);
}

tdvp_history_free(history);
```

### Step Results

Each step produces a `tdvp_result_t`:

```c
typedef struct {
    double time;                // Current time after step
    double energy;              // Energy ⟨H⟩
    double norm;                // State norm
    double truncation_error;    // Truncation error in this step
    uint32_t max_bond_dim;      // Maximum bond dimension reached
    double step_time;           // Wall time for this step (seconds)
} tdvp_result_t;
```

## Observables During Evolution

Measure observables during evolution:

```c
// Define callback
void measure_magnetization(const tn_mps_state_t *mps,
                          double time, void *user_data) {
    double *mag_data = (double *)user_data;
    // Measure and store magnetization
    double mag = tn_mps_expectation_local(mps, S_z, num_sites/2);
    // ... store in mag_data
}

// Evolve with measurements
double mag_history[1000];
tdvp_evolve_with_observables(engine, total_time,
                             measure_magnetization, mag_history,
                             10);  // Measure every 10 steps
```

## Imaginary Time Evolution

Use imaginary time to find ground states:

```c
tdvp_config_t config = tdvp_config_default();
config.evolution_type = TDVP_IMAGINARY_TIME;
config.dt = 0.1;  // Can use larger steps for imaginary time
config.normalize = true;  // Renormalize to prevent collapse

// Start from random state
tn_mps_state_t *mps = tn_mps_random(num_sites, phys_dim, initial_bond_dim);

// Evolve to large imaginary time
tdvp_engine_t *engine = tdvp_engine_create(mps, mpo, &config);
tdvp_evolve_to(engine, 100.0, NULL);  // Project onto ground state

// Check ground state energy
double E0 = tn_mps_expectation_mpo(mps, mpo);
printf("Ground state energy: %.10f\n", E0);
```

## Spin Dynamics (Skyrmion Motion)

TDVP includes specialized support for current-driven skyrmion dynamics:

### Spin-Transfer Torque Parameters

```c
typedef struct {
    double jx;      // Current density x-component
    double jy;      // Current density y-component
    double beta;    // Non-adiabaticity parameter
    double alpha;   // Gilbert damping
} stt_params_t;
```

### Skyrmion Evolution

```c
// Base Hamiltonian (exchange + DMI + anisotropy)
mpo_t *mpo_base = mpo_heisenberg_dmi(lat, J, D, K, B);

// Current drive parameters
stt_params_t stt = {
    .jx = 1e11,    // A/m² (typical current density)
    .jy = 0.0,
    .beta = 0.3,   // Non-adiabaticity
    .alpha = 0.1   // Damping
};

// Configure TDVP
tdvp_config_t config = tdvp_config_default();
config.dt = 0.001;  // Small step for dynamics
config.max_bond_dim = 64;

// Evolve skyrmion
tdvp_history_t *history = tdvp_history_create(10000);
tdvp_evolve_with_stt(mps, mpo_base, &stt, &config, total_time, history);

// Skyrmion moves in response to current
```

### Creating STT Hamiltonian

```c
mpo_t *mpo_stt = mpo_stt_create(lat, &stt);
```

The STT Hamiltonian adds:
$$H_{\text{STT}} = \sum_i \mathbf{j} \cdot \nabla \mathbf{S}_i + \beta \mathbf{j} \cdot (\mathbf{S}_i \times \nabla \mathbf{S}_i)$$

## Matrix Exponential

The Lanczos algorithm efficiently computes $e^{\alpha H}|v\rangle$:

```c
int lanczos_expm(const effective_hamiltonian_t *H_eff,
                 const tensor_t *x,
                 double complex alpha,
                 uint32_t max_iter,
                 double tol,
                 tensor_t *y);
```

**Parameters**:
- For real time: `alpha = -I * dt`
- For imaginary time: `alpha = -dt`

## Numerical Considerations

### Time Step Selection

| Application | Recommended dt | Notes |
|-------------|---------------|-------|
| Ground state (imaginary time) | 0.1 - 1.0 | Can be large |
| Slow dynamics | 0.01 - 0.1 | Energy conserving |
| Fast dynamics | 0.001 - 0.01 | Resolve oscillations |
| Skyrmion braiding | 0.001 - 0.005 | Accurate phases |

### Bond Dimension

| Bond Dim | Use Case | Memory |
|----------|----------|--------|
| 32 | Quick exploration | ~1 MB |
| 64 | Standard accuracy | ~4 MB |
| 128 | High accuracy | ~16 MB |
| 256 | Very high accuracy | ~64 MB |

### Error Sources

1. **Time discretization**: $O(dt^3)$ per step, $O(dt^2)$ total
2. **Truncation error**: Controlled by `svd_cutoff`
3. **Lanczos convergence**: Controlled by `lanczos_tol`

### Energy Conservation

Monitor energy drift to check accuracy:

```c
double E_initial = history->energies[0];
double E_final = history->energies[history->num_steps - 1];
double drift = fabs(E_final - E_initial) / fabs(E_initial);

if (drift > 1e-6) {
    printf("Warning: Energy drift %.2e, consider smaller dt\n", drift);
}
```

## Complete Example

```c
#include "algorithms/tensor_network/tdvp.h"
#include <stdio.h>

int main(void) {
    printf("=== TDVP Dynamics Demo ===\n\n");

    // System parameters
    const uint32_t num_sites = 20;
    const double J = 1.0;   // Exchange coupling
    const double h = 0.5;   // Transverse field

    // Create transverse-field Ising Hamiltonian
    mpo_t *mpo = mpo_tfim_chain(num_sites, J, h);

    // Initial state: all spins up
    tn_mps_state_t *mps = tn_mps_product_state(num_sites, 2, 0);

    // Flip one spin to create excitation
    tn_mps_apply_local(mps, S_x, num_sites / 2);

    // TDVP configuration
    tdvp_config_t config = tdvp_config_default();
    config.dt = 0.02;
    config.max_bond_dim = 64;
    config.verbose = true;

    // Create engine and history
    tdvp_engine_t *engine = tdvp_engine_create(mps, mpo, &config);
    tdvp_history_t *history = tdvp_history_create(500);

    // Evolve to t=10
    printf("Evolving to t=10...\n");
    tdvp_evolve_to(engine, 10.0, history);

    // Print summary
    printf("\nEvolution complete:\n");
    printf("  Steps: %u\n", history->num_steps);
    printf("  Final time: %.4f\n", tdvp_get_time(engine));
    printf("  Initial energy: %.8f\n", history->energies[0]);
    printf("  Final energy: %.8f\n", history->energies[history->num_steps-1]);
    printf("  Energy drift: %.2e\n",
           fabs(history->energies[history->num_steps-1] - history->energies[0]));

    // Cleanup
    tdvp_history_free(history);
    tdvp_engine_free(engine);
    mpo_free(mpo);
    tn_mps_free(mps);

    printf("\n=== Done ===\n");
    return 0;
}
```

## References

**Foundational Papers**:
- Haegeman, J. et al. (2011). Time-dependent variational principle for quantum lattices. *Phys. Rev. Lett.* 107, 070601.
- Haegeman, J. et al. (2016). Unifying time evolution and optimization with matrix product states. *Phys. Rev. B* 94, 165116.

**Applications**:
- Paeckel, S. et al. (2019). Time-evolution methods for matrix-product states. *Annals of Physics* 411, 167998.
- Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. *Annals of Physics* 326, 96-192.

## See Also

- [DMRG Algorithm](dmrg-algorithm.md) - Ground state optimization
- [Tensor Networks](../concepts/tensor-networks.md) - MPS fundamentals
- [Skyrmion Braiding](skyrmion-braiding.md) - Application to topological qubits
- [API: tdvp.h](../api/c/tdvp.md) - Full API reference
