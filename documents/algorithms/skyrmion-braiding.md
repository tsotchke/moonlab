# Skyrmion Braiding

Topological quantum computing with magnetic skyrmions.

## Overview

Magnetic skyrmions are nanoscale spin textures with non-trivial topology that emerge in certain magnetic materials. Their topological protection and controllability make them promising candidates for implementing topological qubits.

Moonlab provides a comprehensive simulation framework for skyrmion-based quantum computing:
- **Skyrmion tracking**: Localize skyrmions from spin configurations
- **Braiding protocols**: Generate exchange paths for topological gates
- **Time evolution**: TDVP dynamics for current-driven motion
- **Phase extraction**: Compute geometric/Berry phases

## Physical Background

### What is a Skyrmion?

A magnetic skyrmion is a localized spin configuration where spins wrap around a sphere, characterized by a topological charge:

$$Q = \frac{1}{4\pi} \int \mathbf{n} \cdot \left(\frac{\partial \mathbf{n}}{\partial x} \times \frac{\partial \mathbf{n}}{\partial y}\right) dx \, dy$$

where $\mathbf{n}(\mathbf{r})$ is the local magnetization direction.

**Key properties**:
- $Q = \pm 1$ for a single skyrmion (sign depends on polarity)
- Topologically protected: cannot be continuously deformed to a uniform state
- Typical size: 1-100 nm in real materials
- Can be created, moved, and annihilated with electric currents

### Skyrmion Types

| Type | Helicity | Materials | DMI Type |
|------|----------|-----------|----------|
| **Néel** | 0 | Multilayers | Interfacial |
| **Bloch** | ±π/2 | B20 compounds | Bulk |
| **Antiskyrmion** | — | Heusler alloys | Anisotropic |

### Skyrmions as Qubits

Following [Psaroudaki & Panagopoulos, Phys. Rev. Lett. 127, 067201 (2021)], a qubit can be encoded in a skyrmion pair:

- $|0\rangle$: Both skyrmions have the same helicity
- $|1\rangle$: Skyrmions have opposite helicity

Braiding (exchanging) the skyrmions implements topological gates:

$$B = e^{i\pi\sigma/4}$$

Two clockwise exchanges give:

$$B^2 = e^{i\pi\sigma/2} = i\sigma$$

This is topologically protected—small deviations in the braiding path only add trivial dynamical phases.

## Skyrmion Tracking

### Single Skyrmion Detection

```c
#include "algorithms/tensor_network/skyrmion_braiding.h"

// Create 2D lattice with spin configuration
lattice_2d_t *lat = lattice_2d_create(32, 32, LATTICE_SQUARE);
double (*spins)[3] = initialize_skyrmion_state(lat);

// Track the skyrmion
skyrmion_t sky;
int found = skyrmion_track(lat, spins, &sky);

if (found == 0) {
    printf("Skyrmion found at (%.2f, %.2f)\n", sky.x, sky.y);
    printf("  Radius: %.2f\n", sky.radius);
    printf("  Charge: %d\n", sky.charge);  // +1 or -1
    printf("  Helicity: %s\n", sky.helicity ? "Bloch" : "Néel");
}
```

### Multiple Skyrmions

```c
// Track up to 10 skyrmions
skyrmion_t skyrmions[10];
int num_found = skyrmion_track_multiple(lat, spins, skyrmions, 10);

printf("Found %d skyrmions:\n", num_found);
for (int i = 0; i < num_found; i++) {
    printf("  [%d] at (%.2f, %.2f), Q=%d\n",
           i, skyrmions[i].x, skyrmions[i].y, skyrmions[i].charge);
}
```

The tracking algorithm:
1. Computes the local topological charge density at each site
2. Finds connected regions with $|q| > $ threshold
3. Computes the centroid of each region
4. Estimates the radius from the charge distribution

## Braiding Paths

### Path Types

| Type | Operation | Phase |
|------|-----------|-------|
| `BRAID_CLOCKWISE` | CW exchange | $e^{i\pi/4}$ |
| `BRAID_COUNTERCLOCKWISE` | CCW exchange | $e^{-i\pi/4}$ |
| `BRAID_FIGURE_EIGHT` | Full figure-8 | $e^{i\pi}$ |
| `BRAID_HALF_EXCHANGE` | Half exchange | $e^{i\pi/8}$ |

### Circular Braiding Path

One skyrmion encircles another:

```c
// Create circular path around center (16, 16) with radius 5
braid_path_t *path = braid_path_circular(
    16.0, 16.0,     // center
    5.0,            // radius
    BRAID_CLOCKWISE,
    32,             // number of segments
    1.0             // velocity
);

printf("Path has %u waypoints over %.2f time units\n",
       path->num_waypoints, path->total_time);

// Access waypoints
for (uint32_t i = 0; i < path->num_waypoints; i++) {
    waypoint_t wp = path->waypoints[i];
    printf("  t=%.2f: (%.2f, %.2f) at v=%.2f\n",
           i * path->total_time / path->num_waypoints,
           wp.x, wp.y, wp.velocity);
}

braid_path_free(path);
```

### Exchange Path

Two skyrmions swap positions:

```c
// Generate exchange paths for two skyrmions
braid_path_t *path1, *path2;

int err = braid_path_exchange(
    10.0, 16.0,     // first skyrmion at (10, 16)
    22.0, 16.0,     // second skyrmion at (22, 16)
    BRAID_CLOCKWISE,
    24,             // segments
    1.0,            // velocity
    &path1, &path2
);

if (err == 0) {
    // path1: moves first skyrmion in arc to (22, 16)
    // path2: moves second skyrmion in arc to (10, 16)
}

braid_path_free(path1);
braid_path_free(path2);
```

## Braiding Dynamics

### Configuration

```c
// Get default configuration
braid_config_t config = braid_config_default();

// Customize for high-accuracy simulation
config.dt = 0.005;              // Smaller time step
config.max_bond_dim = 128;      // Larger bond dimension
config.svd_cutoff = 1e-12;      // Tighter SVD threshold
config.track_skyrmions = true;  // Record positions
config.measure_phase = true;    // Compute Berry phase
config.record_interval = 5;     // Record every 5 steps
config.verbose = true;          // Print progress
```

### Single Braiding

```c
// Prepare MPS state with skyrmion pair
tn_mps_state_t *mps = create_skyrmion_pair_state(lat, x1, y1, x2, y2);

// Create Hamiltonian (Heisenberg + DMI)
mpo_t *mpo = mpo_heisenberg_dmi(lat, J, D);

// Generate braiding path
braid_path_t *path = braid_path_circular(
    (x1 + x2) / 2, (y1 + y2) / 2,  // midpoint
    (x2 - x1) / 2,                  // half separation
    BRAID_CLOCKWISE, 32, 1.0
);

// Execute braiding
braid_result_t *result = skyrmion_braid(mps, mpo, lat, path, &config);

if (result && result->success) {
    printf("Braiding completed!\n");
    printf("Accumulated phase: %.4f + %.4fi\n",
           creal(result->phase), cimag(result->phase));

    // Check energy conservation
    double E_initial = result->energies[0];
    double E_final = result->energies[result->num_records - 1];
    printf("Energy drift: %.2e\n", fabs(E_final - E_initial));
}

braid_result_free(result);
braid_path_free(path);
```

### Double Exchange

Both skyrmions move simultaneously:

```c
braid_path_t *path1, *path2;
braid_path_exchange(x1, y1, x2, y2, BRAID_CLOCKWISE, 24, 1.0, &path1, &path2);

braid_result_t *result = skyrmion_double_braid(
    mps, mpo, lat, path1, path2, &config
);

// The double exchange implements: exp(iπσ/2)
printf("Phase: %.4f\n", carg(result->phase));  // Should be ≈ π/2

braid_path_free(path1);
braid_path_free(path2);
braid_result_free(result);
```

### Monitoring Dynamics

The `braid_result_t` structure contains the full time evolution history:

```c
// Plot skyrmion trajectory
FILE *fp = fopen("trajectory.csv", "w");
fprintf(fp, "t,x,y,energy,charge\n");

for (uint32_t i = 0; i < result->num_records; i++) {
    fprintf(fp, "%.6f,%.4f,%.4f,%.8f,%.4f\n",
            result->times[i],
            result->positions[i][0],  // x
            result->positions[i][1],  // y
            result->energies[i],
            result->charges[i]);      // topological charge
}
fclose(fp);
```

## Topological Qubits

### Qubit Creation

```c
// Define Hamiltonian parameters
hamiltonian_params_t params = {
    .J = 1.0,       // Exchange coupling
    .D = 0.3,       // DMI strength
    .K = 0.1,       // Anisotropy
    .B = 0.2        // External field
};

// Create topological qubit from skyrmion pair
topo_qubit_t *qubit = topo_qubit_create(
    lat,
    &params,
    10.0, 16.0,     // first skyrmion position
    22.0, 16.0,     // second skyrmion position
    64              // MPS bond dimension
);

// Qubit starts in |0⟩
printf("Initial state: |0⟩\n");
printf("  α = %.4f + %.4fi\n", creal(qubit->alpha), cimag(qubit->alpha));
printf("  β = %.4f + %.4fi\n", creal(qubit->beta), cimag(qubit->beta));
```

### Topological Gates

```c
braid_config_t config = braid_config_default();

// Apply single braid gate: exp(iπσ/4)
topo_gate_apply(qubit, TOPO_GATE_BRAID, &config);
printf("After σ-braid: α=%.4f, β=%.4f\n",
       cabs(qubit->alpha), cabs(qubit->beta));

// Apply inverse braid
topo_gate_apply(qubit, TOPO_GATE_BRAID_INV, &config);

// Apply double braid: iσ
topo_gate_apply(qubit, TOPO_GATE_DOUBLE_BRAID, &config);

// Hadamard (requires magic state injection for Ising anyons)
topo_gate_apply(qubit, TOPO_GATE_HADAMARD, &config);
```

### Available Gates

| Gate | Operation | Braid Sequence |
|------|-----------|----------------|
| `TOPO_GATE_IDENTITY` | $I$ | No braiding |
| `TOPO_GATE_BRAID` | $e^{i\pi\sigma/4}$ | Single clockwise exchange |
| `TOPO_GATE_BRAID_INV` | $e^{-i\pi\sigma/4}$ | Single counter-clockwise exchange |
| `TOPO_GATE_DOUBLE_BRAID` | $i\sigma$ | Two clockwise exchanges |
| `TOPO_GATE_HADAMARD` | $H$ | Complex sequence + magic state |

### Measurement

```c
// Measure in Z basis (helicity comparison)
int outcome = topo_qubit_measure_z(qubit);
printf("Measurement result: %+d\n", outcome);  // +1 or -1

// Check fidelity against target state
double fidelity = topo_qubit_fidelity(qubit, 1.0/sqrt(2), 1.0/sqrt(2));
printf("Fidelity to |+⟩: %.4f\n", fidelity);

topo_qubit_free(qubit);
```

## Phase Extraction

### Geometric Phase

The geometric (Berry) phase accumulated during braiding:

```c
// Save initial state
tn_mps_state_t *mps_initial = tn_mps_copy(mps);

// Perform braiding
skyrmion_braid(mps, mpo, lat, path, &config);

// Extract geometric phase
double complex phase = extract_geometric_phase(mps_initial, mps);
printf("Geometric phase: %.6f rad\n", carg(phase));
printf("Magnitude: %.6f (should be ~1)\n", cabs(phase));

tn_mps_free(mps_initial);
```

### Berry Phase from History

When using TDVP with recorded history:

```c
// Configure TDVP to record states
tdvp_config_t tdvp_cfg = tdvp_config_default();
tdvp_cfg.record_history = true;
tdvp_cfg.history_interval = 10;

// Run time evolution
tdvp_history_t *history = tdvp_evolve_with_history(
    mps, mpo, total_time, dt, &tdvp_cfg
);

// Compute Berry phase
double gamma = compute_berry_phase(history);
printf("Berry phase: %.6f rad\n", gamma);

// Expected: γ = π/4 for single braid, π/2 for double braid
if (fabs(gamma - M_PI/4) < 0.1) {
    printf("Consistent with single braid!\n");
}

tdvp_history_free(history);
```

The Berry phase is computed as:

$$\gamma = -\mathrm{Im} \sum_t \ln \langle \psi(t) | \psi(t + dt) \rangle$$

## Complete Example

```c
#include "algorithms/tensor_network/skyrmion_braiding.h"
#include <stdio.h>

int main(void) {
    printf("=== Skyrmion Braiding Demo ===\n\n");

    // Create 32x32 lattice
    lattice_2d_t *lat = lattice_2d_create(32, 32, LATTICE_SQUARE);
    printf("Created %dx%d lattice (%u sites)\n",
           lat->Lx, lat->Ly, lat->num_sites);

    // Hamiltonian parameters
    hamiltonian_params_t params = {
        .J = 1.0,   // Exchange
        .D = 0.3,   // DMI
        .K = 0.1,   // Anisotropy
        .B = 0.2    // Field
    };

    // Create topological qubit
    printf("\nCreating topological qubit...\n");
    topo_qubit_t *qubit = topo_qubit_create(
        lat, &params,
        10.0, 16.0,  // skyrmion 1
        22.0, 16.0,  // skyrmion 2
        64           // bond dim
    );

    printf("  Skyrmion 1: (%.1f, %.1f), Q=%d\n",
           qubit->sky1.x, qubit->sky1.y, qubit->sky1.charge);
    printf("  Skyrmion 2: (%.1f, %.1f), Q=%d\n",
           qubit->sky2.x, qubit->sky2.y, qubit->sky2.charge);
    printf("  Initial state: |0⟩\n");

    // Configure braiding
    braid_config_t config = braid_config_default();
    config.max_bond_dim = 64;
    config.verbose = true;

    // Apply topological NOT (double braid)
    printf("\nApplying topological NOT gate (double braid)...\n");
    topo_gate_apply(qubit, TOPO_GATE_DOUBLE_BRAID, &config);

    // Check result
    printf("\nFinal state:\n");
    printf("  |α|² = %.4f\n", cabs(qubit->alpha) * cabs(qubit->alpha));
    printf("  |β|² = %.4f\n", cabs(qubit->beta) * cabs(qubit->beta));

    double fid_0 = topo_qubit_fidelity(qubit, 1.0, 0.0);
    double fid_1 = topo_qubit_fidelity(qubit, 0.0, 1.0);
    printf("  Fidelity to |0⟩: %.4f\n", fid_0);
    printf("  Fidelity to |1⟩: %.4f\n", fid_1);

    // Measure
    int outcome = topo_qubit_measure_z(qubit);
    printf("\nMeasurement: %d\n", outcome);

    // Cleanup
    topo_qubit_free(qubit);
    lattice_2d_free(lat);

    printf("\n=== Done ===\n");
    return 0;
}
```

## Numerical Considerations

### Bond Dimension

The accuracy of skyrmion braiding simulations depends on the MPS bond dimension:

| Bond Dim | Accuracy | Memory | Speed |
|----------|----------|--------|-------|
| 32 | Qualitative | ~1 MB | Fast |
| 64 | Good | ~4 MB | Medium |
| 128 | High | ~16 MB | Slow |
| 256 | Very high | ~64 MB | Very slow |

For quantitative Berry phase extraction, $\chi \geq 64$ is recommended.

### Time Step

The TDVP time step should be small enough to:
1. Capture the skyrmion dynamics accurately
2. Maintain unitarity of the evolution
3. Resolve the braiding path

Typical values: $dt \in [0.001, 0.01]$

### SVD Cutoff

The SVD truncation threshold affects:
- **Entanglement preservation**: Lower threshold = more accurate
- **Computation speed**: Lower threshold = slower

Recommended: $10^{-10}$ to $10^{-12}$ for phase extraction.

## Physical Parameters

Mapping simulation parameters to real materials:

| Parameter | Symbol | MnSi | FeGe |
|-----------|--------|------|------|
| Exchange | $J$ | 1.0 | 1.0 |
| DMI | $D/J$ | 0.18 | 0.27 |
| Skyrmion size | $\lambda$ | 18 nm | 70 nm |
| Critical field | $B_c$ | 0.2 T | 0.3 T |

Current-driven velocities in real materials: $v \sim 1-100$ m/s for $j \sim 10^{11}$ A/m².

## References

**Foundational Theory**:
- Psaroudaki, C. & Panagopoulos, C. (2021). Skyrmion qubits: A new class of quantum logic elements based on nanoscale magnetization. *Phys. Rev. Lett.* 127, 067201.

**Skyrmion Physics**:
- Fert, A., Cros, V., & Sampaio, J. (2013). Skyrmions on the track. *Nature Nanotech.* 8, 152-156.
- Nagaosa, N. & Tokura, Y. (2013). Topological properties and dynamics of magnetic skyrmions. *Nature Nanotech.* 8, 899-911.

**Braiding Dynamics**:
- Zhang, X. et al. (2015). Skyrmion-skyrmion and skyrmion-edge repulsions in skyrmion-based racetrack memory. *Sci. Rep.* 5, 7643.

**Tensor Network Methods**:
- Haegeman, J. et al. (2011). Time-dependent variational principle for quantum lattices. *Phys. Rev. Lett.* 107, 070601.

## See Also

- [Topological Computing](topological-computing.md) - General topological QC theory
- [Tensor Networks](../concepts/tensor-networks.md) - MPS and DMRG methods
- [TDVP Algorithm](tdvp-algorithm.md) - Time evolution details
- [API: skyrmion_braiding.h](../api/c/skyrmion-braiding.md) - Full API reference
