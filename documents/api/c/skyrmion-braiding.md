# Skyrmion Braiding API

C API for skyrmion-based topological quantum computing.

## Overview

The skyrmion braiding API (`skyrmion_braiding.h`) provides functions for simulating topological qubits encoded in magnetic skyrmion pairs. This includes skyrmion tracking, braiding path generation, dynamics simulation via TDVP, and topological gate implementation.

## Header

```c
#include "algorithms/tensor_network/skyrmion_braiding.h"
```

## Types

### `skyrmion_t`

Represents a single magnetic skyrmion.

```c
typedef struct {
    double x;           // Center x-coordinate
    double y;           // Center y-coordinate
    double radius;      // Effective radius
    int charge;         // Topological charge (+1 or -1)
    int helicity;       // 0 = Néel, 1 = Bloch
} skyrmion_t;
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `x` | `double` | X-coordinate of skyrmion center |
| `y` | `double` | Y-coordinate of skyrmion center |
| `radius` | `double` | Effective skyrmion radius |
| `charge` | `int` | Topological charge Q = ±1 |
| `helicity` | `int` | Helicity type (0 = Néel, 1 = Bloch) |

### `braid_type_t`

Enumeration of braiding path types.

```c
typedef enum {
    BRAID_CLOCKWISE,        // Clockwise exchange
    BRAID_COUNTERCLOCKWISE, // Counter-clockwise exchange
    BRAID_FIGURE_EIGHT,     // Full figure-8 braiding
    BRAID_HALF_EXCHANGE     // Half exchange (for testing)
} braid_type_t;
```

### `waypoint_t`

Single waypoint in a braiding path.

```c
typedef struct {
    double x;           // Target x-coordinate
    double y;           // Target y-coordinate
    double velocity;    // Velocity to this point
} waypoint_t;
```

### `braid_path_t`

Complete braiding path specification.

```c
typedef struct {
    waypoint_t *waypoints;  // Array of waypoints
    uint32_t num_waypoints; // Number of waypoints
    braid_type_t type;      // Type of braid
    double total_time;      // Total time for braid
} braid_path_t;
```

### `braid_config_t`

Configuration for braiding simulations.

```c
typedef struct {
    double dt;              // TDVP time step
    uint32_t max_bond_dim;  // Maximum MPS bond dimension
    double svd_cutoff;      // SVD truncation threshold
    bool track_skyrmions;   // Track skyrmion positions during evolution
    bool measure_phase;     // Measure accumulated phase
    uint32_t record_interval; // Record observables every N steps
    bool verbose;           // Print progress
    double braid_velocity;  // Braiding velocity
    uint32_t braid_segments; // Number of segments in braid path
} braid_config_t;
```

### `braid_result_t`

Results from a braiding operation.

```c
typedef struct {
    double complex phase;       // Total accumulated phase
    double *times;              // Time array
    double *energies;           // Energy at each time
    double *charges;            // Total topological charge
    double (*positions)[2];     // Skyrmion positions [time][x,y]
    uint32_t num_records;       // Number of recorded points
    bool success;               // Whether braiding completed
} braid_result_t;
```

### `topo_qubit_t`

Topological qubit encoded in a skyrmion pair.

```c
typedef struct {
    tn_mps_state_t *mps;    // MPS state encoding the qubit
    lattice_2d_t *lat;      // 2D lattice
    mpo_t *mpo;             // Hamiltonian
    skyrmion_t sky1;        // First skyrmion
    skyrmion_t sky2;        // Second skyrmion
    double complex alpha;   // |0⟩ amplitude
    double complex beta;    // |1⟩ amplitude
} topo_qubit_t;
```

### `topo_gate_type_t`

Topological gate types.

```c
typedef enum {
    TOPO_GATE_IDENTITY,     // Identity (no braiding)
    TOPO_GATE_BRAID,        // Single braid (exp(iπ/4 σ))
    TOPO_GATE_BRAID_INV,    // Inverse braid
    TOPO_GATE_DOUBLE_BRAID, // Double braid (i σ)
    TOPO_GATE_HADAMARD      // Topological Hadamard
} topo_gate_type_t;
```

## Functions

### Skyrmion Tracking

#### `skyrmion_track`

Track a single skyrmion from spin configuration.

```c
int skyrmion_track(const lattice_2d_t *lat,
                   const double (*spins)[3],
                   skyrmion_t *skyrmion);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `const lattice_2d_t*` | 2D lattice structure |
| `spins` | `const double (*)[3]` | Spin configuration [num_sites][3] |
| `skyrmion` | `skyrmion_t*` | Output skyrmion data |

**Returns**: 0 on success, -1 if no skyrmion found.

#### `skyrmion_track_multiple`

Track multiple skyrmions.

```c
int skyrmion_track_multiple(const lattice_2d_t *lat,
                            const double (*spins)[3],
                            skyrmion_t *skyrmions,
                            uint32_t max_skyrmions);
```

**Returns**: Number of skyrmions found.

### Path Generation

#### `braid_path_circular`

Generate a circular braiding path.

```c
braid_path_t *braid_path_circular(double center_x, double center_y,
                                  double radius,
                                  braid_type_t type,
                                  uint32_t num_segments,
                                  double velocity);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `center_x` | `double` | Center of circular path (x) |
| `center_y` | `double` | Center of circular path (y) |
| `radius` | `double` | Radius of circular path |
| `type` | `braid_type_t` | Braiding type (CW or CCW) |
| `num_segments` | `uint32_t` | Number of path segments |
| `velocity` | `double` | Skyrmion velocity |

**Returns**: Braid path or NULL on failure.

#### `braid_path_exchange`

Generate exchange paths for two skyrmions.

```c
int braid_path_exchange(double x1, double y1,
                        double x2, double y2,
                        braid_type_t type,
                        uint32_t num_segments,
                        double velocity,
                        braid_path_t **path1,
                        braid_path_t **path2);
```

**Returns**: 0 on success.

#### `braid_path_free`

Free a braid path.

```c
void braid_path_free(braid_path_t *path);
```

### Braiding Dynamics

#### `braid_config_default`

Get default braiding configuration.

```c
static inline braid_config_t braid_config_default(void);
```

**Default values**:
- `dt`: 0.01
- `max_bond_dim`: 64
- `svd_cutoff`: 1e-10
- `track_skyrmions`: true
- `measure_phase`: true
- `record_interval`: 10
- `verbose`: false

#### `skyrmion_braid`

Execute a braiding protocol.

```c
braid_result_t *skyrmion_braid(tn_mps_state_t *mps,
                               const mpo_t *mpo,
                               const lattice_2d_t *lat,
                               const braid_path_t *path,
                               const braid_config_t *config);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mps` | `tn_mps_state_t*` | Initial MPS state (modified in place) |
| `mpo` | `const mpo_t*` | Base Hamiltonian |
| `lat` | `const lattice_2d_t*` | 2D lattice |
| `path` | `const braid_path_t*` | Braiding path for mobile skyrmion |
| `config` | `const braid_config_t*` | Braiding configuration |

**Returns**: Braiding result or NULL on failure.

#### `skyrmion_double_braid`

Execute double braiding (exchange two skyrmions).

```c
braid_result_t *skyrmion_double_braid(tn_mps_state_t *mps,
                                      const mpo_t *mpo,
                                      const lattice_2d_t *lat,
                                      const braid_path_t *path1,
                                      const braid_path_t *path2,
                                      const braid_config_t *config);
```

#### `braid_result_free`

Free braiding result.

```c
void braid_result_free(braid_result_t *result);
```

### Topological Qubits

#### `topo_qubit_create`

Create a topological qubit from a skyrmion pair.

```c
topo_qubit_t *topo_qubit_create(const lattice_2d_t *lat,
                                const hamiltonian_params_t *params,
                                double x1, double y1,
                                double x2, double y2,
                                uint32_t bond_dim);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat` | `const lattice_2d_t*` | 2D lattice |
| `params` | `const hamiltonian_params_t*` | Hamiltonian parameters |
| `x1, y1` | `double` | First skyrmion position |
| `x2, y2` | `double` | Second skyrmion position |
| `bond_dim` | `uint32_t` | MPS bond dimension |

**Returns**: Topological qubit initialized in |0⟩ state, or NULL on failure.

#### `topo_qubit_free`

Free a topological qubit.

```c
void topo_qubit_free(topo_qubit_t *qubit);
```

#### `topo_gate_apply`

Apply a topological gate to a qubit.

```c
int topo_gate_apply(topo_qubit_t *qubit,
                    topo_gate_type_t gate,
                    const braid_config_t *config);
```

**Returns**: 0 on success.

#### `topo_qubit_measure_z`

Measure a topological qubit in the Z basis.

```c
int topo_qubit_measure_z(const topo_qubit_t *qubit);
```

**Returns**: Measurement result (+1 or -1).

#### `topo_qubit_fidelity`

Get qubit state fidelity against a target state.

```c
double topo_qubit_fidelity(const topo_qubit_t *qubit,
                           double complex target_alpha,
                           double complex target_beta);
```

**Returns**: Fidelity (0 to 1).

### Phase Extraction

#### `extract_geometric_phase`

Extract geometric phase from braiding.

```c
double complex extract_geometric_phase(const tn_mps_state_t *mps_initial,
                                       const tn_mps_state_t *mps_final);
```

**Returns**: Geometric phase as a complex number.

#### `compute_berry_phase`

Compute Berry phase from TDVP history.

```c
double compute_berry_phase(const tdvp_history_t *history);
```

**Returns**: Berry phase $\gamma = -\mathrm{Im} \sum_t \ln \langle \psi(t) | \psi(t+dt) \rangle$.

## Example

```c
#include "algorithms/tensor_network/skyrmion_braiding.h"

int main(void) {
    // Create 32x32 lattice
    lattice_2d_t *lat = lattice_2d_create(32, 32, LATTICE_SQUARE);

    // Hamiltonian parameters
    hamiltonian_params_t params = {
        .J = 1.0, .D = 0.3, .K = 0.1, .B = 0.2
    };

    // Create topological qubit
    topo_qubit_t *qubit = topo_qubit_create(
        lat, &params, 10.0, 16.0, 22.0, 16.0, 64
    );

    // Apply topological gate
    braid_config_t config = braid_config_default();
    topo_gate_apply(qubit, TOPO_GATE_DOUBLE_BRAID, &config);

    // Measure
    int result = topo_qubit_measure_z(qubit);
    printf("Measurement: %d\n", result);

    // Cleanup
    topo_qubit_free(qubit);
    lattice_2d_free(lat);

    return 0;
}
```

## See Also

- [Skyrmion Braiding Algorithm](../../algorithms/skyrmion-braiding.md) - Theory and usage guide
- [TDVP Algorithm](../../algorithms/tdvp-algorithm.md) - Time evolution method
- [Topological Computing API](topological.md) - Anyon models and surface codes
- [Tensor Network API](tensor-network.md) - MPS and MPO operations
