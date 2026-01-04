/**
 * @file skyrmion_braiding.h
 * @brief Skyrmion braiding protocols for topological quantum computing
 *
 * Skyrmions can serve as topological qubits when their exchange (braiding)
 * implements non-Abelian geometric phases. This module provides:
 *
 * 1. Skyrmion pair creation and manipulation
 * 2. Braiding path generation (clockwise/counterclockwise)
 * 3. Time evolution under current-driven motion
 * 4. Geometric phase extraction
 * 5. Topological gate implementation
 *
 * PHYSICAL BACKGROUND:
 * ===================
 * Skyrmion braiding induces Berry phases due to the spin texture's topology.
 * For Ising-type anyons, the braid matrix is:
 *
 *   B = exp(iπ/4 σ)
 *
 * Two clockwise exchanges gives:
 *   B² = exp(iπ/2 σ) = i σ
 *
 * This is a topologically protected gate - errors in the braiding path
 * only affect the phase up to topologically trivial contributions.
 *
 * IMPLEMENTATION:
 * ==============
 * We simulate braiding by:
 * 1. Creating skyrmion pair at positions (x1,y1), (x2,y2)
 * 2. Applying time-dependent potential/current to move skyrmions
 * 3. One skyrmion encircles the other along a defined path
 * 4. Measuring the final state to extract the geometric phase
 *
 * REFERENCES:
 * ===========
 * [1] C. Psaroudaki & C. Panagopoulos, "Skyrmion Qubits: A New Class of
 *     Quantum Logic Gates", Phys. Rev. Lett. 127, 067201 (2021)
 *
 * [2] X. Zhang et al., "Skyrmion-skyrmion and skyrmion-edge repulsions",
 *     Sci. Rep. 5, 7643 (2015)
 *
 * [3] A. Fert et al., "Skyrmions on the track", Nature Nanotech. 8, 152 (2013)
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef SKYRMION_BRAIDING_H
#define SKYRMION_BRAIDING_H

#include "lattice_2d.h"
#include "mpo_2d.h"
#include "tn_state.h"
#include "dmrg.h"
#include "tdvp.h"
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// SKYRMION POSITION TRACKING
// ============================================================================

/**
 * @brief Skyrmion state
 */
typedef struct {
    double x;           /**< Center x-coordinate */
    double y;           /**< Center y-coordinate */
    double radius;      /**< Effective radius */
    int charge;         /**< Topological charge (+1 or -1) */
    int helicity;       /**< 0 = Néel, 1 = Bloch */
} skyrmion_t;

/**
 * @brief Track skyrmion position from spin configuration
 *
 * Finds skyrmion center by computing local topological charge density
 * and finding the centroid of regions with |q| > threshold.
 *
 * @param lat 2D lattice
 * @param spins Spin configuration [num_sites][3]
 * @param skyrmion Output skyrmion data
 * @return 0 on success, -1 if no skyrmion found
 */
int skyrmion_track(const lattice_2d_t *lat,
                    const double (*spins)[3],
                    skyrmion_t *skyrmion);

/**
 * @brief Track multiple skyrmions
 *
 * @param lat 2D lattice
 * @param spins Spin configuration
 * @param skyrmions Output array (caller allocates)
 * @param max_skyrmions Maximum number to find
 * @return Number of skyrmions found
 */
int skyrmion_track_multiple(const lattice_2d_t *lat,
                             const double (*spins)[3],
                             skyrmion_t *skyrmions,
                             uint32_t max_skyrmions);

// ============================================================================
// BRAIDING PATH GENERATION
// ============================================================================

/**
 * @brief Braiding path type
 */
typedef enum {
    BRAID_CLOCKWISE,        /**< Clockwise exchange */
    BRAID_COUNTERCLOCKWISE, /**< Counter-clockwise exchange */
    BRAID_FIGURE_EIGHT,     /**< Full figure-8 braiding */
    BRAID_HALF_EXCHANGE     /**< Half exchange (for testing) */
} braid_type_t;

/**
 * @brief Waypoint in braiding path
 */
typedef struct {
    double x;           /**< Target x-coordinate */
    double y;           /**< Target y-coordinate */
    double velocity;    /**< Velocity to this point */
} waypoint_t;

/**
 * @brief Braiding path specification
 */
typedef struct {
    waypoint_t *waypoints;  /**< Array of waypoints */
    uint32_t num_waypoints; /**< Number of waypoints */
    braid_type_t type;      /**< Type of braid */
    double total_time;      /**< Total time for braid */
} braid_path_t;

/**
 * @brief Generate circular braiding path
 *
 * Creates a path for one skyrmion to encircle another.
 *
 * @param center_x Center of circular path (x)
 * @param center_y Center of circular path (y)
 * @param radius Radius of circular path
 * @param type Braiding type (CW or CCW)
 * @param num_segments Number of path segments
 * @param velocity Skyrmion velocity
 * @return Braid path or NULL on failure
 */
braid_path_t *braid_path_circular(double center_x, double center_y,
                                   double radius,
                                   braid_type_t type,
                                   uint32_t num_segments,
                                   double velocity);

/**
 * @brief Generate exchange path between two positions
 *
 * Creates paths for two skyrmions to exchange positions.
 *
 * @param x1, y1 First skyrmion position
 * @param x2, y2 Second skyrmion position
 * @param type Braiding type
 * @param num_segments Number of segments
 * @param velocity Skyrmion velocity
 * @param path1 Output: path for first skyrmion
 * @param path2 Output: path for second skyrmion
 * @return 0 on success
 */
int braid_path_exchange(double x1, double y1,
                         double x2, double y2,
                         braid_type_t type,
                         uint32_t num_segments,
                         double velocity,
                         braid_path_t **path1,
                         braid_path_t **path2);

/**
 * @brief Free braid path
 */
void braid_path_free(braid_path_t *path);

// ============================================================================
// BRAIDING DYNAMICS
// ============================================================================

/**
 * @brief Braiding simulation configuration
 */
typedef struct {
    double dt;              /**< TDVP time step */
    uint32_t max_bond_dim;  /**< Maximum MPS bond dimension */
    double svd_cutoff;      /**< SVD truncation threshold */
    bool track_skyrmions;   /**< Track skyrmion positions during evolution */
    bool measure_phase;     /**< Measure accumulated phase */
    uint32_t record_interval; /**< Record observables every N steps */
    bool verbose;           /**< Print progress */
    double braid_velocity;  /**< Braiding velocity (higher = faster) */
    uint32_t braid_segments; /**< Number of segments in braid path */
} braid_config_t;

/**
 * @brief Default braiding configuration
 */
static inline braid_config_t braid_config_default(void) {
    return (braid_config_t){
        .dt = 0.01,
        .max_bond_dim = 64,
        .svd_cutoff = 1e-10,
        .track_skyrmions = true,
        .measure_phase = true,
        .record_interval = 10,
        .verbose = false,
        .braid_velocity = 1.0,
        .braid_segments = 8
    };
}

/**
 * @brief Braiding result
 */
typedef struct {
    double complex phase;       /**< Total accumulated phase */
    double *times;              /**< Time array */
    double *energies;           /**< Energy at each time */
    double *charges;            /**< Total topological charge */
    double (*positions)[2];     /**< Skyrmion positions [time][x,y] */
    uint32_t num_records;       /**< Number of recorded points */
    bool success;               /**< Whether braiding completed */
} braid_result_t;

/**
 * @brief Free braiding result
 */
void braid_result_free(braid_result_t *result);

/**
 * @brief Execute braiding protocol
 *
 * Evolves the MPS state while moving skyrmion(s) along the specified path
 * using current-driven dynamics.
 *
 * @param mps Initial MPS state (modified in place)
 * @param mpo Base Hamiltonian
 * @param lat 2D lattice
 * @param path Braiding path for mobile skyrmion
 * @param config Braiding configuration
 * @return Braiding result or NULL on failure
 */
braid_result_t *skyrmion_braid(tn_mps_state_t *mps,
                                const mpo_t *mpo,
                                const lattice_2d_t *lat,
                                const braid_path_t *path,
                                const braid_config_t *config);

/**
 * @brief Execute double braiding (exchange two skyrmions)
 *
 * Both skyrmions move simultaneously along complementary paths.
 *
 * @param mps Initial state
 * @param mpo Hamiltonian
 * @param lat Lattice
 * @param path1 Path for first skyrmion
 * @param path2 Path for second skyrmion
 * @param config Configuration
 * @return Result or NULL on failure
 */
braid_result_t *skyrmion_double_braid(tn_mps_state_t *mps,
                                       const mpo_t *mpo,
                                       const lattice_2d_t *lat,
                                       const braid_path_t *path1,
                                       const braid_path_t *path2,
                                       const braid_config_t *config);

// ============================================================================
// TOPOLOGICAL GATES
// ============================================================================

/**
 * @brief Topological gate type
 */
typedef enum {
    TOPO_GATE_IDENTITY,     /**< Identity (no braiding) */
    TOPO_GATE_BRAID,        /**< Single braid (exp(iπ/4 σ)) */
    TOPO_GATE_BRAID_INV,    /**< Inverse braid */
    TOPO_GATE_DOUBLE_BRAID, /**< Double braid (i σ) */
    TOPO_GATE_HADAMARD      /**< Topological Hadamard (requires magic state) */
} topo_gate_type_t;

/**
 * @brief Topological qubit (encoded in skyrmion pair)
 */
typedef struct {
    tn_mps_state_t *mps;    /**< MPS state encoding the qubit */
    lattice_2d_t *lat;      /**< 2D lattice */
    mpo_t *mpo;             /**< Hamiltonian */
    skyrmion_t sky1;        /**< First skyrmion */
    skyrmion_t sky2;        /**< Second skyrmion */
    double complex alpha;   /**< |0⟩ amplitude */
    double complex beta;    /**< |1⟩ amplitude */
} topo_qubit_t;

/**
 * @brief Create topological qubit from skyrmion pair
 *
 * Initializes qubit in |0⟩ state (both skyrmions with same helicity).
 *
 * @param lat 2D lattice
 * @param params Hamiltonian parameters
 * @param x1, y1 First skyrmion position
 * @param x2, y2 Second skyrmion position
 * @param bond_dim MPS bond dimension
 * @return Topological qubit or NULL on failure
 */
topo_qubit_t *topo_qubit_create(const lattice_2d_t *lat,
                                 const hamiltonian_params_t *params,
                                 double x1, double y1,
                                 double x2, double y2,
                                 uint32_t bond_dim);

/**
 * @brief Free topological qubit
 */
void topo_qubit_free(topo_qubit_t *qubit);

/**
 * @brief Apply topological gate to qubit
 *
 * @param qubit Topological qubit
 * @param gate Gate type
 * @param config Braiding configuration
 * @return 0 on success
 */
int topo_gate_apply(topo_qubit_t *qubit,
                     topo_gate_type_t gate,
                     const braid_config_t *config);

/**
 * @brief Measure topological qubit in Z basis
 *
 * Measures the relative helicity of the two skyrmions.
 *
 * @param qubit Topological qubit
 * @return Measurement result (+1 or -1)
 */
int topo_qubit_measure_z(const topo_qubit_t *qubit);

/**
 * @brief Get qubit state fidelity
 *
 * Compares current state to ideal target state.
 *
 * @param qubit Current qubit
 * @param target_alpha Target |0⟩ amplitude
 * @param target_beta Target |1⟩ amplitude
 * @return Fidelity (0 to 1)
 */
double topo_qubit_fidelity(const topo_qubit_t *qubit,
                            double complex target_alpha,
                            double complex target_beta);

// ============================================================================
// PHASE EXTRACTION
// ============================================================================

/**
 * @brief Extract geometric phase from braiding
 *
 * Computes the phase accumulated during a closed braiding path
 * by comparing initial and final states.
 *
 * @param mps_initial Initial MPS state
 * @param mps_final Final MPS state after braiding
 * @return Geometric phase
 */
double complex extract_geometric_phase(const tn_mps_state_t *mps_initial,
                                        const tn_mps_state_t *mps_final);

/**
 * @brief Compute Berry phase along path
 *
 * Discretized Berry phase: γ = -Im Σ ln⟨ψ(t)|ψ(t+dt)⟩
 *
 * @param history TDVP history with states at each time
 * @return Berry phase
 */
double compute_berry_phase(const tdvp_history_t *history);

#ifdef __cplusplus
}
#endif

#endif /* SKYRMION_BRAIDING_H */
