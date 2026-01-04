/**
 * @file mpo_2d.h
 * @brief Production-grade MPO construction for 2D quantum systems
 *
 * Implements finite-automaton MPO for 2D lattices mapped to 1D via snake ordering.
 * Key insight: nearest-neighbor 2D interactions become long-range 1D interactions
 * with maximum range R = Lx (lattice width).
 *
 * FINITE AUTOMATON APPROACH:
 * ==========================
 * For each bond (i,j) with i < j in snake ordering:
 *   - At site i: Start interaction by applying operator, create carry index
 *   - At sites i+1 to j-1: Pass carry index through with identity
 *   - At site j: Complete interaction by applying second operator
 *
 * BOND DIMENSION SCALING:
 * ======================
 * MPO bond dimension = 2 + num_open_bonds × interaction_types
 *
 * For Heisenberg + DMI on square lattice:
 *   - Heisenberg: XX, YY, ZZ (3 types)
 *   - DMI: XY, YX, XZ, ZX, YZ, ZY (6 types, direction-dependent)
 *   - Total: ~9 types per bond
 *   - Max open bonds: Lx (vertical bonds crossing a horizontal cut)
 *   - Bond dim ≈ 2 + 9 × Lx
 *
 * SUPPORTED HAMILTONIANS:
 * ======================
 * - Heisenberg exchange: -J Σ S_i · S_j
 * - DMI: D Σ d_ij · (S_i × S_j)
 * - Zeeman: -B · Σ S_i
 * - Anisotropy: -K Σ (S_i^z)²
 * - Dipolar (optional): long-range dipole-dipole
 *
 * REFERENCES:
 * ===========
 * [1] Crosswhite & Bacon, Finite automata for caching in matrix product algorithms
 * [2] Stoudenmire & White, Studying 2D systems with DMRG
 * [3] Nagaosa & Tokura, Topological properties of magnetic skyrmions
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef MPO_2D_H
#define MPO_2D_H

#include "dmrg.h"
#include "lattice_2d.h"
#include "tensor.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// INTERACTION TYPES
// ============================================================================

/**
 * @brief Types of two-site interactions in the Hamiltonian
 */
typedef enum {
    INTERACT_XX = 0,    /**< X_i X_j */
    INTERACT_YY,        /**< Y_i Y_j */
    INTERACT_ZZ,        /**< Z_i Z_j */
    INTERACT_XY,        /**< X_i Y_j (for DMI) */
    INTERACT_YX,        /**< Y_i X_j (for DMI) */
    INTERACT_XZ,        /**< X_i Z_j */
    INTERACT_ZX,        /**< Z_i X_j */
    INTERACT_YZ,        /**< Y_i Z_j */
    INTERACT_ZY,        /**< Z_i Y_j */
    NUM_INTERACT_TYPES
} interaction_type_t;

/**
 * @brief DMI type (affects direction vector calculation)
 */
typedef enum {
    DMI_BULK,           /**< Bulk DMI: d_ij parallel to bond */
    DMI_INTERFACIAL,    /**< Interfacial DMI: d_ij perpendicular to bond (in-plane) */
    DMI_NEEL,           /**< Néel-type: d_ij = r_ij/|r_ij| */
    DMI_BLOCH           /**< Bloch-type: d_ij perpendicular to z and r_ij */
} dmi_type_t;

// ============================================================================
// HAMILTONIAN PARAMETERS
// ============================================================================

/**
 * @brief Complete Hamiltonian parameters for 2D magnetic systems
 */
typedef struct {
    /* Exchange coupling */
    double J;               /**< Heisenberg exchange (J > 0 ferromagnetic) */
    bool anisotropic;       /**< If true, use Jx, Jy, Jz separately */
    double Jx;              /**< XX coupling */
    double Jy;              /**< YY coupling */
    double Jz;              /**< ZZ coupling */

    /* Dzyaloshinskii-Moriya interaction */
    double D;               /**< DMI strength */
    dmi_type_t dmi_type;    /**< Type of DMI */

    /* External field */
    double Bx;              /**< Zeeman field x-component */
    double By;              /**< Zeeman field y-component */
    double Bz;              /**< Zeeman field z-component */

    /* Anisotropy */
    double K;               /**< Easy-axis anisotropy along z */
    double Kx;              /**< In-plane anisotropy x */
    double Ky;              /**< In-plane anisotropy y */

    /* Next-nearest neighbor (optional) */
    double J2;              /**< NNN exchange */
    double D2;              /**< NNN DMI */

    /* Dipolar interaction (optional) */
    bool include_dipolar;   /**< Include dipole-dipole interaction */
    double dipolar_strength;/**< Dipolar coupling strength */
    double dipolar_cutoff;  /**< Distance cutoff for dipolar */

    /* Numerical parameters */
    double coupling_cutoff; /**< Ignore couplings smaller than this */
} hamiltonian_params_t;

/**
 * @brief Default Hamiltonian parameters for skyrmion-hosting materials
 *
 * Based on typical values for FeGe or MnSi thin films.
 */
static inline hamiltonian_params_t hamiltonian_params_skyrmion_default(void) {
    return (hamiltonian_params_t){
        .J = 1.0,               /* Exchange sets energy scale */
        .anisotropic = false,
        .Jx = 1.0, .Jy = 1.0, .Jz = 1.0,
        .D = 0.5,               /* D/J ~ 0.5 for stable skyrmions */
        .dmi_type = DMI_INTERFACIAL,
        .Bx = 0.0, .By = 0.0, .Bz = 0.3,  /* Field to stabilize skyrmions */
        .K = 0.1,               /* Small easy-axis anisotropy */
        .Kx = 0.0, .Ky = 0.0,
        .J2 = 0.0, .D2 = 0.0,
        .include_dipolar = false,
        .dipolar_strength = 0.0,
        .dipolar_cutoff = 5.0,
        .coupling_cutoff = 1e-10
    };
}

// ============================================================================
// BOND INTERACTION STRUCTURE
// ============================================================================

/**
 * @brief Single bond interaction specification
 *
 * Describes one interaction term in the Hamiltonian for a specific bond.
 */
typedef struct {
    uint32_t site_i;        /**< First site (in snake ordering, i < j) */
    uint32_t site_j;        /**< Second site */
    interaction_type_t type;/**< Type of interaction */
    double coefficient;     /**< Coupling coefficient */
} bond_interaction_t;

/**
 * @brief Collection of all bond interactions
 */
typedef struct {
    bond_interaction_t *interactions;   /**< Array of interactions */
    uint32_t num_interactions;          /**< Number of interactions */
    uint32_t capacity;                  /**< Allocated capacity */
    uint32_t num_sites;                 /**< Total number of sites */
    uint32_t max_range;                 /**< Maximum |i - j| for any bond */
} bond_list_t;

// ============================================================================
// FINITE AUTOMATON MPO BUILDER
// ============================================================================

/**
 * @brief State of the finite automaton for MPO construction
 *
 * Each MPO bond index represents a state in the automaton.
 * States track which interactions are "open" (started but not closed).
 */
typedef struct {
    /* Automaton states */
    uint32_t state_identity;    /**< Index: identity pass-through */
    uint32_t state_final;       /**< Index: accumulate completed terms */

    /* Open interaction tracking */
    uint32_t *open_bond_states; /**< open_bond_states[bond_idx * NUM_INTERACT_TYPES + type] = state */
    uint32_t num_open_slots;    /**< Number of open bond slots */
    uint32_t bond_dim;          /**< Total MPO bond dimension */

    /* Site-specific data */
    uint32_t current_site;      /**< Site being processed */
    uint32_t num_sites;         /**< Total sites */
} mpo_automaton_t;

/**
 * @brief Create bond interaction list from lattice and parameters
 *
 * Enumerates all nearest-neighbor (and optionally NNN) bonds and
 * computes their interaction coefficients including DMI direction.
 *
 * @param lat 2D lattice structure
 * @param params Hamiltonian parameters
 * @return Bond list or NULL on failure
 */
bond_list_t *bond_list_create(const lattice_2d_t *lat,
                               const hamiltonian_params_t *params);

/**
 * @brief Free bond list
 */
void bond_list_free(bond_list_t *bonds);

/**
 * @brief Print bond list for debugging
 */
void bond_list_print(const bond_list_t *bonds);

// ============================================================================
// MPO CONSTRUCTION
// ============================================================================

/**
 * @brief Create production-grade MPO for 2D Hamiltonian
 *
 * Uses finite automaton approach to handle long-range bonds from snake ordering.
 *
 * @param lat 2D lattice with snake ordering
 * @param params Hamiltonian parameters
 * @return MPO or NULL on failure
 */
mpo_t *mpo_2d_create(const lattice_2d_t *lat, const hamiltonian_params_t *params);

/**
 * @brief Create MPO from explicit bond list
 *
 * Lower-level interface for custom Hamiltonians.
 *
 * @param bonds Bond interaction list
 * @param on_site_z Array of on-site Z coefficients (Zeeman)
 * @param on_site_zz Array of on-site Z² coefficients (anisotropy)
 * @return MPO or NULL on failure
 */
mpo_t *mpo_from_bond_list(const bond_list_t *bonds,
                           const double *on_site_z,
                           const double *on_site_zz);

/**
 * @brief Estimate MPO bond dimension for lattice
 *
 * @param lat 2D lattice
 * @param include_dmi Include DMI terms
 * @return Estimated bond dimension
 */
uint32_t mpo_2d_estimate_bond_dim(const lattice_2d_t *lat, bool include_dmi);

// ============================================================================
// DMI DIRECTION CALCULATION
// ============================================================================

/**
 * @brief Compute DMI direction vector for a bond
 *
 * For interfacial DMI: d_ij = z × r_ij (perpendicular to bond, in-plane)
 * For bulk DMI: d_ij = r_ij (along bond)
 *
 * @param lat Lattice structure
 * @param site_i First site (snake index)
 * @param site_j Second site (snake index)
 * @param dmi_type Type of DMI
 * @param d_out Output direction vector [dx, dy, dz] (normalized)
 */
void compute_dmi_direction(const lattice_2d_t *lat,
                            uint32_t site_i, uint32_t site_j,
                            dmi_type_t dmi_type,
                            double d_out[3]);

/**
 * @brief Get interaction coefficients from DMI vector
 *
 * DMI term: D * d · (S_i × S_j) expands to:
 *   D * (dx*(Y_i Z_j - Z_i Y_j) + dy*(Z_i X_j - X_i Z_j) + dz*(X_i Y_j - Y_i X_j))
 *
 * @param D DMI strength
 * @param d DMI direction vector [dx, dy, dz]
 * @param coeffs Output: coefficients for [XY, YX, XZ, ZX, YZ, ZY]
 */
void dmi_to_coefficients(double D, const double d[3], double coeffs[6]);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Verify MPO correctness by checking operator structure
 *
 * Performs sanity checks on the MPO:
 * - Bond dimensions match at boundaries
 * - Hermiticity of Hamiltonian
 * - No NaN or Inf values
 *
 * @param mpo MPO to verify
 * @return true if MPO passes all checks
 */
bool mpo_verify(const mpo_t *mpo);

/**
 * @brief Print MPO statistics
 */
void mpo_print_info(const mpo_t *mpo);

/**
 * @brief Compute exact Hamiltonian matrix (for small systems)
 *
 * For verification purposes only - exponential in system size.
 *
 * @param mpo MPO representation
 * @return Full Hamiltonian matrix or NULL if too large
 */
tensor_t *mpo_to_matrix(const mpo_t *mpo);

#ifdef __cplusplus
}
#endif

#endif /* MPO_2D_H */
