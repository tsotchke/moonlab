/**
 * @file lattice_2d.h
 * @brief 2D Lattice Support for MPS Simulations
 *
 * Provides infrastructure for simulating 2D quantum systems using MPS via
 * snake ordering. Essential for topological skyrmion simulations.
 *
 * SNAKE ORDERING:
 * ===============
 * Maps 2D lattice to 1D chain while preserving locality:
 *
 *   2D Grid (Lx=4, Ly=3):        Snake Order:
 *
 *   8  9  10 11                  0→1→2→3
 *   4  5  6  7                   7←6←5←4
 *   0  1  2  3                   8→9→10→11
 *
 * This minimizes long-range correlations in the MPS representation.
 *
 * SUPPORTED LATTICES:
 * ==================
 * - Square lattice (nearest neighbor, next-nearest neighbor)
 * - Triangular lattice (6-fold coordination)
 * - Kagome lattice (for frustrated magnetism)
 * - Honeycomb lattice (graphene-like)
 *
 * APPLICATIONS:
 * =============
 * - Magnetic skyrmions
 * - 2D topological phases
 * - Frustrated magnetism
 * - Quantum spin liquids
 *
 * REFERENCES:
 * ===========
 * [1] N. Nagaosa & Y. Tokura, "Topological properties and dynamics of
 *     magnetic skyrmions", Nature Nanotech. 8, 899 (2013)
 *
 * [2] A. Fert et al., "Magnetic skyrmions: advances in physics and
 *     potential applications", Nature Rev. Mat. 2, 17031 (2017)
 *
 * [3] S.S. Pershoguba et al., "Skyrmion qubits: A new class of quantum
 *     logic gates", Phys. Rev. B 105, 054421 (2022)
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef LATTICE_2D_H
#define LATTICE_2D_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// LATTICE TYPES
// ============================================================================

/**
 * @brief Supported 2D lattice geometries
 */
typedef enum {
    LATTICE_SQUARE,        /**< Square lattice (z=4) */
    LATTICE_TRIANGULAR,    /**< Triangular lattice (z=6) */
    LATTICE_HONEYCOMB,     /**< Honeycomb/graphene lattice (z=3) */
    LATTICE_KAGOME         /**< Kagome lattice (frustrated) */
} lattice_type_t;

/**
 * @brief Boundary conditions
 */
typedef enum {
    BC_OPEN,               /**< Open boundaries */
    BC_PERIODIC_X,         /**< Periodic in X only (cylinder) */
    BC_PERIODIC_Y,         /**< Periodic in Y only */
    BC_PERIODIC_XY         /**< Fully periodic (torus) */
} boundary_condition_t;

/**
 * @brief 2D coordinate
 */
typedef struct {
    int32_t x;
    int32_t y;
} coord_2d_t;

/**
 * @brief Bond direction for DMI
 */
typedef struct {
    double dx;             /**< x-component of bond direction */
    double dy;             /**< y-component of bond direction */
    double dz;             /**< z-component (for DMI vector) */
} bond_vector_t;

/**
 * @brief Neighbor information
 */
typedef struct {
    uint32_t site;         /**< Linear index of neighbor */
    bond_vector_t bond;    /**< Bond direction vector */
    double distance;       /**< Distance to neighbor */
    bool valid;            /**< Whether this neighbor exists */
} neighbor_t;

/**
 * @brief 2D Lattice structure
 */
typedef struct {
    uint32_t Lx;                    /**< Width (number of columns) */
    uint32_t Ly;                    /**< Height (number of rows) */
    uint32_t num_sites;             /**< Total number of sites */
    lattice_type_t type;            /**< Lattice geometry */
    boundary_condition_t bc;        /**< Boundary conditions */

    // Precomputed mappings
    uint32_t *snake_to_grid;        /**< snake_index -> grid_index */
    uint32_t *grid_to_snake;        /**< grid_index -> snake_index */
    coord_2d_t *snake_to_coord;     /**< snake_index -> (x, y) */

    // Neighbor lists (precomputed for efficiency)
    neighbor_t **neighbors;         /**< neighbors[site][n] for n=0..max_neighbors-1 */
    uint32_t max_neighbors;         /**< Maximum coordination number */
    uint32_t *num_neighbors;        /**< Actual number of neighbors per site */

    // Lattice constants
    double a;                       /**< Lattice constant */
} lattice_2d_t;

// ============================================================================
// LATTICE CREATION
// ============================================================================

/**
 * @brief Create a 2D lattice with snake ordering
 *
 * @param Lx Width (columns)
 * @param Ly Height (rows)
 * @param type Lattice geometry
 * @param bc Boundary conditions
 * @return Lattice structure or NULL on failure
 */
lattice_2d_t *lattice_2d_create(uint32_t Lx, uint32_t Ly,
                                 lattice_type_t type,
                                 boundary_condition_t bc);

/**
 * @brief Free lattice structure
 */
void lattice_2d_free(lattice_2d_t *lat);

// ============================================================================
// COORDINATE CONVERSION
// ============================================================================

/**
 * @brief Convert 2D grid coordinates to snake (MPS) index
 *
 * Snake ordering alternates direction each row:
 *   Row 0: left-to-right (0, 1, 2, ...)
 *   Row 1: right-to-left (..., Lx+2, Lx+1, Lx)
 *   Row 2: left-to-right (2*Lx, 2*Lx+1, ...)
 *
 * @param lat Lattice structure
 * @param x Column index (0 to Lx-1)
 * @param y Row index (0 to Ly-1)
 * @return Snake index for MPS
 */
static inline uint32_t coord_to_snake(const lattice_2d_t *lat, int32_t x, int32_t y) {
    if (!lat || x < 0 || y < 0 || (uint32_t)x >= lat->Lx || (uint32_t)y >= lat->Ly) {
        return UINT32_MAX;  // Invalid
    }

    // Snake ordering: even rows L→R, odd rows R→L
    if (y % 2 == 0) {
        return y * lat->Lx + x;
    } else {
        return y * lat->Lx + (lat->Lx - 1 - x);
    }
}

/**
 * @brief Convert snake (MPS) index to 2D grid coordinates
 *
 * @param lat Lattice structure
 * @param snake_idx Snake index
 * @return 2D coordinates
 */
static inline coord_2d_t snake_to_coord(const lattice_2d_t *lat, uint32_t snake_idx) {
    coord_2d_t c = {-1, -1};
    if (!lat || snake_idx >= lat->num_sites) return c;

    c.y = snake_idx / lat->Lx;
    uint32_t col_in_row = snake_idx % lat->Lx;

    // Snake ordering: even rows L→R, odd rows R→L
    if (c.y % 2 == 0) {
        c.x = col_in_row;
    } else {
        c.x = lat->Lx - 1 - col_in_row;
    }

    return c;
}

/**
 * @brief Get grid index from coordinates (row-major, no snake)
 */
static inline uint32_t coord_to_grid(const lattice_2d_t *lat, int32_t x, int32_t y) {
    if (!lat || x < 0 || y < 0 || (uint32_t)x >= lat->Lx || (uint32_t)y >= lat->Ly) {
        return UINT32_MAX;
    }
    return y * lat->Lx + x;
}

/**
 * @brief Get coordinates from grid index
 */
static inline coord_2d_t grid_to_coord(const lattice_2d_t *lat, uint32_t grid_idx) {
    coord_2d_t c = {-1, -1};
    if (!lat || grid_idx >= lat->num_sites) return c;
    c.y = grid_idx / lat->Lx;
    c.x = grid_idx % lat->Lx;
    return c;
}

// ============================================================================
// NEIGHBOR FUNCTIONS
// ============================================================================

/**
 * @brief Get neighbors of a site in snake ordering
 *
 * @param lat Lattice structure
 * @param snake_idx Site index in snake ordering
 * @param neighbors Output array (must have space for max_neighbors)
 * @return Number of valid neighbors
 */
uint32_t lattice_2d_get_neighbors(const lattice_2d_t *lat,
                                   uint32_t snake_idx,
                                   neighbor_t *neighbors);

/**
 * @brief Get nearest neighbor in a specific direction
 *
 * @param lat Lattice structure
 * @param snake_idx Site index
 * @param dx Direction in x (-1, 0, +1)
 * @param dy Direction in y (-1, 0, +1)
 * @return Neighbor snake index or UINT32_MAX if invalid
 */
uint32_t lattice_2d_neighbor_at(const lattice_2d_t *lat,
                                 uint32_t snake_idx,
                                 int dx, int dy);

/**
 * @brief Check if two sites are neighbors
 */
bool lattice_2d_are_neighbors(const lattice_2d_t *lat,
                               uint32_t site1, uint32_t site2);

/**
 * @brief Get bond vector between two sites
 *
 * For DMI interaction: D · (S_i × S_j) where D is along d_ij
 *
 * @param lat Lattice structure
 * @param from Source site (snake index)
 * @param to Target site (snake index)
 * @return Bond vector (normalized)
 */
bond_vector_t lattice_2d_bond_vector(const lattice_2d_t *lat,
                                      uint32_t from, uint32_t to);

// ============================================================================
// DISTANCE AND GEOMETRY
// ============================================================================

/**
 * @brief Compute distance between two sites
 *
 * Accounts for periodic boundary conditions.
 *
 * @param lat Lattice structure
 * @param site1 First site (snake index)
 * @param site2 Second site (snake index)
 * @return Euclidean distance in units of lattice constant
 */
double lattice_2d_distance(const lattice_2d_t *lat,
                            uint32_t site1, uint32_t site2);

/**
 * @brief Get real-space coordinates of a site
 *
 * For non-square lattices, returns proper geometric positions.
 *
 * @param lat Lattice structure
 * @param snake_idx Site index
 * @param x Output x-coordinate
 * @param y Output y-coordinate
 */
void lattice_2d_real_coords(const lattice_2d_t *lat,
                             uint32_t snake_idx,
                             double *x, double *y);

// ============================================================================
// LATTICE PROPERTIES
// ============================================================================

/**
 * @brief Get coordination number for lattice type
 */
static inline uint32_t lattice_coordination(lattice_type_t type) {
    switch (type) {
        case LATTICE_SQUARE:     return 4;
        case LATTICE_TRIANGULAR: return 6;
        case LATTICE_HONEYCOMB:  return 3;
        case LATTICE_KAGOME:     return 4;
        default: return 4;
    }
}

/**
 * @brief Print lattice information
 */
void lattice_2d_print_info(const lattice_2d_t *lat);

/**
 * @brief Visualize snake ordering (ASCII art)
 */
void lattice_2d_print_snake(const lattice_2d_t *lat);

// ============================================================================
// SKYRMION-SPECIFIC UTILITIES
// ============================================================================

/**
 * @brief Compute local topological charge contribution
 *
 * For a plaquette (i, j, k), the topological charge contribution is:
 * q_ijk = (1/4π) * solid_angle(S_i, S_j, S_k)
 *
 * This is used to compute the total skyrmion number:
 * Q = Σ q_ijk over all plaquettes
 *
 * @param Si Spin vector at site i (Sx, Sy, Sz)
 * @param Sj Spin vector at site j
 * @param Sk Spin vector at site k
 * @return Topological charge contribution
 */
double compute_plaquette_charge(const double *Si, const double *Sj, const double *Sk);

/**
 * @brief Get plaquettes for topological charge calculation
 *
 * For square lattice, each plaquette is a triangle formed by three corners
 * of a unit cell. Each unit cell contributes 2 triangles.
 *
 * @param lat Lattice structure
 * @param num_plaquettes Output: number of plaquettes
 * @return Array of plaquette indices [p][3] or NULL on failure
 */
uint32_t (*lattice_2d_get_plaquettes(const lattice_2d_t *lat,
                                      uint32_t *num_plaquettes))[3];

/**
 * @brief Initialize skyrmion spin configuration
 *
 * Creates a Néel-type or Bloch-type skyrmion centered at (cx, cy)
 *
 * @param lat Lattice structure
 * @param cx Center x-coordinate
 * @param cy Center y-coordinate
 * @param radius Skyrmion radius (in lattice units)
 * @param helicity 0 = Néel, π/2 = Bloch
 * @param polarity +1 or -1 (up or down core)
 * @param spins Output spin array [num_sites][3]
 */
void skyrmion_init_classical(const lattice_2d_t *lat,
                              double cx, double cy,
                              double radius, double helicity,
                              int polarity, double (*spins)[3]);

#ifdef __cplusplus
}
#endif

#endif // LATTICE_2D_H
