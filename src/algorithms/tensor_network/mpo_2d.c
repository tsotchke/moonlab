/**
 * @file mpo_2d.c
 * @brief Production-grade finite-automaton MPO for 2D quantum systems
 *
 * Implementation of efficient MPO construction for 2D lattices.
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "mpo_2d.h"
#include "lattice_2d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// PAULI MATRICES
// ============================================================================

static const double complex PAULI_I[4] = {1, 0, 0, 1};
static const double complex PAULI_X[4] = {0, 1, 1, 0};
static const double complex PAULI_Y[4] = {0, -I, I, 0};
static const double complex PAULI_Z[4] = {1, 0, 0, -1};

/**
 * @brief Get Pauli matrix for first operator in interaction
 */
static const double complex *get_left_pauli(interaction_type_t type) {
    switch (type) {
        case INTERACT_XX: case INTERACT_XY: case INTERACT_XZ: return PAULI_X;
        case INTERACT_YY: case INTERACT_YX: case INTERACT_YZ: return PAULI_Y;
        case INTERACT_ZZ: case INTERACT_ZX: case INTERACT_ZY: return PAULI_Z;
        default: return PAULI_I;
    }
}

/**
 * @brief Get Pauli matrix for second operator in interaction
 */
static const double complex *get_right_pauli(interaction_type_t type) {
    switch (type) {
        case INTERACT_XX: case INTERACT_YX: case INTERACT_ZX: return PAULI_X;
        case INTERACT_YY: case INTERACT_XY: case INTERACT_ZY: return PAULI_Y;
        case INTERACT_ZZ: case INTERACT_XZ: case INTERACT_YZ: return PAULI_Z;
        default: return PAULI_I;
    }
}

// ============================================================================
// BOND LIST CONSTRUCTION
// ============================================================================

bond_list_t *bond_list_create(const lattice_2d_t *lat,
                               const hamiltonian_params_t *params) {
    if (!lat || !params) return NULL;

    bond_list_t *bonds = (bond_list_t *)calloc(1, sizeof(bond_list_t));
    if (!bonds) return NULL;

    bonds->num_sites = lat->num_sites;
    bonds->capacity = lat->num_sites * 10;  // Estimate: ~10 interactions per site
    bonds->interactions = (bond_interaction_t *)calloc(bonds->capacity,
                                                        sizeof(bond_interaction_t));
    if (!bonds->interactions) {
        free(bonds);
        return NULL;
    }

    bonds->num_interactions = 0;
    bonds->max_range = 0;

    double Jx = params->anisotropic ? params->Jx : params->J;
    double Jy = params->anisotropic ? params->Jy : params->J;
    double Jz = params->anisotropic ? params->Jz : params->J;

    // Enumerate all nearest-neighbor bonds
    for (uint32_t s = 0; s < lat->num_sites; s++) {
        coord_2d_t c = snake_to_coord(lat, s);

        // Check all directions: +x, -x, +y, -y
        int dx_vals[] = {1, -1, 0, 0};
        int dy_vals[] = {0, 0, 1, -1};

        for (int dir = 0; dir < 4; dir++) {
            int nx = c.x + dx_vals[dir];
            int ny = c.y + dy_vals[dir];

            // Check boundaries
            if (nx < 0 || ny < 0 || (uint32_t)nx >= lat->Lx || (uint32_t)ny >= lat->Ly) {
                // Handle periodic boundaries if needed
                if (lat->bc == BC_PERIODIC_XY || lat->bc == BC_PERIODIC_X) {
                    if (nx < 0) nx += lat->Lx;
                    if ((uint32_t)nx >= lat->Lx) nx -= lat->Lx;
                }
                if (lat->bc == BC_PERIODIC_XY || lat->bc == BC_PERIODIC_Y) {
                    if (ny < 0) ny += lat->Ly;
                    if ((uint32_t)ny >= lat->Ly) ny -= lat->Ly;
                }

                // If still out of bounds after periodic wrap, skip
                if (nx < 0 || ny < 0 || (uint32_t)nx >= lat->Lx || (uint32_t)ny >= lat->Ly) {
                    continue;
                }
            }

            uint32_t neighbor = coord_to_snake(lat, nx, ny);
            if (neighbor == UINT32_MAX) continue;

            // Only add each bond once (i < j in snake ordering)
            if (s >= neighbor) continue;

            uint32_t range = neighbor - s;
            if (range > bonds->max_range) {
                bonds->max_range = range;
            }

            // Ensure capacity
            if (bonds->num_interactions + 10 > bonds->capacity) {
                bonds->capacity *= 2;
                bonds->interactions = (bond_interaction_t *)realloc(
                    bonds->interactions,
                    bonds->capacity * sizeof(bond_interaction_t));
                if (!bonds->interactions) {
                    free(bonds);
                    return NULL;
                }
            }

            // Add Heisenberg exchange: XX, YY, ZZ
            if (fabs(Jx) > params->coupling_cutoff) {
                bonds->interactions[bonds->num_interactions++] = (bond_interaction_t){
                    .site_i = s, .site_j = neighbor,
                    .type = INTERACT_XX, .coefficient = -Jx
                };
            }
            if (fabs(Jy) > params->coupling_cutoff) {
                bonds->interactions[bonds->num_interactions++] = (bond_interaction_t){
                    .site_i = s, .site_j = neighbor,
                    .type = INTERACT_YY, .coefficient = -Jy
                };
            }
            if (fabs(Jz) > params->coupling_cutoff) {
                bonds->interactions[bonds->num_interactions++] = (bond_interaction_t){
                    .site_i = s, .site_j = neighbor,
                    .type = INTERACT_ZZ, .coefficient = -Jz
                };
            }

            // Add DMI if present
            if (fabs(params->D) > params->coupling_cutoff) {
                double d[3];
                compute_dmi_direction(lat, s, neighbor, params->dmi_type, d);

                double dmi_coeffs[6];
                dmi_to_coefficients(params->D, d, dmi_coeffs);

                // XY - YX term (from dz component)
                if (fabs(dmi_coeffs[0]) > params->coupling_cutoff) {
                    bonds->interactions[bonds->num_interactions++] = (bond_interaction_t){
                        .site_i = s, .site_j = neighbor,
                        .type = INTERACT_XY, .coefficient = dmi_coeffs[0]
                    };
                }
                if (fabs(dmi_coeffs[1]) > params->coupling_cutoff) {
                    bonds->interactions[bonds->num_interactions++] = (bond_interaction_t){
                        .site_i = s, .site_j = neighbor,
                        .type = INTERACT_YX, .coefficient = dmi_coeffs[1]
                    };
                }

                // XZ - ZX term (from dy component)
                if (fabs(dmi_coeffs[2]) > params->coupling_cutoff) {
                    bonds->interactions[bonds->num_interactions++] = (bond_interaction_t){
                        .site_i = s, .site_j = neighbor,
                        .type = INTERACT_XZ, .coefficient = dmi_coeffs[2]
                    };
                }
                if (fabs(dmi_coeffs[3]) > params->coupling_cutoff) {
                    bonds->interactions[bonds->num_interactions++] = (bond_interaction_t){
                        .site_i = s, .site_j = neighbor,
                        .type = INTERACT_ZX, .coefficient = dmi_coeffs[3]
                    };
                }

                // YZ - ZY term (from dx component)
                if (fabs(dmi_coeffs[4]) > params->coupling_cutoff) {
                    bonds->interactions[bonds->num_interactions++] = (bond_interaction_t){
                        .site_i = s, .site_j = neighbor,
                        .type = INTERACT_YZ, .coefficient = dmi_coeffs[4]
                    };
                }
                if (fabs(dmi_coeffs[5]) > params->coupling_cutoff) {
                    bonds->interactions[bonds->num_interactions++] = (bond_interaction_t){
                        .site_i = s, .site_j = neighbor,
                        .type = INTERACT_ZY, .coefficient = dmi_coeffs[5]
                    };
                }
            }
        }
    }

    return bonds;
}

void bond_list_free(bond_list_t *bonds) {
    if (!bonds) return;
    free(bonds->interactions);
    free(bonds);
}

void bond_list_print(const bond_list_t *bonds) {
    if (!bonds) return;

    printf("Bond list: %u interactions, %u sites, max range %u\n",
           bonds->num_interactions, bonds->num_sites, bonds->max_range);

    for (uint32_t i = 0; i < bonds->num_interactions && i < 20; i++) {
        const bond_interaction_t *b = &bonds->interactions[i];
        const char *type_names[] = {"XX", "YY", "ZZ", "XY", "YX", "XZ", "ZX", "YZ", "ZY"};
        printf("  [%u] sites (%u, %u) type %s coeff %.4f\n",
               i, b->site_i, b->site_j, type_names[b->type], b->coefficient);
    }
    if (bonds->num_interactions > 20) {
        printf("  ... (%u more)\n", bonds->num_interactions - 20);
    }
}

// ============================================================================
// DMI DIRECTION CALCULATION
// ============================================================================

void compute_dmi_direction(const lattice_2d_t *lat,
                            uint32_t site_i, uint32_t site_j,
                            dmi_type_t dmi_type,
                            double d_out[3]) {
    // Get real-space coordinates
    double xi, yi, xj, yj;
    lattice_2d_real_coords(lat, site_i, &xi, &yi);
    lattice_2d_real_coords(lat, site_j, &xj, &yj);

    // Bond vector r_ij
    double rx = xj - xi;
    double ry = yj - yi;
    double rz = 0.0;  // 2D lattice

    // Handle periodic boundaries
    if (lat->bc == BC_PERIODIC_XY || lat->bc == BC_PERIODIC_X) {
        if (rx > lat->Lx * lat->a / 2.0) rx -= lat->Lx * lat->a;
        if (rx < -lat->Lx * lat->a / 2.0) rx += lat->Lx * lat->a;
    }
    if (lat->bc == BC_PERIODIC_XY || lat->bc == BC_PERIODIC_Y) {
        if (ry > lat->Ly * lat->a / 2.0) ry -= lat->Ly * lat->a;
        if (ry < -lat->Ly * lat->a / 2.0) ry += lat->Ly * lat->a;
    }

    double r_norm = sqrt(rx*rx + ry*ry + rz*rz);
    if (r_norm < 1e-10) {
        d_out[0] = d_out[1] = d_out[2] = 0.0;
        return;
    }

    // Normalize bond vector
    double r_hat[3] = {rx/r_norm, ry/r_norm, rz/r_norm};

    switch (dmi_type) {
        case DMI_BULK:
            // d_ij parallel to bond
            d_out[0] = r_hat[0];
            d_out[1] = r_hat[1];
            d_out[2] = r_hat[2];
            break;

        case DMI_INTERFACIAL:
            // d_ij = z × r_ij (perpendicular to bond, in-plane)
            // z × r = (z_y * r_z - z_z * r_y, z_z * r_x - z_x * r_z, z_x * r_y - z_y * r_x)
            //       = (0 - 1*r_y, 1*r_x - 0, 0) = (-r_y, r_x, 0)
            d_out[0] = -r_hat[1];
            d_out[1] = r_hat[0];
            d_out[2] = 0.0;
            break;

        case DMI_NEEL:
            // d_ij = r_ij / |r_ij| (same as bulk)
            d_out[0] = r_hat[0];
            d_out[1] = r_hat[1];
            d_out[2] = r_hat[2];
            break;

        case DMI_BLOCH:
            // d_ij perpendicular to both z and r_ij
            // Same as interfacial for 2D
            d_out[0] = -r_hat[1];
            d_out[1] = r_hat[0];
            d_out[2] = 0.0;
            break;

        default:
            d_out[0] = d_out[1] = d_out[2] = 0.0;
    }
}

void dmi_to_coefficients(double D, const double d[3], double coeffs[6]) {
    // DMI: D * d · (S_i × S_j)
    // S_i × S_j = (Y_i Z_j - Z_i Y_j, Z_i X_j - X_i Z_j, X_i Y_j - Y_i X_j)
    //
    // d · (S_i × S_j) = dx*(Y_i Z_j - Z_i Y_j) + dy*(Z_i X_j - X_i Z_j) + dz*(X_i Y_j - Y_i X_j)
    //
    // Expanding:
    //   = dx*Y_i*Z_j - dx*Z_i*Y_j + dy*Z_i*X_j - dy*X_i*Z_j + dz*X_i*Y_j - dz*Y_i*X_j
    //
    // Grouping by operator type:
    //   XY: +dz, YX: -dz
    //   XZ: -dy, ZX: +dy
    //   YZ: +dx, ZY: -dx

    coeffs[0] = D * d[2];   // XY coefficient
    coeffs[1] = -D * d[2];  // YX coefficient
    coeffs[2] = -D * d[1];  // XZ coefficient
    coeffs[3] = D * d[1];   // ZX coefficient
    coeffs[4] = D * d[0];   // YZ coefficient
    coeffs[5] = -D * d[0];  // ZY coefficient
}

// ============================================================================
// FINITE AUTOMATON MPO CONSTRUCTION
// ============================================================================

/**
 * @brief Active interaction tracking for automaton
 */
typedef struct {
    uint32_t site_j;            /**< Target site where interaction closes */
    interaction_type_t type;    /**< Interaction type */
    double coefficient;         /**< Coupling strength */
    uint32_t state_idx;         /**< Automaton state index */
} active_interaction_t;

/**
 * @brief Create MPO tensor for a single site
 *
 * The automaton has states:
 *   0: Identity (pass-through)
 *   1: Final state (completed terms)
 *   2...: Open interaction states
 *
 * For site s, the MPO tensor W[b_l, sigma, sigma', b_r] encodes:
 *   - Pass-through: W[0, :, :, 0] = I
 *   - On-site terms: W[0, :, :, 1] = h_s * Z + ...
 *   - Start interaction: W[0, :, :, open_state] = coeff * left_op
 *   - Pass open through: W[open_state, :, :, open_state] = I
 *   - Close interaction: W[open_state, :, :, 1] = right_op
 *   - Final pass: W[1, :, :, 1] = I
 */
static mpo_tensor_t create_site_mpo(uint32_t site,
                                      uint32_t num_sites,
                                      const bond_list_t *bonds,
                                      const double *on_site_z,
                                      const double *on_site_zz,
                                      active_interaction_t **active_list,
                                      uint32_t *num_active,
                                      uint32_t *bond_dim) {
    mpo_tensor_t W = {0};
    W.phys_dim = 2;

    // Count how many interactions:
    // - Start at this site (site_i == site)
    // - Are currently active (will close at site_j > site)
    // - Close at this site (site_j == site)

    uint32_t num_starting = 0;
    uint32_t num_closing = 0;

    for (uint32_t i = 0; i < bonds->num_interactions; i++) {
        if (bonds->interactions[i].site_i == site) {
            num_starting++;
        }
    }

    for (uint32_t i = 0; i < *num_active; i++) {
        if ((*active_list)[i].site_j == site) {
            num_closing++;
        }
    }

    // Calculate bond dimensions
    // Left bond: 2 (identity, final) + currently active
    // Right bond: 2 (identity, final) + (active - closing + starting)

    uint32_t b_l = (site == 0) ? 1 : (2 + *num_active);
    uint32_t b_r = (site == num_sites - 1) ? 1 : (2 + *num_active - num_closing + num_starting);

    W.bond_dim_left = b_l;
    W.bond_dim_right = b_r;

    // Create tensor
    uint32_t dims[4] = {b_l, 2, 2, b_r};
    W.W = tensor_create(4, dims);
    if (!W.W) {
        return W;
    }
    memset(W.W->data, 0, W.W->total_size * sizeof(double complex));

    // Helper to set MPO element: W[bl, s, sp, br]
    #define SET_W(bl, s, sp, br, val) \
        W.W->data[(bl) * 4 * b_r + (s) * 2 * b_r + (sp) * b_r + (br)] = (val)

    // Build the automaton transitions

    if (site == 0) {
        // Left boundary: single row
        // W[0, :, :, 0] = on-site + identity for first pass

        // On-site terms go to final state (will be state 1 in next tensor)
        double hz = on_site_z ? on_site_z[site] : 0.0;
        double hzz = on_site_zz ? on_site_zz[site] : 0.0;

        // We output to state indices:
        //   0 = identity pass-through
        //   1 = final state
        //   2+ = open interaction states

        // On-site terms accumulated into state that will become final
        // But at boundary, we go directly to identity + start interactions

        // Identity pass-through: 0 -> 0
        for (int s = 0; s < 2; s++) {
            SET_W(0, s, s, 0, 1.0);  // Identity
        }

        // On-site Z term: 0 -> 1 (final)
        if (fabs(hz) > 1e-15 || fabs(hzz) > 1e-15) {
            for (int s = 0; s < 2; s++) {
                double z_val = (s == 0) ? 1.0 : -1.0;  // Z eigenvalue
                SET_W(0, s, s, 1, hz * z_val + hzz);   // Z + Z^2 = Z + I
            }
        } else {
            // Still need final state pass-through
            for (int s = 0; s < 2; s++) {
                SET_W(0, s, s, 1, 0.0);
            }
        }

        // Start new interactions: 0 -> open_state
        uint32_t open_idx = 2;
        for (uint32_t i = 0; i < bonds->num_interactions; i++) {
            if (bonds->interactions[i].site_i == site) {
                const double complex *left_op = get_left_pauli(bonds->interactions[i].type);
                double coeff = bonds->interactions[i].coefficient;

                for (int s = 0; s < 2; s++) {
                    for (int sp = 0; sp < 2; sp++) {
                        SET_W(0, s, sp, open_idx, coeff * left_op[s * 2 + sp]);
                    }
                }

                // Add to active list
                if (*num_active >= bonds->max_range * NUM_INTERACT_TYPES) {
                    // Reallocate
                    uint32_t new_cap = (*num_active + num_starting) * 2;
                    *active_list = (active_interaction_t *)realloc(
                        *active_list, new_cap * sizeof(active_interaction_t));
                }
                (*active_list)[*num_active] = (active_interaction_t){
                    .site_j = bonds->interactions[i].site_j,
                    .type = bonds->interactions[i].type,
                    .coefficient = bonds->interactions[i].coefficient,
                    .state_idx = open_idx
                };
                (*num_active)++;
                open_idx++;
            }
        }
    }
    else if (site == num_sites - 1) {
        // Right boundary: single column

        // Identity: 0 -> 0
        for (int s = 0; s < 2; s++) {
            SET_W(0, s, s, 0, 1.0);
        }

        // Final state: 1 -> 0 (collect everything)
        for (int s = 0; s < 2; s++) {
            SET_W(1, s, s, 0, 1.0);
        }

        // On-site terms
        double hz = on_site_z ? on_site_z[site] : 0.0;
        double hzz = on_site_zz ? on_site_zz[site] : 0.0;
        if (fabs(hz) > 1e-15 || fabs(hzz) > 1e-15) {
            for (int s = 0; s < 2; s++) {
                double z_val = (s == 0) ? 1.0 : -1.0;
                // Add on-site to identity path: 0 -> 0
                W.W->data[0 * 4 * 1 + s * 2 * 1 + s * 1 + 0] += hz * z_val + hzz;
            }
        }

        // Close all active interactions
        for (uint32_t i = 0; i < *num_active; i++) {
            if ((*active_list)[i].site_j == site) {
                uint32_t bl = (*active_list)[i].state_idx;
                const double complex *right_op = get_right_pauli((*active_list)[i].type);

                for (int s = 0; s < 2; s++) {
                    for (int sp = 0; sp < 2; sp++) {
                        SET_W(bl, s, sp, 0, right_op[s * 2 + sp]);
                    }
                }
            }
        }
    }
    else {
        // Bulk site

        // Build state mapping for right bond
        uint32_t *state_map = (uint32_t *)calloc(*num_active + num_starting, sizeof(uint32_t));
        uint32_t new_num_active = 0;
        active_interaction_t *new_active = (active_interaction_t *)calloc(
            *num_active + num_starting, sizeof(active_interaction_t));

        // 0: identity, 1: final, 2+: open states

        // Identity pass-through: 0 -> 0
        for (int s = 0; s < 2; s++) {
            SET_W(0, s, s, 0, 1.0);
        }

        // Final state pass-through: 1 -> 1
        for (int s = 0; s < 2; s++) {
            SET_W(1, s, s, 1, 1.0);
        }

        // On-site terms: 0 -> 1
        double hz = on_site_z ? on_site_z[site] : 0.0;
        double hzz = on_site_zz ? on_site_zz[site] : 0.0;
        for (int s = 0; s < 2; s++) {
            double z_val = (s == 0) ? 1.0 : -1.0;
            SET_W(0, s, s, 1, hz * z_val + hzz);
        }

        // Process active interactions
        uint32_t next_open_state = 2;
        for (uint32_t i = 0; i < *num_active; i++) {
            uint32_t bl = (*active_list)[i].state_idx;

            if ((*active_list)[i].site_j == site) {
                // Close this interaction: bl -> 1 (final)
                const double complex *right_op = get_right_pauli((*active_list)[i].type);

                for (int s = 0; s < 2; s++) {
                    for (int sp = 0; sp < 2; sp++) {
                        SET_W(bl, s, sp, 1, right_op[s * 2 + sp]);
                    }
                }
                // Don't add to new active list
            } else {
                // Pass through: bl -> new_state
                uint32_t br = next_open_state++;

                for (int s = 0; s < 2; s++) {
                    SET_W(bl, s, s, br, 1.0);  // Identity pass-through
                }

                // Update active list
                new_active[new_num_active] = (*active_list)[i];
                new_active[new_num_active].state_idx = br;
                new_num_active++;
            }
        }

        // Start new interactions: 0 -> new_state
        for (uint32_t i = 0; i < bonds->num_interactions; i++) {
            if (bonds->interactions[i].site_i == site) {
                const double complex *left_op = get_left_pauli(bonds->interactions[i].type);
                double coeff = bonds->interactions[i].coefficient;
                uint32_t br = next_open_state++;

                for (int s = 0; s < 2; s++) {
                    for (int sp = 0; sp < 2; sp++) {
                        SET_W(0, s, sp, br, coeff * left_op[s * 2 + sp]);
                    }
                }

                // Add to new active list
                new_active[new_num_active] = (active_interaction_t){
                    .site_j = bonds->interactions[i].site_j,
                    .type = bonds->interactions[i].type,
                    .coefficient = bonds->interactions[i].coefficient,
                    .state_idx = br
                };
                new_num_active++;
            }
        }

        // Update active list
        free(*active_list);
        *active_list = new_active;
        *num_active = new_num_active;
        free(state_map);
    }

    #undef SET_W

    *bond_dim = b_r;
    return W;
}

mpo_t *mpo_from_bond_list(const bond_list_t *bonds,
                           const double *on_site_z,
                           const double *on_site_zz) {
    if (!bonds) return NULL;

    mpo_t *mpo = (mpo_t *)calloc(1, sizeof(mpo_t));
    if (!mpo) return NULL;

    mpo->num_sites = bonds->num_sites;
    mpo->tensors = (mpo_tensor_t *)calloc(bonds->num_sites, sizeof(mpo_tensor_t));
    if (!mpo->tensors) {
        free(mpo);
        return NULL;
    }

    // Active interaction tracking
    uint32_t active_capacity = bonds->max_range * NUM_INTERACT_TYPES + 10;
    active_interaction_t *active_list = (active_interaction_t *)calloc(
        active_capacity, sizeof(active_interaction_t));
    uint32_t num_active = 0;
    uint32_t max_bond_dim = 0;

    // Build MPO site by site
    for (uint32_t site = 0; site < bonds->num_sites; site++) {
        uint32_t bond_dim;
        mpo->tensors[site] = create_site_mpo(site, bonds->num_sites, bonds,
                                              on_site_z, on_site_zz,
                                              &active_list, &num_active, &bond_dim);

        if (!mpo->tensors[site].W) {
            fprintf(stderr, "Failed to create MPO tensor at site %u\n", site);
            free(active_list);
            mpo_free(mpo);
            return NULL;
        }

        if (bond_dim > max_bond_dim) {
            max_bond_dim = bond_dim;
        }
    }

    free(active_list);
    mpo->max_mpo_bond = max_bond_dim;

    return mpo;
}

mpo_t *mpo_2d_create(const lattice_2d_t *lat, const hamiltonian_params_t *params) {
    if (!lat || !params) return NULL;

    // Create bond list
    bond_list_t *bonds = bond_list_create(lat, params);
    if (!bonds) return NULL;

    // Create on-site arrays
    double *on_site_z = (double *)calloc(lat->num_sites, sizeof(double));
    double *on_site_zz = (double *)calloc(lat->num_sites, sizeof(double));

    if (!on_site_z || !on_site_zz) {
        free(on_site_z);
        free(on_site_zz);
        bond_list_free(bonds);
        return NULL;
    }

    // Fill on-site terms
    for (uint32_t s = 0; s < lat->num_sites; s++) {
        on_site_z[s] = -params->Bz;  // Zeeman: -B · S = -Bz * Sz
        on_site_zz[s] = -params->K;  // Anisotropy: -K * Sz^2
    }

    // Build MPO
    mpo_t *mpo = mpo_from_bond_list(bonds, on_site_z, on_site_zz);

    free(on_site_z);
    free(on_site_zz);
    bond_list_free(bonds);

    return mpo;
}

uint32_t mpo_2d_estimate_bond_dim(const lattice_2d_t *lat, bool include_dmi) {
    if (!lat) return 0;

    // Estimate: 2 (identity + final) + Lx * interaction_types
    uint32_t interaction_types = 3;  // XX, YY, ZZ for Heisenberg
    if (include_dmi) {
        interaction_types += 6;  // XY, YX, XZ, ZX, YZ, ZY
    }

    return 2 + lat->Lx * interaction_types;
}

// ============================================================================
// VERIFICATION UTILITIES
// ============================================================================

bool mpo_verify(const mpo_t *mpo) {
    if (!mpo || !mpo->tensors) return false;

    for (uint32_t i = 0; i < mpo->num_sites; i++) {
        if (!mpo->tensors[i].W) {
            fprintf(stderr, "MPO verify: NULL tensor at site %u\n", i);
            return false;
        }

        // Check bond dimension consistency
        if (i > 0) {
            if (mpo->tensors[i].bond_dim_left != mpo->tensors[i-1].bond_dim_right) {
                fprintf(stderr, "MPO verify: Bond dim mismatch at site %u: left=%u, prev_right=%u\n",
                        i, mpo->tensors[i].bond_dim_left, mpo->tensors[i-1].bond_dim_right);
                return false;
            }
        }

        // Check for NaN/Inf
        for (uint64_t j = 0; j < mpo->tensors[i].W->total_size; j++) {
            if (isnan(creal(mpo->tensors[i].W->data[j])) ||
                isnan(cimag(mpo->tensors[i].W->data[j])) ||
                isinf(creal(mpo->tensors[i].W->data[j])) ||
                isinf(cimag(mpo->tensors[i].W->data[j]))) {
                fprintf(stderr, "MPO verify: NaN/Inf at site %u index %llu\n", i, (unsigned long long)j);
                return false;
            }
        }
    }

    // Check boundary conditions
    if (mpo->tensors[0].bond_dim_left != 1) {
        fprintf(stderr, "MPO verify: Left boundary bond dim should be 1, got %u\n",
                mpo->tensors[0].bond_dim_left);
        return false;
    }

    if (mpo->tensors[mpo->num_sites - 1].bond_dim_right != 1) {
        fprintf(stderr, "MPO verify: Right boundary bond dim should be 1, got %u\n",
                mpo->tensors[mpo->num_sites - 1].bond_dim_right);
        return false;
    }

    return true;
}

void mpo_print_info(const mpo_t *mpo) {
    if (!mpo) {
        printf("MPO: NULL\n");
        return;
    }

    printf("MPO: %u sites, max bond dim %u\n", mpo->num_sites, mpo->max_mpo_bond);
    printf("  Bond dimensions: ");
    for (uint32_t i = 0; i < mpo->num_sites; i++) {
        printf("[%u,%u] ", mpo->tensors[i].bond_dim_left, mpo->tensors[i].bond_dim_right);
    }
    printf("\n");

    // Memory estimate
    size_t total_mem = 0;
    for (uint32_t i = 0; i < mpo->num_sites; i++) {
        if (mpo->tensors[i].W) {
            total_mem += mpo->tensors[i].W->total_size * sizeof(double complex);
        }
    }
    printf("  Total memory: %.2f KB\n", total_mem / 1024.0);
}

tensor_t *mpo_to_matrix(const mpo_t *mpo) {
    if (!mpo || mpo->num_sites > 12) {
        fprintf(stderr, "mpo_to_matrix: System too large (max 12 sites)\n");
        return NULL;
    }

    uint32_t dim = 1 << mpo->num_sites;  // 2^n
    tensor_t *H = tensor_create_matrix(dim, dim);
    if (!H) return NULL;

    memset(H->data, 0, H->total_size * sizeof(double complex));

    // Brute force: compute H[i,j] = <i|H|j> for all basis states
    for (uint32_t row = 0; row < dim; row++) {
        for (uint32_t col = 0; col < dim; col++) {
            // Contract MPO between basis states |row> and |col>
            // Start with left boundary
            double complex *vec = (double complex *)calloc(mpo->max_mpo_bond, sizeof(double complex));
            double complex *vec_new = (double complex *)calloc(mpo->max_mpo_bond, sizeof(double complex));
            vec[0] = 1.0;  // Left boundary

            for (uint32_t site = 0; site < mpo->num_sites; site++) {
                mpo_tensor_t *W = &mpo->tensors[site];
                uint32_t s = (row >> site) & 1;   // Bra state at site
                uint32_t sp = (col >> site) & 1;  // Ket state at site

                memset(vec_new, 0, mpo->max_mpo_bond * sizeof(double complex));

                for (uint32_t bl = 0; bl < W->bond_dim_left; bl++) {
                    for (uint32_t br = 0; br < W->bond_dim_right; br++) {
                        // W[bl, s, sp, br]
                        uint64_t idx = bl * 4 * W->bond_dim_right + s * 2 * W->bond_dim_right +
                                       sp * W->bond_dim_right + br;
                        vec_new[br] += vec[bl] * W->W->data[idx];
                    }
                }

                memcpy(vec, vec_new, mpo->max_mpo_bond * sizeof(double complex));
            }

            H->data[row * dim + col] = vec[0];  // Right boundary contracts to single element
            free(vec);
            free(vec_new);
        }
    }

    return H;
}
