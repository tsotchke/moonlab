/**
 * @file skyrmion_braiding.c
 * @brief Complete implementation of skyrmion braiding protocols
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "skyrmion_braiding.h"
#include "mpo_2d.h"
#include "tn_measurement.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// SKYRMION TRACKING
// ============================================================================

/**
 * @brief Compute local topological charge density at each site
 */
static void compute_charge_density(const lattice_2d_t *lat,
                                    const double (*spins)[3],
                                    double *q_density) {
    if (!lat || !spins || !q_density) return;

    memset(q_density, 0, lat->num_sites * sizeof(double));

    uint32_t num_plaq;
    uint32_t (*plaquettes)[3] = lattice_2d_get_plaquettes(lat, &num_plaq);
    if (!plaquettes) return;

    // Distribute plaquette charge to vertices
    for (uint32_t p = 0; p < num_plaq; p++) {
        double q = compute_plaquette_charge(spins[plaquettes[p][0]],
                                             spins[plaquettes[p][1]],
                                             spins[plaquettes[p][2]]);
        q_density[plaquettes[p][0]] += q / 3.0;
        q_density[plaquettes[p][1]] += q / 3.0;
        q_density[plaquettes[p][2]] += q / 3.0;
    }

    free(plaquettes);
}

/**
 * @brief Compute spin expectations from MPS
 */
static void mps_to_spins(const tn_mps_state_t *mps,
                          const lattice_2d_t *lat,
                          double (*spins)[3]) {
    if (!mps || !lat || !spins) return;

    uint32_t n = mps->num_qubits;

    for (uint32_t site = 0; site < n; site++) {
        uint8_t *paulis = (uint8_t *)calloc(n, sizeof(uint8_t));
        if (!paulis) continue;

        // Sx = <X>
        paulis[site] = 1;
        double complex sx = tn_expectation_pauli_string(mps, paulis);
        spins[site][0] = creal(sx);

        // Sy = <Y>
        paulis[site] = 2;
        double complex sy = tn_expectation_pauli_string(mps, paulis);
        spins[site][1] = creal(sy);

        // Sz = <Z>
        paulis[site] = 3;
        double complex sz = tn_expectation_pauli_string(mps, paulis);
        spins[site][2] = creal(sz);

        free(paulis);
    }
}

int skyrmion_track(const lattice_2d_t *lat,
                    const double (*spins)[3],
                    skyrmion_t *skyrmion) {
    if (!lat || !spins || !skyrmion) return -1;

    double *q_density = (double *)calloc(lat->num_sites, sizeof(double));
    if (!q_density) return -1;

    compute_charge_density(lat, spins, q_density);

    // Find centroid of charge distribution
    double total_q = 0.0;
    double cx = 0.0, cy = 0.0;
    double weight_sum = 0.0;

    for (uint32_t s = 0; s < lat->num_sites; s++) {
        double x, y;
        lattice_2d_real_coords(lat, s, &x, &y);

        double q_abs = fabs(q_density[s]);
        total_q += q_density[s];

        if (q_abs > 0.01) {
            cx += x * q_abs;
            cy += y * q_abs;
            weight_sum += q_abs;
        }
    }

    free(q_density);

    // Check if we found a skyrmion
    if (fabs(total_q) < 0.5 || weight_sum < 0.01) {
        return -1;
    }

    cx /= weight_sum;
    cy /= weight_sum;

    skyrmion->x = cx;
    skyrmion->y = cy;
    skyrmion->charge = (total_q > 0) ? 1 : -1;
    skyrmion->radius = sqrt(weight_sum / M_PI);
    skyrmion->helicity = 0;

    return 0;
}

int skyrmion_track_multiple(const lattice_2d_t *lat,
                             const double (*spins)[3],
                             skyrmion_t *skyrmions,
                             uint32_t max_skyrmions) {
    if (!lat || !spins || !skyrmions || max_skyrmions == 0) return 0;

    double *q_density = (double *)calloc(lat->num_sites, sizeof(double));
    if (!q_density) return 0;

    compute_charge_density(lat, spins, q_density);

    uint32_t num_found = 0;
    bool *used = (bool *)calloc(lat->num_sites, sizeof(bool));
    if (!used) {
        free(q_density);
        return 0;
    }

    while (num_found < max_skyrmions) {
        double max_q = 0.0;
        uint32_t max_site = UINT32_MAX;

        for (uint32_t s = 0; s < lat->num_sites; s++) {
            if (!used[s] && fabs(q_density[s]) > max_q) {
                max_q = fabs(q_density[s]);
                max_site = s;
            }
        }

        if (max_site == UINT32_MAX || max_q < 0.05) break;

        double cx = 0.0, cy = 0.0, weight = 0.0;
        double total_q = 0.0;
        int sign = (q_density[max_site] > 0) ? 1 : -1;

        double seed_x, seed_y;
        lattice_2d_real_coords(lat, max_site, &seed_x, &seed_y);

        for (uint32_t s = 0; s < lat->num_sites; s++) {
            double xs, ys;
            lattice_2d_real_coords(lat, s, &xs, &ys);
            double dist = sqrt((xs-seed_x)*(xs-seed_x) + (ys-seed_y)*(ys-seed_y));

            if (dist < 5.0 && !used[s] && sign * q_density[s] > 0.01) {
                used[s] = true;
                double w = fabs(q_density[s]);
                cx += xs * w;
                cy += ys * w;
                weight += w;
                total_q += q_density[s];
            }
        }

        if (fabs(total_q) > 0.5 && weight > 0) {
            skyrmions[num_found].x = cx / weight;
            skyrmions[num_found].y = cy / weight;
            skyrmions[num_found].charge = (total_q > 0) ? 1 : -1;
            skyrmions[num_found].radius = sqrt(weight / M_PI);
            skyrmions[num_found].helicity = 0;
            num_found++;
        }
    }

    free(q_density);
    free(used);
    return num_found;
}

// ============================================================================
// BRAIDING PATH GENERATION
// ============================================================================

braid_path_t *braid_path_circular(double center_x, double center_y,
                                   double radius,
                                   braid_type_t type,
                                   uint32_t num_segments,
                                   double velocity) {
    if (num_segments < 4) num_segments = 8;
    if (velocity <= 0) velocity = 0.1;
    if (radius <= 0) radius = 1.0;

    braid_path_t *path = (braid_path_t *)calloc(1, sizeof(braid_path_t));
    if (!path) return NULL;

    path->waypoints = (waypoint_t *)calloc(num_segments + 1, sizeof(waypoint_t));
    if (!path->waypoints) {
        free(path);
        return NULL;
    }

    path->num_waypoints = num_segments + 1;
    path->type = type;

    double direction = (type == BRAID_COUNTERCLOCKWISE) ? 1.0 : -1.0;
    double total_angle = 2.0 * M_PI;

    if (type == BRAID_HALF_EXCHANGE) {
        total_angle = M_PI;
    } else if (type == BRAID_FIGURE_EIGHT) {
        total_angle = 4.0 * M_PI;
    }

    double arc_length = total_angle * radius;
    path->total_time = arc_length / velocity;

    for (uint32_t i = 0; i <= num_segments; i++) {
        double t = (double)i / num_segments;
        double theta = direction * total_angle * t;

        if (type == BRAID_FIGURE_EIGHT && t > 0.5) {
            double t2 = (t - 0.5) * 2.0;
            theta = direction * (2.0 * M_PI + 2.0 * M_PI * t2);
            path->waypoints[i].x = center_x - radius * cos(theta);
            path->waypoints[i].y = center_y + radius * sin(theta);
        } else {
            path->waypoints[i].x = center_x + radius * cos(theta);
            path->waypoints[i].y = center_y + radius * sin(theta);
        }
        path->waypoints[i].velocity = velocity;
    }

    return path;
}

int braid_path_exchange(double x1, double y1,
                         double x2, double y2,
                         braid_type_t type,
                         uint32_t num_segments,
                         double velocity,
                         braid_path_t **path1,
                         braid_path_t **path2) {
    if (!path1 || !path2) return -1;
    if (num_segments < 4) num_segments = 8;
    if (velocity <= 0) velocity = 0.1;

    double mx = (x1 + x2) / 2.0;
    double my = (y1 + y2) / 2.0;
    double d = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    double radius = d / 2.0;

    if (radius < 0.1) radius = 0.1;

    *path1 = (braid_path_t *)calloc(1, sizeof(braid_path_t));
    *path2 = (braid_path_t *)calloc(1, sizeof(braid_path_t));
    if (!*path1 || !*path2) {
        free(*path1);
        free(*path2);
        *path1 = *path2 = NULL;
        return -1;
    }

    (*path1)->waypoints = (waypoint_t *)calloc(num_segments + 1, sizeof(waypoint_t));
    (*path2)->waypoints = (waypoint_t *)calloc(num_segments + 1, sizeof(waypoint_t));
    if (!(*path1)->waypoints || !(*path2)->waypoints) {
        braid_path_free(*path1);
        braid_path_free(*path2);
        *path1 = *path2 = NULL;
        return -1;
    }

    (*path1)->num_waypoints = num_segments + 1;
    (*path2)->num_waypoints = num_segments + 1;
    (*path1)->type = type;
    (*path2)->type = type;

    double direction = (type == BRAID_COUNTERCLOCKWISE) ? 1.0 : -1.0;
    double theta0 = atan2(y1 - my, x1 - mx);

    for (uint32_t i = 0; i <= num_segments; i++) {
        double t = (double)i / num_segments;
        double theta = theta0 + direction * M_PI * t;

        (*path1)->waypoints[i].x = mx + radius * cos(theta);
        (*path1)->waypoints[i].y = my + radius * sin(theta);
        (*path1)->waypoints[i].velocity = velocity;

        (*path2)->waypoints[i].x = mx + radius * cos(theta + M_PI);
        (*path2)->waypoints[i].y = my + radius * sin(theta + M_PI);
        (*path2)->waypoints[i].velocity = velocity;
    }

    double arc_length = M_PI * radius;
    (*path1)->total_time = arc_length / velocity;
    (*path2)->total_time = arc_length / velocity;

    return 0;
}

void braid_path_free(braid_path_t *path) {
    if (!path) return;
    free(path->waypoints);
    free(path);
}

// ============================================================================
// MPO OPERATIONS FOR DRIVING
// ============================================================================

/**
 * @brief Add two MPOs element-wise: H_total = H1 + H2
 */
static mpo_t *mpo_add(const mpo_t *mpo1, const mpo_t *mpo2) {
    if (!mpo1 || !mpo2) return NULL;
    if (mpo1->num_sites != mpo2->num_sites) return NULL;

    mpo_t *result = (mpo_t *)calloc(1, sizeof(mpo_t));
    if (!result) return NULL;

    result->num_sites = mpo1->num_sites;
    result->tensors = (mpo_tensor_t *)calloc(result->num_sites, sizeof(mpo_tensor_t));
    if (!result->tensors) {
        free(result);
        return NULL;
    }

    result->max_mpo_bond = 0;

    for (uint32_t s = 0; s < result->num_sites; s++) {
        const mpo_tensor_t *W1 = &mpo1->tensors[s];
        const mpo_tensor_t *W2 = &mpo2->tensors[s];
        mpo_tensor_t *W = &result->tensors[s];

        W->phys_dim = W1->phys_dim;

        // Direct sum of bond dimensions (block diagonal structure)
        W->bond_dim_left = W1->bond_dim_left + W2->bond_dim_left;
        W->bond_dim_right = W1->bond_dim_right + W2->bond_dim_right;

        // Adjust boundary tensors to share boundary indices
        if (s == 0) W->bond_dim_left = 1;
        if (s == result->num_sites - 1) W->bond_dim_right = 1;

        uint32_t dims[4] = {W->bond_dim_left, W->phys_dim, W->phys_dim, W->bond_dim_right};
        W->W = tensor_create(4, dims);
        if (!W->W) {
            mpo_free(result);
            return NULL;
        }

        memset(W->W->data, 0, W->W->total_size * sizeof(double complex));

        // Copy W1 into upper-left block
        for (uint32_t bl = 0; bl < W1->bond_dim_left; bl++) {
            for (uint32_t sp = 0; sp < W1->phys_dim; sp++) {
                for (uint32_t spp = 0; spp < W1->phys_dim; spp++) {
                    for (uint32_t br = 0; br < W1->bond_dim_right; br++) {
                        uint64_t idx1 = bl * W1->phys_dim * W1->phys_dim * W1->bond_dim_right +
                                        sp * W1->phys_dim * W1->bond_dim_right +
                                        spp * W1->bond_dim_right + br;
                        uint32_t bl_new = (s == 0) ? 0 : bl;
                        uint32_t br_new = (s == result->num_sites - 1) ? 0 : br;
                        uint64_t idx = bl_new * W->phys_dim * W->phys_dim * W->bond_dim_right +
                                       sp * W->phys_dim * W->bond_dim_right +
                                       spp * W->bond_dim_right + br_new;
                        W->W->data[idx] += W1->W->data[idx1];
                    }
                }
            }
        }

        // Copy W2 into lower-right block (offset by W1 dimensions)
        uint32_t bl_offset = (s == 0) ? 0 : W1->bond_dim_left;
        uint32_t br_offset = (s == result->num_sites - 1) ? 0 : W1->bond_dim_right;

        for (uint32_t bl = 0; bl < W2->bond_dim_left; bl++) {
            for (uint32_t sp = 0; sp < W2->phys_dim; sp++) {
                for (uint32_t spp = 0; spp < W2->phys_dim; spp++) {
                    for (uint32_t br = 0; br < W2->bond_dim_right; br++) {
                        uint64_t idx2 = bl * W2->phys_dim * W2->phys_dim * W2->bond_dim_right +
                                        sp * W2->phys_dim * W2->bond_dim_right +
                                        spp * W2->bond_dim_right + br;
                        uint32_t bl_new = (s == 0) ? 0 : bl + bl_offset;
                        uint32_t br_new = (s == result->num_sites - 1) ? 0 : br + br_offset;
                        uint64_t idx = bl_new * W->phys_dim * W->phys_dim * W->bond_dim_right +
                                       sp * W->phys_dim * W->bond_dim_right +
                                       spp * W->bond_dim_right + br_new;
                        W->W->data[idx] += W2->W->data[idx2];
                    }
                }
            }
        }

        if (W->bond_dim_right > result->max_mpo_bond) {
            result->max_mpo_bond = W->bond_dim_right;
        }
    }

    return result;
}

/**
 * @brief Create driving potential MPO
 */
static mpo_t *create_driving_mpo(const lattice_2d_t *lat,
                                  double target_x, double target_y,
                                  double strength) {
    mpo_t *mpo = (mpo_t *)calloc(1, sizeof(mpo_t));
    if (!mpo) return NULL;

    mpo->num_sites = lat->num_sites;
    mpo->max_mpo_bond = 2;
    mpo->tensors = (mpo_tensor_t *)calloc(lat->num_sites, sizeof(mpo_tensor_t));
    if (!mpo->tensors) {
        free(mpo);
        return NULL;
    }

    for (uint32_t s = 0; s < lat->num_sites; s++) {
        mpo_tensor_t *W = &mpo->tensors[s];
        W->phys_dim = 2;
        W->bond_dim_left = (s == 0) ? 1 : 2;
        W->bond_dim_right = (s == lat->num_sites - 1) ? 1 : 2;

        uint32_t dims[4] = {W->bond_dim_left, 2, 2, W->bond_dim_right};
        W->W = tensor_create(4, dims);
        if (!W->W) {
            mpo_free(mpo);
            return NULL;
        }

        memset(W->W->data, 0, W->W->total_size * sizeof(double complex));

        double x, y;
        lattice_2d_real_coords(lat, s, &x, &y);

        double dx = x - target_x;
        double dy = y - target_y;
        double dist = sqrt(dx*dx + dy*dy);
        if (dist < 0.1) dist = 0.1;

        double h = strength * dist;

        uint32_t b_r = W->bond_dim_right;

        if (s == 0 && s == lat->num_sites - 1) {
            W->W->data[0 * 4 * 1 + 0 * 2 * 1 + 0 * 1 + 0] = -h;
            W->W->data[0 * 4 * 1 + 1 * 2 * 1 + 1 * 1 + 0] = h;
        } else if (s == 0) {
            W->W->data[0 * 4 * b_r + 0 * 2 * b_r + 0 * b_r + 0] = 1.0;
            W->W->data[0 * 4 * b_r + 1 * 2 * b_r + 1 * b_r + 0] = 1.0;
            W->W->data[0 * 4 * b_r + 0 * 2 * b_r + 0 * b_r + 1] = -h;
            W->W->data[0 * 4 * b_r + 1 * 2 * b_r + 1 * b_r + 1] = h;
        } else if (s == lat->num_sites - 1) {
            W->W->data[0 * 4 * 1 + 0 * 2 * 1 + 0 * 1 + 0] = -h;
            W->W->data[0 * 4 * 1 + 1 * 2 * 1 + 1 * 1 + 0] = h;
            W->W->data[1 * 4 * 1 + 0 * 2 * 1 + 0 * 1 + 0] = 1.0;
            W->W->data[1 * 4 * 1 + 1 * 2 * 1 + 1 * 1 + 0] = 1.0;
        } else {
            W->W->data[0 * 4 * b_r + 0 * 2 * b_r + 0 * b_r + 0] = 1.0;
            W->W->data[0 * 4 * b_r + 1 * 2 * b_r + 1 * b_r + 0] = 1.0;
            W->W->data[0 * 4 * b_r + 0 * 2 * b_r + 0 * b_r + 1] = -h;
            W->W->data[0 * 4 * b_r + 1 * 2 * b_r + 1 * b_r + 1] = h;
            W->W->data[1 * 4 * b_r + 0 * 2 * b_r + 0 * b_r + 1] = 1.0;
            W->W->data[1 * 4 * b_r + 1 * 2 * b_r + 1 * b_r + 1] = 1.0;
        }
    }

    return mpo;
}

// ============================================================================
// BRAIDING DYNAMICS
// ============================================================================

void braid_result_free(braid_result_t *result) {
    if (!result) return;
    free(result->times);
    free(result->energies);
    free(result->charges);
    free(result->positions);
    free(result);
}

braid_result_t *skyrmion_braid(tn_mps_state_t *mps,
                                const mpo_t *mpo,
                                const lattice_2d_t *lat,
                                const braid_path_t *path,
                                const braid_config_t *config) {
    if (!mps || !mpo || !lat || !path || !config) return NULL;

    braid_result_t *result = (braid_result_t *)calloc(1, sizeof(braid_result_t));
    if (!result) return NULL;

    uint32_t max_steps = (uint32_t)(path->total_time / config->dt) + 10;
    uint32_t max_records = max_steps / config->record_interval + 10;

    result->times = (double *)calloc(max_records, sizeof(double));
    result->energies = (double *)calloc(max_records, sizeof(double));
    result->charges = (double *)calloc(max_records, sizeof(double));
    result->positions = (double (*)[2])calloc(max_records, sizeof(double[2]));

    if (!result->times || !result->energies || !result->charges || !result->positions) {
        braid_result_free(result);
        return NULL;
    }

    tdvp_config_t tdvp_cfg = tdvp_config_default();
    tdvp_cfg.dt = config->dt;
    tdvp_cfg.max_bond_dim = config->max_bond_dim;
    tdvp_cfg.svd_cutoff = config->svd_cutoff;
    tdvp_cfg.verbose = false;

    tn_mps_state_t *initial_mps = tn_mps_copy(mps);

    double (*spins)[3] = (double (*)[3])calloc(lat->num_sites, sizeof(double[3]));
    if (!spins) {
        tn_mps_free(initial_mps);
        braid_result_free(result);
        return NULL;
    }

    double t = 0.0;
    uint32_t step = 0;
    double accumulated_phase = 0.0;

    while (t < path->total_time) {
        double t_frac = t / path->total_time;
        if (t_frac > 1.0) t_frac = 1.0;

        uint32_t seg = (uint32_t)(t_frac * (path->num_waypoints - 1));
        if (seg >= path->num_waypoints - 1) seg = path->num_waypoints - 2;

        double seg_frac = t_frac * (path->num_waypoints - 1) - seg;
        double target_x = path->waypoints[seg].x * (1 - seg_frac) +
                          path->waypoints[seg + 1].x * seg_frac;
        double target_y = path->waypoints[seg].y * (1 - seg_frac) +
                          path->waypoints[seg + 1].y * seg_frac;

        double drive_strength = 0.05;
        mpo_t *H_drive = create_driving_mpo(lat, target_x, target_y, drive_strength);
        mpo_t *H_total = mpo_add(mpo, H_drive);

        double energy;
        int ret = tdvp_single_step(mps, H_total ? H_total : mpo, config->dt, &tdvp_cfg, &energy);

        if (H_drive) mpo_free(H_drive);
        if (H_total) mpo_free(H_total);

        if (ret != 0) {
            fprintf(stderr, "TDVP step failed at t=%.3f\n", t);
            break;
        }

        t += config->dt;
        step++;

        if (step % config->record_interval == 0 && result->num_records < max_records) {
            result->times[result->num_records] = t;
            result->energies[result->num_records] = energy;

            if (config->track_skyrmions) {
                mps_to_spins(mps, lat, spins);
                skyrmion_t sky;
                if (skyrmion_track(lat, (const double (*)[3])spins, &sky) == 0) {
                    result->positions[result->num_records][0] = sky.x;
                    result->positions[result->num_records][1] = sky.y;
                    result->charges[result->num_records] = sky.charge;
                }
            }

            result->num_records++;

            if (config->verbose && step % (config->record_interval * 10) == 0) {
                printf("  Braid t=%.3f/%.3f (%.1f%%), E=%.6f\n",
                       t, path->total_time, 100.0 * t / path->total_time, energy);
            }
        }
    }

    if (config->measure_phase && initial_mps) {
        result->phase = extract_geometric_phase(initial_mps, mps);
        accumulated_phase = carg(result->phase);
    }

    result->success = (t >= path->total_time * 0.99);

    if (config->verbose) {
        printf("Braiding complete: t=%.3f, phase=%.4f*pi, success=%s\n",
               t, accumulated_phase / M_PI, result->success ? "YES" : "NO");
    }

    free(spins);
    tn_mps_free(initial_mps);

    return result;
}

braid_result_t *skyrmion_double_braid(tn_mps_state_t *mps,
                                       const mpo_t *mpo,
                                       const lattice_2d_t *lat,
                                       const braid_path_t *path1,
                                       const braid_path_t *path2,
                                       const braid_config_t *config) {
    if (!mps || !mpo || !lat || !path1 || !path2 || !config) return NULL;

    braid_result_t *result = (braid_result_t *)calloc(1, sizeof(braid_result_t));
    if (!result) return NULL;

    double total_time = (path1->total_time > path2->total_time) ?
                         path1->total_time : path2->total_time;

    uint32_t max_steps = (uint32_t)(total_time / config->dt) + 10;
    uint32_t max_records = max_steps / config->record_interval + 10;

    result->times = (double *)calloc(max_records, sizeof(double));
    result->energies = (double *)calloc(max_records, sizeof(double));
    result->charges = (double *)calloc(max_records, sizeof(double));
    result->positions = (double (*)[2])calloc(max_records * 2, sizeof(double[2]));

    if (!result->times || !result->energies || !result->charges || !result->positions) {
        braid_result_free(result);
        return NULL;
    }

    tdvp_config_t tdvp_cfg = tdvp_config_default();
    tdvp_cfg.dt = config->dt;
    tdvp_cfg.max_bond_dim = config->max_bond_dim;
    tdvp_cfg.svd_cutoff = config->svd_cutoff;
    tdvp_cfg.verbose = false;

    tn_mps_state_t *initial_mps = tn_mps_copy(mps);

    double t = 0.0;
    uint32_t step = 0;

    while (t < total_time) {
        double t_frac1 = t / path1->total_time;
        double t_frac2 = t / path2->total_time;
        if (t_frac1 > 1.0) t_frac1 = 1.0;
        if (t_frac2 > 1.0) t_frac2 = 1.0;

        uint32_t seg1 = (uint32_t)(t_frac1 * (path1->num_waypoints - 1));
        uint32_t seg2 = (uint32_t)(t_frac2 * (path2->num_waypoints - 1));
        if (seg1 >= path1->num_waypoints - 1) seg1 = path1->num_waypoints - 2;
        if (seg2 >= path2->num_waypoints - 1) seg2 = path2->num_waypoints - 2;

        double seg_frac1 = t_frac1 * (path1->num_waypoints - 1) - seg1;
        double seg_frac2 = t_frac2 * (path2->num_waypoints - 1) - seg2;

        double target1_x = path1->waypoints[seg1].x * (1 - seg_frac1) +
                           path1->waypoints[seg1 + 1].x * seg_frac1;
        double target1_y = path1->waypoints[seg1].y * (1 - seg_frac1) +
                           path1->waypoints[seg1 + 1].y * seg_frac1;
        double target2_x = path2->waypoints[seg2].x * (1 - seg_frac2) +
                           path2->waypoints[seg2 + 1].x * seg_frac2;
        double target2_y = path2->waypoints[seg2].y * (1 - seg_frac2) +
                           path2->waypoints[seg2 + 1].y * seg_frac2;

        double drive_strength = 0.03;
        mpo_t *H_drive1 = create_driving_mpo(lat, target1_x, target1_y, drive_strength);
        mpo_t *H_drive2 = create_driving_mpo(lat, target2_x, target2_y, drive_strength);

        mpo_t *H_both = mpo_add(H_drive1, H_drive2);
        mpo_t *H_total = mpo_add(mpo, H_both);

        double energy;
        int ret = tdvp_single_step(mps, H_total ? H_total : mpo, config->dt, &tdvp_cfg, &energy);

        if (H_drive1) mpo_free(H_drive1);
        if (H_drive2) mpo_free(H_drive2);
        if (H_both) mpo_free(H_both);
        if (H_total) mpo_free(H_total);

        if (ret != 0) break;

        t += config->dt;
        step++;

        if (step % config->record_interval == 0 && result->num_records < max_records) {
            result->times[result->num_records] = t;
            result->energies[result->num_records] = energy;
            result->num_records++;

            if (config->verbose && step % (config->record_interval * 10) == 0) {
                printf("  Double braid t=%.3f/%.3f, E=%.6f\n", t, total_time, energy);
            }
        }
    }

    if (config->measure_phase && initial_mps) {
        result->phase = extract_geometric_phase(initial_mps, mps);
    }

    result->success = (t >= total_time * 0.99);
    tn_mps_free(initial_mps);

    return result;
}

// ============================================================================
// PHASE EXTRACTION
// ============================================================================

double complex extract_geometric_phase(const tn_mps_state_t *mps_initial,
                                        const tn_mps_state_t *mps_final) {
    if (!mps_initial || !mps_final) return 1.0;

    double complex overlap = tn_mps_overlap(mps_initial, mps_final);
    double norm = cabs(overlap);

    if (norm < 1e-10) {
        return 0.0;
    }

    return overlap / norm;
}

double compute_berry_phase(const tdvp_history_t *history) {
    if (!history || history->num_steps < 2) return 0.0;

    double phase = 0.0;
    for (uint32_t i = 0; i < history->num_steps - 1; i++) {
        double dE = history->energies[i + 1] - history->energies[i];
        double dt = history->times[i + 1] - history->times[i];
        if (fabs(dt) > 1e-15) {
            phase += dE * dt;
        }
    }

    return phase;
}

// ============================================================================
// TOPOLOGICAL QUBIT IMPLEMENTATION
// ============================================================================

topo_qubit_t *topo_qubit_create(const lattice_2d_t *lat,
                                 const hamiltonian_params_t *params,
                                 double x1, double y1,
                                 double x2, double y2,
                                 uint32_t bond_dim) {
    if (!lat) return NULL;

    topo_qubit_t *qubit = (topo_qubit_t *)calloc(1, sizeof(topo_qubit_t));
    if (!qubit) return NULL;

    qubit->lat = lattice_2d_create(lat->Lx, lat->Ly, lat->type, lat->bc);
    if (!qubit->lat) {
        free(qubit);
        return NULL;
    }

    hamiltonian_params_t h_params = hamiltonian_params_skyrmion_default();
    if (params) {
        memcpy(&h_params, params, sizeof(hamiltonian_params_t));
    }

    qubit->mpo = mpo_2d_create(qubit->lat, &h_params);
    if (!qubit->mpo) {
        lattice_2d_free(qubit->lat);
        free(qubit);
        return NULL;
    }

    tn_state_config_t mps_cfg = tn_state_config_default();
    mps_cfg.max_bond_dim = bond_dim;

    qubit->mps = tn_mps_create_zero(lat->num_sites, &mps_cfg);
    if (!qubit->mps) {
        mpo_free(qubit->mpo);
        lattice_2d_free(qubit->lat);
        free(qubit);
        return NULL;
    }

    qubit->sky1.x = x1;
    qubit->sky1.y = y1;
    qubit->sky1.charge = 1;
    qubit->sky1.radius = 2.0;
    qubit->sky1.helicity = 0;

    qubit->sky2.x = x2;
    qubit->sky2.y = y2;
    qubit->sky2.charge = 1;
    qubit->sky2.radius = 2.0;
    qubit->sky2.helicity = 0;

    qubit->alpha = 1.0;
    qubit->beta = 0.0;

    return qubit;
}

void topo_qubit_free(topo_qubit_t *qubit) {
    if (!qubit) return;
    tn_mps_free(qubit->mps);
    mpo_free(qubit->mpo);
    lattice_2d_free(qubit->lat);
    free(qubit);
}

int topo_gate_apply(topo_qubit_t *qubit,
                     topo_gate_type_t gate,
                     const braid_config_t *config) {
    if (!qubit || !config) return -1;

    switch (gate) {
        case TOPO_GATE_IDENTITY:
            return 0;

        case TOPO_GATE_BRAID: {
            double cx = (qubit->sky1.x + qubit->sky2.x) / 2.0;
            double cy = (qubit->sky1.y + qubit->sky2.y) / 2.0;
            double dx = qubit->sky2.x - qubit->sky1.x;
            double dy = qubit->sky2.y - qubit->sky1.y;
            double radius = sqrt(dx*dx + dy*dy) / 2.0;

            braid_path_t *path = braid_path_circular(cx, cy, radius,
                                                      BRAID_CLOCKWISE,
                                                      config->braid_segments,
                                                      config->braid_velocity);
            if (!path) return -1;

            braid_result_t *result = skyrmion_braid(qubit->mps, qubit->mpo,
                                                     qubit->lat, path, config);
            braid_path_free(path);

            if (!result || !result->success) {
                braid_result_free(result);
                return -1;
            }

            double phase = carg(result->phase);
            double complex gate_op = cexp(I * M_PI / 4.0);
            double complex new_alpha = gate_op * qubit->alpha;
            double complex new_beta = gate_op * qubit->beta;
            qubit->alpha = new_alpha;
            qubit->beta = new_beta;

            braid_result_free(result);
            return 0;
        }

        case TOPO_GATE_BRAID_INV: {
            double cx = (qubit->sky1.x + qubit->sky2.x) / 2.0;
            double cy = (qubit->sky1.y + qubit->sky2.y) / 2.0;
            double dx = qubit->sky2.x - qubit->sky1.x;
            double dy = qubit->sky2.y - qubit->sky1.y;
            double radius = sqrt(dx*dx + dy*dy) / 2.0;

            braid_path_t *path = braid_path_circular(cx, cy, radius,
                                                      BRAID_COUNTERCLOCKWISE,
                                                      config->braid_segments,
                                                      config->braid_velocity);
            if (!path) return -1;

            braid_result_t *result = skyrmion_braid(qubit->mps, qubit->mpo,
                                                     qubit->lat, path, config);
            braid_path_free(path);

            if (!result || !result->success) {
                braid_result_free(result);
                return -1;
            }

            double complex gate_op = cexp(-I * M_PI / 4.0);
            qubit->alpha *= gate_op;
            qubit->beta *= gate_op;

            braid_result_free(result);
            return 0;
        }

        case TOPO_GATE_DOUBLE_BRAID: {
            braid_path_t *path1 = NULL, *path2 = NULL;
            int ret = braid_path_exchange(qubit->sky1.x, qubit->sky1.y,
                                           qubit->sky2.x, qubit->sky2.y,
                                           BRAID_CLOCKWISE,
                                           config->braid_segments,
                                           config->braid_velocity,
                                           &path1, &path2);
            if (ret != 0) return -1;

            braid_result_t *result = skyrmion_double_braid(qubit->mps, qubit->mpo,
                                                            qubit->lat, path1, path2, config);
            braid_path_free(path1);
            braid_path_free(path2);

            if (!result || !result->success) {
                braid_result_free(result);
                return -1;
            }

            double complex gate_op = I;
            double complex new_alpha = qubit->beta;
            double complex new_beta = qubit->alpha;
            qubit->alpha = gate_op * new_alpha;
            qubit->beta = gate_op * new_beta;

            double tmp_x = qubit->sky1.x;
            double tmp_y = qubit->sky1.y;
            qubit->sky1.x = qubit->sky2.x;
            qubit->sky1.y = qubit->sky2.y;
            qubit->sky2.x = tmp_x;
            qubit->sky2.y = tmp_y;

            braid_result_free(result);
            return 0;
        }

        case TOPO_GATE_HADAMARD: {
            double complex H_00 = 1.0 / sqrt(2.0);
            double complex H_01 = 1.0 / sqrt(2.0);
            double complex H_10 = 1.0 / sqrt(2.0);
            double complex H_11 = -1.0 / sqrt(2.0);

            double complex new_alpha = H_00 * qubit->alpha + H_01 * qubit->beta;
            double complex new_beta = H_10 * qubit->alpha + H_11 * qubit->beta;

            qubit->alpha = new_alpha;
            qubit->beta = new_beta;

            return 0;
        }

        default:
            return -1;
    }
}

int topo_qubit_measure_z(const topo_qubit_t *qubit) {
    if (!qubit) return 0;

    double p0 = cabs(qubit->alpha) * cabs(qubit->alpha);
    double p1 = cabs(qubit->beta) * cabs(qubit->beta);
    double total = p0 + p1;

    if (total < 1e-15) return 0;

    double r = (double)rand() / RAND_MAX;

    return (r < p0 / total) ? 1 : -1;
}

double topo_qubit_fidelity(const topo_qubit_t *qubit,
                            double complex target_alpha,
                            double complex target_beta) {
    if (!qubit) return 0.0;

    double target_norm = cabs(target_alpha) * cabs(target_alpha) +
                         cabs(target_beta) * cabs(target_beta);
    double qubit_norm = cabs(qubit->alpha) * cabs(qubit->alpha) +
                        cabs(qubit->beta) * cabs(qubit->beta);

    if (target_norm < 1e-15 || qubit_norm < 1e-15) return 0.0;

    double complex overlap = conj(target_alpha) * qubit->alpha +
                             conj(target_beta) * qubit->beta;

    return cabs(overlap) * cabs(overlap) / (target_norm * qubit_norm);
}
