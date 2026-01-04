/**
 * @file dmrg.c
 * @brief DMRG implementation for ground state preparation
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "dmrg.h"
#include "mpo_2d.h"
#include "svd_compress.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static double get_time_sec(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

static double complex_dot(const double complex *a, const double complex *b, uint64_t n) {
    double complex sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += conj(a[i]) * b[i];
    }
    return sum;
}

static double vector_norm(const double complex *v, uint64_t n) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
    }
    return sqrt(sum);
}

static void vector_scale(double complex *v, double complex alpha, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        v[i] *= alpha;
    }
}

static void vector_axpy(double complex *y, double complex alpha,
                        const double complex *x, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

// ============================================================================
// MPO CREATION
// ============================================================================

void mpo_free(mpo_t *mpo) {
    if (!mpo) return;

    if (mpo->tensors) {
        for (uint32_t i = 0; i < mpo->num_sites; i++) {
            if (mpo->tensors[i].W) {
                tensor_free(mpo->tensors[i].W);
            }
        }
        free(mpo->tensors);
    }
    free(mpo);
}

/**
 * @brief Create MPO for transverse field Ising model
 *
 * The MPO has bond dimension 3:
 *   W = | I    0   0  |
 *       | Z    0   0  |
 *       | -hX  -JZ  I |
 *
 * For boundary sites, use truncated versions.
 */
mpo_t *mpo_tfim_create(uint32_t num_sites, double J, double h) {
    if (num_sites < 2) return NULL;

    mpo_t *mpo = (mpo_t *)calloc(1, sizeof(mpo_t));
    if (!mpo) return NULL;

    mpo->num_sites = num_sites;
    mpo->max_mpo_bond = 3;
    mpo->tensors = (mpo_tensor_t *)calloc(num_sites, sizeof(mpo_tensor_t));

    if (!mpo->tensors) {
        free(mpo);
        return NULL;
    }

    // Pauli matrices
    double complex I_mat[4] = {1, 0, 0, 1};
    double complex X_mat[4] = {0, 1, 1, 0};
    double complex Z_mat[4] = {1, 0, 0, -1};
    double complex zero_mat[4] = {0, 0, 0, 0};

    for (uint32_t site = 0; site < num_sites; site++) {
        mpo_tensor_t *W = &mpo->tensors[site];
        W->phys_dim = 2;

        if (site == 0) {
            // Left boundary: W[0] is a row vector [1 x 3]
            // W = [-hX, -JZ, I]
            W->bond_dim_left = 1;
            W->bond_dim_right = 3;

            uint32_t dims[4] = {1, 2, 2, 3};  // [b_l][s][s'][b_r]
            W->W = tensor_create(4, dims);
            if (!W->W) goto error;

            // Fill tensor: W[0,s,s',b] for b=0,1,2
            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx_base = s * 2 + sp;
                    // b=0: -h*X
                    W->W->data[0 * 4 * 3 + idx_base * 3 + 0] = -h * X_mat[s * 2 + sp];
                    // b=1: -J*Z
                    W->W->data[0 * 4 * 3 + idx_base * 3 + 1] = -J * Z_mat[s * 2 + sp];
                    // b=2: I
                    W->W->data[0 * 4 * 3 + idx_base * 3 + 2] = I_mat[s * 2 + sp];
                }
            }
        }
        else if (site == num_sites - 1) {
            // Right boundary: W[n-1] is a column vector [3 x 1]
            // W = [I; Z; -hX]
            W->bond_dim_left = 3;
            W->bond_dim_right = 1;

            uint32_t dims[4] = {3, 2, 2, 1};
            W->W = tensor_create(4, dims);
            if (!W->W) goto error;

            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx_base = s * 2 + sp;
                    // b=0: I
                    W->W->data[0 * 4 * 1 + idx_base * 1 + 0] = I_mat[s * 2 + sp];
                    // b=1: Z
                    W->W->data[1 * 4 * 1 + idx_base * 1 + 0] = Z_mat[s * 2 + sp];
                    // b=2: -h*X
                    W->W->data[2 * 4 * 1 + idx_base * 1 + 0] = -h * X_mat[s * 2 + sp];
                }
            }
        }
        else {
            // Bulk: 3x3 MPO tensor
            W->bond_dim_left = 3;
            W->bond_dim_right = 3;

            uint32_t dims[4] = {3, 2, 2, 3};
            W->W = tensor_create(4, dims);
            if (!W->W) goto error;

            // Initialize to zero
            memset(W->W->data, 0, W->W->total_size * sizeof(double complex));

            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;

                    // W[0,0] = I (top-left)
                    W->W->data[0 * 4 * 3 + idx * 3 + 0] = I_mat[idx];
                    // W[1,0] = Z (connects ZZ term)
                    W->W->data[1 * 4 * 3 + idx * 3 + 0] = Z_mat[idx];
                    // W[2,0] = -h*X (on-site term)
                    W->W->data[2 * 4 * 3 + idx * 3 + 0] = -h * X_mat[idx];
                    // W[2,1] = -J*Z (starts ZZ term)
                    W->W->data[2 * 4 * 3 + idx * 3 + 1] = -J * Z_mat[idx];
                    // W[2,2] = I (bottom-right)
                    W->W->data[2 * 4 * 3 + idx * 3 + 2] = I_mat[idx];
                }
            }
        }
    }

    return mpo;

error:
    mpo_free(mpo);
    return NULL;
}

mpo_t *mpo_heisenberg_create(uint32_t num_sites, double J, double Delta, double h) {
    // Heisenberg XXZ Hamiltonian:
    // H = J * Σ (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1}) - h Σ Z_i
    //
    // MPO has bond dimension 5 with structure:
    // W[0] = [I, X, Y, Z, -h*Z]  (left boundary, 1×5)
    // W[bulk] = [[I, 0, 0, 0, 0],      (5×5)
    //            [X, 0, 0, 0, 0],
    //            [Y, 0, 0, 0, 0],
    //            [Z, 0, 0, 0, 0],
    //            [-h*Z, J*X, J*Y, J*Δ*Z, I]]
    // W[N-1] = [I, X, Y, Z, -h*Z]^T  (right boundary, 5×1)

    if (num_sites < 2) return NULL;

    mpo_t *mpo = (mpo_t *)calloc(1, sizeof(mpo_t));
    if (!mpo) return NULL;

    mpo->num_sites = num_sites;
    mpo->max_mpo_bond = 5;
    mpo->tensors = (mpo_tensor_t *)calloc(num_sites, sizeof(mpo_tensor_t));
    if (!mpo->tensors) {
        free(mpo);
        return NULL;
    }

    // Pauli matrices (2×2)
    double complex I_mat[4] = {1, 0, 0, 1};
    double complex X_mat[4] = {0, 1, 1, 0};
    double complex Y_mat[4] = {0, -I, I, 0};
    double complex Z_mat[4] = {1, 0, 0, -1};
    double complex Zero_mat[4] = {0, 0, 0, 0};

    for (uint32_t site = 0; site < num_sites; site++) {
        mpo_tensor_t *W = &mpo->tensors[site];

        if (site == 0) {
            // Left boundary: W[0] has shape [1, 2, 2, 5]
            W->bond_dim_left = 1;
            W->bond_dim_right = 5;
            uint32_t dims[4] = {1, 2, 2, 5};
            W->W = tensor_create(4, dims);
            if (!W->W) { mpo_free(mpo); return NULL; }
            tensor_zero(W->W);

            // Fill: [I, X, Y, Z, -h*Z]
            // W[0, s, s', b] where s,s' are physical, b is right bond
            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;
                    // b=0: I
                    W->W->data[0 * 2*2*5 + s * 2*5 + sp * 5 + 0] = I_mat[idx];
                    // b=1: X
                    W->W->data[0 * 2*2*5 + s * 2*5 + sp * 5 + 1] = X_mat[idx];
                    // b=2: Y
                    W->W->data[0 * 2*2*5 + s * 2*5 + sp * 5 + 2] = Y_mat[idx];
                    // b=3: Z
                    W->W->data[0 * 2*2*5 + s * 2*5 + sp * 5 + 3] = Z_mat[idx];
                    // b=4: -h*Z
                    W->W->data[0 * 2*2*5 + s * 2*5 + sp * 5 + 4] = -h * Z_mat[idx];
                }
            }
        } else if (site == num_sites - 1) {
            // Right boundary: W[N-1] has shape [5, 2, 2, 1]
            W->bond_dim_left = 5;
            W->bond_dim_right = 1;
            uint32_t dims[4] = {5, 2, 2, 1};
            W->W = tensor_create(4, dims);
            if (!W->W) { mpo_free(mpo); return NULL; }
            tensor_zero(W->W);

            // Fill: [I, X, Y, Z, -h*Z]^T (column vector in bond space)
            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;
                    // b_l=0: -h*Z (on-site from left boundary propagation)
                    W->W->data[0 * 2*2*1 + s * 2*1 + sp * 1 + 0] = -h * Z_mat[idx];
                    // b_l=1: J*X (coupling from left X)
                    W->W->data[1 * 2*2*1 + s * 2*1 + sp * 1 + 0] = J * X_mat[idx];
                    // b_l=2: J*Y (coupling from left Y)
                    W->W->data[2 * 2*2*1 + s * 2*1 + sp * 1 + 0] = J * Y_mat[idx];
                    // b_l=3: J*Delta*Z (coupling from left Z)
                    W->W->data[3 * 2*2*1 + s * 2*1 + sp * 1 + 0] = J * Delta * Z_mat[idx];
                    // b_l=4: I (identity to close)
                    W->W->data[4 * 2*2*1 + s * 2*1 + sp * 1 + 0] = I_mat[idx];
                }
            }
        } else {
            // Bulk sites: W has shape [5, 2, 2, 5]
            W->bond_dim_left = 5;
            W->bond_dim_right = 5;
            uint32_t dims[4] = {5, 2, 2, 5};
            W->W = tensor_create(4, dims);
            if (!W->W) { mpo_free(mpo); return NULL; }
            tensor_zero(W->W);

            // MPO transfer matrix structure:
            // [[I, 0, 0, 0, 0],
            //  [X, 0, 0, 0, 0],
            //  [Y, 0, 0, 0, 0],
            //  [Z, 0, 0, 0, 0],
            //  [-h*Z, J*X, J*Y, J*Δ*Z, I]]

            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;

                    // Row 0 (from left I): propagate I to right
                    W->W->data[0 * 2*2*5 + s * 2*5 + sp * 5 + 0] = I_mat[idx];

                    // Row 1 (from left X): create X for coupling
                    W->W->data[1 * 2*2*5 + s * 2*5 + sp * 5 + 0] = X_mat[idx];

                    // Row 2 (from left Y): create Y for coupling
                    W->W->data[2 * 2*2*5 + s * 2*5 + sp * 5 + 0] = Y_mat[idx];

                    // Row 3 (from left Z): create Z for coupling
                    W->W->data[3 * 2*2*5 + s * 2*5 + sp * 5 + 0] = Z_mat[idx];

                    // Row 4 (operator row):
                    // Col 0: on-site -h*Z
                    W->W->data[4 * 2*2*5 + s * 2*5 + sp * 5 + 0] = -h * Z_mat[idx];
                    // Col 1: J*X to couple with next X
                    W->W->data[4 * 2*2*5 + s * 2*5 + sp * 5 + 1] = J * X_mat[idx];
                    // Col 2: J*Y to couple with next Y
                    W->W->data[4 * 2*2*5 + s * 2*5 + sp * 5 + 2] = J * Y_mat[idx];
                    // Col 3: J*Delta*Z to couple with next Z
                    W->W->data[4 * 2*2*5 + s * 2*5 + sp * 5 + 3] = J * Delta * Z_mat[idx];
                    // Col 4: I to propagate operator row
                    W->W->data[4 * 2*2*5 + s * 2*5 + sp * 5 + 4] = I_mat[idx];
                }
            }
        }
    }

    return mpo;
}

/**
 * @brief Create MPO for Kitaev chain / XY model
 *
 * H = -J_XX sum_i X_i X_{i+1} - J_YY sum_i Y_i Y_{i+1} - h sum_i Z_i
 *
 * At the Kitaev sweet spot (Delta = t):
 *   J_XX = (t + Delta)/2 = t
 *   J_YY = (t - Delta)/2 = 0
 *   h = mu/2
 *
 * @param num_sites Number of lattice sites
 * @param J_XX XX coupling strength
 * @param J_YY YY coupling strength
 * @param h Transverse Z-field strength
 * @return MPO representation or NULL on failure
 */
mpo_t *mpo_kitaev_create(uint32_t num_sites, double J_XX, double J_YY, double h) {
    if (num_sites < 2) return NULL;

    mpo_t *mpo = (mpo_t *)calloc(1, sizeof(mpo_t));
    if (!mpo) return NULL;

    mpo->num_sites = num_sites;
    mpo->max_mpo_bond = 4;
    mpo->tensors = (mpo_tensor_t *)calloc(num_sites, sizeof(mpo_tensor_t));

    if (!mpo->tensors) {
        free(mpo);
        return NULL;
    }

    // Pauli matrices (stored as [row][col] -> data[row*2 + col])
    double complex I_mat[4] = {1, 0, 0, 1};
    double complex X_mat[4] = {0, 1, 1, 0};
    double complex Y_mat[4] = {0, -I, I, 0};
    double complex Z_mat[4] = {1, 0, 0, -1};

    for (uint32_t site = 0; site < num_sites; site++) {
        mpo_tensor_t *W = &mpo->tensors[site];
        W->phys_dim = 2;

        if (site == 0) {
            // Left boundary: W[0] is [1 x 4]
            // W = [-h*Z, -J_XX*X, -J_YY*Y, I]
            W->bond_dim_left = 1;
            W->bond_dim_right = 4;

            uint32_t dims[4] = {1, 2, 2, 4};
            W->W = tensor_create(4, dims);
            if (!W->W) goto error;

            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;
                    // b=0: -h*Z
                    W->W->data[0 * 4 * 4 + idx * 4 + 0] = -h * Z_mat[idx];
                    // b=1: -J_XX*X
                    W->W->data[0 * 4 * 4 + idx * 4 + 1] = -J_XX * X_mat[idx];
                    // b=2: -J_YY*Y
                    W->W->data[0 * 4 * 4 + idx * 4 + 2] = -J_YY * Y_mat[idx];
                    // b=3: I
                    W->W->data[0 * 4 * 4 + idx * 4 + 3] = I_mat[idx];
                }
            }
        }
        else if (site == num_sites - 1) {
            // Right boundary: W[n-1] is [4 x 1]
            // W = [I; X; Y; -h*Z]^T
            W->bond_dim_left = 4;
            W->bond_dim_right = 1;

            uint32_t dims[4] = {4, 2, 2, 1};
            W->W = tensor_create(4, dims);
            if (!W->W) goto error;

            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;
                    // b=0: I (completes identity)
                    W->W->data[0 * 4 * 1 + idx * 1 + 0] = I_mat[idx];
                    // b=1: X (completes XX)
                    W->W->data[1 * 4 * 1 + idx * 1 + 0] = X_mat[idx];
                    // b=2: Y (completes YY)
                    W->W->data[2 * 4 * 1 + idx * 1 + 0] = Y_mat[idx];
                    // b=3: -h*Z (on-site Z term)
                    W->W->data[3 * 4 * 1 + idx * 1 + 0] = -h * Z_mat[idx];
                }
            }
        }
        else {
            // Bulk: 4x4 MPO tensor
            // W = [[I, 0, 0, 0],
            //      [X, 0, 0, 0],
            //      [Y, 0, 0, 0],
            //      [-h*Z, -J_XX*X, -J_YY*Y, I]]
            W->bond_dim_left = 4;
            W->bond_dim_right = 4;

            uint32_t dims[4] = {4, 2, 2, 4};
            W->W = tensor_create(4, dims);
            if (!W->W) goto error;

            memset(W->W->data, 0, W->W->total_size * sizeof(double complex));

            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;
                    // Row 0: [I, 0, 0, 0]
                    W->W->data[0 * 4 * 4 + idx * 4 + 0] = I_mat[idx];
                    // Row 1: [X, 0, 0, 0]
                    W->W->data[1 * 4 * 4 + idx * 4 + 0] = X_mat[idx];
                    // Row 2: [Y, 0, 0, 0]
                    W->W->data[2 * 4 * 4 + idx * 4 + 0] = Y_mat[idx];
                    // Row 3: [-h*Z, -J_XX*X, -J_YY*Y, I]
                    W->W->data[3 * 4 * 4 + idx * 4 + 0] = -h * Z_mat[idx];
                    W->W->data[3 * 4 * 4 + idx * 4 + 1] = -J_XX * X_mat[idx];
                    W->W->data[3 * 4 * 4 + idx * 4 + 2] = -J_YY * Y_mat[idx];
                    W->W->data[3 * 4 * 4 + idx * 4 + 3] = I_mat[idx];
                }
            }
        }
    }

    return mpo;

error:
    mpo_free(mpo);
    return NULL;
}

// ============================================================================
// 2D SKYRMION MPO
// ============================================================================

/**
 * @brief Helper: Convert 2D grid coordinates to snake (MPS) index
 */
static inline uint32_t grid_to_snake_idx(uint32_t x, uint32_t y, uint32_t Lx) {
    if (y % 2 == 0) {
        return y * Lx + x;
    } else {
        return y * Lx + (Lx - 1 - x);
    }
}

/**
 * @brief Helper: Get 2D coordinates from snake index
 */
static inline void snake_to_grid_coords(uint32_t s, uint32_t Lx,
                                        uint32_t *x, uint32_t *y) {
    *y = s / Lx;
    uint32_t col = s % Lx;
    if (*y % 2 == 0) {
        *x = col;
    } else {
        *x = Lx - 1 - col;
    }
}

mpo_t *mpo_skyrmion_create(uint32_t num_sites, uint32_t Lx, uint32_t Ly,
                           double J, double D, double B, double K) {
    if (num_sites < 4 || Lx < 2 || Ly < 2 || Lx * Ly != num_sites) {
        return NULL;
    }

    // For 2D systems with snake ordering, we need a larger MPO bond dimension
    // to capture long-range interactions that arise from mapping 2D → 1D
    //
    // The Hamiltonian is:
    // H = -J Σ (X_i X_j + Y_i Y_j + Z_i Z_j)  [Heisenberg exchange]
    //     + D Σ d_ij · (S_i × S_j)            [DMI]
    //     - B Σ Z_i                           [Zeeman field]
    //     - K Σ Z_i²                          [Anisotropy]
    //
    // DMI term: d_ij · (S_i × S_j) for interfacial DMI, d_ij ⊥ bond
    // For bond along x: d = (0, 0, 1), giving D * (X_i Y_j - Y_i X_j)
    // For bond along y: d = (0, 0, 1), giving D * (Y_i X_j - X_i Y_j)

    mpo_t *mpo = (mpo_t *)calloc(1, sizeof(mpo_t));
    if (!mpo) return NULL;

    mpo->num_sites = num_sites;
    // Bond dimension: need to track X, Y, Z for each "open" interaction
    // For simplicity, use a larger fixed bond dimension
    mpo->max_mpo_bond = 6;  // I, X, Y, Z, ZZ_aniso, final
    mpo->tensors = (mpo_tensor_t *)calloc(num_sites, sizeof(mpo_tensor_t));

    if (!mpo->tensors) {
        free(mpo);
        return NULL;
    }

    // Pauli matrices
    double complex I_mat[4] = {1, 0, 0, 1};
    double complex X_mat[4] = {0, 1, 1, 0};
    double complex Y_mat[4] = {0, -I, I, 0};
    double complex Z_mat[4] = {1, 0, 0, -1};
    double complex Z2_mat[4] = {1, 0, 0, 1};  // Z² = I

    // For each site, build MPO tensor
    // This is a simplified version that handles nearest-neighbor interactions
    // in the snake-ordered chain

    for (uint32_t site = 0; site < num_sites; site++) {
        mpo_tensor_t *W = &mpo->tensors[site];
        W->phys_dim = 2;

        // Get 2D coordinates
        uint32_t x, y;
        snake_to_grid_coords(site, Lx, &x, &y);

        if (site == 0) {
            // Left boundary: [1 x 6]
            W->bond_dim_left = 1;
            W->bond_dim_right = 6;

            uint32_t dims[4] = {1, 2, 2, 6};
            W->W = tensor_create(4, dims);
            if (!W->W) goto error;

            // On-site terms: -B*Z - K*Z² (but Z²=I, so just shift)
            // MPO: [on-site, -J*X, -J*Y, -J*Z, 0, I]
            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;
                    // b=0: -B*Z - K (on-site)
                    W->W->data[idx * 6 + 0] = -B * Z_mat[idx] - K * Z2_mat[idx];
                    // b=1: -J*X (starts XX)
                    W->W->data[idx * 6 + 1] = -J * X_mat[idx];
                    // b=2: -J*Y (starts YY)
                    W->W->data[idx * 6 + 2] = -J * Y_mat[idx];
                    // b=3: -J*Z (starts ZZ)
                    W->W->data[idx * 6 + 3] = -J * Z_mat[idx];
                    // b=4: D*X for DMI (XY - YX term)
                    W->W->data[idx * 6 + 4] = D * X_mat[idx];
                    // b=5: I
                    W->W->data[idx * 6 + 5] = I_mat[idx];
                }
            }
        }
        else if (site == num_sites - 1) {
            // Right boundary: [6 x 1]
            W->bond_dim_left = 6;
            W->bond_dim_right = 1;

            uint32_t dims[4] = {6, 2, 2, 1};
            W->W = tensor_create(4, dims);
            if (!W->W) goto error;

            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;
                    // b=0: I (completes on-site from previous)
                    W->W->data[0 * 4 + idx] = I_mat[idx];
                    // b=1: X (completes XX)
                    W->W->data[1 * 4 + idx] = X_mat[idx];
                    // b=2: Y (completes YY)
                    W->W->data[2 * 4 + idx] = Y_mat[idx];
                    // b=3: Z (completes ZZ)
                    W->W->data[3 * 4 + idx] = Z_mat[idx];
                    // b=4: Y for DMI
                    W->W->data[4 * 4 + idx] = Y_mat[idx];
                    // b=5: -B*Z - K (on-site)
                    W->W->data[5 * 4 + idx] = -B * Z_mat[idx] - K * Z2_mat[idx];
                }
            }
        }
        else {
            // Bulk: [6 x 6]
            W->bond_dim_left = 6;
            W->bond_dim_right = 6;

            uint32_t dims[4] = {6, 2, 2, 6};
            W->W = tensor_create(4, dims);
            if (!W->W) goto error;

            memset(W->W->data, 0, W->W->total_size * sizeof(double complex));

            for (int s = 0; s < 2; s++) {
                for (int sp = 0; sp < 2; sp++) {
                    int idx = s * 2 + sp;

                    // Row 0: [I, 0, 0, 0, 0, 0] - identity pass-through
                    W->W->data[0 * 4 * 6 + idx * 6 + 0] = I_mat[idx];

                    // Row 1: [X, 0, 0, 0, 0, 0] - complete XX
                    W->W->data[1 * 4 * 6 + idx * 6 + 0] = X_mat[idx];

                    // Row 2: [Y, 0, 0, 0, 0, 0] - complete YY
                    W->W->data[2 * 4 * 6 + idx * 6 + 0] = Y_mat[idx];

                    // Row 3: [Z, 0, 0, 0, 0, 0] - complete ZZ
                    W->W->data[3 * 4 * 6 + idx * 6 + 0] = Z_mat[idx];

                    // Row 4: [Y, 0, 0, 0, 0, 0] - complete DMI (X*Y)
                    W->W->data[4 * 4 * 6 + idx * 6 + 0] = Y_mat[idx];

                    // Row 5: [on-site, -J*X, -J*Y, -J*Z, D*X, I]
                    W->W->data[5 * 4 * 6 + idx * 6 + 0] = -B * Z_mat[idx] - K * Z2_mat[idx];
                    W->W->data[5 * 4 * 6 + idx * 6 + 1] = -J * X_mat[idx];
                    W->W->data[5 * 4 * 6 + idx * 6 + 2] = -J * Y_mat[idx];
                    W->W->data[5 * 4 * 6 + idx * 6 + 3] = -J * Z_mat[idx];
                    W->W->data[5 * 4 * 6 + idx * 6 + 4] = D * X_mat[idx];
                    W->W->data[5 * 4 * 6 + idx * 6 + 5] = I_mat[idx];
                }
            }
        }
    }

    return mpo;

error:
    mpo_free(mpo);
    return NULL;
}

mpo_t *mpo_2d_heisenberg_dmi_create(uint32_t num_sites,
                                     uint32_t num_bonds,
                                     const uint32_t (*bonds)[2],
                                     const double (*bond_vectors)[3],
                                     double J, double D, double B, double K) {
    // Create a bond list from the explicit bond specification
    // This handles arbitrary 2D lattices with long-range bonds (in 1D ordering)

    if (num_sites < 2 || num_bonds == 0 || !bonds) return NULL;

    // Allocate bond list
    bond_list_t *bond_list = (bond_list_t *)calloc(1, sizeof(bond_list_t));
    if (!bond_list) return NULL;

    bond_list->num_sites = num_sites;
    bond_list->max_range = 0;

    // Count interactions: for each bond, we have XX + YY + ZZ (3 terms)
    // Plus DMI terms if D != 0: XY - YX (2 terms per bond)
    uint32_t interactions_per_bond = 3;  // XX, YY, ZZ
    if (D != 0.0 && bond_vectors != NULL) {
        interactions_per_bond += 2;  // XY, YX for DMI
    }

    bond_list->capacity = num_bonds * interactions_per_bond;
    bond_list->interactions = (bond_interaction_t *)calloc(
        bond_list->capacity, sizeof(bond_interaction_t));

    if (!bond_list->interactions) {
        free(bond_list);
        return NULL;
    }

    // Fill in the interactions for each bond
    uint32_t idx = 0;
    for (uint32_t b = 0; b < num_bonds; b++) {
        uint32_t i = bonds[b][0];
        uint32_t j = bonds[b][1];

        // Ensure i < j for consistent ordering
        if (i > j) {
            uint32_t tmp = i;
            i = j;
            j = tmp;
        }

        // Track max range
        uint32_t range = j - i;
        if (range > bond_list->max_range) {
            bond_list->max_range = range;
        }

        // Add Heisenberg exchange terms: J * (X_i X_j + Y_i Y_j + Z_i Z_j)
        bond_list->interactions[idx++] = (bond_interaction_t){
            .site_i = i, .site_j = j,
            .type = INTERACT_XX, .coefficient = J
        };
        bond_list->interactions[idx++] = (bond_interaction_t){
            .site_i = i, .site_j = j,
            .type = INTERACT_YY, .coefficient = J
        };
        bond_list->interactions[idx++] = (bond_interaction_t){
            .site_i = i, .site_j = j,
            .type = INTERACT_ZZ, .coefficient = J
        };

        // Add DMI terms if D != 0 and bond_vectors provided
        // DMI: D * d_ij · (S_i × S_j) where d_ij is the DMI vector
        // For interfacial DMI with d_ij in z-direction:
        // D * (X_i Y_j - Y_i X_j)
        if (D != 0.0 && bond_vectors != NULL) {
            // Get bond direction for DMI
            double dx = bond_vectors[b][0];
            double dy = bond_vectors[b][1];
            double dz = bond_vectors[b][2];

            // For interfacial DMI: d_ij perpendicular to bond
            // Simplified: assume DMI along z gives X_i Y_j - Y_i X_j term
            // The coefficient depends on the bond direction
            double dmi_coeff = D * dz;  // z-component of DMI vector

            if (fabs(dmi_coeff) > 1e-12) {
                bond_list->interactions[idx++] = (bond_interaction_t){
                    .site_i = i, .site_j = j,
                    .type = INTERACT_XY, .coefficient = dmi_coeff
                };
                bond_list->interactions[idx++] = (bond_interaction_t){
                    .site_i = i, .site_j = j,
                    .type = INTERACT_YX, .coefficient = -dmi_coeff
                };
            }
        }
    }
    bond_list->num_interactions = idx;

    // Create on-site terms arrays
    double *on_site_z = (double *)calloc(num_sites, sizeof(double));
    double *on_site_zz = (double *)calloc(num_sites, sizeof(double));

    if (!on_site_z || !on_site_zz) {
        free(on_site_z);
        free(on_site_zz);
        bond_list_free(bond_list);
        return NULL;
    }

    // Zeeman term: -B * Z_i and anisotropy: -K * Z_i^2 = -K (constant shift)
    for (uint32_t i = 0; i < num_sites; i++) {
        on_site_z[i] = -B;   // Zeeman field
        on_site_zz[i] = -K;  // Anisotropy (Z^2 = I, so this is just a constant)
    }

    // Create MPO from bond list
    mpo_t *mpo = mpo_from_bond_list(bond_list, on_site_z, on_site_zz);

    // Cleanup
    free(on_site_z);
    free(on_site_zz);
    bond_list_free(bond_list);

    return mpo;
}

// ============================================================================
// DMRG RESULT
// ============================================================================

void dmrg_result_free(dmrg_result_t *result) {
    if (!result) return;
    if (result->sweep_energies) free(result->sweep_energies);
    free(result);
}

// ============================================================================
// LANCZOS EIGENSOLVER
// ============================================================================

void lanczos_result_free(lanczos_result_t *result) {
    if (!result) return;
    if (result->eigenvector) tensor_free(result->eigenvector);
    free(result);
}

/**
 * @brief Apply effective Hamiltonian H_eff to theta vector
 *
 * For two-site DMRG:
 * y[l,s1,s2,r] = sum_{l',s1',s2',r'} H_eff[l,s1,s2,r; l',s1',s2',r'] * x[l',s1',s2',r']
 *
 * H_eff is built from L, W_left, W_right, R environments.
 */
int effective_hamiltonian_apply(const effective_hamiltonian_t *H_eff,
                                const tensor_t *x,
                                tensor_t *y) {
    if (!H_eff || !x || !y) return -1;

    uint32_t chi_l = H_eff->chi_l;
    uint32_t chi_r = H_eff->chi_r;
    uint32_t d = H_eff->phys_dim;

    // Initialize y to zero
    memset(y->data, 0, y->total_size * sizeof(double complex));

    if (H_eff->two_site) {
        // Validate environment tensors
        if (!H_eff->L || !H_eff->R) return -1;
        if (!H_eff->L->data || !H_eff->R->data) return -1;
        if (!H_eff->W_left || !H_eff->W_right) return -1;
        if (!H_eff->W_left->W || !H_eff->W_right->W) return -1;
        if (!H_eff->W_left->W->data || !H_eff->W_right->W->data) return -1;

        // Two-site: theta has shape [chi_l][d][d][chi_r]
        uint32_t b_l = H_eff->W_left->bond_dim_left;
        uint32_t b_m = H_eff->W_left->bond_dim_right;  // = W_right->bond_dim_left
        uint32_t b_r = H_eff->W_right->bond_dim_right;

        tensor_t *L = H_eff->L;  // [chi_l][b_l][chi_l]
        tensor_t *R = H_eff->R;  // [chi_r][b_r][chi_r]
        tensor_t *W1 = H_eff->W_left->W;   // [b_l][d][d][b_m]
        tensor_t *W2 = H_eff->W_right->W;  // [b_m][d][d][b_r]

        // Contract: y = L @ W1 @ W2 @ R @ x
        // Optimized sequential contraction to reduce O(chi^4) to O(chi^2 * b^2)
        //
        // Step 1: temp1[lp,s1p,s2p,br,r] = x[lp,s1p,s2p,rp] @ R[r,br,rp]
        // Step 2: temp2[lp,s1p,bm,s2,r] = temp1[lp,s1p,s2p,br,r] @ W2[bm,s2,s2p,br]
        // Step 3: temp3[lp,bl,s1,s2,r] = temp2[lp,s1p,bm,s2,r] @ W1[bl,s1,s1p,bm]
        // Step 4: y[l,s1,s2,r] = temp3[lp,bl,s1,s2,r] @ L[l,bl,lp]

        // Allocate intermediate tensors
        size_t temp1_size = chi_l * d * d * b_r * chi_r;
        size_t temp2_size = chi_l * d * b_m * d * chi_r;
        size_t temp3_size = chi_l * b_l * d * d * chi_r;

        double complex *temp1 = (double complex *)calloc(temp1_size, sizeof(double complex));
        double complex *temp2 = (double complex *)calloc(temp2_size, sizeof(double complex));
        double complex *temp3 = (double complex *)calloc(temp3_size, sizeof(double complex));

        if (!temp1 || !temp2 || !temp3) {
            free(temp1); free(temp2); free(temp3);
            return -1;
        }

        // Step 1: temp1[lp,s1p,s2p,br,r] = sum_rp x[lp,s1p,s2p,rp] * R[r,br,rp]
        for (uint32_t lp = 0; lp < chi_l; lp++) {
            for (uint32_t s1p = 0; s1p < d; s1p++) {
                for (uint32_t s2p = 0; s2p < d; s2p++) {
                    for (uint32_t br = 0; br < b_r; br++) {
                        for (uint32_t r = 0; r < chi_r; r++) {
                            double complex sum = 0.0;
                            for (uint32_t rp = 0; rp < chi_r; rp++) {
                                uint64_t x_idx = lp * d * d * chi_r + s1p * d * chi_r + s2p * chi_r + rp;
                                uint64_t R_idx = r * b_r * chi_r + br * chi_r + rp;
                                sum += x->data[x_idx] * R->data[R_idx];
                            }
                            uint64_t t1_idx = lp * d * d * b_r * chi_r + s1p * d * b_r * chi_r + s2p * b_r * chi_r + br * chi_r + r;
                            temp1[t1_idx] = sum;
                        }
                    }
                }
            }
        }

        // Step 2: temp2[lp,s1p,bm,s2,r] = sum_{s2p,br} temp1[lp,s1p,s2p,br,r] * W2[bm,s2,s2p,br]
        for (uint32_t lp = 0; lp < chi_l; lp++) {
            for (uint32_t s1p = 0; s1p < d; s1p++) {
                for (uint32_t bm = 0; bm < b_m; bm++) {
                    for (uint32_t s2 = 0; s2 < d; s2++) {
                        for (uint32_t r = 0; r < chi_r; r++) {
                            double complex sum = 0.0;
                            for (uint32_t s2p = 0; s2p < d; s2p++) {
                                for (uint32_t br = 0; br < b_r; br++) {
                                    uint64_t t1_idx = lp * d * d * b_r * chi_r + s1p * d * b_r * chi_r + s2p * b_r * chi_r + br * chi_r + r;
                                    uint64_t W2_idx = bm * d * d * b_r + s2 * d * b_r + s2p * b_r + br;
                                    sum += temp1[t1_idx] * W2->data[W2_idx];
                                }
                            }
                            uint64_t t2_idx = lp * d * b_m * d * chi_r + s1p * b_m * d * chi_r + bm * d * chi_r + s2 * chi_r + r;
                            temp2[t2_idx] = sum;
                        }
                    }
                }
            }
        }

        free(temp1);

        // Step 3: temp3[lp,bl,s1,s2,r] = sum_{s1p,bm} temp2[lp,s1p,bm,s2,r] * W1[bl,s1,s1p,bm]
        for (uint32_t lp = 0; lp < chi_l; lp++) {
            for (uint32_t bl = 0; bl < b_l; bl++) {
                for (uint32_t s1 = 0; s1 < d; s1++) {
                    for (uint32_t s2 = 0; s2 < d; s2++) {
                        for (uint32_t r = 0; r < chi_r; r++) {
                            double complex sum = 0.0;
                            for (uint32_t s1p = 0; s1p < d; s1p++) {
                                for (uint32_t bm = 0; bm < b_m; bm++) {
                                    uint64_t t2_idx = lp * d * b_m * d * chi_r + s1p * b_m * d * chi_r + bm * d * chi_r + s2 * chi_r + r;
                                    uint64_t W1_idx = bl * d * d * b_m + s1 * d * b_m + s1p * b_m + bm;
                                    sum += temp2[t2_idx] * W1->data[W1_idx];
                                }
                            }
                            uint64_t t3_idx = lp * b_l * d * d * chi_r + bl * d * d * chi_r + s1 * d * chi_r + s2 * chi_r + r;
                            temp3[t3_idx] = sum;
                        }
                    }
                }
            }
        }

        free(temp2);

        // Step 4: y[l,s1,s2,r] = sum_{lp,bl} temp3[lp,bl,s1,s2,r] * L[l,bl,lp]
        for (uint32_t l = 0; l < chi_l; l++) {
            for (uint32_t s1 = 0; s1 < d; s1++) {
                for (uint32_t s2 = 0; s2 < d; s2++) {
                    for (uint32_t r = 0; r < chi_r; r++) {
                        double complex sum = 0.0;
                        for (uint32_t lp = 0; lp < chi_l; lp++) {
                            for (uint32_t bl = 0; bl < b_l; bl++) {
                                uint64_t t3_idx = lp * b_l * d * d * chi_r + bl * d * d * chi_r + s1 * d * chi_r + s2 * chi_r + r;
                                uint64_t L_idx = l * b_l * chi_l + bl * chi_l + lp;
                                sum += temp3[t3_idx] * L->data[L_idx];
                            }
                        }
                        uint64_t y_idx = l * d * d * chi_r + s1 * d * chi_r + s2 * chi_r + r;
                        y->data[y_idx] = sum;
                    }
                }
            }
        }

        free(temp3);
    }
    else {
        // One-site DMRG - similar but simpler
        // theta has shape [chi_l][d][chi_r]
        // Implementation similar to above but with only one W tensor
    }

    return 0;
}

/**
 * @brief Solve tridiagonal eigenvalue problem using QL algorithm
 *
 * Finds smallest eigenvalue and eigenvector of tridiagonal matrix T.
 * T has diagonal alpha[0..n-1] and off-diagonal beta[1..n-1].
 *
 * @param alpha Diagonal elements
 * @param beta Off-diagonal elements (beta[0] unused)
 * @param n Matrix size
 * @param eigenvalue Output: smallest eigenvalue
 * @param eigenvector Output: corresponding eigenvector (length n)
 */
static void tridiag_smallest_eigenpair(const double *alpha, const double *beta,
                                        uint32_t n, double *eigenvalue,
                                        double *eigenvector) {
    if (n == 0) return;
    if (n == 1) {
        *eigenvalue = alpha[0];
        eigenvector[0] = 1.0;
        return;
    }

    // Copy to working arrays
    // Note: e needs n+1 elements because QL algorithm accesses e[i+2] where i can be n-2
    double *d = (double *)malloc(n * sizeof(double));
    double *e = (double *)malloc((n + 1) * sizeof(double));
    double *z = (double *)calloc(n * n, sizeof(double));

    if (!d || !e || !z) {
        free(d);
        free(e);
        free(z);
        *eigenvalue = 0.0;
        return;
    }
    e[n] = 0.0;  // Initialize the extra element

    for (uint32_t i = 0; i < n; i++) {
        d[i] = alpha[i];
        e[i] = (i > 0) ? beta[i] : 0.0;
        z[i * n + i] = 1.0;  // Identity matrix
    }

    // QL algorithm with implicit shifts
    const int MAX_ITER = 30;
    const double EPS = 1e-15;

    for (uint32_t l = 0; l < n; l++) {
        int iter = 0;
        uint32_t m;

        do {
            // Find small off-diagonal element
            for (m = l; m < n - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m + 1]) <= EPS * dd) break;
            }

            if (m != l) {
                if (iter++ >= MAX_ITER) break;

                // Form shift
                double g = (d[l + 1] - d[l]) / (2.0 * e[l + 1]);
                double r = sqrt(g * g + 1.0);
                g = d[m] - d[l] + e[l + 1] / (g + (g >= 0 ? r : -r));

                double s = 1.0, c = 1.0, p = 0.0;

                for (int i = m - 1; i >= (int)l; i--) {
                    double f = s * e[i + 1];
                    double b = c * e[i + 1];

                    if (fabs(f) >= fabs(g)) {
                        c = g / f;
                        r = sqrt(c * c + 1.0);
                        e[i + 2] = f * r;
                        s = 1.0 / r;
                        c *= s;
                    } else {
                        s = f / g;
                        r = sqrt(s * s + 1.0);
                        e[i + 2] = g * r;
                        c = 1.0 / r;
                        s *= c;
                    }

                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;

                    // Update eigenvector matrix
                    for (uint32_t k = 0; k < n; k++) {
                        f = z[k * n + i + 1];
                        z[k * n + i + 1] = s * z[k * n + i] + c * f;
                        z[k * n + i] = c * z[k * n + i] - s * f;
                    }
                }

                d[l] -= p;
                e[l + 1] = g;
                e[m + 1] = 0.0;
            }
        } while (m != l);
    }

    // Find smallest eigenvalue
    uint32_t min_idx = 0;
    double min_val = d[0];
    for (uint32_t i = 1; i < n; i++) {
        if (d[i] < min_val) {
            min_val = d[i];
            min_idx = i;
        }
    }

    *eigenvalue = min_val;
    for (uint32_t i = 0; i < n; i++) {
        eigenvector[i] = z[i * n + min_idx];
    }

    free(d);
    free(e);
    free(z);
}

/**
 * @brief Lanczos algorithm for ground state
 *
 * Finds the smallest eigenvalue and corresponding eigenvector.
 */
lanczos_result_t *lanczos_ground_state(const effective_hamiltonian_t *H_eff,
                                        const tensor_t *initial_guess,
                                        uint32_t max_iter,
                                        double tol) {
    if (!H_eff) return NULL;

    uint32_t chi_l = H_eff->chi_l;
    uint32_t chi_r = H_eff->chi_r;
    uint32_t d = H_eff->phys_dim;
    uint64_t vec_size = H_eff->two_site ? chi_l * d * d * chi_r : chi_l * d * chi_r;

    lanczos_result_t *result = (lanczos_result_t *)calloc(1, sizeof(lanczos_result_t));
    if (!result) return NULL;

    // Allocate Lanczos vectors
    double complex *v_prev = (double complex *)calloc(vec_size, sizeof(double complex));
    double complex *v_curr = (double complex *)calloc(vec_size, sizeof(double complex));
    double complex *v_next = (double complex *)calloc(vec_size, sizeof(double complex));
    double complex *w = (double complex *)calloc(vec_size, sizeof(double complex));

    // Tridiagonal matrix elements
    double *alpha = (double *)calloc(max_iter, sizeof(double));
    double *beta = (double *)calloc(max_iter + 1, sizeof(double));

    // Store all Lanczos vectors for eigenvector reconstruction
    double complex **V = (double complex **)calloc(max_iter, sizeof(double complex *));

    if (!v_prev || !v_curr || !v_next || !w || !alpha || !beta || !V) {
        goto cleanup;
    }

    // Initialize with provided guess or random
    if (initial_guess && initial_guess->total_size == vec_size) {
        memcpy(v_curr, initial_guess->data, vec_size * sizeof(double complex));
    } else {
        for (uint64_t i = 0; i < vec_size; i++) {
            v_curr[i] = (double)rand() / RAND_MAX - 0.5;
        }
    }

    // Normalize
    double norm = vector_norm(v_curr, vec_size);
    if (norm < 1e-15) {
        v_curr[0] = 1.0;
        norm = 1.0;
    }
    vector_scale(v_curr, 1.0 / norm, vec_size);

    beta[0] = 0.0;
    double prev_eigenvalue = 1e30;
    uint32_t num_iter = 0;

    // Create temporary tensors
    uint32_t dims[4];
    if (H_eff->two_site) {
        dims[0] = chi_l; dims[1] = d; dims[2] = d; dims[3] = chi_r;
    } else {
        dims[0] = chi_l; dims[1] = d; dims[2] = chi_r; dims[3] = 0;
    }

    tensor_t *x_tensor = tensor_create(H_eff->two_site ? 4 : 3, dims);
    tensor_t *y_tensor = tensor_create(H_eff->two_site ? 4 : 3, dims);

    if (!x_tensor || !y_tensor) {
        if (x_tensor) tensor_free(x_tensor);
        if (y_tensor) tensor_free(y_tensor);
        goto cleanup;
    }

    // Lanczos iteration
    for (uint32_t iter = 0; iter < max_iter; iter++) {
        num_iter = iter + 1;

        // Store current vector
        V[iter] = (double complex *)malloc(vec_size * sizeof(double complex));
        if (!V[iter]) goto cleanup;
        memcpy(V[iter], v_curr, vec_size * sizeof(double complex));

        // Validate H_eff pointers
        if (!H_eff->L || !H_eff->L->data || !H_eff->R || !H_eff->R->data) {
            goto cleanup;
        }

        // w = H @ v_curr
        memcpy(x_tensor->data, v_curr, vec_size * sizeof(double complex));
        int apply_ret = effective_hamiltonian_apply(H_eff, x_tensor, y_tensor);
        if (apply_ret != 0) {
            fprintf(stderr, "      [Lanczos] H_eff apply failed at iter %u\n", iter);
            goto cleanup;
        }
        memcpy(w, y_tensor->data, vec_size * sizeof(double complex));

        // alpha[iter] = v_curr^H @ w (Rayleigh quotient)
        double complex alpha_c = complex_dot(v_curr, w, vec_size);
        alpha[iter] = creal(alpha_c);

        // w = w - alpha * v_curr - beta[iter] * v_prev
        vector_axpy(w, -alpha[iter], v_curr, vec_size);
        if (iter > 0) {
            vector_axpy(w, -beta[iter], v_prev, vec_size);
        }

        // Reorthogonalize (important for numerical stability)
        for (uint32_t j = 0; j <= iter; j++) {
            double complex overlap = complex_dot(V[j], w, vec_size);
            vector_axpy(w, -overlap, V[j], vec_size);
        }

        // beta[iter+1] = ||w||
        beta[iter + 1] = vector_norm(w, vec_size);

        // Check for convergence
        if (iter >= 2) {
            double *evec = (double *)malloc((iter + 1) * sizeof(double));
            if (!evec) goto cleanup;
            double eval;
            tridiag_smallest_eigenpair(alpha, beta, iter + 1, &eval, evec);
            free(evec);

            if (fabs(eval - prev_eigenvalue) < tol) {
                result->converged = true;
                result->eigenvalue = eval;
                result->num_iterations = iter + 1;
                break;
            }
            prev_eigenvalue = eval;
        }

        // Check for invariant subspace
        if (beta[iter + 1] < 1e-14) {
            result->converged = true;
            result->num_iterations = iter + 1;
            break;
        }

        // v_next = w / beta
        memcpy(v_next, w, vec_size * sizeof(double complex));
        vector_scale(v_next, 1.0 / beta[iter + 1], vec_size);

        // Shift vectors
        memcpy(v_prev, v_curr, vec_size * sizeof(double complex));
        memcpy(v_curr, v_next, vec_size * sizeof(double complex));
    }

    tensor_free(x_tensor);
    tensor_free(y_tensor);

    // Solve tridiagonal eigenvalue problem for final result
    double *tridiag_evec = (double *)malloc(num_iter * sizeof(double));
    double tridiag_eval;
    tridiag_smallest_eigenpair(alpha, beta, num_iter, &tridiag_eval, tridiag_evec);

    result->eigenvalue = tridiag_eval;
    if (!result->converged) {
        result->num_iterations = num_iter;
    }

    // Reconstruct eigenvector: psi = sum_i evec[i] * V[i]
    uint32_t edims[4];
    if (H_eff->two_site) {
        edims[0] = chi_l; edims[1] = d; edims[2] = d; edims[3] = chi_r;
        result->eigenvector = tensor_create(4, edims);
    } else {
        edims[0] = chi_l; edims[1] = d; edims[2] = chi_r;
        result->eigenvector = tensor_create(3, edims);
    }

    if (result->eigenvector) {
        memset(result->eigenvector->data, 0, vec_size * sizeof(double complex));
        for (uint32_t i = 0; i < num_iter; i++) {
            for (uint64_t j = 0; j < vec_size; j++) {
                result->eigenvector->data[j] += tridiag_evec[i] * V[i][j];
            }
        }

        // Normalize the eigenvector
        norm = vector_norm(result->eigenvector->data, vec_size);
        if (norm > 1e-15) {
            vector_scale(result->eigenvector->data, 1.0 / norm, vec_size);
        }
    }

    free(tridiag_evec);

    // Cleanup Lanczos vectors
    for (uint32_t i = 0; i < num_iter; i++) {
        if (V[i]) free(V[i]);
    }
    free(V);
    free(v_prev);
    free(v_curr);
    free(v_next);
    free(w);
    free(alpha);
    free(beta);

    return result;

cleanup:
    if (V) {
        for (uint32_t i = 0; i < max_iter; i++) {
            if (V[i]) free(V[i]);
        }
        free(V);
    }
    free(v_prev);
    free(v_curr);
    free(v_next);
    free(w);
    free(alpha);
    free(beta);
    lanczos_result_free(result);
    return NULL;
}

// ============================================================================
// DMRG ENVIRONMENTS
// ============================================================================

dmrg_environments_t *dmrg_environments_create(uint32_t num_sites) {
    dmrg_environments_t *env = (dmrg_environments_t *)calloc(1, sizeof(dmrg_environments_t));
    if (!env) return NULL;

    env->num_sites = num_sites;
    env->L = (tensor_t **)calloc(num_sites, sizeof(tensor_t *));
    env->R = (tensor_t **)calloc(num_sites, sizeof(tensor_t *));

    if (!env->L || !env->R) {
        dmrg_environments_free(env);
        return NULL;
    }

    return env;
}

void dmrg_environments_free(dmrg_environments_t *env) {
    if (!env) return;

    if (env->L) {
        for (uint32_t i = 0; i < env->num_sites; i++) {
            if (env->L[i]) tensor_free(env->L[i]);
        }
        free(env->L);
    }

    if (env->R) {
        for (uint32_t i = 0; i < env->num_sites; i++) {
            if (env->R[i]) tensor_free(env->R[i]);
        }
        free(env->R);
    }

    free(env);
}

/**
 * @brief Initialize left boundary environment
 *
 * L[0] = identity for left boundary
 */
static tensor_t *create_left_boundary(uint32_t chi, uint32_t mpo_bond) {
    uint32_t dims[3] = {chi, mpo_bond, chi};
    tensor_t *L = tensor_create(3, dims);
    if (!L) return NULL;

    memset(L->data, 0, L->total_size * sizeof(double complex));

    // L[0,0,0] = 1 for left boundary (identity)
    // For general: L[l,b,l'] = delta_{l,l'} * delta_{b,0}
    for (uint32_t l = 0; l < chi; l++) {
        L->data[l * mpo_bond * chi + 0 * chi + l] = 1.0;
    }

    return L;
}

/**
 * @brief Initialize right boundary environment
 */
static tensor_t *create_right_boundary(uint32_t chi, uint32_t mpo_bond) {
    uint32_t dims[3] = {chi, mpo_bond, chi};
    tensor_t *R = tensor_create(3, dims);
    if (!R) return NULL;

    memset(R->data, 0, R->total_size * sizeof(double complex));

    // R[r,b,r'] = delta_{r,r'} * delta_{b,mpo_bond-1}
    for (uint32_t r = 0; r < chi; r++) {
        R->data[r * mpo_bond * chi + (mpo_bond - 1) * chi + r] = 1.0;
    }

    return R;
}

int dmrg_init_right_environments(dmrg_environments_t *env,
                                  const tn_mps_state_t *mps,
                                  const mpo_t *mpo) {
    if (!env || !mps || !mpo) return -1;

    uint32_t n = mps->num_qubits;

    // R[n-1] is the right boundary
    uint32_t chi_r = mps->tensors[n - 1]->dims[2];
    uint32_t b_r = mpo->tensors[n - 1].bond_dim_right;

    if (env->R[n - 1]) tensor_free(env->R[n - 1]);
    env->R[n - 1] = create_right_boundary(chi_r, b_r);
    if (!env->R[n - 1]) return -1;

    // Build R[i] from R[i+1] by contracting with A[i+1] and W[i+1]
    for (int i = n - 2; i >= 0; i--) {
        tensor_t *A = mps->tensors[i + 1];
        if (!A || !A->data) return -1;

        mpo_tensor_t *W = &mpo->tensors[i + 1];
        tensor_t *R_next = env->R[i + 1];
        if (!R_next || !R_next->data) return -1;

        uint32_t chi_l = A->dims[0];
        uint32_t d = A->dims[1];
        uint32_t chi_r_old = A->dims[2];
        uint32_t b_l = W->bond_dim_left;
        uint32_t b_r_old = W->bond_dim_right;

        // R[i] = contract A^* @ W @ A @ R[i+1]
        uint32_t new_dims[3] = {chi_l, b_l, chi_l};
        tensor_t *R_new = tensor_create(3, new_dims);
        if (!R_new) return -1;

        memset(R_new->data, 0, R_new->total_size * sizeof(double complex));

        for (uint32_t l = 0; l < chi_l; l++) {
            for (uint32_t bl = 0; bl < b_l; bl++) {
                for (uint32_t lp = 0; lp < chi_l; lp++) {
                    double complex sum = 0.0;
                    for (uint32_t r = 0; r < chi_r_old; r++) {
                        for (uint32_t s = 0; s < d; s++) {
                            for (uint32_t sp = 0; sp < d; sp++) {
                                for (uint32_t rp = 0; rp < chi_r_old; rp++) {
                                    for (uint32_t br = 0; br < b_r_old; br++) {
                                        uint64_t A_idx = l * d * chi_r_old + s * chi_r_old + r;
                                        uint64_t Ap_idx = lp * d * chi_r_old + sp * chi_r_old + rp;
                                        uint64_t W_idx = bl * d * d * b_r_old + s * d * b_r_old + sp * b_r_old + br;
                                        uint64_t R_idx = r * b_r_old * chi_r_old + br * chi_r_old + rp;
                                        sum += conj(A->data[A_idx]) * W->W->data[W_idx] *
                                               A->data[Ap_idx] * R_next->data[R_idx];
                                    }
                                }
                            }
                        }
                    }
                    R_new->data[l * b_l * chi_l + bl * chi_l + lp] = sum;
                }
            }
        }

        if (env->R[i]) tensor_free(env->R[i]);
        env->R[i] = R_new;
    }

    return 0;
}

int dmrg_init_left_environments(dmrg_environments_t *env,
                                 const tn_mps_state_t *mps,
                                 const mpo_t *mpo) {
    if (!env || !mps || !mpo) return -1;

    uint32_t n = mps->num_qubits;

    // L[0] is the left boundary
    uint32_t chi_l = mps->tensors[0]->dims[0];  // Should be 1 for left boundary
    uint32_t b_l = mpo->tensors[0].bond_dim_left;

    if (env->L[0]) tensor_free(env->L[0]);
    env->L[0] = create_left_boundary(chi_l, b_l);
    if (!env->L[0]) return -1;

    // Build L[i] from L[i-1] by contracting with A[i-1] and W[i-1]
    for (uint32_t i = 1; i < n; i++) {
        tensor_t *A = mps->tensors[i - 1];  // [chi_l][d][chi_r]
        mpo_tensor_t *W = &mpo->tensors[i - 1];  // [b_l][d][d][b_r]
        tensor_t *L_prev = env->L[i - 1];  // [chi_l'][b_l][chi_l']

        uint32_t chi_l_old = A->dims[0];
        uint32_t d = A->dims[1];
        uint32_t chi_r = A->dims[2];
        uint32_t b_l_old = W->bond_dim_left;
        uint32_t b_r = W->bond_dim_right;

        // L[i] = contract L[i-1] @ A^* @ W @ A
        // L[i][r, b_r, r'] = sum_{l,s,s',l',b_l}
        //   conj(A[l,s,r]) * L[i-1][l,b_l,l'] * W[b_l,s,s',b_r] * A[l',s',r']

        uint32_t new_dims[3] = {chi_r, b_r, chi_r};
        tensor_t *L_new = tensor_create(3, new_dims);
        if (!L_new) return -1;

        memset(L_new->data, 0, L_new->total_size * sizeof(double complex));

        for (uint32_t r = 0; r < chi_r; r++) {
            for (uint32_t br = 0; br < b_r; br++) {
                for (uint32_t rp = 0; rp < chi_r; rp++) {
                    double complex sum = 0.0;

                    for (uint32_t l = 0; l < chi_l_old; l++) {
                        for (uint32_t s = 0; s < d; s++) {
                            for (uint32_t sp = 0; sp < d; sp++) {
                                for (uint32_t lp = 0; lp < chi_l_old; lp++) {
                                    for (uint32_t bl = 0; bl < b_l_old; bl++) {
                                        uint64_t A_idx = l * d * chi_r + s * chi_r + r;
                                        uint64_t Ap_idx = lp * d * chi_r + sp * chi_r + rp;
                                        uint64_t L_idx = l * b_l_old * chi_l_old + bl * chi_l_old + lp;
                                        uint64_t W_idx = bl * d * d * b_r + s * d * b_r + sp * b_r + br;

                                        sum += conj(A->data[A_idx]) * L_prev->data[L_idx] *
                                               W->W->data[W_idx] * A->data[Ap_idx];
                                    }
                                }
                            }
                        }
                    }

                    L_new->data[r * b_r * chi_r + br * chi_r + rp] = sum;
                }
            }
        }

        if (env->L[i]) tensor_free(env->L[i]);
        env->L[i] = L_new;
    }

    return 0;
}

int dmrg_update_left_environment(dmrg_environments_t *env,
                                  const tn_mps_state_t *mps,
                                  const mpo_t *mpo,
                                  uint32_t site) {
    // Similar to init_right but going left to right
    // L[site+1] = contract L[site] @ A[site] @ W[site]

    if (!env || !mps || !mpo || site >= mps->num_qubits - 1) return -1;
    if (!env->L[site]) {
        fprintf(stderr, "DMRG: L[%u] is NULL in update_left\n", site);
        return -1;
    }

    tensor_t *L_prev = env->L[site];
    tensor_t *A = mps->tensors[site];
    mpo_tensor_t *W = &mpo->tensors[site];

    uint32_t chi_l = A->dims[0];

    // L_prev should have dimensions [chi_l][b_l][chi_l]
    if (L_prev->dims[0] != chi_l || L_prev->dims[2] != chi_l) {
        fprintf(stderr, "DMRG: L[%u] dimension mismatch: L has [%u,%u,%u], expected chi_l=%u\n",
                site, L_prev->dims[0], L_prev->dims[1], L_prev->dims[2], chi_l);
        return 0;  // Skip update
    }
    uint32_t d = A->dims[1];
    uint32_t chi_r = A->dims[2];
    uint32_t b_l = W->bond_dim_left;
    uint32_t b_r = W->bond_dim_right;

    uint32_t new_dims[3] = {chi_r, b_r, chi_r};
    tensor_t *L_new = tensor_create(3, new_dims);
    if (!L_new) return -1;

    memset(L_new->data, 0, L_new->total_size * sizeof(double complex));

    for (uint32_t r = 0; r < chi_r; r++) {
        for (uint32_t br = 0; br < b_r; br++) {
            for (uint32_t rp = 0; rp < chi_r; rp++) {
                double complex sum = 0.0;

                for (uint32_t l = 0; l < chi_l; l++) {
                    for (uint32_t s = 0; s < d; s++) {
                        for (uint32_t sp = 0; sp < d; sp++) {
                            for (uint32_t lp = 0; lp < chi_l; lp++) {
                                for (uint32_t bl = 0; bl < b_l; bl++) {
                                    uint64_t L_idx = l * b_l * chi_l + bl * chi_l + lp;
                                    uint64_t A_idx = l * d * chi_r + s * chi_r + r;
                                    uint64_t Ap_idx = lp * d * chi_r + sp * chi_r + rp;
                                    uint64_t W_idx = bl * d * d * b_r + s * d * b_r + sp * b_r + br;

                                    sum += conj(A->data[A_idx]) * L_prev->data[L_idx] *
                                           W->W->data[W_idx] * A->data[Ap_idx];
                                }
                            }
                        }
                    }
                }

                L_new->data[r * b_r * chi_r + br * chi_r + rp] = sum;
            }
        }
    }

    if (env->L[site + 1]) tensor_free(env->L[site + 1]);
    env->L[site + 1] = L_new;

    return 0;
}

int dmrg_update_right_environment(dmrg_environments_t *env,
                                   const tn_mps_state_t *mps,
                                   const mpo_t *mpo,
                                   uint32_t site) {
    // Update R[site-1] after optimizing at site
    if (!env || !mps || !mpo || site == 0) return -1;
    if (!env->R[site]) {
        fprintf(stderr, "DMRG: R[%u] is NULL in update_right\n", site);
        return -1;
    }

    // Similar contraction as init_right_environments but for single site
    tensor_t *A = mps->tensors[site];
    mpo_tensor_t *W = &mpo->tensors[site];
    tensor_t *R_next = env->R[site];

    // Debug dimension check
    uint32_t chi_l = A->dims[0];
    uint32_t d = A->dims[1];
    uint32_t chi_r = A->dims[2];

    // R_next should have dimensions [chi_r][b_r][chi_r]
    if (R_next->dims[0] != chi_r || R_next->dims[2] != chi_r) {
        // Dimension mismatch after SVD bond dimension change
        // Rebuild R environments from site to boundary
        uint32_t n = mps->num_qubits;
        uint32_t b_r = mpo->tensors[n - 1].bond_dim_right;

        // Start from rightmost site with identity
        tensor_free(env->R[n - 1]);
        uint32_t A_chi = mps->tensors[n - 1]->dims[2];
        uint32_t dims_R[3] = {A_chi, b_r, A_chi};
        env->R[n - 1] = tensor_create(3, dims_R);
        if (!env->R[n - 1]) return -1;
        memset(env->R[n - 1]->data, 0, env->R[n - 1]->total_size * sizeof(double complex));
        env->R[n - 1]->data[0] = 1.0;  // Identity at boundary

        // Contract from right to site
        for (int s = (int)n - 2; s >= (int)site; s--) {
            tensor_t *As = mps->tensors[s];
            mpo_tensor_t *Ws = &mpo->tensors[s];
            tensor_t *Rs_next = env->R[s + 1];

            uint32_t chi_ls = As->dims[0];
            uint32_t ds = As->dims[1];
            uint32_t chi_rs = As->dims[2];
            uint32_t b_ls = Ws->bond_dim_left;
            uint32_t b_rs = Ws->bond_dim_right;

            uint32_t new_dims[3] = {chi_ls, b_ls, chi_ls};
            tensor_t *R_new = tensor_create(3, new_dims);
            if (!R_new) return -1;
            memset(R_new->data, 0, R_new->total_size * sizeof(double complex));

            for (uint32_t l = 0; l < chi_ls; l++) {
                for (uint32_t bl = 0; bl < b_ls; bl++) {
                    for (uint32_t lp = 0; lp < chi_ls; lp++) {
                        double complex sum = 0.0;
                        for (uint32_t r = 0; r < chi_rs; r++) {
                            for (uint32_t ss = 0; ss < ds; ss++) {
                                for (uint32_t sp = 0; sp < ds; sp++) {
                                    for (uint32_t rp = 0; rp < chi_rs; rp++) {
                                        for (uint32_t br = 0; br < b_rs; br++) {
                                            if (r >= Rs_next->dims[0] || rp >= Rs_next->dims[2]) continue;
                                            uint64_t A_idx = l * ds * chi_rs + ss * chi_rs + r;
                                            uint64_t Ap_idx = lp * ds * chi_rs + sp * chi_rs + rp;
                                            uint64_t W_idx = bl * ds * ds * b_rs + ss * ds * b_rs + sp * b_rs + br;
                                            uint64_t R_idx = r * b_rs * chi_rs + br * chi_rs + rp;
                                            if (R_idx < Rs_next->total_size && W_idx < Ws->W->total_size) {
                                                sum += conj(As->data[A_idx]) * Ws->W->data[W_idx] *
                                                       As->data[Ap_idx] * Rs_next->data[R_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        R_new->data[l * b_ls * chi_ls + bl * chi_ls + lp] = sum;
                    }
                }
            }

            tensor_free(env->R[s]);
            env->R[s] = R_new;
        }

        // Now R_next should be correct - get updated reference
        R_next = env->R[site];
        chi_r = mps->tensors[site]->dims[2];
    }

    uint32_t b_l = W->bond_dim_left;
    uint32_t b_r = W->bond_dim_right;

    uint32_t new_dims[3] = {chi_l, b_l, chi_l};
    tensor_t *R_new = tensor_create(3, new_dims);
    if (!R_new) return -1;

    memset(R_new->data, 0, R_new->total_size * sizeof(double complex));

    for (uint32_t l = 0; l < chi_l; l++) {
        for (uint32_t bl = 0; bl < b_l; bl++) {
            for (uint32_t lp = 0; lp < chi_l; lp++) {
                double complex sum = 0.0;

                for (uint32_t r = 0; r < chi_r; r++) {
                    for (uint32_t s = 0; s < d; s++) {
                        for (uint32_t sp = 0; sp < d; sp++) {
                            for (uint32_t rp = 0; rp < chi_r; rp++) {
                                for (uint32_t br = 0; br < b_r; br++) {
                                    uint64_t A_idx = l * d * chi_r + s * chi_r + r;
                                    uint64_t Ap_idx = lp * d * chi_r + sp * chi_r + rp;
                                    uint64_t W_idx = bl * d * d * b_r + s * d * b_r + sp * b_r + br;
                                    uint64_t R_idx = r * b_r * chi_r + br * chi_r + rp;

                                    sum += conj(A->data[A_idx]) * W->W->data[W_idx] *
                                           A->data[Ap_idx] * R_next->data[R_idx];
                                }
                            }
                        }
                    }
                }

                R_new->data[l * b_l * chi_l + bl * chi_l + lp] = sum;
            }
        }
    }

    if (env->R[site - 1]) tensor_free(env->R[site - 1]);
    env->R[site - 1] = R_new;

    return 0;
}

// ============================================================================
// DMRG CORE ALGORITHM
// ============================================================================

int dmrg_optimize_two_site(tn_mps_state_t *mps,
                            const mpo_t *mpo,
                            dmrg_environments_t *env,
                            uint32_t site,
                            dmrg_sweep_direction_t direction,
                            const dmrg_config_t *config,
                            double *energy) {
    if (!mps || !mpo || !env || !config || !energy) return -1;
    if (site >= mps->num_qubits - 1) return -1;

    tensor_t *A = mps->tensors[site];      // [chi_l][d][chi_m]
    tensor_t *B = mps->tensors[site + 1];  // [chi_m][d][chi_r]

    if (!A || !B) {
        fprintf(stderr, "DMRG: NULL tensor at site %u or %u\n", site, site + 1);
        return -1;
    }

    uint32_t chi_l = A->dims[0];
    uint32_t d = A->dims[1];
    uint32_t chi_m = A->dims[2];
    uint32_t chi_r = B->dims[2];

    // Validate environments
    if (!env->L[site]) {
        fprintf(stderr, "DMRG: L[%u] is NULL\n", site);
        return -1;
    }
    if (!env->L[site]->data) {
        fprintf(stderr, "DMRG: L[%u]->data is NULL\n", site);
        return -1;
    }
    if (!env->R[site + 1]) {
        fprintf(stderr, "DMRG: R[%u] is NULL\n", site + 1);
        return -1;
    }
    if (!env->R[site + 1]->data) {
        fprintf(stderr, "DMRG: R[%u]->data is NULL (tensor exists but data=%p)\n",
                site + 1, (void*)env->R[site + 1]->data);
        fprintf(stderr, "      R[%u] dims=[%u,%u,%u]\n", site + 1,
                env->R[site + 1]->dims[0], env->R[site + 1]->dims[1], env->R[site + 1]->dims[2]);
        return -1;
    }

    // Check dimension compatibility
    if (env->L[site]->dims[0] != chi_l || env->L[site]->dims[2] != chi_l) {
        fprintf(stderr, "DMRG site %u: L dim mismatch: L=[%u,%u,%u] but chi_l=%u\n",
                site, env->L[site]->dims[0], env->L[site]->dims[1],
                env->L[site]->dims[2], chi_l);
        return -1;
    }
    if (env->R[site + 1]->dims[0] != chi_r || env->R[site + 1]->dims[2] != chi_r) {
        fprintf(stderr, "DMRG site %u: R dim mismatch: R=[%u,%u,%u] but chi_r=%u\n",
                site, env->R[site + 1]->dims[0], env->R[site + 1]->dims[1],
                env->R[site + 1]->dims[2], chi_r);
        return -1;
    }

    // Build effective Hamiltonian
    if (!mpo->tensors[site].W || !mpo->tensors[site + 1].W) {
        return -1;
    }
    effective_hamiltonian_t H_eff = {
        .L = env->L[site],
        .R = env->R[site + 1],
        .W_left = &mpo->tensors[site],
        .W_right = &mpo->tensors[site + 1],
        .chi_l = chi_l,
        .chi_r = chi_r,
        .phys_dim = d,
        .two_site = true
    };

    // Form initial theta from A @ B
    uint32_t theta_dims[4] = {chi_l, d, d, chi_r};
    tensor_t *theta = tensor_create(4, theta_dims);
    if (!theta) return -1;

    // theta[l,s1,s2,r] = sum_m A[l,s1,m] * B[m,s2,r]
    if (!A->data || !B->data) {
        tensor_free(theta);
        return -1;
    }

    for (uint32_t l = 0; l < chi_l; l++) {
        for (uint32_t s1 = 0; s1 < d; s1++) {
            for (uint32_t s2 = 0; s2 < d; s2++) {
                for (uint32_t r = 0; r < chi_r; r++) {
                    double complex sum = 0.0;
                    for (uint32_t m = 0; m < chi_m; m++) {
                        uint64_t A_idx = l * d * chi_m + s1 * chi_m + m;
                        uint64_t B_idx = m * d * chi_r + s2 * chi_r + r;
                        sum += A->data[A_idx] * B->data[B_idx];
                    }
                    uint64_t theta_idx = l * d * d * chi_r + s1 * d * chi_r + s2 * chi_r + r;
                    theta->data[theta_idx] = sum;
                }
            }
        }
    }

    // Run Lanczos to find ground state
    lanczos_result_t *lanczos = lanczos_ground_state(&H_eff, theta,
                                                       config->lanczos_max_iter,
                                                       config->lanczos_tol);
    tensor_free(theta);

    if (!lanczos || !lanczos->eigenvector) {
        if (lanczos) lanczos_result_free(lanczos);
        return -1;
    }

    *energy = lanczos->eigenvalue;

    // SVD the optimized theta to get new A and B
    uint32_t mat_dims[2] = {chi_l * d, d * chi_r};
    tensor_t *mat = tensor_reshape(lanczos->eigenvector, 2, mat_dims);
    if (!mat) {
        lanczos_result_free(lanczos);
        return -1;
    }

    // ============================================================
    // SUBSPACE EXPANSION: Add noise to enable bond dimension growth
    // ============================================================
    // In two-site DMRG, the SVD can only produce bonds up to min(chi_l*d, d*chi_r).
    // Adding small noise before SVD allows exploration of new directions in
    // Hilbert space that would otherwise be missed.
    //
    // Key insight: The noise "seeds" small components in directions orthogonal
    // to the current wavefunction. During optimization, if these directions
    // lower the energy, they grow; otherwise they get truncated.
    // ============================================================
    if (config->noise_strength > 0.0) {
        // Compute norm of the wavefunction for scaling
        double wf_norm = 0.0;
        for (uint64_t i = 0; i < mat->total_size; i++) {
            wf_norm += cabs(mat->data[i]) * cabs(mat->data[i]);
        }
        wf_norm = sqrt(wf_norm);

        // Scale noise relative to wavefunction norm
        double noise_scale = config->noise_strength * wf_norm;

        // Add random noise to each element
        // Use reproducible but varied seed based on site position
        unsigned int local_seed = 12345 + site * 97;
        for (uint64_t i = 0; i < mat->total_size; i++) {
            // Linear congruential generator for reproducibility
            local_seed = local_seed * 1103515245 + 12345;
            double r1 = ((double)((local_seed >> 16) & 0x7FFF) / 32767.0 - 0.5) * 2.0;
            local_seed = local_seed * 1103515245 + 12345;
            double r2 = ((double)((local_seed >> 16) & 0x7FFF) / 32767.0 - 0.5) * 2.0;

            mat->data[i] += noise_scale * (r1 + I * r2);
        }

        // Renormalize to preserve wavefunction norm
        double new_norm = 0.0;
        for (uint64_t i = 0; i < mat->total_size; i++) {
            new_norm += cabs(mat->data[i]) * cabs(mat->data[i]);
        }
        new_norm = sqrt(new_norm);

        if (new_norm > 1e-15) {
            double scale = wf_norm / new_norm;
            for (uint64_t i = 0; i < mat->total_size; i++) {
                mat->data[i] *= scale;
            }
        }
    }

    // SVD with truncation
    svd_compress_config_t svd_cfg = svd_compress_config_default();
    svd_cfg.max_bond_dim = config->max_bond_dim;
    svd_cfg.cutoff = config->svd_cutoff;

    svd_compress_result_t *svd = svd_compress(mat, &svd_cfg);
    tensor_free(mat);
    lanczos_result_free(lanczos);

    if (!svd) return -1;

    uint32_t new_bond = svd->bond_dim;

    // ============================================================
    // DENSITY MATRIX PERTURBATION: Boost smaller singular values
    // ============================================================
    // The density matrix eigenvalues are σ_i². By boosting smaller σ_i,
    // we effectively mix in components that would otherwise be truncated.
    // This helps DMRG escape local minima by maintaining some weight
    // in the "discarded" space.
    //
    // Method: σ_i → σ_i + ε * σ_max
    // This adds a small uniform boost proportional to the largest singular value.
    // ============================================================
    if (config->dm_perturbation > 0.0 && new_bond > 0) {
        double s_max = svd->singular_values[0];  // Largest singular value

        // Skip perturbation if singular values are too small
        if (s_max > 1e-15) {
            double pert = config->dm_perturbation * s_max;

            // Boost all singular values
            for (uint32_t i = 0; i < new_bond; i++) {
                svd->singular_values[i] += pert;
            }

            // Renormalize to preserve the total weight (trace of density matrix)
            // Original: sum(σ_i²), after perturbation we want to preserve this
            double sum_sq_orig = 0.0;
            double sum_sq_new = 0.0;

            // Compute sums
            for (uint32_t i = 0; i < new_bond; i++) {
                double s_new = svd->singular_values[i];
                double s_orig = s_new - pert;  // Original before perturbation
                sum_sq_orig += s_orig * s_orig;
                sum_sq_new += s_new * s_new;
            }

            // Scale to preserve norm
            if (sum_sq_new > 1e-15 && sum_sq_orig > 1e-15) {
                double scale = sqrt(sum_sq_orig / sum_sq_new);
                for (uint32_t i = 0; i < new_bond; i++) {
                    svd->singular_values[i] *= scale;
                }
            }
        }
    }

    // Absorb singular values based on sweep direction for proper canonical form
    if (direction == DMRG_SWEEP_LEFT_TO_RIGHT) {
        // L->R sweep: absorb S into right tensor for left-canonical form
        // A = U, B = S @ V^†
        for (uint32_t i = 0; i < new_bond; i++) {
            for (uint32_t j = 0; j < svd->right->dims[1]; j++) {
                svd->right->data[i * svd->right->dims[1] + j] *= svd->singular_values[i];
            }
        }
    } else {
        // R->L sweep: absorb S into left tensor for right-canonical form
        // A = U @ S, B = V^†
        for (uint32_t i = 0; i < svd->left->dims[0]; i++) {
            for (uint32_t j = 0; j < new_bond; j++) {
                svd->left->data[i * svd->left->dims[1] + j] *= svd->singular_values[j];
            }
        }
    }

    // Reshape to new MPS tensors
    uint32_t A_dims[3] = {chi_l, d, new_bond};
    tensor_t *A_new = tensor_reshape(svd->left, 3, A_dims);

    uint32_t B_dims[3] = {new_bond, d, chi_r};
    tensor_t *B_new = tensor_reshape(svd->right, 3, B_dims);

    if (!A_new || !B_new) {
        if (A_new) tensor_free(A_new);
        if (B_new) tensor_free(B_new);
        svd_compress_result_free(svd);
        return -1;
    }

    // Update MPS tensors
    tensor_free(mps->tensors[site]);
    tensor_free(mps->tensors[site + 1]);
    mps->tensors[site] = A_new;
    mps->tensors[site + 1] = B_new;
    mps->bond_dims[site] = new_bond;

    svd_compress_result_free(svd);
    return 0;
}

int dmrg_sweep(tn_mps_state_t *mps,
               const mpo_t *mpo,
               dmrg_environments_t *env,
               const dmrg_config_t *config,
               double *energy) {
    if (!mps || !mpo || !env || !config || !energy) return -1;

    uint32_t n = mps->num_qubits;
    double site_energy = 0.0;

    // Rebuild all environments from current MPS before L->R sweep
    if (dmrg_init_left_environments(env, mps, mpo) != 0) return -1;
    if (dmrg_init_right_environments(env, mps, mpo) != 0) return -1;

    // Left-to-right sweep
    for (uint32_t site = 0; site < n - 1; site++) {
        if (dmrg_optimize_two_site(mps, mpo, env, site, DMRG_SWEEP_LEFT_TO_RIGHT, config, &site_energy) != 0) {
            return -1;
        }
        if (site < n - 2) {
            dmrg_update_left_environment(env, mps, mpo, site);
        }
    }

    // Rebuild right environments before R->L sweep
    dmrg_init_right_environments(env, mps, mpo);

    // Right-to-left sweep
    for (int site = n - 2; site >= 0; site--) {
        if (dmrg_optimize_two_site(mps, mpo, env, site, DMRG_SWEEP_RIGHT_TO_LEFT, config, &site_energy) != 0) {
            return -1;
        }
        if (site > 0) {
            dmrg_update_right_environment(env, mps, mpo, site + 1);
        }
    }

    *energy = site_energy;
    return 0;
}

dmrg_result_t *dmrg_ground_state(tn_mps_state_t *mps,
                                  const mpo_t *mpo,
                                  const dmrg_config_t *config) {
    if (!mps || !mpo || !config) return NULL;

    double start_time = get_time_sec();

    dmrg_result_t *result = (dmrg_result_t *)calloc(1, sizeof(dmrg_result_t));
    if (!result) return NULL;

    // Compute total sweeps including warmup
    uint32_t total_sweeps = config->max_sweeps + config->warmup_sweeps;
    result->sweep_energies = (double *)calloc(total_sweeps, sizeof(double));
    if (!result->sweep_energies) {
        free(result);
        return NULL;
    }

    // Create environments
    dmrg_environments_t *env = dmrg_environments_create(mps->num_qubits);
    if (!env) {
        dmrg_result_free(result);
        return NULL;
    }

    // Initialize left boundary
    uint32_t b_l = mpo->tensors[0].bond_dim_left;
    env->L[0] = create_left_boundary(mps->tensors[0]->dims[0], b_l);
    if (!env->L[0]) {
        dmrg_environments_free(env);
        dmrg_result_free(result);
        return NULL;
    }

    // Initialize right environments
    if (dmrg_init_right_environments(env, mps, mpo) != 0) {
        dmrg_environments_free(env);
        dmrg_result_free(result);
        return NULL;
    }

    // ============================================================
    // DMRG SWEEPS with NOISE DECAY and WARMUP
    // ============================================================
    // Phase 1: Warmup sweeps with larger noise to explore bond dimension space
    // Phase 2: Production sweeps with decaying noise for precision
    // ============================================================

    // Create mutable copy of config for noise decay
    dmrg_config_t sweep_config = *config;

    double prev_energy = 1e30;
    uint32_t sweep_idx = 0;

    // ---- PHASE 1: WARMUP SWEEPS ----
    // Use larger noise and possibly limited bond dimension for rapid exploration
    if (config->warmup_sweeps > 0) {
        sweep_config.noise_strength = config->warmup_noise;

        // Temporarily limit bond dimension during warmup for speed
        uint32_t saved_max_bond = sweep_config.max_bond_dim;
        if (config->warmup_bond_dim > 0 && config->warmup_bond_dim < config->max_bond_dim) {
            sweep_config.max_bond_dim = config->warmup_bond_dim;
        }

        if (config->verbose) {
            fprintf(stderr, "=== WARMUP PHASE: %u sweeps with noise=%.2e, max_chi=%u ===\n",
                    config->warmup_sweeps, sweep_config.noise_strength, sweep_config.max_bond_dim);
        }

        for (uint32_t warmup = 0; warmup < config->warmup_sweeps; warmup++) {
            double energy;

            if (dmrg_sweep(mps, mpo, env, &sweep_config, &energy) != 0) {
                dmrg_environments_free(env);
                dmrg_result_free(result);
                return NULL;
            }

            result->sweep_energies[sweep_idx] = energy;
            sweep_idx++;

            if (config->verbose) {
                fprintf(stderr, "Warmup %u: E = %.12f\n", warmup + 1, energy);
            }

            // Decay warmup noise slightly
            sweep_config.noise_strength *= 0.7;
            prev_energy = energy;
        }

        // Restore full bond dimension for production sweeps
        sweep_config.max_bond_dim = saved_max_bond;
    }

    // ---- PHASE 2: PRODUCTION SWEEPS ----
    // Normal DMRG with noise decay for convergence
    sweep_config.noise_strength = config->noise_strength;

    if (config->verbose && config->warmup_sweeps > 0) {
        fprintf(stderr, "=== PRODUCTION PHASE: up to %u sweeps with noise=%.2e (decay=%.2f) ===\n",
                config->max_sweeps, sweep_config.noise_strength, config->noise_decay);
    }

    for (uint32_t sweep = 0; sweep < config->max_sweeps; sweep++) {
        double energy;

        if (dmrg_sweep(mps, mpo, env, &sweep_config, &energy) != 0) {
            dmrg_environments_free(env);
            dmrg_result_free(result);
            return NULL;
        }

        result->sweep_energies[sweep_idx] = energy;
        sweep_idx++;
        result->num_sweeps = sweep_idx;

        if (config->verbose) {
            fprintf(stderr, "Sweep %u: E = %.12f, dE = %.2e, noise = %.2e\n",
                    sweep + 1, energy, fabs(energy - prev_energy), sweep_config.noise_strength);
        }

        // Check convergence
        if (fabs(energy - prev_energy) < config->energy_tol) {
            result->converged = true;
            result->ground_energy = energy;
            break;
        }

        prev_energy = energy;
        result->ground_energy = energy;

        // Decay noise after each sweep
        // This allows exploration early and precision later
        if (sweep_config.noise_strength > 0 && config->noise_decay > 0) {
            sweep_config.noise_strength *= config->noise_decay;

            // Turn off noise when it gets too small
            if (sweep_config.noise_strength < 1e-12) {
                sweep_config.noise_strength = 0.0;
            }
        }
    }

    dmrg_environments_free(env);

    result->total_time = get_time_sec() - start_time;

    return result;
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

tn_mps_state_t *dmrg_tfim_ground_state(uint32_t num_sites,
                                        double g,
                                        const dmrg_config_t *config,
                                        dmrg_result_t **result) {
    // Use default config if not provided
    dmrg_config_t cfg = config ? *config : dmrg_config_default();

    // Create MPO for TFIM: H = -J sum ZZ - h sum X, with g = h/J
    // We set J = 1, h = g
    mpo_t *mpo = mpo_tfim_create(num_sites, 1.0, g);
    if (!mpo) return NULL;

    // Create initial MPS with random perturbations for variational flexibility
    // Start with larger bond dimensions to allow DMRG to explore more states
    tn_state_config_t mps_cfg = tn_state_config_default();
    mps_cfg.max_bond_dim = cfg.max_bond_dim;
    mps_cfg.svd_cutoff = cfg.svd_cutoff;

    // Initial bond dimension (smaller than max, but larger than 1)
    uint32_t chi_init = (cfg.max_bond_dim > 8) ? 8 : cfg.max_bond_dim;
    if (chi_init < 2) chi_init = 2;

    // Create MPS with chi_init bond dimension
    tn_mps_state_t *mps = (tn_mps_state_t *)calloc(1, sizeof(tn_mps_state_t));
    if (!mps) {
        mpo_free(mpo);
        return NULL;
    }
    mps->num_qubits = num_sites;
    mps->config = mps_cfg;
    mps->tensors = (tensor_t **)calloc(num_sites, sizeof(tensor_t *));
    mps->bond_dims = (uint32_t *)calloc(num_sites - 1, sizeof(uint32_t));
    if (!mps->tensors || !mps->bond_dims) {
        tn_mps_free(mps);
        mpo_free(mpo);
        return NULL;
    }

    // Create tensors with bond dimension chi_init
    // Use random initialization with small magnitude
    srand(42);  // Fixed seed for reproducibility
    for (uint32_t i = 0; i < num_sites; i++) {
        uint32_t chi_l = (i == 0) ? 1 : chi_init;
        uint32_t chi_r = (i == num_sites - 1) ? 1 : chi_init;
        uint32_t dims[3] = {chi_l, 2, chi_r};

        mps->tensors[i] = tensor_create(3, dims);
        if (!mps->tensors[i]) {
            tn_mps_free(mps);
            mpo_free(mpo);
            return NULL;
        }

        // Initialize with small random values
        for (uint64_t j = 0; j < mps->tensors[i]->total_size; j++) {
            double re = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            double im = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            mps->tensors[i]->data[j] = re + I * im;
        }

        // Add |+> component for the first bond index
        // This ensures we start near a reasonable state
        uint32_t idx0[3] = {0, 0, 0};
        uint32_t idx1[3] = {0, 1, 0};
        mps->tensors[i]->data[0] += 1.0 / sqrt(2.0);  // |0> component
        tensor_set(mps->tensors[i], idx1, tensor_get(mps->tensors[i], idx1) + 1.0 / sqrt(2.0));

        if (i < num_sites - 1) {
            mps->bond_dims[i] = chi_init;
        }
    }

    // Normalize the MPS (approximate)
    double norm = tn_mps_norm(mps);
    if (norm > 1e-10) {
        double scale = 1.0 / norm;
        for (uint32_t i = 0; i < num_sites; i++) {
            for (uint64_t j = 0; j < mps->tensors[i]->total_size; j++) {
                mps->tensors[i]->data[j] *= pow(scale, 1.0 / num_sites);
            }
        }
    }

    // Run DMRG
    dmrg_result_t *dmrg_res = dmrg_ground_state(mps, mpo, &cfg);

    mpo_free(mpo);

    if (!dmrg_res) {
        tn_mps_free(mps);
        return NULL;
    }

    if (result) {
        *result = dmrg_res;
    } else {
        dmrg_result_free(dmrg_res);
    }

    return mps;
}

double dmrg_compute_energy(const tn_mps_state_t *mps, const mpo_t *mpo) {
    if (!mps || !mpo || mps->num_qubits != mpo->num_sites) return NAN;

    // E = <psi|H|psi> computed via transfer matrix contraction
    // Similar to environment contraction

    uint32_t n = mps->num_qubits;

    // Start with left boundary
    uint32_t chi_0 = mps->tensors[0]->dims[0];  // Should be 1
    uint32_t b_0 = mpo->tensors[0].bond_dim_left;  // Should be 1

    tensor_t *T = create_left_boundary(chi_0, b_0);
    if (!T) return NAN;

    // Contract through the chain
    for (uint32_t site = 0; site < n; site++) {
        tensor_t *A = mps->tensors[site];
        mpo_tensor_t *W = &mpo->tensors[site];

        uint32_t chi_l = A->dims[0];
        uint32_t d = A->dims[1];
        uint32_t chi_r = A->dims[2];
        uint32_t b_l = W->bond_dim_left;
        uint32_t b_r = W->bond_dim_right;

        // T_new[r,b_r,r'] = sum_{l,s,s',l',b_l} T[l,b_l,l'] * conj(A[l,s,r]) * W[b_l,s,s',b_r] * A[l',s',r']
        uint32_t new_dims[3] = {chi_r, b_r, chi_r};
        tensor_t *T_new = tensor_create(3, new_dims);
        if (!T_new) {
            tensor_free(T);
            return NAN;
        }

        memset(T_new->data, 0, T_new->total_size * sizeof(double complex));

        for (uint32_t r = 0; r < chi_r; r++) {
            for (uint32_t br = 0; br < b_r; br++) {
                for (uint32_t rp = 0; rp < chi_r; rp++) {
                    double complex sum = 0.0;

                    for (uint32_t l = 0; l < chi_l; l++) {
                        for (uint32_t s = 0; s < d; s++) {
                            for (uint32_t sp = 0; sp < d; sp++) {
                                for (uint32_t lp = 0; lp < chi_l; lp++) {
                                    for (uint32_t bl = 0; bl < b_l; bl++) {
                                        uint64_t T_idx = l * b_l * chi_l + bl * chi_l + lp;
                                        uint64_t A_idx = l * d * chi_r + s * chi_r + r;
                                        uint64_t Ap_idx = lp * d * chi_r + sp * chi_r + rp;
                                        uint64_t W_idx = bl * d * d * b_r + s * d * b_r + sp * b_r + br;

                                        sum += T->data[T_idx] * conj(A->data[A_idx]) *
                                               W->W->data[W_idx] * A->data[Ap_idx];
                                    }
                                }
                            }
                        }
                    }

                    T_new->data[r * b_r * chi_r + br * chi_r + rp] = sum;
                }
            }
        }

        tensor_free(T);
        T = T_new;
    }

    // Extract energy from final contraction
    // T should be [1][1][1] for normalized MPS
    double energy = creal(T->data[0]);
    tensor_free(T);

    return energy;
}

double dmrg_energy_variance(const tn_mps_state_t *mps, const mpo_t *mpo) {
    // Var(E) = <H^2> - <H>^2
    // For a true eigenstate, Var(E) = 0
    // Computed by contracting MPS with TWO MPO layers

    if (!mps || !mpo || mps->num_qubits != mpo->num_sites) return NAN;

    // First compute <H>
    double E = dmrg_compute_energy(mps, mpo);
    if (isnan(E)) return NAN;

    // Now compute <H^2> via double-layer MPO contraction
    // Transfer tensor T has shape [chi_l, b_l, b_l, chi_l'] for double MPO layer
    // This contracts: <psi| H H |psi>

    uint32_t n = mps->num_qubits;

    // Initial left boundary: T[chi=1, b=1, b'=1, chi'=1] = 1
    uint32_t chi_0 = mps->tensors[0]->dims[0];  // Should be 1
    uint32_t b_0 = mpo->tensors[0].bond_dim_left;  // Should be 1

    // 4D transfer tensor: [chi_l, b_l, b_l, chi_l']
    uint32_t T_dims[4] = {chi_0, b_0, b_0, chi_0};
    tensor_t *T = tensor_create(4, T_dims);
    if (!T) return NAN;

    memset(T->data, 0, T->total_size * sizeof(double complex));
    T->data[0] = 1.0;  // Left boundary identity

    // Contract through the chain
    for (uint32_t site = 0; site < n; site++) {
        tensor_t *A = mps->tensors[site];
        mpo_tensor_t *W = &mpo->tensors[site];

        uint32_t chi_l = A->dims[0];
        uint32_t d = A->dims[1];
        uint32_t chi_r = A->dims[2];
        uint32_t b_l = W->bond_dim_left;
        uint32_t b_r = W->bond_dim_right;

        // New transfer tensor: [chi_r, b_r, b_r, chi_r]
        uint32_t new_dims[4] = {chi_r, b_r, b_r, chi_r};
        tensor_t *T_new = tensor_create(4, new_dims);
        if (!T_new) {
            tensor_free(T);
            return NAN;
        }

        memset(T_new->data, 0, T_new->total_size * sizeof(double complex));

        // Contract: T_new[r, br1, br2, rp] =
        //   sum_{l,s,m,t,lp,bl1,bl2} T[l, bl1, bl2, lp]
        //     * conj(A[l,s,r]) * W[bl1,s,m,br1] * W[bl2,m,t,br2] * A[lp,t,rp]
        //
        // This computes <psi| H(1) H(2) |psi> where:
        //   s = bra physical index
        //   m = intermediate physical index (contracted between two H's)
        //   t = ket physical index

        for (uint32_t r = 0; r < chi_r; r++) {
            for (uint32_t br1 = 0; br1 < b_r; br1++) {
                for (uint32_t br2 = 0; br2 < b_r; br2++) {
                    for (uint32_t rp = 0; rp < chi_r; rp++) {
                        double complex sum = 0.0;

                        for (uint32_t l = 0; l < chi_l; l++) {
                            for (uint32_t lp = 0; lp < chi_l; lp++) {
                                for (uint32_t bl1 = 0; bl1 < b_l; bl1++) {
                                    for (uint32_t bl2 = 0; bl2 < b_l; bl2++) {
                                        // T[l, bl1, bl2, lp]
                                        uint64_t T_idx = l * b_l * b_l * chi_l +
                                                         bl1 * b_l * chi_l +
                                                         bl2 * chi_l + lp;
                                        double complex T_val = T->data[T_idx];
                                        if (cabs(T_val) < 1e-15) continue;

                                        for (uint32_t s = 0; s < d; s++) {
                                            for (uint32_t m = 0; m < d; m++) {
                                                for (uint32_t t = 0; t < d; t++) {
                                                    // A[l,s,r] (bra, conjugated)
                                                    uint64_t A_bra_idx = l * d * chi_r + s * chi_r + r;
                                                    // W[bl1,s,m,br1] (first H layer)
                                                    uint64_t W1_idx = bl1 * d * d * b_r + s * d * b_r + m * b_r + br1;
                                                    // W[bl2,m,t,br2] (second H layer)
                                                    uint64_t W2_idx = bl2 * d * d * b_r + m * d * b_r + t * b_r + br2;
                                                    // A[lp,t,rp] (ket)
                                                    uint64_t A_ket_idx = lp * d * chi_r + t * chi_r + rp;

                                                    sum += T_val *
                                                           conj(A->data[A_bra_idx]) *
                                                           W->W->data[W1_idx] *
                                                           W->W->data[W2_idx] *
                                                           A->data[A_ket_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        T_new->data[r * b_r * b_r * chi_r + br1 * b_r * chi_r + br2 * chi_r + rp] = sum;
                    }
                }
            }
        }

        tensor_free(T);
        T = T_new;
    }

    // Extract <H^2> from final contraction (should be [1,1,1,1])
    double H2 = creal(T->data[0]);
    tensor_free(T);

    // Variance = <H^2> - <H>^2
    double variance = H2 - E * E;

    // Handle numerical precision issues (variance can't be negative)
    if (variance < 0.0) {
        if (variance > -1e-10) {
            variance = 0.0;  // Small negative due to precision
        }
        // Otherwise return the value (might indicate numerical issues)
    }

    return variance;
}
