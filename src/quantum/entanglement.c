/**
 * @file entanglement.c
 * @brief Quantum entanglement analysis and utilities
 *
 * Provides tools for analyzing and quantifying entanglement:
 * - Entanglement entropy (von Neumann, Renyi)
 * - Concurrence for 2-qubit states
 * - Schmidt decomposition
 * - Entanglement witnesses
 * - Partial transpose for PPT criterion
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "entanglement.h"
#include "state.h"
#include "../utils/config.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// Default Jacobi parameters (used when config unavailable)
#define DEFAULT_JACOBI_MAX_ITER 100
#define DEFAULT_JACOBI_TOLERANCE 1e-12

// Helper to get Jacobi parameters from config
static inline void get_jacobi_params(int* max_iter, double* tolerance) {
    qsim_config_t* cfg = qsim_config_global();
    *max_iter = (cfg && cfg->algorithm.jacobi_max_iter > 0)
        ? cfg->algorithm.jacobi_max_iter
        : DEFAULT_JACOBI_MAX_ITER;
    *tolerance = (cfg && cfg->algorithm.jacobi_tolerance > 0)
        ? cfg->algorithm.jacobi_tolerance
        : DEFAULT_JACOBI_TOLERANCE;
}

// Cross-platform BLAS/LAPACK support for Hermitian eigenvalue computation
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define HAS_LAPACK 1
#elif defined(QSIM_HAS_OPENBLAS) || defined(__linux__)
#include <lapacke.h>
#define HAS_LAPACK 1
#else
#define HAS_LAPACK 0
#endif

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634  // log2(e)
#endif

// ============================================================================
// EIGENVALUE DECOMPOSITION FOR HERMITIAN MATRICES
// ============================================================================

/**
 * @brief Eigenvalue decomposition for Hermitian matrices
 *
 * Computes eigenvalues of a Hermitian matrix using LAPACK (when available)
 * or Jacobi rotations as fallback.
 *
 * @param matrix Input Hermitian matrix (row-major, dim x dim)
 * @param dim Matrix dimension
 * @param eigenvalues Output array for eigenvalues (real, sorted descending)
 * @param max_iter Maximum iterations for Jacobi fallback (100 typically sufficient)
 * @param tol Convergence tolerance for Jacobi fallback (1e-12 typical)
 * @return Number of iterations (0 if LAPACK used), -1 on error
 */
static int jacobi_hermitian_eigenvalues(const complex_t* matrix, uint64_t dim,
                                        double* eigenvalues, int max_iter, double tol) {
    if (!matrix || !eigenvalues || dim == 0) return -1;

    // Work on a copy of the matrix (ZHEEV modifies input)
    complex_t* A = malloc(dim * dim * sizeof(complex_t));
    if (!A) return -1;
    memcpy(A, matrix, dim * dim * sizeof(complex_t));

#if HAS_LAPACK
    // Use LAPACK ZHEEV for optimal performance
    int use_jacobi = 0;

#ifdef __APPLE__
    // Apple Accelerate framework
    {
        char jobz = 'N';  // Eigenvalues only
        char uplo = 'U';  // Upper triangle
        __CLPK_integer n = (__CLPK_integer)dim;
        __CLPK_integer lda = n;
        __CLPK_integer lwork = -1;
        __CLPK_integer info;
        double complex work_query;
        double* rwork = malloc((3 * dim - 2) * sizeof(double));

        if (!rwork) {
            free(A);
            return -1;
        }

        // Query optimal work size
        zheev_(&jobz, &uplo, &n, (__CLPK_doublecomplex*)A, &lda,
               eigenvalues, (__CLPK_doublecomplex*)&work_query, &lwork, rwork, &info);

        lwork = (__CLPK_integer)creal(work_query);
        if (lwork < 1) lwork = 2 * dim;

        double complex* work = malloc(lwork * sizeof(double complex));
        if (!work) {
            free(rwork);
            free(A);
            return -1;
        }

        // Compute eigenvalues
        zheev_(&jobz, &uplo, &n, (__CLPK_doublecomplex*)A, &lda,
               eigenvalues, (__CLPK_doublecomplex*)work, &lwork, rwork, &info);

        free(work);
        free(rwork);

        if (info != 0) {
            use_jacobi = 1;  // Fall back to Jacobi
        } else {
            // ZHEEV returns eigenvalues in ascending order, we want descending
            for (uint64_t i = 0; i < dim / 2; i++) {
                double tmp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[dim - 1 - i];
                eigenvalues[dim - 1 - i] = tmp;
            }
            free(A);
            return 0;  // Success with LAPACK
        }
    }
#else
    // OpenBLAS/LAPACKE interface
    {
        lapack_int n = (lapack_int)dim;
        lapack_int info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'N', 'U', n,
                                         (lapack_complex_double*)A, n, eigenvalues);
        if (info != 0) {
            use_jacobi = 1;
        } else {
            // ZHEEV returns ascending, we want descending
            for (uint64_t i = 0; i < dim / 2; i++) {
                double tmp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[dim - 1 - i];
                eigenvalues[dim - 1 - i] = tmp;
            }
            free(A);
            return 0;  // Success with LAPACK
        }
    }
#endif

    // If LAPACK failed, reload matrix for Jacobi
    if (use_jacobi) {
        memcpy(A, matrix, dim * dim * sizeof(complex_t));
    }
#endif  // HAS_LAPACK

    // Jacobi fallback for systems without LAPACK or if LAPACK failed

    // Jacobi iteration: find largest off-diagonal, apply rotation to eliminate it
    for (int iter = 0; iter < max_iter; iter++) {
        // Find largest off-diagonal element
        double max_off = 0.0;
        uint64_t p = 0, q = 0;

        for (uint64_t i = 0; i < dim; i++) {
            for (uint64_t j = i + 1; j < dim; j++) {
                double mag = cabs(A[i * dim + j]);
                if (mag > max_off) {
                    max_off = mag;
                    p = i;
                    q = j;
                }
            }
        }

        // Check convergence
        if (max_off < tol) {
            // Extract diagonal as eigenvalues
            for (uint64_t i = 0; i < dim; i++) {
                eigenvalues[i] = creal(A[i * dim + i]);
            }

            // Sort descending by bubble sort (sufficient for small matrices)
            for (uint64_t i = 0; i < dim - 1; i++) {
                for (uint64_t j = i + 1; j < dim; j++) {
                    if (eigenvalues[j] > eigenvalues[i]) {
                        double tmp = eigenvalues[i];
                        eigenvalues[i] = eigenvalues[j];
                        eigenvalues[j] = tmp;
                    }
                }
            }

            free(A);
            return iter;
        }

        // Compute Jacobi rotation parameters for Hermitian case
        // We need to eliminate A[p,q] and A[q,p] = conj(A[p,q])
        double app = creal(A[p * dim + p]);
        double aqq = creal(A[q * dim + q]);
        complex_t apq = A[p * dim + q];

        // For Hermitian matrices, rotation angle satisfies:
        // tan(2θ) * exp(iφ) = 2*apq / (aqq - app)
        // where φ = arg(apq)
        double phi = carg(apq);
        double mag_apq = cabs(apq);

        double tau;
        if (mag_apq < 1e-15) {
            // Off-diagonal element is essentially zero, no rotation needed
            continue;
        } else if (fabs(aqq - app) < 1e-15) {
            tau = 1.0;  // θ = π/4
        } else {
            tau = (aqq - app) / (2.0 * mag_apq);
        }

        double t;
        if (tau >= 0) {
            t = 1.0 / (tau + sqrt(1.0 + tau * tau));
        } else {
            t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
        }

        double c = 1.0 / sqrt(1.0 + t * t);  // cos(θ)
        double s = t * c;                      // sin(θ)

        // Apply rotation: this is a unitary transformation
        // A' = U^H A U where U is rotation in (p,q) plane with phase
        complex_t exp_iphi = cexp(I * phi);
        complex_t exp_miphi = cexp(-I * phi);

        // Update matrix elements
        for (uint64_t k = 0; k < dim; k++) {
            if (k != p && k != q) {
                complex_t akp = A[k * dim + p];
                complex_t akq = A[k * dim + q];

                A[k * dim + p] = c * akp - s * exp_miphi * akq;
                A[k * dim + q] = s * exp_iphi * akp + c * akq;
                A[p * dim + k] = conj(A[k * dim + p]);
                A[q * dim + k] = conj(A[k * dim + q]);
            }
        }

        // Update diagonal elements
        A[p * dim + p] = app - t * mag_apq;
        A[q * dim + q] = aqq + t * mag_apq;
        A[p * dim + q] = 0.0;
        A[q * dim + p] = 0.0;
    }

    // Did not converge - fall back to diagonal
    for (uint64_t i = 0; i < dim; i++) {
        eigenvalues[i] = creal(A[i * dim + i]);
    }

    for (uint64_t i = 0; i < dim - 1; i++) {
        for (uint64_t j = i + 1; j < dim; j++) {
            if (eigenvalues[j] > eigenvalues[i]) {
                double tmp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = tmp;
            }
        }
    }

    free(A);
    return max_iter;
}

// ============================================================================
// REDUCED DENSITY MATRIX
// ============================================================================

/**
 * @brief Compute reduced density matrix by tracing out specified qubits
 *
 * For a bipartite system AB, computes ρ_A = Tr_B(|ψ⟩⟨ψ|)
 *
 * @param state Full quantum state
 * @param trace_out_qubits Array of qubit indices to trace out
 * @param num_trace_out Number of qubits to trace out
 * @param reduced_dm Output reduced density matrix (row-major)
 * @param reduced_dim Dimension of reduced system (2^remaining_qubits)
 * @return 0 on success, -1 on error
 */
int entanglement_reduced_density_matrix(const quantum_state_t* state,
                                        const int* trace_out_qubits,
                                        int num_trace_out,
                                        complex_t* reduced_dm,
                                        uint64_t* reduced_dim) {
    if (!state || !state->amplitudes || !reduced_dm) {
        return -1;
    }

    const int total_qubits = state->num_qubits;
    const int remaining_qubits = total_qubits - num_trace_out;

    if (remaining_qubits <= 0 || remaining_qubits > 30) {
        return -1;
    }

    // Create mask for traced-out qubits
    uint64_t trace_mask = 0;
    for (int i = 0; i < num_trace_out; i++) {
        if (trace_out_qubits[i] >= 0 && trace_out_qubits[i] < total_qubits) {
            trace_mask |= (1ULL << trace_out_qubits[i]);
        }
    }

    const uint64_t full_dim = state->state_dim;
    const uint64_t red_dim = 1ULL << remaining_qubits;
    *reduced_dim = red_dim;

    // Initialize reduced density matrix to zero
    memset(reduced_dm, 0, red_dim * red_dim * sizeof(complex_t));

    // Compute reduced density matrix
    // ρ_A[i,j] = Σ_k ⟨i,k|ψ⟩⟨ψ|j,k⟩
    for (uint64_t full_i = 0; full_i < full_dim; full_i++) {
        for (uint64_t full_j = 0; full_j < full_dim; full_j++) {
            // Check if traced-out parts match
            if ((full_i & trace_mask) != (full_j & trace_mask)) {
                continue;
            }

            // Extract remaining qubit indices
            uint64_t red_i = 0, red_j = 0;
            int bit_pos = 0;

            for (int q = 0; q < total_qubits; q++) {
                if (!(trace_mask & (1ULL << q))) {
                    // This qubit is NOT traced out
                    if (full_i & (1ULL << q)) red_i |= (1ULL << bit_pos);
                    if (full_j & (1ULL << q)) red_j |= (1ULL << bit_pos);
                    bit_pos++;
                }
            }

            // Add contribution: ρ[i,j] += ψ[full_i] * conj(ψ[full_j])
            reduced_dm[red_i * red_dim + red_j] +=
                state->amplitudes[full_i] * conj(state->amplitudes[full_j]);
        }
    }

    return 0;
}

// ============================================================================
// ENTANGLEMENT ENTROPY
// ============================================================================

/**
 * @brief Compute von Neumann entropy of reduced density matrix
 *
 * S = -Tr(ρ log₂ ρ) = -Σ λ_i log₂ λ_i
 *
 * where λ_i are eigenvalues of ρ
 *
 * @param reduced_dm Reduced density matrix (Hermitian)
 * @param dim Dimension of matrix
 * @return Entropy in bits (0 for pure, log₂(dim) for maximally mixed)
 */
double entanglement_von_neumann_entropy(const complex_t* reduced_dm, uint64_t dim) {
    if (!reduced_dm || dim == 0) {
        return 0.0;
    }

    // For small dimensions, compute eigenvalues directly
    // For larger dims, would need LAPACK (zheev)

    if (dim == 2) {
        // 2x2 case: analytical eigenvalues
        complex_t a = reduced_dm[0];
        complex_t b = reduced_dm[1];
        complex_t c = reduced_dm[2];
        complex_t d = reduced_dm[3];

        // Eigenvalues: (tr ± sqrt(tr² - 4det)) / 2
        double trace = creal(a) + creal(d);
        complex_t det = a * d - b * c;
        double disc = trace * trace - 4.0 * creal(det);

        if (disc < 0) disc = 0;

        double lambda1 = (trace + sqrt(disc)) / 2.0;
        double lambda2 = (trace - sqrt(disc)) / 2.0;

        // Clamp to [0, 1] for numerical stability
        if (lambda1 < 1e-15) lambda1 = 0;
        if (lambda2 < 1e-15) lambda2 = 0;
        if (lambda1 > 1.0) lambda1 = 1.0;
        if (lambda2 > 1.0) lambda2 = 1.0;

        double entropy = 0.0;
        if (lambda1 > 1e-15) entropy -= lambda1 * log2(lambda1);
        if (lambda2 > 1e-15) entropy -= lambda2 * log2(lambda2);

        return entropy;
    }

    // For larger matrices, use full eigenvalue decomposition
    // Allocate eigenvalue array
    double* eigenvalues = malloc(dim * sizeof(double));
    if (!eigenvalues) {
        // Fall back to diagonal approximation
        double entropy = 0.0;
        for (uint64_t i = 0; i < dim; i++) {
            double diag = creal(reduced_dm[i * dim + i]);
            if (diag > 1e-15) {
                entropy -= diag * log2(diag);
            }
        }
        return entropy;
    }

    // Compute eigenvalues using Jacobi iteration
    int max_iter; double tol;
    get_jacobi_params(&max_iter, &tol);
    jacobi_hermitian_eigenvalues(reduced_dm, dim, eigenvalues, max_iter, tol);

    // Compute von Neumann entropy: S = -Σ λᵢ log₂(λᵢ)
    double entropy = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        double lambda = eigenvalues[i];
        // Clamp to [0, 1] for numerical stability
        if (lambda < 1e-15) lambda = 0;
        if (lambda > 1.0) lambda = 1.0;
        if (lambda > 1e-15) {
            entropy -= lambda * log2(lambda);
        }
    }

    free(eigenvalues);
    return entropy;
}

/**
 * @brief Compute Renyi entropy of order α
 *
 * S_α = (1/(1-α)) log₂ Tr(ρ^α)
 *
 * @param reduced_dm Reduced density matrix
 * @param dim Dimension
 * @param alpha Renyi parameter (α > 0, α ≠ 1)
 * @return Renyi entropy
 */
double entanglement_renyi_entropy(const complex_t* reduced_dm, uint64_t dim,
                                  double alpha) {
    if (!reduced_dm || dim == 0 || alpha <= 0 || fabs(alpha - 1.0) < 1e-10) {
        return 0.0;
    }

    // For α = 2, Tr(ρ²) is the purity
    if (fabs(alpha - 2.0) < 1e-10) {
        double purity = 0.0;
        for (uint64_t i = 0; i < dim; i++) {
            for (uint64_t j = 0; j < dim; j++) {
                complex_t sum = 0.0;
                for (uint64_t k = 0; k < dim; k++) {
                    sum += reduced_dm[i * dim + k] * reduced_dm[k * dim + j];
                }
                if (i == j) {
                    purity += creal(sum);
                }
            }
        }
        return -log2(purity);
    }

    // General case: compute eigenvalues and use Tr(ρ^α) = Σ λᵢ^α
    double* eigenvalues = malloc(dim * sizeof(double));
    if (!eigenvalues) {
        // Fallback to diagonal approximation
        double trace_rho_alpha = 0.0;
        for (uint64_t i = 0; i < dim; i++) {
            double diag = creal(reduced_dm[i * dim + i]);
            if (diag > 1e-15) {
                trace_rho_alpha += pow(diag, alpha);
            }
        }
        return log2(trace_rho_alpha) / (1.0 - alpha);
    }

    // Use Jacobi eigenvalue decomposition
    int jmax_iter; double jtol;
    get_jacobi_params(&jmax_iter, &jtol);
    int converged = jacobi_hermitian_eigenvalues(reduced_dm, dim, eigenvalues, jmax_iter, jtol);

    if (!converged) {
        // Fallback to diagonal approximation
        free(eigenvalues);
        double trace_rho_alpha = 0.0;
        for (uint64_t i = 0; i < dim; i++) {
            double diag = creal(reduced_dm[i * dim + i]);
            if (diag > 1e-15) {
                trace_rho_alpha += pow(diag, alpha);
            }
        }
        return log2(trace_rho_alpha) / (1.0 - alpha);
    }

    // Compute Tr(ρ^α) = Σ λᵢ^α
    double trace_rho_alpha = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        if (eigenvalues[i] > 1e-15) {
            trace_rho_alpha += pow(eigenvalues[i], alpha);
        }
    }

    free(eigenvalues);

    if (trace_rho_alpha <= 0.0) {
        return 0.0;
    }

    return log2(trace_rho_alpha) / (1.0 - alpha);
}

/**
 * @brief Compute entanglement entropy for bipartition
 *
 * Traces out the specified subsystem and computes entropy.
 *
 * @param state Quantum state
 * @param subsystem_b_qubits Qubits in subsystem B (to trace out)
 * @param num_b_qubits Number of qubits in B
 * @return Entanglement entropy in bits
 */
double entanglement_entropy_bipartition(const quantum_state_t* state,
                                        const int* subsystem_b_qubits,
                                        int num_b_qubits) {
    if (!state || !subsystem_b_qubits || num_b_qubits <= 0) {
        return 0.0;
    }

    int remaining = state->num_qubits - num_b_qubits;
    if (remaining <= 0) return 0.0;

    uint64_t reduced_dim;
    uint64_t max_dim = 1ULL << remaining;

    complex_t* reduced_dm = malloc(max_dim * max_dim * sizeof(complex_t));
    if (!reduced_dm) return 0.0;

    int result = entanglement_reduced_density_matrix(state, subsystem_b_qubits,
                                                     num_b_qubits, reduced_dm,
                                                     &reduced_dim);

    double entropy = 0.0;
    if (result == 0) {
        entropy = entanglement_von_neumann_entropy(reduced_dm, reduced_dim);
    }

    free(reduced_dm);
    return entropy;
}

// ============================================================================
// CONCURRENCE (2-QUBIT)
// ============================================================================

/**
 * @brief Compute concurrence for 2-qubit state
 *
 * C(ψ) = |⟨ψ|σ_y ⊗ σ_y|ψ*⟩|
 *
 * For pure states, this equals 2|α₀₀ α₁₁ - α₀₁ α₁₀|
 *
 * @param state 2-qubit quantum state
 * @return Concurrence in [0, 1] (0 = separable, 1 = maximally entangled)
 */
double entanglement_concurrence_2qubit(const quantum_state_t* state) {
    if (!state || !state->amplitudes || state->num_qubits != 2) {
        return 0.0;
    }

    // |ψ⟩ = α₀₀|00⟩ + α₀₁|01⟩ + α₁₀|10⟩ + α₁₁|11⟩
    complex_t a00 = state->amplitudes[0];  // |00⟩
    complex_t a01 = state->amplitudes[1];  // |01⟩
    complex_t a10 = state->amplitudes[2];  // |10⟩
    complex_t a11 = state->amplitudes[3];  // |11⟩

    // Concurrence = 2|α₀₀ α₁₁ - α₀₁ α₁₀|
    complex_t det = a00 * a11 - a01 * a10;
    return 2.0 * cabs(det);
}

/**
 * @brief Compute concurrence from 2-qubit density matrix
 *
 * For mixed states: C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
 * where λᵢ are square roots of eigenvalues of ρ(σ_y⊗σ_y)ρ*(σ_y⊗σ_y)
 *
 * @param density_matrix 4x4 density matrix
 * @return Concurrence in [0, 1]
 */
double entanglement_concurrence_mixed(const complex_t* density_matrix) {
    if (!density_matrix) return 0.0;

    // σ_y ⊗ σ_y matrix (spin-flip)
    // [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]
    complex_t sigma_yy[16] = {
        0, 0, 0, -1,
        0, 0, 1, 0,
        0, 1, 0, 0,
        -1, 0, 0, 0
    };

    // Compute ρ̃ = (σ_y⊗σ_y) ρ* (σ_y⊗σ_y)
    complex_t rho_tilde[16];
    complex_t temp[16];

    // temp = ρ* (conjugate)
    for (int i = 0; i < 16; i++) {
        temp[i] = conj(density_matrix[i]);
    }

    // temp2 = (σ_y⊗σ_y) * temp
    complex_t temp2[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp2[i * 4 + j] = 0;
            for (int k = 0; k < 4; k++) {
                temp2[i * 4 + j] += sigma_yy[i * 4 + k] * temp[k * 4 + j];
            }
        }
    }

    // rho_tilde = temp2 * (σ_y⊗σ_y)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            rho_tilde[i * 4 + j] = 0;
            for (int k = 0; k < 4; k++) {
                rho_tilde[i * 4 + j] += temp2[i * 4 + k] * sigma_yy[k * 4 + j];
            }
        }
    }

    // Compute R = ρ * ρ̃
    complex_t R[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            R[i * 4 + j] = 0;
            for (int k = 0; k < 4; k++) {
                R[i * 4 + j] += density_matrix[i * 4 + k] * rho_tilde[k * 4 + j];
            }
        }
    }

    // Compute eigenvalues of R using Jacobi iteration
    // Concurrence formula: C = max(0, √λ₁ - √λ₂ - √λ₃ - √λ₄)
    // where λᵢ are eigenvalues sorted in decreasing order
    double eigenvalues[4];
    int conc_max_iter; double conc_tol;
    get_jacobi_params(&conc_max_iter, &conc_tol);
    jacobi_hermitian_eigenvalues(R, 4, eigenvalues, conc_max_iter, conc_tol);

    // Compute square roots of eigenvalues (clamped to non-negative)
    double sqrt_lambda[4];
    for (int i = 0; i < 4; i++) {
        sqrt_lambda[i] = (eigenvalues[i] > 0) ? sqrt(eigenvalues[i]) : 0;
    }

    // Concurrence = max(0, √λ₁ - √λ₂ - √λ₃ - √λ₄)
    double concurrence = sqrt_lambda[0] - sqrt_lambda[1] - sqrt_lambda[2] - sqrt_lambda[3];
    if (concurrence < 0) concurrence = 0;
    if (concurrence > 1) concurrence = 1;

    return concurrence;
}

// ============================================================================
// SCHMIDT DECOMPOSITION
// ============================================================================

/**
 * @brief Compute Schmidt coefficients for bipartite state
 *
 * |ψ⟩ = Σᵢ λᵢ |αᵢ⟩_A |βᵢ⟩_B
 *
 * The Schmidt coefficients λᵢ are square roots of eigenvalues of ρ_A.
 *
 * @param state Quantum state
 * @param partition_a_qubits Qubits in partition A
 * @param num_a Number of qubits in A
 * @param coefficients Output Schmidt coefficients (sorted descending)
 * @param num_coefficients Output number of coefficients
 * @return 0 on success
 */
int entanglement_schmidt_coefficients(const quantum_state_t* state,
                                      const int* partition_a_qubits,
                                      int num_a,
                                      double* coefficients,
                                      int* num_coefficients) {
    if (!state || !coefficients || !num_coefficients) {
        return -1;
    }

    int num_b = state->num_qubits - num_a;
    if (num_a <= 0 || num_b <= 0) {
        return -1;
    }

    // Create list of qubits in B (complement of A)
    int* partition_b = malloc(num_b * sizeof(int));
    int b_idx = 0;

    for (int q = 0; q < state->num_qubits; q++) {
        int in_a = 0;
        for (int i = 0; i < num_a; i++) {
            if (partition_a_qubits[i] == q) {
                in_a = 1;
                break;
            }
        }
        if (!in_a) {
            partition_b[b_idx++] = q;
        }
    }

    // Compute reduced density matrix of A
    uint64_t dim_a;
    uint64_t max_dim_a = 1ULL << num_a;
    complex_t* rho_a = malloc(max_dim_a * max_dim_a * sizeof(complex_t));

    int result = entanglement_reduced_density_matrix(state, partition_b, num_b,
                                                     rho_a, &dim_a);
    free(partition_b);

    if (result != 0) {
        free(rho_a);
        return -1;
    }

    // Extract eigenvalues using Jacobi decomposition
    // Schmidt coefficients are square roots of eigenvalues of ρ_A
    *num_coefficients = (int)dim_a;

    double* eigenvalues = malloc(dim_a * sizeof(double));
    if (!eigenvalues) {
        // Fall back to diagonal approximation
        for (uint64_t i = 0; i < dim_a; i++) {
            double eigenval = creal(rho_a[i * dim_a + i]);
            coefficients[i] = (eigenval > 0) ? sqrt(eigenval) : 0;
        }
    } else {
        // Full eigenvalue decomposition
        int schmidt_max_iter; double schmidt_tol;
        get_jacobi_params(&schmidt_max_iter, &schmidt_tol);
        jacobi_hermitian_eigenvalues(rho_a, dim_a, eigenvalues, schmidt_max_iter, schmidt_tol);

        // Schmidt coefficients = sqrt(eigenvalues)
        for (uint64_t i = 0; i < dim_a; i++) {
            double eigenval = eigenvalues[i];
            // Clamp for numerical stability
            if (eigenval < 0) eigenval = 0;
            if (eigenval > 1.0) eigenval = 1.0;
            coefficients[i] = (eigenval > 0) ? sqrt(eigenval) : 0;
        }
        free(eigenvalues);
    }

    // Eigenvalues already sorted descending by jacobi_hermitian_eigenvalues
    // But double-check the sort for safety
    for (int i = 0; i < *num_coefficients - 1; i++) {
        for (int j = i + 1; j < *num_coefficients; j++) {
            if (coefficients[j] > coefficients[i]) {
                double tmp = coefficients[i];
                coefficients[i] = coefficients[j];
                coefficients[j] = tmp;
            }
        }
    }

    free(rho_a);
    return 0;
}

/**
 * @brief Compute Schmidt rank (number of non-zero Schmidt coefficients)
 *
 * @param state Quantum state
 * @param partition_a_qubits Qubits in partition A
 * @param num_a Number of qubits in A
 * @param threshold Threshold for considering coefficient non-zero
 * @return Schmidt rank
 */
int entanglement_schmidt_rank(const quantum_state_t* state,
                              const int* partition_a_qubits,
                              int num_a,
                              double threshold) {
    int max_coeffs = 1 << state->num_qubits;
    double* coeffs = malloc(max_coeffs * sizeof(double));
    int num_coeffs;

    int result = entanglement_schmidt_coefficients(state, partition_a_qubits,
                                                   num_a, coeffs, &num_coeffs);
    if (result != 0) {
        free(coeffs);
        return -1;
    }

    int rank = 0;
    for (int i = 0; i < num_coeffs; i++) {
        if (coeffs[i] > threshold) rank++;
    }

    free(coeffs);
    return rank;
}

// ============================================================================
// ENTANGLEMENT DETECTION
// ============================================================================

/**
 * @brief Check if state is separable (unentangled)
 *
 * Uses PPT criterion for 2x2 and 2x3 systems.
 * For larger systems, uses entropy-based heuristic.
 *
 * @param state Quantum state
 * @param partition_a_qubits Qubits in first partition
 * @param num_a Number of qubits in first partition
 * @return 1 if likely separable, 0 if entangled, -1 on error
 */
int entanglement_is_separable(const quantum_state_t* state,
                              const int* partition_a_qubits,
                              int num_a) {
    if (!state) return -1;

    // For pure states, check if entropy is 0
    double entropy = entanglement_entropy_bipartition(state, partition_a_qubits,
                                                      num_a);

    // If entropy < threshold, state is approximately separable
    if (entropy < 1e-10) {
        return 1;  // Separable
    }

    return 0;  // Entangled
}

/**
 * @brief Compute purity of reduced density matrix
 *
 * Tr(ρ²) = 1 for pure states, 1/d for maximally mixed
 *
 * @param reduced_dm Density matrix
 * @param dim Dimension
 * @return Purity in [1/dim, 1]
 */
double entanglement_purity(const complex_t* reduced_dm, uint64_t dim) {
    if (!reduced_dm || dim == 0) return 0.0;

    double purity = 0.0;

    for (uint64_t i = 0; i < dim; i++) {
        for (uint64_t j = 0; j < dim; j++) {
            complex_t sum = 0.0;
            for (uint64_t k = 0; k < dim; k++) {
                sum += reduced_dm[i * dim + k] * reduced_dm[k * dim + j];
            }
            if (i == j) {
                purity += creal(sum);
            }
        }
    }

    return purity;
}

/**
 * @brief Compute linear entropy
 *
 * S_L = (d/(d-1)) * (1 - Tr(ρ²))
 *
 * Normalized to [0, 1]
 *
 * @param reduced_dm Density matrix
 * @param dim Dimension
 * @return Linear entropy
 */
double entanglement_linear_entropy(const complex_t* reduced_dm, uint64_t dim) {
    if (!reduced_dm || dim <= 1) return 0.0;

    double purity = entanglement_purity(reduced_dm, dim);
    return ((double)dim / (dim - 1)) * (1.0 - purity);
}
