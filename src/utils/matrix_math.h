#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <stddef.h>
#include <complex.h>

typedef double _Complex complex_t;

/**
 * @file matrix_math.h
 * @brief Production-quality matrix mathematics for quantum operations
 * 
 * Implements:
 * - Eigenvalue decomposition for Hermitian matrices
 * - Matrix operations (multiply, transpose, trace)
 * - Numerical linear algebra utilities
 */

/**
 * @brief Compute eigenvalues and eigenvectors of Hermitian matrix.
 *
 * Complex-Hermitian Jacobi: each rotation absorbs the off-diagonal
 * phase via D = diag(1, e^{-i phi}) before applying the standard real
 * Givens step.  Eigenvalues and eigenvectors are correct for both
 * real-symmetric and complex-Hermitian inputs (residual
 * ||H v - lambda v|| < 1e-14 on a 2x2 unit smoke).
 *
 * The previous version used real-valued Givens rotations which only
 * diagonalised the real part of A; the workaround in
 * `src/algorithms/topology_realspace/chern_marker.c` (sign-function
 * via Schulz iteration on the gapped projector) is no longer needed
 * for correctness, though it remains as a faster path for the
 * specific topological-marker case.
 *
 * @param matrix Input Hermitian matrix (n×n), row-major order
 * @param n Dimension of matrix
 * @param eigenvalues Output: array of n real eigenvalues (sorted descending)
 * @param eigenvectors Output: matrix of eigenvectors (n×n, column-major)
 * @param max_iterations Maximum iterations (default 50*n²)
 * @param tolerance Convergence tolerance (default 1e-10)
 * @return 0 on success, -1 on error
 */
int hermitian_eigen_decomposition(
    const complex_t *matrix,
    size_t n,
    double *eigenvalues,
    complex_t *eigenvectors,
    int max_iterations,
    double tolerance
);

/**
 * @brief Matrix multiplication: C = A × B
 * @param a First matrix (m×k)
 * @param b Second matrix (k×n)
 * @param c Output matrix (m×n)
 * @param m Rows in A
 * @param k Cols in A, rows in B
 * @param n Cols in B
 */
void matrix_multiply(
    const complex_t *a,
    const complex_t *b,
    complex_t *c,
    size_t m,
    size_t k,
    size_t n
);

/**
 * @brief Matrix trace: Tr(A) = Σ Aᵢᵢ
 * @param matrix Square matrix (n×n)
 * @param n Dimension
 * @return Trace value
 */
complex_t matrix_trace(const complex_t *matrix, size_t n);

/**
 * @brief Check if matrix is Hermitian: A = A†
 * @param matrix Matrix to check (n×n)
 * @param n Dimension
 * @param tolerance Tolerance for comparison
 * @return 1 if Hermitian, 0 otherwise
 */
int matrix_is_hermitian(const complex_t *matrix, size_t n, double tolerance);

/**
 * @brief Conjugate transpose: A† (dagger)
 * @param matrix Input matrix (m×n)
 * @param result Output matrix (n×m)
 * @param m Rows in input
 * @param n Cols in input
 */
void matrix_conjugate_transpose(
    const complex_t *matrix,
    complex_t *result,
    size_t m,
    size_t n
);

/**
 * @brief Matrix norm (Frobenius): ||A|| = √(Σ|Aᵢⱼ|²)
 * @param matrix Matrix (m×n)
 * @param m Rows
 * @param n Cols
 * @return Frobenius norm
 */
double matrix_frobenius_norm(const complex_t *matrix, size_t m, size_t n);

#endif /* MATRIX_MATH_H */
