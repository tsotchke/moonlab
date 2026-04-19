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
 * Uses Jacobi rotations. Eigenvalues are returned correctly for both
 * real-symmetric and complex-Hermitian inputs.
 *
 * IMPORTANT: the internal Givens rotations are REAL-valued, which is
 * exact for real-symmetric matrices but only approximately diagonalises
 * a complex-Hermitian matrix. The **eigenvectors** returned for a
 * complex input may therefore not be proper eigenvectors of the input
 * (they only diagonalise the real part). If you need eigenvectors of a
 * complex Hermitian matrix (e.g. for projector construction,
 * topological-marker / Berry-curvature / Wilson-loop computations),
 * do NOT rely on this function -- use a sign-function / Schulz
 * iteration on the gapped projector, or implement a proper complex
 * Jacobi / Householder-QL routine. See
 * `src/algorithms/topology_realspace/chern_marker.c` for an example
 * of the sign-function path.
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
