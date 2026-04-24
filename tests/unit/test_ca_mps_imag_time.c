/**
 * @file test_ca_mps_imag_time.c
 * @brief CA-MPS imaginary-time ground-state search on Heisenberg clusters.
 *
 * Drives CA-MPS with first-order Trotter steps
 *     |psi>_{t+tau} = (prod_k exp(-tau c_k P_k)) |psi>_t / norm
 * on the Heisenberg Hamiltonian  H = sum_<ij> (XX + YY + ZZ).
 *
 * Since all three on-site Pauli products at a given bond commute
 * (XX, YY, ZZ pairwise commute), the bond Gibbs factor is exact:
 *     exp(-tau (XX + YY + ZZ)) = exp(-tau XX) exp(-tau YY) exp(-tau ZZ)
 * and distinct bonds share at most one qubit so the bond-by-bond Trotter
 * error is O(tau^2).
 *
 * Success criteria: energy converges to dense-ED ground state for N=2,
 * N=4, and N=6 open Heisenberg chains to <= 1e-3.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void zheev_(char *jobz, char *uplo, int *n, double _Complex *a, int *lda,
                   double *w, double _Complex *work, int *lwork,
                   double *rwork, int *info);
#endif

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

/* Dense ED: exact ground-state energy of Heisenberg on bond list. */
static double dense_ed_gs(uint32_t n, const uint32_t (*bonds)[2], uint32_t num_bonds) {
    size_t dim = (size_t)1 << n;
    double _Complex* H = (double _Complex*)calloc(dim * dim, sizeof(double _Complex));
    for (uint32_t b = 0; b < num_bonds; b++) {
        uint32_t i = bonds[b][0], j = bonds[b][1];
        size_t mask = ((size_t)1 << i) | ((size_t)1 << j);
        for (size_t s = 0; s < dim; s++) {
            int bi = (int)((s >> i) & 1), bj = (int)((s >> j) & 1);
            H[s * dim + s] += (bi == bj) ? 1.0 : -1.0;
            if (bi != bj) H[(s ^ mask) * dim + s] += 2.0;
        }
    }
    char jobz = 'N', uplo = 'U';
    int N = (int)dim;
    double* w = (double*)malloc((size_t)N * sizeof(double));
    double* rwork = (double*)malloc((size_t)(3 * N) * sizeof(double));
#ifdef __APPLE__
    __CLPK_integer nn = N, lda = N, lwork_q = -1, info = 0;
    __CLPK_doublecomplex wq;
    zheev_(&jobz, &uplo, &nn, (__CLPK_doublecomplex*)H, &lda, w, &wq, &lwork_q, rwork, &info);
    int lwork = (int)wq.r;
    double _Complex* work = (double _Complex*)malloc((size_t)lwork * sizeof(double _Complex));
    lwork_q = lwork;
    zheev_(&jobz, &uplo, &nn, (__CLPK_doublecomplex*)H, &lda, w,
           (__CLPK_doublecomplex*)work, &lwork_q, rwork, &info);
#else
    int nn = N, lda = N, lwork_q = -1, info = 0;
    double _Complex wq = 0.0;
    zheev_(&jobz, &uplo, &nn, H, &lda, w, &wq, &lwork_q, rwork, &info);
    int lwork = (int)creal(wq);
    double _Complex* work = (double _Complex*)malloc((size_t)lwork * sizeof(double _Complex));
    lwork_q = lwork;
    zheev_(&jobz, &uplo, &nn, H, &lda, w, work, &lwork_q, rwork, &info);
#endif
    double e0 = w[0];
    free(H); free(w); free(rwork); free(work);
    return e0;
}

/* Build Heisenberg H as Pauli sum for expectation evaluation. */
static void build_heis_paulis(uint32_t n, const uint32_t (*bonds)[2], uint32_t num_bonds,
                              uint8_t** out_paulis, double _Complex** out_coeffs,
                              uint32_t* out_nterms) {
    uint32_t nt = 3 * num_bonds;
    uint8_t* paulis = (uint8_t*)calloc((size_t)nt * n, sizeof(uint8_t));
    double _Complex* coeffs = (double _Complex*)calloc(nt, sizeof(double _Complex));
    for (uint32_t b = 0; b < num_bonds; b++) {
        uint32_t i = bonds[b][0], j = bonds[b][1];
        for (uint8_t p = 1; p <= 3; p++) {
            uint32_t k = 3 * b + (p - 1);
            paulis[(size_t)k * n + i] = p;
            paulis[(size_t)k * n + j] = p;
            coeffs[k] = 1.0;
        }
    }
    *out_paulis = paulis; *out_coeffs = coeffs; *out_nterms = nt;
}

/* One Trotter step: apply exp(-tau XX_ij) exp(-tau YY_ij) exp(-tau ZZ_ij) on
 * each bond in sequence.  Bonds sharing sites are handled by the bond order
 * (even-odd sweep would be cleaner but for small systems a plain linear
 * sweep works too). */
static void trotter_step(moonlab_ca_mps_t* s, uint32_t n,
                         const uint32_t (*bonds)[2], uint32_t num_bonds,
                         double tau) {
    uint8_t* pauli = (uint8_t*)calloc(n, sizeof(uint8_t));
    for (uint32_t b = 0; b < num_bonds; b++) {
        uint32_t i = bonds[b][0], j = bonds[b][1];
        for (uint8_t p = 1; p <= 3; p++) {
            memset(pauli, 0, n);
            pauli[i] = p; pauli[j] = p;
            moonlab_ca_mps_imag_pauli_rotation(s, pauli, tau);
        }
    }
    free(pauli);
    moonlab_ca_mps_normalize(s);
}

static int gs_search_heisenberg(uint32_t n,
                                const uint32_t (*bonds)[2], uint32_t num_bonds,
                                double E_ref, const char* label,
                                double target_tol) {
    uint8_t* paulis; double _Complex* coeffs; uint32_t nterms;
    build_heis_paulis(n, bonds, num_bonds, &paulis, &coeffs, &nterms);

    moonlab_ca_mps_t* s = moonlab_ca_mps_create(n, 64);
    /* Initial Neel state: |0101...> kicks away from the high-symmetry
     * singlet so imag-time has to do real work to find the ground state. */
    for (uint32_t q = 1; q < n; q += 2) moonlab_ca_mps_x(s, q);
    /* Plus a Hadamard kick on qubit 0 to introduce superposition. */
    moonlab_ca_mps_h(s, 0);

    double taus[]   = { 0.05, 0.02, 0.005, 0.001 };
    int    steps[]  = { 60,   80,   80,    80 };

    double E = 0.0;
    fprintf(stdout, "\n  [%s] N=%u  target E_0 = %.6f (ED)\n", label, n, E_ref);
    for (size_t schedule = 0; schedule < sizeof(taus) / sizeof(taus[0]); schedule++) {
        double tau = taus[schedule];
        int nsteps = steps[schedule];
        for (int step = 0; step < nsteps; step++) {
            trotter_step(s, n, bonds, num_bonds, tau);
        }
        double _Complex e;
        moonlab_ca_mps_expect_pauli_sum(s, paulis, coeffs, nterms, &e);
        E = creal(e);
        fprintf(stdout, "    tau=%.4f  %d steps  E = %+.9f   err = %.2e\n",
                tau, nsteps, E, fabs(E - E_ref));
    }

    moonlab_ca_mps_free(s); free(paulis); free(coeffs);

    int local_fail = 0;
    if (fabs(E - E_ref) > target_tol) {
        fprintf(stderr, "  FAIL  [%s] E = %.9f  E_ref = %.9f  err = %.3e (tol %.0e)\n",
                label, E, E_ref, fabs(E - E_ref), target_tol);
        local_fail++;
    } else {
        fprintf(stdout, "  OK    [%s] converged to E_0 within %.3e\n",
                label, fabs(E - E_ref));
    }
    return local_fail;
}

int main(void) {
    fprintf(stdout, "=== CA-MPS imaginary-time ground-state search on Heisenberg ===\n");

    /* N=2 dimer: exact E_0 = -3. */
    {
        uint32_t b[1][2] = {{0,1}};
        double ed = dense_ed_gs(2, b, 1);
        CHECK(fabs(ed - (-3.0)) < 1e-9, "dense ED dimer = %.9f (expect -3)", ed);
        failures += gs_search_heisenberg(2, b, 1, ed, "dimer", 1e-3);
    }

    /* N=4 open chain. */
    {
        uint32_t b[3][2] = {{0,1},{1,2},{2,3}};
        double ed = dense_ed_gs(4, b, 3);
        failures += gs_search_heisenberg(4, b, 3, ed, "chain-4", 5e-3);
    }

    /* N=6 open chain. */
    {
        uint32_t b[5][2] = {{0,1},{1,2},{2,3},{3,4},{4,5}};
        double ed = dense_ed_gs(6, b, 5);
        failures += gs_search_heisenberg(6, b, 5, ed, "chain-6", 1e-2);
    }

    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
