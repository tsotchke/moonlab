/*
 * gpu_metal_svd.c
 *
 * CPU double-precision SVD bridge for the Metal MPS truncation path.
 *
 * The Metal tensor kernels operate in single precision (float2). Its old
 * one-sided Jacobi truncation never converged, assumed sorted singular
 * values it did not produce, and silently skipped the decomposition when the
 * pipeline was nil -- so the MPS gate/truncation returned wrong numbers.
 *
 * This bridge runs the truncation SVD honestly on the CPU in double precision
 * via the shared, LAPACK-backed tensor_svd (zgesvd). It lives in a plain C
 * translation unit so it can include tensor.h (which uses C99 `double complex`
 * that does not compile through the Objective-C++ frontend of gpu_metal.mm).
 * gpu_metal.mm lifts its float2 theta matrix to interleaved doubles, calls
 * this, then picks the rank in double and writes the truncated factors back.
 */

#include "../algorithms/tensor_network/tensor.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Full economy SVD of an m x n complex matrix supplied as interleaved doubles
 * (row-major, [re, im] per entry). On success writes the full economy factors:
 *
 *   S_out  : min(m,n) singular values, descending
 *   U_out  : m x min(m,n) interleaved doubles, row-major
 *   Vt_out : min(m,n) x n interleaved doubles, row-major  (this is V^H)
 *
 * The caller sizes the output buffers for the full min(m,n) factors and picks
 * the truncation rank itself from S_out. Returns min(m,n) on success, 0 on
 * failure (allocation, or the SVD reported non-finite / negative values).
 */
int moonlab_metal_svd_double(const double *theta_interleaved,
                             uint32_t m, uint32_t n,
                             double *U_out, double *S_out, double *Vt_out)
{
    if (!theta_interleaved || !U_out || !S_out || !Vt_out || m == 0 || n == 0) {
        return 0;
    }
    const uint32_t k = (m < n) ? m : n;

    double complex *data =
        (double complex *)malloc((size_t)m * (size_t)n * sizeof(double complex));
    if (!data) return 0;
    for (size_t i = 0; i < (size_t)m * (size_t)n; i++) {
        data[i] = theta_interleaved[2 * i] + theta_interleaved[2 * i + 1] * I;
    }

    uint32_t dims[2] = { m, n };
    tensor_t *mat = tensor_create_with_data(2, dims, data);
    free(data);
    if (!mat) return 0;

    /* Full economy SVD: no truncation here (cutoff = 0, max_rank = 0), so the
     * caller sees every singular value and selects the rank in double. */
    tensor_svd_result_t *svd = tensor_svd(mat, 0, 0.0);
    tensor_free(mat);
    if (!svd || !svd->U || !svd->Vh || !svd->S) {
        if (svd) tensor_svd_free(svd);
        return 0;
    }

    const uint32_t sk = svd->k;                 /* == min(m,n) for a full SVD */
    const double complex *U = svd->U->data;     /* m  x sk, row-major */
    const double complex *V = svd->Vh->data;    /* sk x n,  row-major */

    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < k; j++) {
            double complex v = (j < sk) ? U[(size_t)i * sk + j] : 0.0;
            U_out[2 * ((size_t)i * k + j)]     = creal(v);
            U_out[2 * ((size_t)i * k + j) + 1] = cimag(v);
        }
    }
    for (uint32_t j = 0; j < k; j++) {
        S_out[j] = (j < sk) ? svd->S[j] : 0.0;
    }
    for (uint32_t j = 0; j < k; j++) {
        for (uint32_t l = 0; l < n; l++) {
            double complex v = (j < sk) ? V[(size_t)j * (size_t)n + l] : 0.0;
            Vt_out[2 * ((size_t)j * (size_t)n + l)]     = creal(v);
            Vt_out[2 * ((size_t)j * (size_t)n + l) + 1] = cimag(v);
        }
    }

    tensor_svd_free(svd);
    return (int)k;
}

/*
 * Honest double-precision MPS single-/two-site <Z>/<ZZ> contraction.
 *
 * The Metal transfer-matrix kernels read the MPS buffers as float2 and never
 * divided by the norm, producing garbage. tn_measurement.c uploads the tensors
 * as raw double complex (t->data), so gpu_metal.mm hands their pointers here
 * as interleaved doubles and we contract left-to-right in double, then divide
 * by the norm. This C TU keeps the C99 `double complex` math out of the
 * Objective-C++ frontend of gpu_metal.mm.
 *
 * tensors[s] points at tensor s laid out row-major [Dl][2][Dr] as interleaved
 * doubles (re, im). Bond dims: Dl_s = (s>0)?bond_dims[s-1]:1, Dr_s =
 * (s<num_sites-1)?bond_dims[s]:1. op_site/op_site2 (< 0 = none) insert a
 * diagonal Z (weights +1 on |0>, -1 on |1>).
 */
static double complex mps_contract_scalar(const double *const *tensors,
                                          const uint32_t *bond_dims,
                                          uint32_t num_sites,
                                          int op_site, int op_site2) {
    static const double z_w[2] = { 1.0, -1.0 };
    uint32_t Ldim = 1;
    double complex *L = (double complex *)calloc(1, sizeof(double complex));
    if (!L) return 0.0;
    L[0] = 1.0;

    for (uint32_t s = 0; s < num_sites; s++) {
        uint32_t Dl = (s > 0) ? bond_dims[s - 1] : 1;
        uint32_t Dr = (s < num_sites - 1) ? bond_dims[s] : 1;
        const double *Ad = tensors[s];   /* interleaved [Dl][2][Dr] */
        double complex *Lnew =
            (double complex *)calloc((size_t)Dr * Dr, sizeof(double complex));
        if (!Lnew) { free(L); return 0.0; }

        for (uint32_t p = 0; p < 2; p++) {
            double w = 1.0;
            if (op_site  >= 0 && s == (uint32_t)op_site)  w *= z_w[p];
            if (op_site2 >= 0 && s == (uint32_t)op_site2) w *= z_w[p];
            if (w == 0.0) continue;
            for (uint32_t b = 0; b < Dr; b++) {
                for (uint32_t bp = 0; bp < Dr; bp++) {
                    double complex acc = 0.0;
                    for (uint32_t a = 0; a < Dl; a++) {
                        size_t ia = ((size_t)a * 2 + p) * Dr + b;
                        double complex ca = Ad[2*ia] - Ad[2*ia+1] * I; /* conj */
                        double complex row = 0.0;
                        for (uint32_t ap = 0; ap < Dl; ap++) {
                            size_t ib = ((size_t)ap * 2 + p) * Dr + bp;
                            double complex av = Ad[2*ib] + Ad[2*ib+1] * I;
                            row += L[(size_t)a * Ldim + ap] * av;
                        }
                        acc += ca * row;
                    }
                    Lnew[(size_t)b * Dr + bp] += w * acc;
                }
            }
        }
        free(L);
        L = Lnew;
        Ldim = Dr;
    }

    double complex val = L[0];
    free(L);
    return val;
}

double moonlab_metal_mps_expectation_z(const double *const *tensors,
                                       const uint32_t *bond_dims,
                                       uint32_t num_sites, uint32_t site) {
    if (!tensors || !bond_dims || site >= num_sites) return 0.0;
    double num  = creal(mps_contract_scalar(tensors, bond_dims, num_sites, (int)site, -1));
    double norm = creal(mps_contract_scalar(tensors, bond_dims, num_sites, -1, -1));
    return (fabs(norm) > 1e-300) ? (num / norm) : 0.0;
}

double moonlab_metal_mps_expectation_zz(const double *const *tensors,
                                        const uint32_t *bond_dims,
                                        uint32_t num_sites,
                                        uint32_t site_i, uint32_t site_j) {
    if (!tensors || !bond_dims) return 0.0;
    if (site_i > site_j) { uint32_t t = site_i; site_i = site_j; site_j = t; }
    if (site_j >= num_sites) return 0.0;
    double num  = creal(mps_contract_scalar(tensors, bond_dims, num_sites,
                                            (int)site_i, (int)site_j));
    double norm = creal(mps_contract_scalar(tensors, bond_dims, num_sites, -1, -1));
    return (fabs(norm) > 1e-300) ? (num / norm) : 0.0;
}
