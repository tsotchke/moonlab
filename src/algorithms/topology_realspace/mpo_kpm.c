/**
 * @file mpo_kpm.c
 * @brief Implementation notes for the MPO-level Chebyshev / Jackson
 *        kernel polynomial method.
 *
 * All of the non-trivial work is concentrated in two helpers:
 *
 *   - @c mpo_kpm_mps_combine builds @f$\alpha|A\rangle + \beta|B\rangle@f$
 *     by block-diagonal concatenation; every result tensor is filled
 *     by direct memcpy of the two source tensors at the appropriate
 *     strides.  Shifts and scales from the Chebyshev recurrence are
 *     carried by @f$\alpha, \beta@f$ and baked into the boundary
 *     site-0 tensor.
 *   - @c mpo_kpm_chebyshev_moments runs the three-term recurrence
 *     @f$|v_{n+1}\rangle = 2\hat{\tilde H}|v_n\rangle - |v_{n-1}\rangle@f$
 *     in the rescaled-inline form
 *     @f$|v_{n+1}\rangle = (2/b)(\hat H|v_n\rangle) - (2a/b)|v_n\rangle - |v_{n-1}\rangle@f$
 *     using two @c mpo_kpm_mps_combine calls per step (one to fold
 *     @f$\hat H|v_n\rangle@f$ and @f$|v_n\rangle@f$, one to subtract
 *     @f$|v_{n-1}\rangle@f$), each followed by an SVD truncation to
 *     @c params->max_bond_dim.
 */

#include "mpo_kpm.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------- */
/* Parameter defaults                                                */
/* ---------------------------------------------------------------- */

mpo_kpm_params_t mpo_kpm_params_default(void) {
    mpo_kpm_params_t p;
    p.n_cheby       = 80;
    p.E_shift       = 0.0;
    p.E_scale       = 1.0;
    p.max_bond_dim  = 128;
    p.svd_cutoff    = 1e-12;
    p.use_jackson   = 1;
    return p;
}

/* ---------------------------------------------------------------- */
/* Jackson kernel and sign-function coefficients                     */
/* ---------------------------------------------------------------- */

void mpo_kpm_jackson_weights(size_t n_cheby, double* g_out) {
    if (n_cheby == 0 || !g_out) return;
    const double Np1 = (double)(n_cheby + 1);
    const double denom = Np1;
    const double pi_by_Np1 = M_PI / Np1;
    const double cot_pi_Np1 = cos(pi_by_Np1) / sin(pi_by_Np1);
    for (size_t n = 0; n < n_cheby; n++) {
        const double a = (double)(n_cheby - n + 1) * cos((double)n * pi_by_Np1);
        const double b = sin((double)n * pi_by_Np1) * cot_pi_Np1;
        g_out[n] = (a + b) / denom;
    }
}

void mpo_kpm_sign_coefficients(size_t n_cheby, double* c_out) {
    if (n_cheby == 0 || !c_out) return;
    c_out[0] = 0.0;
    for (size_t n = 1; n < n_cheby; n++) {
        if (n % 2 == 0) {
            c_out[n] = 0.0;
        } else {
            /* sin(n pi / 2) for odd n alternates between +1 (n = 1, 5, ...)
             * and -1 (n = 3, 7, ...); equivalently (-1)^((n-1)/2). */
            const int k = (int)((n - 1) / 2);
            const double sign = (k % 2 == 0) ? 1.0 : -1.0;
            c_out[n] = 4.0 / (M_PI * (double)n) * sign;
        }
    }
}

/* ---------------------------------------------------------------- */
/* DMRG mpo_t -> gate-API tn_mpo_t adapter                           */
/* ---------------------------------------------------------------- */

tn_mpo_t* mpo_kpm_mpo_to_tn_mpo(const mpo_t* H) {
    if (!H || H->num_sites == 0 || !H->tensors) return NULL;

    tn_mpo_t* out = (tn_mpo_t*)calloc(1, sizeof(tn_mpo_t));
    if (!out) return NULL;

    out->num_sites = H->num_sites;
    out->tensors   = (tensor_t**)calloc(out->num_sites, sizeof(tensor_t*));
    if (H->num_sites >= 2) {
        out->bond_dims = (uint32_t*)calloc(out->num_sites - 1, sizeof(uint32_t));
    } else {
        out->bond_dims = NULL;
    }

    if (!out->tensors || (H->num_sites >= 2 && !out->bond_dims)) {
        tn_mpo_free(out);
        return NULL;
    }

    for (uint32_t i = 0; i < H->num_sites; i++) {
        const tensor_t* Wsrc = H->tensors[i].W;
        if (!Wsrc) {
            tn_mpo_free(out);
            return NULL;
        }
        out->tensors[i] = tensor_copy(Wsrc);
        if (!out->tensors[i]) {
            tn_mpo_free(out);
            return NULL;
        }
    }
    for (uint32_t i = 0; i + 1 < H->num_sites; i++) {
        out->bond_dims[i] = H->tensors[i].bond_dim_right;
    }

    return out;
}

/* ---------------------------------------------------------------- */
/* MPS linear combination via block-diagonal construction            */
/* ---------------------------------------------------------------- */

/* For a rank-3 MPS tensor [dL, d, dR] return the stride offset in
 * complex elements for the (l, s, r) entry. */
static inline size_t mps_idx(uint32_t l, uint32_t s, uint32_t r,
                             uint32_t dL, uint32_t d, uint32_t dR) {
    (void)dL;
    return ((size_t)l * d + s) * dR + r;
}

/* Build the site-0 tensor of the combined MPS: shape [1, d, dR_A + dR_B],
 * with alpha folded into the A columns and beta into the B columns. */
static tensor_t* combine_first_site(
    const tensor_t* A, double complex alpha,
    const tensor_t* B, double complex beta)
{
    const uint32_t d    = A->dims[1];
    const uint32_t dRA  = A->dims[2];
    const uint32_t dRB  = B->dims[2];
    const uint32_t dRC  = dRA + dRB;

    const uint32_t cdims[3] = {1, d, dRC};
    tensor_t* C = tensor_create(3, cdims);
    if (!C) return NULL;

    for (uint32_t s = 0; s < d; s++) {
        for (uint32_t r = 0; r < dRA; r++) {
            C->data[mps_idx(0, s, r, 1, d, dRC)] =
                alpha * A->data[mps_idx(0, s, r, 1, d, dRA)];
        }
        for (uint32_t r = 0; r < dRB; r++) {
            C->data[mps_idx(0, s, dRA + r, 1, d, dRC)] =
                beta * B->data[mps_idx(0, s, r, 1, d, dRB)];
        }
    }
    return C;
}

/* Interior site: shape [dL_A + dL_B, d, dR_A + dR_B], block diagonal. */
static tensor_t* combine_interior_site(const tensor_t* A, const tensor_t* B)
{
    const uint32_t d   = A->dims[1];
    const uint32_t dLA = A->dims[0];
    const uint32_t dRA = A->dims[2];
    const uint32_t dLB = B->dims[0];
    const uint32_t dRB = B->dims[2];
    const uint32_t dLC = dLA + dLB;
    const uint32_t dRC = dRA + dRB;

    const uint32_t cdims[3] = {dLC, d, dRC};
    tensor_t* C = tensor_create(3, cdims);
    if (!C) return NULL;

    for (uint32_t s = 0; s < d; s++) {
        /* A block: rows [0, dLA), cols [0, dRA) */
        for (uint32_t l = 0; l < dLA; l++) {
            for (uint32_t r = 0; r < dRA; r++) {
                C->data[mps_idx(l, s, r, dLC, d, dRC)] =
                    A->data[mps_idx(l, s, r, dLA, d, dRA)];
            }
        }
        /* B block: rows [dLA, dLC), cols [dRA, dRC) */
        for (uint32_t l = 0; l < dLB; l++) {
            for (uint32_t r = 0; r < dRB; r++) {
                C->data[mps_idx(dLA + l, s, dRA + r, dLC, d, dRC)] =
                    B->data[mps_idx(l, s, r, dLB, d, dRB)];
            }
        }
    }
    return C;
}

/* Site L-1: shape [dL_A + dL_B, d, 1], stacked vertically. */
static tensor_t* combine_last_site(const tensor_t* A, const tensor_t* B)
{
    const uint32_t d   = A->dims[1];
    const uint32_t dLA = A->dims[0];
    const uint32_t dLB = B->dims[0];
    const uint32_t dLC = dLA + dLB;

    const uint32_t cdims[3] = {dLC, d, 1};
    tensor_t* C = tensor_create(3, cdims);
    if (!C) return NULL;

    for (uint32_t s = 0; s < d; s++) {
        for (uint32_t l = 0; l < dLA; l++) {
            C->data[mps_idx(l, s, 0, dLC, d, 1)] =
                A->data[mps_idx(l, s, 0, dLA, d, 1)];
        }
        for (uint32_t l = 0; l < dLB; l++) {
            C->data[mps_idx(dLA + l, s, 0, dLC, d, 1)] =
                B->data[mps_idx(l, s, 0, dLB, d, 1)];
        }
    }
    return C;
}

tn_mps_state_t* mpo_kpm_mps_combine(
    const tn_mps_state_t* A, double complex alpha,
    const tn_mps_state_t* B, double complex beta)
{
    if (!A || !B) return NULL;
    if (A->num_qubits != B->num_qubits) return NULL;
    const uint32_t L = A->num_qubits;
    if (L == 0) return NULL;

    /* Derive an MPS shell from A's config, then overwrite every
     * tensor with the block-diagonal combination. */
    tn_mps_state_t* C = tn_mps_copy(A);
    if (!C) return NULL;

    /* Replace every tensor. */
    for (uint32_t i = 0; i < L; i++) {
        const tensor_t* Ai = A->tensors[i];
        const tensor_t* Bi = B->tensors[i];
        if (!Ai || !Bi || Ai->dims[1] != Bi->dims[1]) {
            tn_mps_free(C);
            return NULL;
        }
        tensor_t* Ci = NULL;
        if (L == 1) {
            /* Edge case: single-site MPS. Shape is [1, d, 1].  There
             * is no bond to broaden; just scale and add. */
            const uint32_t d = Ai->dims[1];
            const uint32_t cdims[3] = {1, d, 1};
            Ci = tensor_create(3, cdims);
            if (Ci) {
                for (uint32_t s = 0; s < d; s++) {
                    Ci->data[s] = alpha * Ai->data[s] + beta * Bi->data[s];
                }
            }
        } else if (i == 0) {
            Ci = combine_first_site(Ai, alpha, Bi, beta);
        } else if (i == L - 1) {
            Ci = combine_last_site(Ai, Bi);
        } else {
            Ci = combine_interior_site(Ai, Bi);
        }
        if (!Ci) {
            tn_mps_free(C);
            return NULL;
        }
        tensor_free(C->tensors[i]);
        C->tensors[i] = Ci;
    }
    /* Refresh bond dims and invalidate canonical form. */
    if (C->bond_dims) {
        for (uint32_t i = 0; i + 1 < L; i++) {
            C->bond_dims[i] = C->tensors[i]->dims[2];
        }
    }
    C->canonical = TN_CANONICAL_NONE;
    C->canonical_center = -1;
    /* tn_mps_overlap / norm_squared read state->norm as an outer
     * multiplier (lazy normalisation bookkeeping).  A freshly-built
     * combined MPS has no accumulated scale, so set norm = 1 and
     * zero the log factor; the Chebyshev caller applies any rescale
     * through alpha/beta rather than through this field. */
    C->norm = 1.0;
    C->log_norm_factor = 0.0;
    return C;
}

/* ---------------------------------------------------------------- */
/* Apply-H helper                                                    */
/* ---------------------------------------------------------------- */

/* Contract one MPS tensor [dL, d, dR] with one MPO tensor
 * [bL, d, d', bR] over the physical axis, returning a new MPS tensor
 * [dL * bL, d', dR * bR] with the correct interleaved-axis layout.
 *
 * tensor_contract puts surviving A-axes before surviving B-axes, so
 * the raw output has shape [dL, dR, bL, d', bR].  We permute to
 * [dL, bL, d', dR, bR] before the reshape so that the merged left
 * bond is ordered (dL major, bL minor) and the merged right bond
 * is ordered (dR major, bR minor).  The in-tree tn_apply_mpo in
 * tn_gates.c reshapes without this permutation and therefore
 * computes a different (and not physically meaningful) quantity;
 * mpo_kpm needs the correct MPO-MPS action, so we implement the
 * local step here rather than call into that path.
 */
static tensor_t* site_apply_mpo(const tensor_t* mps_t,
                                const tensor_t* mpo_t)
{
    if (!mps_t || !mpo_t) return NULL;
    if (mps_t->rank != 3 || mpo_t->rank != 4) return NULL;
    if (mps_t->dims[1] != mpo_t->dims[1]) return NULL;

    const uint32_t dL = mps_t->dims[0];
    const uint32_t dR = mps_t->dims[2];
    const uint32_t bL = mpo_t->dims[0];
    const uint32_t dPO = mpo_t->dims[2];
    const uint32_t bR = mpo_t->dims[3];

    uint32_t axes_mps[1] = {1};
    uint32_t axes_mpo[1] = {1};
    tensor_t* raw = tensor_contract(mps_t, mpo_t, axes_mps, axes_mpo, 1);
    /* raw shape [dL, dR, bL, dPO, bR] */
    if (!raw) return NULL;

    uint32_t perm[5] = {0, 2, 3, 1, 4};
    tensor_t* perm_t = tensor_transpose(raw, perm);
    tensor_free(raw);
    if (!perm_t) return NULL;
    /* perm_t shape [dL, bL, dPO, dR, bR] */

    uint32_t new_dims[3] = {dL * bL, dPO, dR * bR};
    tensor_t* out = tensor_reshape(perm_t, 3, new_dims);
    tensor_free(perm_t);
    return out;
}

/* Copy @p src, apply @p H site-by-site, SVD-truncate back to
 * @p max_bond, return the result. */
static tn_mps_state_t* apply_H_copy(const tn_mpo_t* H,
                                    const tn_mps_state_t* src,
                                    uint32_t max_bond)
{
    if (!H || !src) return NULL;
    if (H->num_sites != src->num_qubits) return NULL;

    tn_mps_state_t* out = tn_mps_copy(src);
    if (!out) return NULL;
    out->config.max_bond_dim = max_bond;

    for (uint32_t i = 0; i < src->num_qubits; i++) {
        tensor_t* new_site = site_apply_mpo(out->tensors[i], H->tensors[i]);
        if (!new_site) { tn_mps_free(out); return NULL; }
        tensor_free(out->tensors[i]);
        out->tensors[i] = new_site;
    }
    if (out->bond_dims) {
        for (uint32_t i = 0; i + 1 < out->num_qubits; i++) {
            out->bond_dims[i] = out->tensors[i]->dims[2];
        }
    }
    out->canonical = TN_CANONICAL_NONE;
    out->canonical_center = -1;

    if (max_bond > 0) {
        double err = 0.0;
        tn_mps_truncate(out, max_bond, &err);
    }
    return out;
}

/* Combine and truncate in one step.  Destroys A and B; returns the
 * truncated sum. */
static tn_mps_state_t* combine_and_truncate(
    tn_mps_state_t* A, double complex alpha,
    tn_mps_state_t* B, double complex beta,
    uint32_t max_bond)
{
    tn_mps_state_t* C = mpo_kpm_mps_combine(A, alpha, B, beta);
    tn_mps_free(A);
    tn_mps_free(B);
    if (!C) return NULL;
    if (max_bond > 0) {
        double err = 0.0;
        tn_mps_truncate(C, max_bond, &err);
    }
    return C;
}

/* ---------------------------------------------------------------- */
/* Streaming Chebyshev moments                                       */
/* ---------------------------------------------------------------- */

int mpo_kpm_chebyshev_moments(
    const tn_mpo_t* H,
    const tn_mps_state_t* bra,
    const tn_mps_state_t* ket,
    const mpo_kpm_params_t* params,
    double complex* moments_out)
{
    if (!H || !bra || !ket || !params || !moments_out) return -1;
    if (params->n_cheby == 0) return -2;
    if (params->E_scale <= 0.0) return -3;

    const double a = params->E_shift;
    const double b = params->E_scale;
    const uint32_t max_bond = params->max_bond_dim;

    /* v_0 = ket (working copy, since we mutate in the recurrence). */
    tn_mps_state_t* v_prev = tn_mps_copy(ket);
    if (!v_prev) return -4;
    v_prev->config.max_bond_dim = max_bond;

    /* μ_0 = <bra|v_0>. */
    moments_out[0] = tn_mps_overlap(bra, v_prev);
    if (params->n_cheby == 1) {
        tn_mps_free(v_prev);
        return 0;
    }

    /* v_1 = H_tilde |ket> = (1/b)(H|ket> - a|ket>).
     * Build it as a two-term linear combine of H|ket> (scale 1/b) and
     * |ket> (scale -a/b). */
    tn_mps_state_t* Hk = apply_H_copy(H, v_prev, max_bond);
    if (!Hk) { tn_mps_free(v_prev); return -5; }

    tn_mps_state_t* ket_copy = tn_mps_copy(v_prev);
    if (!ket_copy) { tn_mps_free(v_prev); tn_mps_free(Hk); return -6; }

    tn_mps_state_t* v_curr = combine_and_truncate(
        Hk, (double complex)(1.0 / b),
        ket_copy, (double complex)(-a / b),
        max_bond);
    if (!v_curr) { tn_mps_free(v_prev); return -7; }

    moments_out[1] = tn_mps_overlap(bra, v_curr);

    /* Recurrence: v_{n+1} = 2 H_tilde v_n - v_{n-1}.
     * In bare-H form:
     *   v_{n+1} = (2/b) H v_n - (2 a / b) v_n - v_{n-1}.
     * Done as two linear-combines:
     *   tmp1 = (2/b) H v_n + (-2 a / b) v_n
     *   v_{n+1} = tmp1 + (-1) v_{n-1}
     */
    for (size_t n = 2; n < params->n_cheby; n++) {
        tn_mps_state_t* Hv = apply_H_copy(H, v_curr, max_bond);
        if (!Hv) { tn_mps_free(v_prev); tn_mps_free(v_curr); return -8; }

        tn_mps_state_t* v_curr_copy = tn_mps_copy(v_curr);
        if (!v_curr_copy) {
            tn_mps_free(v_prev); tn_mps_free(v_curr); tn_mps_free(Hv);
            return -9;
        }

        tn_mps_state_t* tmp1 = combine_and_truncate(
            Hv,          (double complex)(2.0 / b),
            v_curr_copy, (double complex)(-2.0 * a / b),
            max_bond);
        if (!tmp1) { tn_mps_free(v_prev); tn_mps_free(v_curr); return -10; }

        tn_mps_state_t* v_next = mpo_kpm_mps_combine(
            tmp1, 1.0, v_prev, -1.0);
        tn_mps_free(tmp1);
        if (!v_next) { tn_mps_free(v_prev); tn_mps_free(v_curr); return -11; }
        if (max_bond > 0) {
            double err = 0.0;
            tn_mps_truncate(v_next, max_bond, &err);
        }

        moments_out[n] = tn_mps_overlap(bra, v_next);

        /* Roll: v_prev := v_curr ; v_curr := v_next. */
        tn_mps_free(v_prev);
        v_prev = v_curr;
        v_curr = v_next;
    }

    tn_mps_free(v_prev);
    tn_mps_free(v_curr);
    return 0;
}

/* ---------------------------------------------------------------- */
/* High-level reconstructions                                        */
/* ---------------------------------------------------------------- */

static double complex reconstruct_sign_from_moments(
    const double complex* moments,
    const mpo_kpm_params_t* params)
{
    const size_t N = params->n_cheby;
    double* c = (double*)malloc(N * sizeof(double));
    double* g = NULL;
    if (!c) return 0.0;
    mpo_kpm_sign_coefficients(N, c);
    if (params->use_jackson) {
        g = (double*)malloc(N * sizeof(double));
        if (!g) { free(c); return 0.0; }
        mpo_kpm_jackson_weights(N, g);
    }
    double complex acc = 0.0;
    for (size_t n = 0; n < N; n++) {
        const double w = (params->use_jackson ? g[n] : 1.0) * c[n];
        acc += w * moments[n];
    }
    free(c);
    if (g) free(g);
    return acc;
}

double complex mpo_kpm_sign_matrix_element(
    const tn_mpo_t* H,
    const tn_mps_state_t* bra,
    const tn_mps_state_t* ket,
    const mpo_kpm_params_t* params)
{
    if (!params) return 0.0;
    double complex* moments =
        (double complex*)malloc(params->n_cheby * sizeof(double complex));
    if (!moments) return 0.0;
    int rc = mpo_kpm_chebyshev_moments(H, bra, ket, params, moments);
    double complex out = (rc == 0)
        ? reconstruct_sign_from_moments(moments, params) : 0.0;
    free(moments);
    return out;
}

double complex mpo_kpm_projector_matrix_element(
    const tn_mpo_t* H,
    const tn_mps_state_t* bra,
    const tn_mps_state_t* ket,
    const mpo_kpm_params_t* params)
{
    if (!params || !bra || !ket) return 0.0;
    const double complex overlap = tn_mps_overlap(bra, ket);
    const double complex sign_me =
        mpo_kpm_sign_matrix_element(H, bra, ket, params);
    return 0.5 * (overlap - sign_me);
}
