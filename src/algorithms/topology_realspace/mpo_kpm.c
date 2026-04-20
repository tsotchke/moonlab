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

/* ---------------------------------------------------------------- */
/* Full MPS projector / sign application                             */
/* ---------------------------------------------------------------- */

/* Zero MPS with the same chain length and physical dimension as
 * @p tmpl, bond dimension 1 everywhere, and all amplitudes = 0.
 * Used as the initial accumulator for the Chebyshev sum. */
static tn_mps_state_t* zero_mps_like(const tn_mps_state_t* tmpl) {
    if (!tmpl || tmpl->num_qubits == 0) return NULL;
    tn_state_config_t cfg = tmpl->config;
    tn_mps_state_t* z = tn_mps_create_zero(tmpl->num_qubits, &cfg);
    if (!z) return NULL;
    /* tn_mps_create_zero builds |00...0>, i.e. amplitude-1 on one
     * basis state. Zero every tensor to produce the null vector. */
    for (uint32_t i = 0; i < z->num_qubits; i++) {
        tensor_t* t = z->tensors[i];
        if (t && t->data) {
            memset(t->data, 0, t->total_size * sizeof(double complex));
        }
    }
    z->norm = 1.0;
    z->log_norm_factor = 0.0;
    z->canonical = TN_CANONICAL_NONE;
    z->canonical_center = -1;
    return z;
}

tn_mps_state_t* mpo_kpm_apply_sign(
    const tn_mpo_t* H,
    const tn_mps_state_t* ket,
    const mpo_kpm_params_t* params)
{
    if (!H || !ket || !params) return NULL;
    if (params->n_cheby == 0) return NULL;
    if (params->E_scale <= 0.0) return NULL;

    const double a = params->E_shift;
    const double b = params->E_scale;
    const uint32_t max_bond = params->max_bond_dim;
    const size_t N = params->n_cheby;

    /* Precompute Jackson + sign coefficients once. */
    double* cs = (double*)malloc(N * sizeof(double));
    double* gs = (double*)malloc(N * sizeof(double));
    if (!cs || !gs) { free(cs); free(gs); return NULL; }
    mpo_kpm_sign_coefficients(N, cs);
    if (params->use_jackson) {
        mpo_kpm_jackson_weights(N, gs);
    } else {
        for (size_t n = 0; n < N; n++) gs[n] = 1.0;
    }

    /* Running state: v_prev = T_{n-1} |ket>, v_curr = T_n |ket>,
     * acc = sum_{m=0}^{n} g_m c_m T_m |ket>. */
    tn_mps_state_t* v_prev = tn_mps_copy(ket);
    if (!v_prev) { free(cs); free(gs); return NULL; }
    v_prev->config.max_bond_dim = max_bond;

    tn_mps_state_t* acc = zero_mps_like(ket);
    if (!acc) { tn_mps_free(v_prev); free(cs); free(gs); return NULL; }

    /* c_0 = 0; nothing to add at n = 0. */

    if (N == 1) { free(cs); free(gs); tn_mps_free(v_prev); return acc; }

    /* v_1 = (1/b) H |ket> + (-a/b) |ket>. */
    tn_mps_state_t* Hk = apply_H_copy(H, v_prev, max_bond);
    if (!Hk) {
        tn_mps_free(v_prev); tn_mps_free(acc);
        free(cs); free(gs); return NULL;
    }
    tn_mps_state_t* ket_copy = tn_mps_copy(v_prev);
    if (!ket_copy) {
        tn_mps_free(v_prev); tn_mps_free(acc); tn_mps_free(Hk);
        free(cs); free(gs); return NULL;
    }
    tn_mps_state_t* v_curr = combine_and_truncate(
        Hk, (double complex)(1.0 / b),
        ket_copy, (double complex)(-a / b),
        max_bond);
    if (!v_curr) {
        tn_mps_free(v_prev); tn_mps_free(acc);
        free(cs); free(gs); return NULL;
    }

    /* acc += g_1 c_1 * v_1. */
    {
        tn_mps_state_t* v1_copy = tn_mps_copy(v_curr);
        if (!v1_copy) {
            tn_mps_free(v_prev); tn_mps_free(v_curr); tn_mps_free(acc);
            free(cs); free(gs); return NULL;
        }
        tn_mps_state_t* next_acc = combine_and_truncate(
            acc, 1.0,
            v1_copy, (double complex)(gs[1] * cs[1]),
            max_bond);
        if (!next_acc) {
            tn_mps_free(v_prev); tn_mps_free(v_curr);
            free(cs); free(gs); return NULL;
        }
        acc = next_acc;
    }

    for (size_t n = 2; n < N; n++) {
        /* v_next = (2/b) H |v_curr> - (2 a / b) |v_curr> - |v_prev>. */
        tn_mps_state_t* Hv = apply_H_copy(H, v_curr, max_bond);
        tn_mps_state_t* v_curr_copy = tn_mps_copy(v_curr);
        if (!Hv || !v_curr_copy) {
            if (Hv) tn_mps_free(Hv);
            if (v_curr_copy) tn_mps_free(v_curr_copy);
            tn_mps_free(v_prev); tn_mps_free(v_curr); tn_mps_free(acc);
            free(cs); free(gs); return NULL;
        }
        tn_mps_state_t* tmp1 = combine_and_truncate(
            Hv,          (double complex)(2.0 / b),
            v_curr_copy, (double complex)(-2.0 * a / b),
            max_bond);
        if (!tmp1) {
            tn_mps_free(v_prev); tn_mps_free(v_curr); tn_mps_free(acc);
            free(cs); free(gs); return NULL;
        }
        tn_mps_state_t* v_next = mpo_kpm_mps_combine(tmp1, 1.0, v_prev, -1.0);
        tn_mps_free(tmp1);
        if (!v_next) {
            tn_mps_free(v_prev); tn_mps_free(v_curr); tn_mps_free(acc);
            free(cs); free(gs); return NULL;
        }
        if (max_bond > 0) {
            double err = 0.0;
            tn_mps_truncate(v_next, max_bond, &err);
        }

        /* acc += g_n c_n * v_next. */
        const double w = gs[n] * cs[n];
        if (w != 0.0) {
            tn_mps_state_t* vn_copy = tn_mps_copy(v_next);
            if (!vn_copy) {
                tn_mps_free(v_prev); tn_mps_free(v_curr); tn_mps_free(v_next);
                tn_mps_free(acc);
                free(cs); free(gs); return NULL;
            }
            tn_mps_state_t* next_acc = combine_and_truncate(
                acc, 1.0,
                vn_copy, (double complex)w,
                max_bond);
            if (!next_acc) {
                tn_mps_free(v_prev); tn_mps_free(v_curr); tn_mps_free(v_next);
                free(cs); free(gs); return NULL;
            }
            acc = next_acc;
        }

        /* Roll. */
        tn_mps_free(v_prev);
        v_prev = v_curr;
        v_curr = v_next;
    }

    tn_mps_free(v_prev);
    tn_mps_free(v_curr);
    free(cs);
    free(gs);
    return acc;
}

tn_mps_state_t* mpo_kpm_apply_projector(
    const tn_mpo_t* H,
    const tn_mps_state_t* ket,
    const mpo_kpm_params_t* params)
{
    if (!params || !ket || !H) return NULL;
    tn_mps_state_t* sign_ket = mpo_kpm_apply_sign(H, ket, params);
    if (!sign_ket) return NULL;
    tn_mps_state_t* ket_copy = tn_mps_copy(ket);
    if (!ket_copy) { tn_mps_free(sign_ket); return NULL; }
    tn_mps_state_t* out = mpo_kpm_mps_combine(
        ket_copy, 0.5, sign_ket, -0.5);
    tn_mps_free(ket_copy);
    tn_mps_free(sign_ket);
    if (!out) return NULL;
    if (params->max_bond_dim > 0) {
        double err = 0.0;
        tn_mps_truncate(out, params->max_bond_dim, &err);
    }
    return out;
}

/* ---------------------------------------------------------------- */
/* Diagonal single-site-sum MPO: D = sum_i f_i * O_i                  */
/* ---------------------------------------------------------------- */

/* Build a single MPO tensor for one site of the finite-automaton
 * chain.  Layout [b_l, phys_in, phys_out, b_r]:
 *   b_l = 0 "not-yet-fired":
 *     b_r = 0 -> identity pass-through
 *     b_r = 1 -> fire f_i * O  (terminate into accumulator channel)
 *   b_l = 1 "already-fired":
 *     b_r = 1 -> identity pass-through (carry accumulator)
 *
 * Boundary conventions:
 *   - Site 0: left bond dim 1 with b_l = 0 forced (start state).
 *   - Site L-1: right bond dim 1 with b_r = 1 forced (accept state).
 */
static tensor_t* diagonal_mpo_site_tensor(uint32_t site, uint32_t L,
                                          const double complex op[4],
                                          double f_i)
{
    const uint32_t d = 2;
    uint32_t b_l = (site == 0) ? 1 : 2;
    uint32_t b_r = (site == L - 1) ? 1 : 2;

    const uint32_t dims[4] = {b_l, d, d, b_r};
    tensor_t* W = tensor_create(4, dims);
    if (!W) return NULL;
    const uint32_t ld_pi = d * b_r;         /* stride along phys_in */
    const uint32_t ld_bl = d * ld_pi;       /* stride along b_l */

    /* Identity 2x2 in (phys_in, phys_out) = diag(1). */
    double complex I4[4] = {1.0, 0.0, 0.0, 1.0};

    /* Logical (bl, br) entries to fill:
     *   (0, 0): I                         (always present, except
     *                                      at site L-1 where b_r == 1
     *                                      and we need the "fire"
     *                                      channel, so this entry
     *                                      is instead encoded at
     *                                      (0, 0) as f * O --
     *                                      handled below)
     *   (0, 1): f_i * O_i                 (fire-and-terminate) --
     *                                      when b_r > 1
     *   (1, 1): I                         (carry accumulator) --
     *                                      when b_l > 1
     */

    /* (0, 0) slot.  On internal sites this is "pass the
     * not-yet-fired state through as identity"; on site L-1 the
     * right boundary collapses to b_r = 1 and the only way to
     * terminate without having fired earlier is to fire at this
     * last opportunity, which means (0, 0) becomes f * O. */
    for (uint32_t pi = 0; pi < d; pi++) {
        for (uint32_t po = 0; po < d; po++) {
            const uint32_t idx = 0 * ld_bl + pi * ld_pi + po * b_r + 0;
            if (site == L - 1) {
                W->data[idx] = f_i * op[pi * d + po];
            } else {
                W->data[idx] = I4[pi * d + po];
            }
        }
    }

    /* (0, 1) fire-slot: only exists when b_r >= 2, i.e. not on the
     * right boundary.  At site 0 this lives in (b_l=0, b_r=1). */
    if (b_r == 2) {
        for (uint32_t pi = 0; pi < d; pi++) {
            for (uint32_t po = 0; po < d; po++) {
                const uint32_t idx = 0 * ld_bl + pi * ld_pi + po * b_r + 1;
                W->data[idx] = f_i * op[pi * d + po];
            }
        }
    }

    /* (1, 1) carry-slot: only exists when b_l >= 2, i.e. not on
     * the left boundary. */
    if (b_l == 2 && b_r >= 1) {
        const uint32_t br = (b_r == 1) ? 0 : 1;
        for (uint32_t pi = 0; pi < d; pi++) {
            for (uint32_t po = 0; po < d; po++) {
                const uint32_t idx = 1 * ld_bl + pi * ld_pi + po * b_r + br;
                W->data[idx] = I4[pi * d + po];
            }
        }
    }

    return W;
}

tn_mpo_t* mpo_kpm_diagonal_sum_mpo(uint32_t num_sites,
                                   const double complex op[4],
                                   const double* f_per_site)
{
    if (num_sites == 0 || !op || !f_per_site) return NULL;

    tn_mpo_t* mpo = (tn_mpo_t*)calloc(1, sizeof(tn_mpo_t));
    if (!mpo) return NULL;
    mpo->num_sites = num_sites;
    mpo->tensors   = (tensor_t**)calloc(num_sites, sizeof(tensor_t*));
    if (num_sites >= 2) {
        mpo->bond_dims = (uint32_t*)calloc(num_sites - 1, sizeof(uint32_t));
    }
    if (!mpo->tensors || (num_sites >= 2 && !mpo->bond_dims)) {
        tn_mpo_free(mpo);
        return NULL;
    }

    for (uint32_t i = 0; i < num_sites; i++) {
        tensor_t* W = diagonal_mpo_site_tensor(i, num_sites, op,
                                                f_per_site[i]);
        if (!W) { tn_mpo_free(mpo); return NULL; }
        mpo->tensors[i] = W;
    }
    for (uint32_t i = 0; i + 1 < num_sites; i++) {
        mpo->bond_dims[i] = mpo->tensors[i]->dims[3];
    }
    return mpo;
}

tn_mps_state_t* mpo_kpm_apply_mpo(const tn_mpo_t* op,
                                  const tn_mps_state_t* in,
                                  uint32_t max_bond_dim)
{
    return apply_H_copy(op, in, max_bond_dim);
}

/* ================================================================ */
/* MPO-level Chebyshev machinery (P5.08 step 3)                     */
/* ================================================================ */

/* ---------------------------------------------------------------- */
/* MPO constructors                                                   */
/* ---------------------------------------------------------------- */

tn_mpo_t* mpo_kpm_identity_mpo(uint32_t num_sites) {
    if (num_sites == 0) return NULL;
    tn_mpo_t* mpo = (tn_mpo_t*)calloc(1, sizeof(tn_mpo_t));
    if (!mpo) return NULL;
    mpo->num_sites = num_sites;
    mpo->tensors = (tensor_t**)calloc(num_sites, sizeof(tensor_t*));
    if (num_sites >= 2) {
        mpo->bond_dims = (uint32_t*)calloc(num_sites - 1, sizeof(uint32_t));
    }
    if (!mpo->tensors || (num_sites >= 2 && !mpo->bond_dims)) {
        tn_mpo_free(mpo);
        return NULL;
    }
    for (uint32_t i = 0; i < num_sites; i++) {
        const uint32_t dims[4] = {1, 2, 2, 1};
        tensor_t* t = tensor_create(4, dims);
        if (!t) { tn_mpo_free(mpo); return NULL; }
        t->data[0 * 2 * 2 * 1 + 0 * 2 * 1 + 0 * 1 + 0] = 1.0; /* [0,0,0,0] */
        t->data[0 * 2 * 2 * 1 + 1 * 2 * 1 + 1 * 1 + 0] = 1.0; /* [0,1,1,0] */
        mpo->tensors[i] = t;
    }
    for (uint32_t i = 0; i + 1 < num_sites; i++) mpo->bond_dims[i] = 1;
    return mpo;
}

/* ---------------------------------------------------------------- */
/* MPO bond SVD compression                                           */
/* ---------------------------------------------------------------- */

/* Reshape a rank-4 MPO site [bL, d_in, d_out, bR] to a rank-3
 * MPS-like [bL, d_in * d_out, bR] by merging the two physical axes.
 * Returns a new tensor; caller owns it. */
static tensor_t* mpo_site_as_mps(const tensor_t* s4) {
    if (!s4 || s4->rank != 4) return NULL;
    const uint32_t bL = s4->dims[0], d_in = s4->dims[1];
    const uint32_t d_out = s4->dims[2], bR = s4->dims[3];
    const uint32_t dims[3] = {bL, d_in * d_out, bR};
    tensor_t* out = tensor_create(3, dims);
    if (!out) return NULL;
    memcpy(out->data, s4->data,
           s4->total_size * sizeof(double complex));
    return out;
}

/* Inverse: rank-3 [bL, d*d, bR] -> rank-4 [bL, d, d, bR]. */
static tensor_t* mps_back_to_mpo_site(const tensor_t* s3,
                                       uint32_t d_in, uint32_t d_out) {
    if (!s3 || s3->rank != 3) return NULL;
    if (s3->dims[1] != d_in * d_out) return NULL;
    const uint32_t bL = s3->dims[0], bR = s3->dims[2];
    const uint32_t dims[4] = {bL, d_in, d_out, bR};
    tensor_t* out = tensor_create(4, dims);
    if (!out) return NULL;
    memcpy(out->data, s3->data,
           s3->total_size * sizeof(double complex));
    return out;
}

/* Single-bond SVD compression between two adjacent MPO sites.  Wraps
 * the MPS bond compressor by reshaping; the physical-axis layout is
 * preserved exactly. */
static int mpo_bond_svd_compress(tensor_t** left, tensor_t** right,
                                  uint32_t max_bond) {
    if (!left || !right || !*left || !*right) return -1;
    if ((*left)->rank != 4 || (*right)->rank != 4) return -1;

    const uint32_t dL_in  = (*left)->dims[1];
    const uint32_t dL_out = (*left)->dims[2];
    const uint32_t dR_in  = (*right)->dims[1];
    const uint32_t dR_out = (*right)->dims[2];

    tensor_t* L3 = mpo_site_as_mps(*left);
    tensor_t* R3 = mpo_site_as_mps(*right);
    if (!L3 || !R3) { tensor_free(L3); tensor_free(R3); return -2; }

    svd_compress_config_t cfg = svd_compress_config_fixed(max_bond);
    svd_compress_result_t* res = svd_compress_bond(L3, R3, &cfg, true);
    tensor_free(L3);
    tensor_free(R3);
    if (!res) return -3;

    tensor_t* L4_new = mps_back_to_mpo_site(res->left,  dL_in, dL_out);
    tensor_t* R4_new = mps_back_to_mpo_site(res->right, dR_in, dR_out);

    res->left = NULL; res->right = NULL;  /* ownership transferred */
    svd_compress_result_free(res);

    if (!L4_new || !R4_new) {
        tensor_free(L4_new); tensor_free(R4_new);
        return -4;
    }
    tensor_free(*left);
    tensor_free(*right);
    *left = L4_new;
    *right = R4_new;
    return 0;
}

/* Sweep SVD compression right-to-left after bringing to
 * left-canonical form via QR-like splits.  Conservative simple
 * version: iterate all adjacent bonds L-to-R, then R-to-L, applying
 * mpo_bond_svd_compress with the bond cap.  Sufficient for keeping
 * MPO bond dim bounded during the Chebyshev recurrence. */
static int mpo_truncate_all_bonds(tn_mpo_t* mpo, uint32_t max_bond) {
    if (!mpo || max_bond == 0) return 0;
    const uint32_t L = mpo->num_sites;
    if (L < 2) return 0;
    /* Left-to-right sweep. */
    for (uint32_t i = 0; i + 1 < L; i++) {
        int rc = mpo_bond_svd_compress(&mpo->tensors[i],
                                        &mpo->tensors[i + 1],
                                        max_bond);
        if (rc != 0) return rc;
    }
    /* Right-to-left sweep. */
    for (int i = (int)L - 2; i >= 0; i--) {
        int rc = mpo_bond_svd_compress(&mpo->tensors[i],
                                        &mpo->tensors[i + 1],
                                        max_bond);
        if (rc != 0) return rc;
    }
    if (mpo->bond_dims) {
        for (uint32_t i = 0; i + 1 < L; i++) {
            mpo->bond_dims[i] = mpo->tensors[i]->dims[3];
        }
    }
    return 0;
}

/* ---------------------------------------------------------------- */
/* MPO x MPO multiplication                                           */
/* ---------------------------------------------------------------- */

/* Per-site: C[(aL,bL), s_in, s_out, (aR,bR)] =
 *   sum_t A[aL, s_in, t, aR] * B[bL, t, s_out, bR].
 *
 * tensor_contract(A, axes={2}, B, axes={1}) gives
 *   raw shape [aL, s_in, aR, bL, s_out, bR];
 * permute {0, 3, 1, 4, 2, 5} -> [aL, bL, s_in, s_out, aR, bR];
 * reshape to [aL*bL, s_in, s_out, aR*bR].
 */
static tensor_t* mpo_site_multiply(const tensor_t* A, const tensor_t* B) {
    if (!A || !B || A->rank != 4 || B->rank != 4) return NULL;
    if (A->dims[2] != B->dims[1]) return NULL;
    const uint32_t aL = A->dims[0], s_in  = A->dims[1];
    const uint32_t aR = A->dims[3];
    const uint32_t bL = B->dims[0], s_out = B->dims[2];
    const uint32_t bR = B->dims[3];

    uint32_t axes_a[1] = {2};
    uint32_t axes_b[1] = {1};
    tensor_t* raw = tensor_contract(A, B, axes_a, axes_b, 1);
    if (!raw) return NULL;
    uint32_t perm[6] = {0, 3, 1, 4, 2, 5};
    tensor_t* p = tensor_transpose(raw, perm);
    tensor_free(raw);
    if (!p) return NULL;
    const uint32_t new_dims[4] = {aL * bL, s_in, s_out, aR * bR};
    tensor_t* out = tensor_reshape(p, 4, new_dims);
    tensor_free(p);
    return out;
}

tn_mpo_t* mpo_kpm_mpo_multiply(const tn_mpo_t* A, const tn_mpo_t* B,
                                uint32_t max_bond_dim)
{
    if (!A || !B) return NULL;
    if (A->num_sites != B->num_sites) return NULL;
    const uint32_t L = A->num_sites;

    tn_mpo_t* C = (tn_mpo_t*)calloc(1, sizeof(tn_mpo_t));
    if (!C) return NULL;
    C->num_sites = L;
    C->tensors = (tensor_t**)calloc(L, sizeof(tensor_t*));
    if (L >= 2) C->bond_dims = (uint32_t*)calloc(L - 1, sizeof(uint32_t));
    if (!C->tensors || (L >= 2 && !C->bond_dims)) {
        tn_mpo_free(C); return NULL;
    }

    for (uint32_t i = 0; i < L; i++) {
        C->tensors[i] = mpo_site_multiply(A->tensors[i], B->tensors[i]);
        if (!C->tensors[i]) { tn_mpo_free(C); return NULL; }
    }
    for (uint32_t i = 0; i + 1 < L; i++) {
        C->bond_dims[i] = C->tensors[i]->dims[3];
    }
    if (max_bond_dim > 0) mpo_truncate_all_bonds(C, max_bond_dim);
    return C;
}

/* ---------------------------------------------------------------- */
/* MPO alpha A + beta B via block-diagonal construction               */
/* ---------------------------------------------------------------- */

static tensor_t* mpo_combine_first_site(
    const tensor_t* A, double complex alpha,
    const tensor_t* B, double complex beta)
{
    const uint32_t d_in = A->dims[1];
    const uint32_t d_out = A->dims[2];
    const uint32_t aR = A->dims[3];
    const uint32_t bR = B->dims[3];
    const uint32_t new_dims[4] = {1, d_in, d_out, aR + bR};
    tensor_t* C = tensor_create(4, new_dims);
    if (!C) return NULL;
    for (uint32_t pi = 0; pi < d_in; pi++) {
        for (uint32_t po = 0; po < d_out; po++) {
            for (uint32_t r = 0; r < aR; r++) {
                const size_t ai = ((size_t)0 * d_in + pi) * d_out * aR
                                + (size_t)po * aR + r;
                const size_t ci = ((size_t)0 * d_in + pi) * d_out * (aR + bR)
                                + (size_t)po * (aR + bR) + r;
                C->data[ci] = alpha * A->data[ai];
            }
            for (uint32_t r = 0; r < bR; r++) {
                const size_t bi = ((size_t)0 * d_in + pi) * d_out * bR
                                + (size_t)po * bR + r;
                const size_t ci = ((size_t)0 * d_in + pi) * d_out * (aR + bR)
                                + (size_t)po * (aR + bR) + (aR + r);
                C->data[ci] = beta * B->data[bi];
            }
        }
    }
    return C;
}

static tensor_t* mpo_combine_interior_site(const tensor_t* A,
                                           const tensor_t* B)
{
    const uint32_t d_in = A->dims[1];
    const uint32_t d_out = A->dims[2];
    const uint32_t aL = A->dims[0], aR = A->dims[3];
    const uint32_t bL = B->dims[0], bR = B->dims[3];
    const uint32_t new_dims[4] = {aL + bL, d_in, d_out, aR + bR};
    tensor_t* C = tensor_create(4, new_dims);
    if (!C) return NULL;
    for (uint32_t pi = 0; pi < d_in; pi++) {
        for (uint32_t po = 0; po < d_out; po++) {
            for (uint32_t l = 0; l < aL; l++) {
                for (uint32_t r = 0; r < aR; r++) {
                    const size_t ai = ((size_t)l * d_in + pi) * d_out * aR
                                    + (size_t)po * aR + r;
                    const size_t ci = ((size_t)l * d_in + pi) * d_out * (aR + bR)
                                    + (size_t)po * (aR + bR) + r;
                    C->data[ci] = A->data[ai];
                }
            }
            for (uint32_t l = 0; l < bL; l++) {
                for (uint32_t r = 0; r < bR; r++) {
                    const size_t bi = ((size_t)l * d_in + pi) * d_out * bR
                                    + (size_t)po * bR + r;
                    const size_t ci = ((size_t)(aL + l) * d_in + pi)
                                      * d_out * (aR + bR)
                                    + (size_t)po * (aR + bR) + (aR + r);
                    C->data[ci] = B->data[bi];
                }
            }
        }
    }
    return C;
}

static tensor_t* mpo_combine_last_site(const tensor_t* A, const tensor_t* B)
{
    const uint32_t d_in = A->dims[1];
    const uint32_t d_out = A->dims[2];
    const uint32_t aL = A->dims[0];
    const uint32_t bL = B->dims[0];
    const uint32_t new_dims[4] = {aL + bL, d_in, d_out, 1};
    tensor_t* C = tensor_create(4, new_dims);
    if (!C) return NULL;
    for (uint32_t pi = 0; pi < d_in; pi++) {
        for (uint32_t po = 0; po < d_out; po++) {
            for (uint32_t l = 0; l < aL; l++) {
                const size_t ai = ((size_t)l * d_in + pi) * d_out
                                + (size_t)po;
                const size_t ci = ((size_t)l * d_in + pi) * d_out
                                + (size_t)po;
                C->data[ci] = A->data[ai];
            }
            for (uint32_t l = 0; l < bL; l++) {
                const size_t bi = ((size_t)l * d_in + pi) * d_out
                                + (size_t)po;
                const size_t ci = ((size_t)(aL + l) * d_in + pi) * d_out
                                + (size_t)po;
                C->data[ci] = B->data[bi];
            }
        }
    }
    return C;
}

tn_mpo_t* mpo_kpm_mpo_combine(const tn_mpo_t* A, double complex alpha,
                               const tn_mpo_t* B, double complex beta,
                               uint32_t max_bond_dim)
{
    if (!A || !B) return NULL;
    if (A->num_sites != B->num_sites) return NULL;
    const uint32_t L = A->num_sites;
    if (L == 0) return NULL;

    tn_mpo_t* C = (tn_mpo_t*)calloc(1, sizeof(tn_mpo_t));
    if (!C) return NULL;
    C->num_sites = L;
    C->tensors = (tensor_t**)calloc(L, sizeof(tensor_t*));
    if (L >= 2) C->bond_dims = (uint32_t*)calloc(L - 1, sizeof(uint32_t));
    if (!C->tensors || (L >= 2 && !C->bond_dims)) {
        tn_mpo_free(C); return NULL;
    }

    for (uint32_t i = 0; i < L; i++) {
        const tensor_t* Ai = A->tensors[i];
        const tensor_t* Bi = B->tensors[i];
        if (!Ai || !Bi) { tn_mpo_free(C); return NULL; }
        if (Ai->dims[1] != Bi->dims[1] || Ai->dims[2] != Bi->dims[2]) {
            tn_mpo_free(C); return NULL;
        }
        tensor_t* Ci = NULL;
        if (L == 1) {
            /* Single-site: [1, d_in, d_out, 1]; just add. */
            const uint32_t d_in = Ai->dims[1], d_out = Ai->dims[2];
            const uint32_t dims[4] = {1, d_in, d_out, 1};
            Ci = tensor_create(4, dims);
            if (Ci) {
                for (size_t k = 0; k < Ai->total_size; k++) {
                    Ci->data[k] = alpha * Ai->data[k] + beta * Bi->data[k];
                }
            }
        } else if (i == 0) {
            Ci = mpo_combine_first_site(Ai, alpha, Bi, beta);
        } else if (i == L - 1) {
            Ci = mpo_combine_last_site(Ai, Bi);
        } else {
            Ci = mpo_combine_interior_site(Ai, Bi);
        }
        if (!Ci) { tn_mpo_free(C); return NULL; }
        C->tensors[i] = Ci;
    }
    for (uint32_t i = 0; i + 1 < L; i++) {
        C->bond_dims[i] = C->tensors[i]->dims[3];
    }
    if (max_bond_dim > 0) mpo_truncate_all_bonds(C, max_bond_dim);
    return C;
}

/* ---------------------------------------------------------------- */
/* MPO-level Chebyshev recurrence for sign(H_tilde)                   */
/* ---------------------------------------------------------------- */

tn_mpo_t* mpo_kpm_sign_mpo(const tn_mpo_t* H,
                            const mpo_kpm_params_t* params)
{
    if (!H || !params) return NULL;
    if (params->n_cheby == 0) return NULL;
    if (params->E_scale <= 0.0) return NULL;
    const uint32_t L = H->num_sites;
    const uint32_t max_bond = params->max_bond_dim;
    const double a = params->E_shift;
    const double b = params->E_scale;
    const size_t N = params->n_cheby;

    /* Coefficients. */
    double* cs = (double*)malloc(N * sizeof(double));
    double* gs = (double*)malloc(N * sizeof(double));
    if (!cs || !gs) { free(cs); free(gs); return NULL; }
    mpo_kpm_sign_coefficients(N, cs);
    if (params->use_jackson) mpo_kpm_jackson_weights(N, gs);
    else for (size_t n = 0; n < N; n++) gs[n] = 1.0;

    /* T_0 = I, T_1 = H_tilde = (H - a I) / b.
     * Build T_1 as combine(H, 1/b, I, -a/b). */
    tn_mpo_t* I_mpo = mpo_kpm_identity_mpo(L);
    if (!I_mpo) { free(cs); free(gs); return NULL; }
    tn_mpo_t* T_prev = mpo_kpm_identity_mpo(L);       /* T_0 */
    tn_mpo_t* T_curr = mpo_kpm_mpo_combine(H, 1.0 / b,
                                            I_mpo, -a / b,
                                            max_bond);  /* T_1 */
    if (!T_prev || !T_curr) {
        tn_mpo_free(I_mpo);
        if (T_prev) tn_mpo_free(T_prev);
        if (T_curr) tn_mpo_free(T_curr);
        free(cs); free(gs); return NULL;
    }

    /* Accumulator: g_0 c_0 T_0 + g_1 c_1 T_1.  c_0 = 0 so start from
     * a zero-valued MPO and add g_1 c_1 T_1.  Build zero as scalar 0
     * times identity via combine. */
    tn_mpo_t* zero_mpo = mpo_kpm_mpo_combine(I_mpo, 0.0, I_mpo, 0.0,
                                              max_bond);
    tn_mpo_free(I_mpo);
    if (!zero_mpo) {
        tn_mpo_free(T_prev); tn_mpo_free(T_curr);
        free(cs); free(gs); return NULL;
    }
    tn_mpo_t* acc = mpo_kpm_mpo_combine(zero_mpo, 1.0,
                                         T_curr, gs[1] * cs[1],
                                         max_bond);
    tn_mpo_free(zero_mpo);
    if (!acc) {
        tn_mpo_free(T_prev); tn_mpo_free(T_curr);
        free(cs); free(gs); return NULL;
    }

    /* Recurrence: T_{n+1} = 2 H_tilde T_n - T_{n-1}.
     * We don't have H_tilde as a standalone MPO (we'd need to rebuild
     * it), but we can use T_curr which at iteration n IS T_n, and
     * compute 2 H_tilde T_n via MPO multiply:
     *   T_next = 2 (H * T_curr / b) - (2 a / b) T_curr - T_prev.
     * The (H T_curr / b) is an mpo_multiply; the scalar factors are
     * folded into subsequent combines. */
    for (size_t n = 2; n < N; n++) {
        tn_mpo_t* H_T = mpo_kpm_mpo_multiply(H, T_curr, max_bond);
        if (!H_T) break;
        /* 2 H_T / b - (2 a / b) T_curr */
        tn_mpo_t* tmp1 = mpo_kpm_mpo_combine(
            H_T, 2.0 / b,
            T_curr, -2.0 * a / b,
            max_bond);
        tn_mpo_free(H_T);
        if (!tmp1) break;
        /* T_next = tmp1 - T_prev. */
        tn_mpo_t* T_next = mpo_kpm_mpo_combine(tmp1, 1.0,
                                                 T_prev, -1.0,
                                                 max_bond);
        tn_mpo_free(tmp1);
        if (!T_next) break;

        /* acc += g_n c_n * T_next. */
        const double w = gs[n] * cs[n];
        if (w != 0.0) {
            tn_mpo_t* next_acc = mpo_kpm_mpo_combine(acc, 1.0,
                                                       T_next, w,
                                                       max_bond);
            if (!next_acc) {
                tn_mpo_free(T_next);
                break;
            }
            tn_mpo_free(acc);
            acc = next_acc;
        }
        /* Roll: T_prev := T_curr; T_curr := T_next. */
        tn_mpo_free(T_prev);
        T_prev = T_curr;
        T_curr = T_next;
    }
    tn_mpo_free(T_prev);
    tn_mpo_free(T_curr);
    free(cs); free(gs);
    return acc;
}

tn_mpo_t* mpo_kpm_projector_mpo(const tn_mpo_t* H,
                                 const mpo_kpm_params_t* params)
{
    if (!H || !params) return NULL;
    tn_mpo_t* sign_mpo = mpo_kpm_sign_mpo(H, params);
    if (!sign_mpo) return NULL;
    tn_mpo_t* I_mpo = mpo_kpm_identity_mpo(H->num_sites);
    if (!I_mpo) { tn_mpo_free(sign_mpo); return NULL; }
    tn_mpo_t* P = mpo_kpm_mpo_combine(I_mpo, 0.5, sign_mpo, -0.5,
                                       params->max_bond_dim);
    tn_mpo_free(I_mpo);
    tn_mpo_free(sign_mpo);
    return P;
}

/* ---------------------------------------------------------------- */
/* Dense -> MPO via successive SVD                                   */
/* ---------------------------------------------------------------- */

tn_mpo_t* mpo_kpm_mpo_from_dense(const double complex* M,
                                  uint32_t L,
                                  double svd_cutoff)
{
    if (!M || L == 0) return NULL;
    const size_t N  = (size_t)1 << L;
    const size_t N2 = N * N;

    tn_mpo_t* mpo = (tn_mpo_t*)calloc(1, sizeof(tn_mpo_t));
    if (!mpo) return NULL;
    mpo->num_sites = L;
    mpo->tensors = (tensor_t**)calloc(L, sizeof(tensor_t*));
    if (L >= 2) {
        mpo->bond_dims = (uint32_t*)calloc(L - 1, sizeof(uint32_t));
    }
    if (!mpo->tensors || (L >= 2 && !mpo->bond_dims)) {
        tn_mpo_free(mpo); return NULL;
    }

    /* Bit-interleaved layout: reshape M into a 4^L-long buffer.  At
     * each chain site s the rank-4 MPO tensor has shape
     * [b_l, phys_in, phys_out, b_r]; after we reshape the flat buffer
     * into [rest_rows * 4, ...] the 4 decomposes (in row-major) as
     * (phys_in * 2 + phys_out).  Therefore:
     *   - stride-2 axis (odd bit at position 2k+1) = phys_in.
     *   - stride-1 axis (even bit at position 2k)  = phys_out.
     *
     * For M acting as (M psi)_i = sum_j M_ij psi_j:
     *   - phys_in reads the state -> column index j.
     *   - phys_out writes the state -> row index i.
     *
     * So bit_k(j) must go to the 2k+1 position, bit_k(i) to 2k.
     * A prior version had these swapped, which stored M^T and, for
     * complex-Hermitian inputs with nonzero imaginary off-diagonals,
     * produced an MPO representing conj(M) instead of M.  Fixed. */
    double complex* rest = (double complex*)malloc(N2 * sizeof(double complex));
    if (!rest) { tn_mpo_free(mpo); return NULL; }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            size_t T_idx = 0;
            for (uint32_t k = 0; k < L; k++) {
                if ((j >> k) & 1u) T_idx |= (size_t)1u << (2u * k + 1u);
                if ((i >> k) & 1u) T_idx |= (size_t)1u << (2u * k);
            }
            rest[T_idx] = M[i * N + j];
        }
    }

    size_t rest_rows = 1;
    size_t rest_cols = N2;

    /* Successive SVD from the left: at each step we reshape
     *   rest [rest_rows, rest_cols] -> [rest_rows * 4, rest_cols / 4]
     * SVD to U S V^H, keep U as the MPO tensor at this site, and
     * pass S V^H forward as the new rest. */
    for (uint32_t s = 0; s < L - 1; s++) {
        const size_t mat_rows = rest_rows * 4;
        const size_t mat_cols = rest_cols / 4;

        uint32_t mdims[2] = {(uint32_t)mat_rows, (uint32_t)mat_cols};
        tensor_t* mat = tensor_create(2, mdims);
        if (!mat) { free(rest); tn_mpo_free(mpo); return NULL; }
        memcpy(mat->data, rest, mat_rows * mat_cols * sizeof(double complex));

        tensor_svd_result_t* svd = tensor_svd(mat, 0, svd_cutoff);
        tensor_free(mat);
        if (!svd) { free(rest); tn_mpo_free(mpo); return NULL; }

        const uint32_t r = svd->k;
        uint32_t wdims[4] = {(uint32_t)rest_rows, 2, 2, r};
        tensor_t* W = tensor_create(4, wdims);
        if (!W) {
            tensor_svd_free(svd); free(rest); tn_mpo_free(mpo);
            return NULL;
        }
        memcpy(W->data, svd->U->data,
               (size_t)rest_rows * 4 * r * sizeof(double complex));
        mpo->tensors[s] = W;

        /* new rest = S V^H, shape [r, mat_cols]. */
        double complex* new_rest = (double complex*)malloc(
            (size_t)r * mat_cols * sizeof(double complex));
        if (!new_rest) {
            tensor_svd_free(svd); free(rest); tn_mpo_free(mpo);
            return NULL;
        }
        for (uint32_t i = 0; i < r; i++) {
            for (size_t j = 0; j < mat_cols; j++) {
                new_rest[(size_t)i * mat_cols + j] =
                    svd->S[i] * svd->Vh->data[(size_t)i * mat_cols + j];
            }
        }
        tensor_svd_free(svd);
        free(rest);
        rest = new_rest;
        rest_rows = r;
        rest_cols = mat_cols;
    }

    /* Last site: rest has shape [rest_rows, 4] ==> MPO tensor
     * [rest_rows, 2, 2, 1]. */
    uint32_t wdims_last[4] = {(uint32_t)rest_rows, 2, 2, 1};
    tensor_t* W_last = tensor_create(4, wdims_last);
    if (!W_last) { free(rest); tn_mpo_free(mpo); return NULL; }
    memcpy(W_last->data, rest,
           (size_t)rest_rows * 4 * sizeof(double complex));
    mpo->tensors[L - 1] = W_last;
    free(rest);

    for (uint32_t i = 0; i + 1 < L; i++) {
        mpo->bond_dims[i] = mpo->tensors[i]->dims[3];
    }
    return mpo;
}
