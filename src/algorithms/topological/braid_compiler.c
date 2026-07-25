/**
 * @file braid_compiler.c
 * @brief Braid words, and compilation of target unitaries into them.
 *
 * Three layers:
 *
 *   - braid_word_t: a word in the generators sigma_i^{+-1}, applied to a
 *     fusion_tree_t by braid_anyons(), plus its matrix on the fusion space.
 *   - Exact compilation for Ising.  The image of the braid group in PSU(2) on
 *     4 sigma anyons is the 24-element single-qubit Clifford group, and on 6
 *     sigma anyons it is a finite subgroup of the two-qubit Clifford group;
 *     both are enumerated by breadth-first search of the Cayley graph, so
 *     every reachable target gets an exact (~1e-16) shortest braid word.
 *   - Solovay-Kitaev compilation for Fibonacci.  The image is dense but not
 *     exhaustive in PSU(2), so a caller-supplied epsilon is met by the
 *     standard recursion of Dawson and Nielsen (quant-ph/0505030) over a base
 *     net built by exhaustive enumeration of short braid words.  The returned
 *     word's distance to the target is measured before it is returned, so the
 *     epsilon is a checked guarantee rather than an asymptotic claim.
 *
 * @stability evolving
 * @since v1.2.0
 */

#include "topological.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// BRAID WORDS
// ============================================================================

braid_word_t *braid_word_create(void) {
    braid_word_t *w = calloc(1, sizeof(braid_word_t));
    return w;
}

void braid_word_free(braid_word_t *w) {
    if (!w) return;
    free(w->gens);
    free(w);
}

static qs_error_t bw_reserve(braid_word_t *w, uint32_t need) {
    if (w->capacity >= need) return QS_SUCCESS;
    uint32_t cap = w->capacity ? w->capacity : 16;
    while (cap < need) cap *= 2;
    braid_gen_t *g = realloc(w->gens, (size_t)cap * sizeof(braid_gen_t));
    if (!g) return QS_ERROR_OUT_OF_MEMORY;
    w->gens = g;
    w->capacity = cap;
    return QS_SUCCESS;
}

qs_error_t braid_word_append(braid_word_t *w, uint32_t position, bool clockwise) {
    if (!w) return QS_ERROR_INVALID_STATE;
    qs_error_t e = bw_reserve(w, w->length + 1);
    if (e != QS_SUCCESS) return e;
    w->gens[w->length].position = position;
    w->gens[w->length].clockwise = clockwise ? 1u : 0u;
    w->length++;
    return QS_SUCCESS;
}

qs_error_t braid_word_append_word(braid_word_t *dst, const braid_word_t *src) {
    if (!dst || !src) return QS_ERROR_INVALID_STATE;
    qs_error_t e = bw_reserve(dst, dst->length + src->length);
    if (e != QS_SUCCESS) return e;
    memcpy(dst->gens + dst->length, src->gens,
           (size_t)src->length * sizeof(braid_gen_t));
    dst->length += src->length;
    return QS_SUCCESS;
}

qs_error_t braid_word_append_inverse(braid_word_t *dst, const braid_word_t *src) {
    if (!dst || !src) return QS_ERROR_INVALID_STATE;
    qs_error_t e = bw_reserve(dst, dst->length + src->length);
    if (e != QS_SUCCESS) return e;
    for (uint32_t i = 0; i < src->length; i++) {
        const braid_gen_t *g = &src->gens[src->length - 1 - i];
        dst->gens[dst->length + i].position = g->position;
        dst->gens[dst->length + i].clockwise = g->clockwise ? 0u : 1u;
    }
    dst->length += src->length;
    return QS_SUCCESS;
}

braid_word_t *braid_word_clone(const braid_word_t *w) {
    if (!w) return NULL;
    braid_word_t *c = braid_word_create();
    if (!c) return NULL;
    if (braid_word_append_word(c, w) != QS_SUCCESS) { braid_word_free(c); return NULL; }
    return c;
}

uint32_t braid_word_length(const braid_word_t *w) { return w ? w->length : 0u; }

uint32_t braid_word_reduce(braid_word_t *w) {
    if (!w || w->length == 0) return 0;
    uint32_t top = 0;
    for (uint32_t i = 0; i < w->length; i++) {
        if (top > 0 &&
            w->gens[top - 1].position == w->gens[i].position &&
            w->gens[top - 1].clockwise != w->gens[i].clockwise) {
            top--;                       /* sigma_i sigma_i^{-1} cancels */
        } else {
            w->gens[top++] = w->gens[i];
        }
    }
    w->length = top;
    return top;
}

qs_error_t braid_word_apply(const braid_word_t *w, fusion_tree_t *tree) {
    if (!w || !tree) return QS_ERROR_INVALID_STATE;
    for (uint32_t i = 0; i < w->length; i++) {
        qs_error_t e = braid_anyons(tree, w->gens[i].position,
                                    w->gens[i].clockwise != 0);
        if (e != QS_SUCCESS) return e;
    }
    return QS_SUCCESS;
}

/* Apply a word whose generator positions are relative to `offset`. */
static qs_error_t bw_apply_shifted(const braid_word_t *w, fusion_tree_t *tree,
                                   uint32_t offset) {
    for (uint32_t i = 0; i < w->length; i++) {
        qs_error_t e = braid_anyons(tree, w->gens[i].position + offset,
                                    w->gens[i].clockwise != 0);
        if (e != QS_SUCCESS) return e;
    }
    return QS_SUCCESS;
}

qs_error_t braid_word_matrix(const braid_word_t *w, anyon_system_t *sys,
                             const anyon_charge_t *charges, uint32_t num_anyons,
                             anyon_charge_t total_charge,
                             double complex *out, uint32_t *out_dim) {
    if (!sys || !charges) return QS_ERROR_INVALID_STATE;
    uint32_t dim = fusion_count_paths(sys, charges, num_anyons, total_charge);
    if (out_dim) *out_dim = dim;
    if (dim == 0) return QS_ERROR_INVALID_DIMENSION;
    if (!out) return QS_SUCCESS;

    for (uint32_t j = 0; j < dim; j++) {
        fusion_tree_t *t = fusion_tree_create(sys, charges, num_anyons, total_charge);
        if (!t) return QS_ERROR_OUT_OF_MEMORY;
        fusion_tree_set_basis_state(t, j);
        qs_error_t e = w ? braid_word_apply(w, t) : QS_SUCCESS;
        if (e != QS_SUCCESS) { fusion_tree_free(t); return e; }
        for (uint32_t i = 0; i < dim; i++) out[(size_t)i * dim + j] = t->amplitudes[i];
        fusion_tree_free(t);
    }
    return QS_SUCCESS;
}

// ============================================================================
// SMALL DENSE COMPLEX MATRIX HELPERS
// ============================================================================

static void mat_mul(uint32_t d, const double complex *a, const double complex *b,
                    double complex *out) {
    for (uint32_t i = 0; i < d; i++)
        for (uint32_t j = 0; j < d; j++) {
            double complex s = 0.0;
            for (uint32_t k = 0; k < d; k++) s += a[(size_t)i * d + k] * b[(size_t)k * d + j];
            out[(size_t)i * d + j] = s;
        }
}

static void mat_dag(uint32_t d, const double complex *a, double complex *out) {
    for (uint32_t i = 0; i < d; i++)
        for (uint32_t j = 0; j < d; j++)
            out[(size_t)i * d + j] = conj(a[(size_t)j * d + i]);
}

static void mat_eye(uint32_t d, double complex *out) {
    memset(out, 0, (size_t)d * d * sizeof(double complex));
    for (uint32_t i = 0; i < d; i++) out[(size_t)i * d + i] = 1.0;
}

/* |tr(A^dag B)| / d: 1 exactly when A and B agree up to a global phase. */
static double mat_projective_overlap(uint32_t d, const double complex *a,
                                     const double complex *b) {
    double complex t = 0.0;
    for (uint32_t i = 0; i < d; i++)
        for (uint32_t j = 0; j < d; j++)
            t += conj(a[(size_t)i * d + j]) * b[(size_t)i * d + j];
    return cabs(t) / (double)d;
}

/* ||A - e^{i phi} B||_F minimised over phi, computed as a difference rather
 * than from 2 - |tr|: the latter cancels catastrophically once the two agree
 * to machine precision and would report sqrt(eps) instead of eps. */
static double mat_phase_aligned_frobenius(uint32_t d, const double complex *a,
                                          const double complex *b) {
    double complex t = 0.0;
    for (uint32_t i = 0; i < d * d; i++) t += conj(b[i]) * a[i];
    double complex ph = (cabs(t) > 1e-300) ? t / cabs(t) : 1.0;
    double s = 0.0;
    for (uint32_t i = 0; i < d * d; i++) {
        double e = cabs(a[i] - ph * b[i]);
        s += e * e;
    }
    return sqrt(s);
}

// ============================================================================
// SU(2) HELPERS
// ============================================================================

/* Rescale a 2x2 unitary to determinant 1. */
static int su2_normalize(const double complex u[4], double complex out[4]) {
    double complex det = u[0] * u[3] - u[1] * u[2];
    if (cabs(det) < 1e-12) return 0;
    double complex s = csqrt(det);
    for (int i = 0; i < 4; i++) out[i] = u[i] / s;
    return 1;
}

/* U = w I + i(x sx + y sy + z sz)  <->  q = (w,x,y,z), |q| = 1 for SU(2). */
static void su2_to_quat(const double complex u[4], double q[4]) {
    q[0] = creal(u[0]);
    q[3] = cimag(u[0]);
    q[2] = creal(u[1]);
    q[1] = cimag(u[1]);
    double n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n > 0.0) for (int i = 0; i < 4; i++) q[i] /= n;
}

double su2_projective_distance(const double complex a[4], const double complex b[4]) {
    double complex na[4], nb[4];
    if (!su2_normalize(a, na) || !su2_normalize(b, nb)) return 2.0;
    /* min_phi ||A - e^{i phi} B||_op = min_phi ||A - e^{i phi} B||_F / sqrt(2)
     * on SU(2); the Frobenius form is evaluated directly to stay accurate all
     * the way down to machine precision. */
    return mat_phase_aligned_frobenius(2, na, nb) / M_SQRT2;
}

/* R_n(theta) = cos(theta/2) I - i sin(theta/2) (n . sigma). */
static void su2_rotation(const double n[3], double theta, double complex out[4]) {
    double c = cos(theta / 2.0), s = sin(theta / 2.0);
    out[0] = c - I * s * n[2];
    out[1] = -I * s * (n[0] - I * n[1]);
    out[2] = -I * s * (n[0] + I * n[1]);
    out[3] = c + I * s * n[2];
}

/* Inverse of su2_rotation: theta in [0, pi] with a unit axis.
 *
 * The angle comes from atan2 of the vector part, not from acos of the scalar
 * part: at the depths the Solovay-Kitaev recursion reaches, theta is ~1e-9 and
 * cos(theta/2) rounds to exactly 1, which would silently truncate the
 * remaining correction to zero. */
static void su2_axis_angle(const double complex u[4], double n[3], double *theta) {
    double q[4];
    su2_to_quat(u, q);
    if (q[0] < 0.0) for (int i = 0; i < 4; i++) q[i] = -q[i];  /* theta <= pi */
    double s = sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    *theta = 2.0 * atan2(s, q[0]);
    if (s < 1e-300) { n[0] = 0.0; n[1] = 0.0; n[2] = 1.0; return; }
    /* q = (cos(t/2), -sin(t/2) n) for the convention above. */
    n[0] = -q[1] / s; n[1] = -q[2] / s; n[2] = -q[3] / s;
}

/* Balanced group-commutator decomposition: Delta = V W V^dag W^dag.
 *
 * V and W are rotations by a common angle phi about orthogonal axes, so the
 * commutator's rotation angle is a fixed function f(phi); phi is recovered by
 * bracketing and bisecting f, which avoids transcribing the closed form and
 * makes the construction self-checking (the caller verifies the result).  The
 * commutator's axis is then rotated onto Delta's. */
static int su2_gc_decompose(const double complex delta[4],
                            double complex V[4], double complex W[4]) {
    double n[3], theta;
    su2_axis_angle(delta, n, &theta);
    if (theta < 1e-300) { mat_eye(2, V); mat_eye(2, W); return 1; }

    const double ex[3] = {1.0, 0.0, 0.0}, ey[3] = {0.0, 1.0, 0.0};
    double complex vx[4], wy[4], vd[4], wd[4], t1[4], t2[4], c[4];
    double cn[3], ctheta = 0.0;

    /* f(phi) = rotation angle of [R_x(phi), R_y(phi)] rises monotonically from
     * 0 to a peak; locate the peak, then bisect [0, peak] for f(phi) = theta.
     * Bisecting the whole interval rather than a scan-sized bracket keeps the
     * tiny-theta case (~1e-9, which is where the recursion spends its depth)
     * as accurate as the large-theta one. */
    double lo = 0.0, hi = -1.0;
    double prev = 0.0;
    for (int k = 1; k <= 400; k++) {
        double phi = M_PI * k / 400.0;
        su2_rotation(ex, phi, vx);  su2_rotation(ey, phi, wy);
        mat_dag(2, vx, vd);         mat_dag(2, wy, wd);
        mat_mul(2, vx, wy, t1); mat_mul(2, t1, vd, t2); mat_mul(2, t2, wd, c);
        su2_axis_angle(c, cn, &ctheta);
        if (ctheta < prev) break;   /* past the peak */
        prev = ctheta;
        hi = phi;
    }
    if (hi < 0.0 || prev < theta) return 0;   /* out of the construction's reach */

    for (int it = 0; it < 200; it++) {
        double mid = 0.5 * (lo + hi);
        su2_rotation(ex, mid, vx);  su2_rotation(ey, mid, wy);
        mat_dag(2, vx, vd);         mat_dag(2, wy, wd);
        mat_mul(2, vx, wy, t1); mat_mul(2, t1, vd, t2); mat_mul(2, t2, wd, c);
        su2_axis_angle(c, cn, &ctheta);
        if (ctheta < theta) lo = mid; else hi = mid;
    }
    double phi = 0.5 * (lo + hi);
    su2_rotation(ex, phi, vx);  su2_rotation(ey, phi, wy);
    mat_dag(2, vx, vd);         mat_dag(2, wy, wd);
    mat_mul(2, vx, wy, t1); mat_mul(2, t1, vd, t2); mat_mul(2, t2, wd, c);
    su2_axis_angle(c, cn, &ctheta);

    /* S: rotation carrying the commutator's axis onto Delta's. */
    double dot = cn[0]*n[0] + cn[1]*n[1] + cn[2]*n[2];
    double ax[3] = { cn[1]*n[2] - cn[2]*n[1],
                     cn[2]*n[0] - cn[0]*n[2],
                     cn[0]*n[1] - cn[1]*n[0] };
    double axn = sqrt(ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2]);
    double complex S[4], Sd[4];
    if (axn < 1e-12) {
        if (dot > 0.0) {
            mat_eye(2, S);
        } else {
            double perp[3] = {1.0, 0.0, 0.0};
            if (fabs(cn[0]) > 0.9) { perp[0] = 0.0; perp[1] = 1.0; }
            double p[3] = { perp[1]*cn[2] - perp[2]*cn[1],
                            perp[2]*cn[0] - perp[0]*cn[2],
                            perp[0]*cn[1] - perp[1]*cn[0] };
            double pn = sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
            for (int i = 0; i < 3; i++) p[i] /= pn;
            su2_rotation(p, M_PI, S);
        }
    } else {
        for (int i = 0; i < 3; i++) ax[i] /= axn;
        if (dot > 1.0) dot = 1.0;
        if (dot < -1.0) dot = -1.0;
        su2_rotation(ax, acos(dot), S);
    }
    mat_dag(2, S, Sd);
    mat_mul(2, S, vx, t1); mat_mul(2, t1, Sd, V);
    mat_mul(2, S, wy, t1); mat_mul(2, t1, Sd, W);
    return 1;
}

// ============================================================================
// GENERATOR MATRICES ON A UNIFORM-CHARGE FUSION SPACE
// ============================================================================

/* Every compiler here works on a tree whose external charges are all equal, so
 * braid_anyons() leaves the external list invariant and the generators have
 * well-defined matrices that compose. */
typedef struct {
    uint32_t dim;
    uint32_t num_gen;              /* 2 * (num_anyons - 1): sigma_i and its inverse */
    double complex *g;             /* num_gen matrices of dim x dim */
} gen_set_t;

static void gen_set_free(gen_set_t *gs) {
    if (!gs) return;
    free(gs->g);
    free(gs);
}

static gen_set_t *gen_set_build(anyon_system_t *sys, anyon_charge_t charge,
                                uint32_t num_anyons, anyon_charge_t total) {
    anyon_charge_t *ch = malloc(num_anyons * sizeof(anyon_charge_t));
    if (!ch) return NULL;
    for (uint32_t i = 0; i < num_anyons; i++) ch[i] = charge;

    uint32_t dim = fusion_count_paths(sys, ch, num_anyons, total);
    if (dim == 0) { free(ch); return NULL; }

    gen_set_t *gs = calloc(1, sizeof(gen_set_t));
    if (!gs) { free(ch); return NULL; }
    gs->dim = dim;
    gs->num_gen = 2 * (num_anyons - 1);
    gs->g = calloc((size_t)gs->num_gen * dim * dim, sizeof(double complex));
    if (!gs->g) { free(ch); free(gs); return NULL; }

    braid_word_t *w = braid_word_create();
    for (uint32_t p = 0; p + 1 < num_anyons; p++) {
        for (uint32_t inv = 0; inv < 2; inv++) {
            w->length = 0;
            braid_word_append(w, p, inv == 0);
            if (braid_word_matrix(w, sys, ch, num_anyons, total,
                                  gs->g + (size_t)(2u * p + inv) * dim * dim,
                                  NULL) != QS_SUCCESS) {
                braid_word_free(w); free(ch); gen_set_free(gs); return NULL;
            }
        }
    }
    braid_word_free(w);
    free(ch);
    return gs;
}

/* Symbol s encodes generator sigma_{s/2}^{+-1}: even = clockwise. */
static inline uint32_t sym_position(uint32_t s) { return s >> 1; }
static inline bool sym_clockwise(uint32_t s) { return (s & 1u) == 0u; }
static inline uint32_t sym_inverse(uint32_t s) { return s ^ 1u; }

// ============================================================================
// EXACT COMPILATION FOR ISING: BFS OVER A FINITE BRAID-GROUP IMAGE
// ============================================================================

typedef struct {
    uint32_t dim;
    uint32_t count;
    double complex *mats;      /* count * dim * dim */
    uint8_t *words;            /* flat symbol storage */
    uint32_t *word_off;
    uint8_t *word_len;
    /* bucket index on the phase-invariant modulus pattern */
    int32_t *buckets;
    uint32_t nbuckets;
    int32_t *chain;
} finite_group_t;

static void finite_group_free(finite_group_t *fg) {
    if (!fg) return;
    free(fg->mats); free(fg->words); free(fg->word_off); free(fg->word_len);
    free(fg->buckets); free(fg->chain);
    free(fg);
}

static uint64_t fg_fingerprint(uint32_t d, const double complex *m) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < d * d; i++) {
        long long q = (long long)llround(cabs(m[i]) * 1e9);
        for (int b = 0; b < 8; b++) {
            h ^= (uint64_t)((q >> (8 * b)) & 0xff);
            h *= 1099511628211ull;
        }
    }
    return h;
}

static int32_t fg_lookup(const finite_group_t *fg, const double complex *m) {
    uint64_t h = fg_fingerprint(fg->dim, m);
    int32_t i = fg->buckets[h & (fg->nbuckets - 1)];
    while (i >= 0) {
        if (mat_projective_overlap(fg->dim, fg->mats + (size_t)i * fg->dim * fg->dim, m)
            > 1.0 - 1e-9) return i;
        i = fg->chain[i];
    }
    return -1;
}

/* Enumerate the group generated by `gs` and record a shortest word for each
 * element.  Returns NULL if the image is larger than `max_elems` (i.e. not
 * finite at the scale we can enumerate). */
static finite_group_t *finite_group_build(const gen_set_t *gs, uint32_t max_elems,
                                          uint32_t max_word) {
    const uint32_t d = gs->dim, dd = d * d;
    finite_group_t *fg = calloc(1, sizeof(finite_group_t));
    if (!fg) return NULL;
    fg->dim = d;
    fg->nbuckets = 1u << 16;
    fg->buckets = malloc(fg->nbuckets * sizeof(int32_t));
    fg->mats = malloc((size_t)max_elems * dd * sizeof(double complex));
    fg->chain = malloc((size_t)max_elems * sizeof(int32_t));
    fg->word_off = malloc((size_t)max_elems * sizeof(uint32_t));
    fg->word_len = malloc((size_t)max_elems);
    fg->words = malloc((size_t)max_elems * (max_word + 1));
    if (!fg->buckets || !fg->mats || !fg->chain || !fg->word_off ||
        !fg->word_len || !fg->words) { finite_group_free(fg); return NULL; }
    for (uint32_t i = 0; i < fg->nbuckets; i++) fg->buckets[i] = -1;

    double complex *tmp = malloc(dd * sizeof(double complex));
    if (!tmp) { finite_group_free(fg); return NULL; }

    /* identity */
    mat_eye(d, fg->mats);
    fg->word_off[0] = 0; fg->word_len[0] = 0;
    uint64_t h = fg_fingerprint(d, fg->mats);
    fg->chain[0] = fg->buckets[h & (fg->nbuckets - 1)];
    fg->buckets[h & (fg->nbuckets - 1)] = 0;
    fg->count = 1;
    uint32_t words_used = 0;

    for (uint32_t head = 0; head < fg->count; head++) {
        if (fg->word_len[head] >= max_word) continue;
        for (uint32_t s = 0; s < gs->num_gen; s++) {
            /* new = G_s * cur : appending the symbol applies it last */
            mat_mul(d, gs->g + (size_t)s * dd, fg->mats + (size_t)head * dd, tmp);
            if (fg_lookup(fg, tmp) >= 0) continue;
            if (fg->count >= max_elems) { free(tmp); finite_group_free(fg); return NULL; }
            uint32_t idx = fg->count++;
            memcpy(fg->mats + (size_t)idx * dd, tmp, dd * sizeof(double complex));
            uint32_t len = fg->word_len[head];
            fg->word_off[idx] = words_used;
            memcpy(fg->words + words_used, fg->words + fg->word_off[head], len);
            fg->words[words_used + len] = (uint8_t)s;
            fg->word_len[idx] = (uint8_t)(len + 1);
            words_used += len + 1;
            uint64_t hh = fg_fingerprint(d, tmp);
            fg->chain[idx] = fg->buckets[hh & (fg->nbuckets - 1)];
            fg->buckets[hh & (fg->nbuckets - 1)] = (int32_t)idx;
        }
    }
    free(tmp);
    return fg;
}

static braid_word_t *fg_word(const finite_group_t *fg, int32_t idx) {
    braid_word_t *w = braid_word_create();
    if (!w) return NULL;
    const uint8_t *sym = fg->words + fg->word_off[idx];
    for (uint32_t i = 0; i < fg->word_len[idx]; i++) {
        braid_word_append(w, sym_position(sym[i]), sym_clockwise(sym[i]));
    }
    return w;
}

/* Cached one- and two-qubit Ising braid-group images. */
static finite_group_t *g_ising1 = NULL;   /* 4 sigma, total charge vacuum, dim 2 */
static finite_group_t *g_ising2 = NULL;   /* 6 sigma, total charge vacuum, dim 4 */

static finite_group_t *ising_group(anyon_system_t *sys, uint32_t num_anyons) {
    finite_group_t **slot = (num_anyons == 4) ? &g_ising1 : &g_ising2;
    if (*slot) return *slot;
    gen_set_t *gs = gen_set_build(sys, ISING_SIGMA, num_anyons, ISING_VACUUM);
    if (!gs) return NULL;
    *slot = finite_group_build(gs, (num_anyons == 4) ? 256u : 65536u,
                               (num_anyons == 4) ? 12u : 24u);
    gen_set_free(gs);
    return *slot;
}

uint32_t ising_braid_group_order(anyon_system_t *sys, uint32_t num_anyons) {
    if (!sys || (num_anyons != 4 && num_anyons != 6)) return 0;
    finite_group_t *fg = ising_group(sys, num_anyons);
    return fg ? fg->count : 0u;
}

braid_word_t *ising_compile_clifford(anyon_system_t *sys,
                                     const double complex target[4],
                                     double *achieved_error) {
    if (!sys || !target) return NULL;
    finite_group_t *fg = ising_group(sys, 4);
    if (!fg) return NULL;
    int32_t idx = fg_lookup(fg, target);
    if (idx < 0) return NULL;
    if (achieved_error) {
        *achieved_error = su2_projective_distance(target,
                                                  fg->mats + (size_t)idx * 4);
    }
    return fg_word(fg, idx);
}

braid_word_t *ising_compile_clifford2(anyon_system_t *sys,
                                      const double complex target[16],
                                      double *achieved_error) {
    if (!sys || !target) return NULL;
    finite_group_t *fg = ising_group(sys, 6);
    if (!fg) return NULL;
    int32_t idx = fg_lookup(fg, target);
    if (idx < 0) return NULL;
    if (achieved_error) {
        *achieved_error = mat_phase_aligned_frobenius(4, target,
                                                      fg->mats + (size_t)idx * 16) / 2.0;
    }
    return fg_word(fg, idx);
}

// ============================================================================
// SOLOVAY-KITAEV FOR FIBONACCI
// ============================================================================

/* Base net: every braid word up to FNET_MAX_LEN, thinned so that no two kept
 * elements are within FNET_CELL of each other in su2_projective_distance.  The
 * thinned set therefore covers the enumerated set to FNET_CELL, which is the
 * eps_0 the Solovay-Kitaev recursion starts from. */
#define FNET_MAX_LEN  14
#define FNET_CELL     0.0225

typedef struct {
    double q[4];
    uint32_t off;
    uint8_t len;
} fnet_entry_t;

static struct {
    int built;
    int failed;
    fnet_entry_t *e;
    uint32_t n, cap;
    uint8_t *words;
    size_t wn, wcap;
    int32_t *buckets;
    uint32_t nbuckets;
    int32_t *chain_next;
    uint32_t *chain_entry;
    uint32_t chain_n, chain_cap;
    double complex gen[4][4];       /* sigma_1, sigma_1^-1, sigma_2, sigma_2^-1 */
} g_fnet;

static uint64_t cell_hash(const double q[4], double h) {
    uint64_t x = 1469598103934665603ull;
    for (int i = 0; i < 4; i++) {
        int32_t c = (int32_t)floor(q[i] / h);
        for (int b = 0; b < 4; b++) {
            x ^= (uint64_t)((c >> (8 * b)) & 0xff);
            x *= 1099511628211ull;
        }
    }
    return x;
}

static uint64_t cell_hash_off(const double q[4], double h, const int off[4]) {
    uint64_t x = 1469598103934665603ull;
    for (int i = 0; i < 4; i++) {
        int32_t c = (int32_t)floor(q[i] / h) + off[i];
        for (int b = 0; b < 4; b++) {
            x ^= (uint64_t)((c >> (8 * b)) & 0xff);
            x *= 1099511628211ull;
        }
    }
    return x;
}

static void fnet_chain_add(uint64_t h, uint32_t entry) {
    if (g_fnet.chain_n == g_fnet.chain_cap) {
        uint32_t cap = g_fnet.chain_cap ? g_fnet.chain_cap * 2 : 1024;
        g_fnet.chain_next = realloc(g_fnet.chain_next, cap * sizeof(int32_t));
        g_fnet.chain_entry = realloc(g_fnet.chain_entry, cap * sizeof(uint32_t));
        g_fnet.chain_cap = cap;
    }
    uint32_t b = (uint32_t)(h & (g_fnet.nbuckets - 1));
    g_fnet.chain_next[g_fnet.chain_n] = g_fnet.buckets[b];
    g_fnet.chain_entry[g_fnet.chain_n] = entry;
    g_fnet.buckets[b] = (int32_t)g_fnet.chain_n;
    g_fnet.chain_n++;
}

/* Scan the cube of cells within `radius` cells of q.  Both +q and -q are
 * indexed, so the projective metric coincides with the Euclidean one on the
 * stored copies and a radius-r scan is exact out to r * FNET_CELL. */
static int32_t fnet_scan(const double q[4], int radius, double *dist) {
    int32_t best = -1;
    double bestd = 1e30;
    int off[4];
    for (off[0] = -radius; off[0] <= radius; off[0]++)
    for (off[1] = -radius; off[1] <= radius; off[1]++)
    for (off[2] = -radius; off[2] <= radius; off[2]++)
    for (off[3] = -radius; off[3] <= radius; off[3]++) {
        uint64_t h = cell_hash_off(q, FNET_CELL, off);
        int32_t c = g_fnet.buckets[h & (g_fnet.nbuckets - 1)];
        while (c >= 0) {
            uint32_t ei = g_fnet.chain_entry[c];
            const double *p = g_fnet.e[ei].q;
            double dot = q[0]*p[0] + q[1]*p[1] + q[2]*p[2] + q[3]*p[3];
            double dq[4];
            if (dot < 0.0) { for (int i = 0; i < 4; i++) dq[i] = q[i] + p[i]; }
            else           { for (int i = 0; i < 4; i++) dq[i] = q[i] - p[i]; }
            double d = sqrt(dq[0]*dq[0] + dq[1]*dq[1] + dq[2]*dq[2] + dq[3]*dq[3]);
            if (d < bestd) { bestd = d; best = (int32_t)ei; }
            c = g_fnet.chain_next[c];
        }
    }
    if (dist) *dist = bestd;
    return best;
}

/* Nearest stored element, widening the search until the result is certified:
 * a hit at distance d found within radius r is nearest overall once
 * d <= (r-1) * FNET_CELL, because anything closer would have been in the
 * scanned cube. */
static int32_t fnet_nearest(const double q[4], double *dist) {
    for (int r = 1; r <= 12; r++) {
        double d;
        int32_t best = fnet_scan(q, r, &d);
        if (best >= 0 && d <= (double)(r - 1) * FNET_CELL) {
            if (dist) *dist = d;
            return best;
        }
        if (r == 12 && best >= 0) { if (dist) *dist = d; return best; }
    }
    if (dist) *dist = 1e30;
    return -1;
}

static void fnet_insert(const double q[4], const uint8_t *sym, uint32_t len) {
    if (g_fnet.n == g_fnet.cap) {
        uint32_t cap = g_fnet.cap ? g_fnet.cap * 2 : 4096;
        g_fnet.e = realloc(g_fnet.e, (size_t)cap * sizeof(fnet_entry_t));
        g_fnet.cap = cap;
    }
    if (g_fnet.wn + len > g_fnet.wcap) {
        size_t cap = g_fnet.wcap ? g_fnet.wcap * 2 : 65536;
        while (cap < g_fnet.wn + len) cap *= 2;
        g_fnet.words = realloc(g_fnet.words, cap);
        g_fnet.wcap = cap;
    }
    uint32_t idx = g_fnet.n++;
    memcpy(g_fnet.e[idx].q, q, 4 * sizeof(double));
    g_fnet.e[idx].off = (uint32_t)g_fnet.wn;
    g_fnet.e[idx].len = (uint8_t)len;
    memcpy(g_fnet.words + g_fnet.wn, sym, len);
    g_fnet.wn += len;

    double nq[4] = { -q[0], -q[1], -q[2], -q[3] };
    fnet_chain_add(cell_hash(q, FNET_CELL), idx);
    fnet_chain_add(cell_hash(nq, FNET_CELL), idx);
}

/* Depth-first enumeration of all words of exactly `target_len` symbols, with
 * immediate cancellations pruned. */
static void fnet_dfs(uint32_t depth, uint32_t target_len, uint8_t *sym,
                     const double complex *cur) {
    if (depth == target_len) {
        double complex nu[4];
        if (!su2_normalize(cur, nu)) return;
        double q[4];
        su2_to_quat(nu, q);
        double d;
        if (g_fnet.n > 0) {
            /* Only the "is anything within one cell" question matters here, so
             * the single-radius scan is exact for it and stays cheap.  Most
             * words land in an already-occupied cell, so try that one first:
             * it turns 81 hash probes into one for the common case. */
            (void)fnet_scan(q, 0, &d);
            if (d < FNET_CELL) return;
            (void)fnet_scan(q, 1, &d);
            if (d < FNET_CELL) return;
        }
        fnet_insert(q, sym, target_len);
        return;
    }
    double complex next[4];
    for (uint32_t s = 0; s < 4; s++) {
        if (depth > 0 && sym[depth - 1] == sym_inverse(s)) continue;
        mat_mul(2, g_fnet.gen[s], cur, next);
        sym[depth] = (uint8_t)s;
        fnet_dfs(depth + 1, target_len, sym, next);
    }
}

/* Fibonacci qubit: 3 tau anyons, total charge tau, dimension 2.
 * Path 0 is e_1 = vacuum = |0>_L, path 1 is e_1 = tau = |1>_L. */
static int fnet_build(anyon_system_t *sys) {
    if (g_fnet.built) return 1;
    if (g_fnet.failed) return 0;

    gen_set_t *gs = gen_set_build(sys, FIB_TAU, 3, FIB_TAU);
    if (!gs || gs->dim != 2) { gen_set_free(gs); g_fnet.failed = 1; return 0; }
    for (uint32_t s = 0; s < 4; s++)
        memcpy(g_fnet.gen[s], gs->g + (size_t)s * 4, 4 * sizeof(double complex));
    gen_set_free(gs);

    g_fnet.nbuckets = 1u << 21;
    g_fnet.buckets = malloc(g_fnet.nbuckets * sizeof(int32_t));
    if (!g_fnet.buckets) { g_fnet.failed = 1; return 0; }
    for (uint32_t i = 0; i < g_fnet.nbuckets; i++) g_fnet.buckets[i] = -1;

    double complex id[4];
    mat_eye(2, id);
    uint8_t sym[FNET_MAX_LEN + 1];
    /* Shortest words first, so the net prefers cheap elements. */
    for (uint32_t L = 0; L <= FNET_MAX_LEN; L++) fnet_dfs(0, L, sym, id);

    g_fnet.built = 1;
    return 1;
}

static void bw_matrix_fib(const braid_word_t *w, double complex out[4]) {
    double complex acc[4], tmp[4];
    mat_eye(2, acc);
    for (uint32_t i = 0; i < w->length; i++) {
        uint32_t s = 2 * w->gens[i].position + (w->gens[i].clockwise ? 0u : 1u);
        mat_mul(2, g_fnet.gen[s], acc, tmp);
        memcpy(acc, tmp, 4 * sizeof(double complex));
    }
    memcpy(out, acc, 4 * sizeof(double complex));
}

static int sk_recurse(const double complex U[4], int depth, braid_word_t *out) {
    if (depth == 0) {
        double complex nu[4];
        if (!su2_normalize(U, nu)) return 0;
        double q[4];
        su2_to_quat(nu, q);
        double d;
        int32_t idx = fnet_nearest(q, &d);
        if (idx < 0) return 0;
        const uint8_t *sym = g_fnet.words + g_fnet.e[idx].off;
        for (uint32_t i = 0; i < g_fnet.e[idx].len; i++) {
            braid_word_append(out, sym_position(sym[i]), sym_clockwise(sym[i]));
        }
        return 1;
    }

    braid_word_t *u = braid_word_create();
    if (!u) return 0;
    if (!sk_recurse(U, depth - 1, u)) { braid_word_free(u); return 0; }

    double complex Um[4], Un[4], Ud[4], delta[4], Ur[4];
    bw_matrix_fib(u, Um);
    if (!su2_normalize(Um, Un) || !su2_normalize(U, Ur)) { braid_word_free(u); return 0; }
    /* Pick the SU(2) lift of the approximation closest to the target so that
     * Delta sits near +I, where the commutator construction is well behaved. */
    double complex t = 0.0;
    for (int i = 0; i < 4; i++) t += conj(Un[i]) * Ur[i];
    if (creal(t) < 0.0) for (int i = 0; i < 4; i++) Un[i] = -Un[i];
    mat_dag(2, Un, Ud);
    mat_mul(2, Ur, Ud, delta);

    double complex V[4], W[4];
    if (!su2_gc_decompose(delta, V, W)) { braid_word_free(u); return 0; }

    braid_word_t *v = braid_word_create();
    braid_word_t *w = braid_word_create();
    int ok = v && w && sk_recurse(V, depth - 1, v) && sk_recurse(W, depth - 1, w);
    if (ok) {
        /* target ~ V W V^dag W^dag U_{n-1}: rightmost factor is applied first. */
        ok = braid_word_append_word(out, u) == QS_SUCCESS &&
             braid_word_append_inverse(out, w) == QS_SUCCESS &&
             braid_word_append_inverse(out, v) == QS_SUCCESS &&
             braid_word_append_word(out, w) == QS_SUCCESS &&
             braid_word_append_word(out, v) == QS_SUCCESS;
    }
    braid_word_free(u); braid_word_free(v); braid_word_free(w);
    return ok;
}

braid_word_t *fibonacci_compile_su2(anyon_system_t *sys,
                                    const double complex target[4],
                                    double epsilon, double *achieved_error) {
    if (!sys || !target || !(epsilon > 0.0)) return NULL;
    if (!fnet_build(sys)) return NULL;

    braid_word_t *best = NULL;
    double bestd = 1e30;
    int stalled = 0;
    for (int depth = 0; depth <= 8; depth++) {
        braid_word_t *w = braid_word_create();
        if (!w) break;
        if (!sk_recurse(target, depth, w)) { braid_word_free(w); break; }
        braid_word_reduce(w);
        double complex m[4];
        bw_matrix_fib(w, m);
        double d = su2_projective_distance(target, m);
        if (d < 0.8 * bestd) {
            braid_word_free(best);
            best = w; bestd = d; stalled = 0;
        } else {
            if (d < bestd) { braid_word_free(best); best = w; bestd = d; }
            else braid_word_free(w);
            /* Once rounding in the length-5^n matrix product dominates the
             * recursion's own residual, deeper is only longer. */
            if (++stalled >= 2) break;
        }
        if (bestd <= epsilon) break;
    }
    if (!best) return NULL;
    if (bestd > epsilon) { braid_word_free(best); return NULL; }
    if (achieved_error) *achieved_error = bestd;
    return best;
}

uint32_t fibonacci_braid_net_size(anyon_system_t *sys, double *covering_radius) {
    if (!sys || !fnet_build(sys)) return 0;
    if (covering_radius) {
        /* Deterministic quasi-random sweep of SU(2): the worst nearest-net
         * distance over it is the eps_0 the recursion actually starts from. */
        double worst = 0.0;
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (int k = 0; k < 20000; k++) {
            double q[4];
            double n = 0.0;
            for (int i = 0; i < 4; i++) {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                q[i] = ((double)(s >> 11) / 9007199254740992.0) * 2.0 - 1.0;
                n += q[i] * q[i];
            }
            if (n < 1e-12) continue;
            n = sqrt(n);
            for (int i = 0; i < 4; i++) q[i] /= n;
            double d;
            if (fnet_nearest(q, &d) >= 0 && d > worst) worst = d;
        }
        *covering_radius = worst;
    }
    return g_fnet.n;
}

braid_word_t *fibonacci_exact_phase_gate(uint32_t m) {
    if (m > 9) return NULL;
    /* sigma_1 ~ diag(1, e^{3 i pi / 5}) projectively, so sigma_1^k realises
     * diag(1, e^{i pi (3k mod 10) / 5}) exactly; 3 is invertible mod 10. */
    uint32_t k = 0;
    while (((3u * k) % 10u) != m) k++;
    braid_word_t *w = braid_word_create();
    if (!w) return NULL;
    for (uint32_t i = 0; i < k; i++) braid_word_append(w, 0, true);
    return w;
}

// ============================================================================
// REGISTER GATES
// ============================================================================

/* Standard targets. */
static const double complex GATE_X[4] = { 0.0, 1.0, 1.0, 0.0 };
static const double complex GATE_H[4] = { M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2 };

static qs_error_t register_apply_target(anyonic_register_t *reg, uint32_t qubit,
                                        const double complex target[4],
                                        double epsilon, double *achieved) {
    if (!reg || !reg->tree || qubit >= reg->num_logical_qubits) {
        return QS_ERROR_INVALID_QUBIT;
    }
    if (!(epsilon > 0.0)) epsilon = ANYONIC_GATE_DEFAULT_EPSILON;

    braid_word_t *w = NULL;
    double err = 0.0;
    if (reg->sys->type == ANYON_MODEL_FIBONACCI) {
        w = fibonacci_compile_su2(reg->sys, target, epsilon, &err);
    } else {
        w = ising_compile_clifford(reg->sys, target, &err);
        if (!w) return QS_ERROR_NOT_SUPPORTED;   /* not a Clifford */
    }
    if (!w) return QS_ERROR_NOT_SUPPORTED;

    /* Braids inside a block preserve every other block's charge, so the word
     * acts as U (x) I on the register. */
    qs_error_t e = bw_apply_shifted(w, reg->tree, qubit * 4);
    braid_word_free(w);
    if (achieved) *achieved = err;
    return e;
}

qs_error_t anyonic_apply_unitary(anyonic_register_t *reg, uint32_t qubit,
                                 const double complex target[4],
                                 double epsilon, double *achieved) {
    if (!target) return QS_ERROR_INVALID_STATE;
    return register_apply_target(reg, qubit, target, epsilon, achieved);
}

qs_error_t anyonic_not(anyonic_register_t *reg, uint32_t qubit) {
    return register_apply_target(reg, qubit, GATE_X,
                                 ANYONIC_GATE_DEFAULT_EPSILON, NULL);
}

qs_error_t anyonic_hadamard(anyonic_register_t *reg, uint32_t qubit) {
    return register_apply_target(reg, qubit, GATE_H,
                                 ANYONIC_GATE_DEFAULT_EPSILON, NULL);
}

qs_error_t anyonic_T_gate(anyonic_register_t *reg, uint32_t qubit, double precision) {
    if (!reg || qubit >= reg->num_logical_qubits) return QS_ERROR_INVALID_QUBIT;
    if (reg->sys->type != ANYON_MODEL_FIBONACCI) {
        /* Ising braiding generates exactly the Clifford group, and T is not in
         * it.  There is no approximation to fall back on either: the group is
         * finite, so the distance from T to it is bounded away from zero. */
        return QS_ERROR_NOT_SUPPORTED;
    }
    const double complex T[4] = { 1.0, 0.0, 0.0, cexp(I * M_PI / 4.0) };
    return register_apply_target(reg, qubit, T, precision, NULL);
}
