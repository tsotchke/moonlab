/**
 * @file test_analyticity_oracle.c
 * @brief Adversarial pillar P7 -- parametric analyticity oracle.
 *
 * The expectation value of any observable after a parameterised circuit is an
 * EXACT trigonometric polynomial in each rotation angle. A parameter that
 * appears in q rotation gates (each of the form exp(-i(theta + phi_g) P / 2),
 * P a Pauli) gives
 *
 *     f(theta) = <psi(theta)| O |psi(theta)> = sum_{k=-q..q} c_k e^{ik theta},
 *
 * because each of the q gates contributes e^{+-i theta/2} to the ket and its
 * conjugate to the bra, so the total harmonic content spans exactly -q..q. The
 * identity is exact; only floating-point accumulation intervenes. That makes it
 * a razor: a backend that is subtly wrong in a way a single-point comparison
 * misses (a mis-signed generator, a dropped cross term, a truncation that leaks
 * a spurious harmonic) shows up as a violated degree bound or a fit that fails
 * to reproduce f at angles it never saw.
 *
 * The pillar has two families.
 *
 * 1. TRIGONOMETRIC POLYNOMIAL IDENTITY. Seed-deterministic circuits on 4-8
 *    qubits (rx/ry/rz among Cliffords, forward-direction adjacent 2q gates
 *    only) carry a shared parameter at q = 1, 2, 3 occurrences. The 2q+1
 *    equally spaced angles determine c_k exactly by DFT; the fitted polynomial
 *    then has to reproduce f at 16 independent random angles it was never
 *    given. Run on the dense state vector and on tn_mps at an exact bond
 *    dimension. An over-sampled DFT at 2q+3 nodes additionally pins the DEGREE:
 *    the harmonics at +-(q+1) must vanish. A Clifford-closure family (all
 *    Cliffords plus q shared rz gates at pi/2-multiple offsets) closes the loop
 *    on the Aaronson-Gottesman tableau: the polynomial fitted from non-Clifford
 *    angles must predict the tableau's exact algebraic value at theta = m pi/2.
 *
 * 2. NOISE CONTINUATION. With a depolarizing channel of rate p applied at k
 *    fixed sites, the expectation is a polynomial in p of degree <= k. Fitted
 *    at k+1 Chebyshev-spaced rates in [0, 0.1] it must reproduce the value at
 *    other rates in the same regime AND continue to a different regime
 *    (p = 0.3, 0.5, 0.75 -- the last being where depolarizing saturates the
 *    maximally mixed state). Two independent noise paths are gated:
 *      - the DETERMINISTIC channel path, moonlab_mpdo_* (matrix-product
 *        density operator, exact for the 1q channels and 1q unitaries it
 *        supports -- bond dimension is untouched by a 1q Kraus map, so there is
 *        no truncation);
 *      - the state-vector path, whose channels (noise_depolarizing_single) are
 *        a Monte-Carlo unravelling rather than a channel, so the polynomial
 *        check runs on the EXACT per-Pauli-branch expectation: all 4^k Kraus
 *        branches enumerated through the production function with deterministic
 *        branch-selecting uniforms, each weighted by its exact Kraus weight
 *        ((1-p) for I, p/3 for each of X, Y, Z). This reaches the entangled,
 *        multi-qubit-observable regime the MPDO path cannot represent.
 *    A third probe pins the two noise paths against each other on the product
 *    states where both are exact.
 *
 * Extrapolating a fit out of its sampling window amplifies floating-point
 * error by exactly the Lebesgue sum A(t) = sum_j |L_j(t)| of the Lagrange
 * basis at the evaluation point. That factor is COMPUTED, not guessed, and the
 * gate is tol_base * (1 + A(t)). The identity itself stays exact; nothing is
 * loosened to make a path pass.
 *
 * Emits event: analyticity_oracle
 */
#include "oracle_common.h"
#include "../../src/quantum/noise.h"
#include "../../src/quantum/noise_mpdo.h"

/* ------------------------------------------------------------------ */
/* Tolerances                                                          */
/* ------------------------------------------------------------------ */

/* Trig-polynomial identity: exact in exact arithmetic. The base tolerance is
 * scaled by circuit size because each of the 2q+1 DFT samples carries the fp
 * error of a full circuit evaluation. */
#define TRIG_TOL_BASE   1e-11
/* Excess-harmonic amplitude at |k| = q+1. Same accumulation, no fit involved. */
#define DEGREE_TOL_BASE 1e-11
/* Polynomial-in-p identity. The p-dependence lives entirely in the branch
 * weights, so the raw evaluation is accurate to a few ulp; the fit tolerance is
 * the conditioning-scaled version of this. */
#define NOISE_TOL_BASE  1e-13
/* MPDO channel vs state-vector Kraus-branch sum: two independent exact routes
 * to the same number. */
#define NOISE_XPATH_TOL 1e-12

#define A_MAX_QUBITS     8
#define A_MAX_GATES      160
#define A_MAX_Q          3
#define A_VERIFY_ANGLES  16
#define A_MAX_NOISE_K    3
#define A_LAYERS         3

static const uint64_t BASE_SEED = 0xA11A17C1719ULL;

/* ------------------------------------------------------------------ */
/* Parametric circuit representation                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *g;    /* gate mnemonic, as understood by oracle_apply_* */
    int q0, q1;       /* qubits (q1 = -1 for 1q gates) */
    double fixed;     /* angle when param == 0 */
    int param;        /* 1 => angle is (theta + offset) */
    double offset;    /* per-occurrence phase offset */
} anal_gate_t;

typedef struct {
    char id[128];
    int n;
    int q;                       /* shared-parameter occurrence count */
    int num_gates;
    int clifford_family;         /* every non-parametric gate is Clifford */
    anal_gate_t gates[A_MAX_GATES];
    double wz[A_MAX_QUBITS];     /* observable weights on <Z_i> */
    double wzz[A_MAX_QUBITS];    /* observable weights on <Z_i Z_{i+1}> */
    int noise_k;                 /* depolarizing sites */
    int noise_site[A_MAX_NOISE_K];
    int noise_at[A_MAX_NOISE_K]; /* apply channel before gate index noise_at[i] */
} anal_circuit_t;

static const char *const CLIFF_1Q[] = { "h", "s", "sdg", "x", "y", "z" };
static const char *const ROT_1Q[]   = { "rx", "ry", "rz" };

/* Max Schmidt rank across any cut is 2^floor(n/2); 2^ceil(n/2) covers it. */
static uint32_t exact_bond_cap(int n) { return 1u << ((n + 1) / 2); }

static int eval_dense(const anal_circuit_t *c, double theta, double *out);
static void trig_fit(const double *f, int deg, double complex *ck);

/* Total amplitude of the non-constant harmonics of f(theta), read off the DFT
 * at the 2q+1 nodes. A circuit whose shared parameter cannot reach the
 * observable (an rz at the end of a Z-basis light cone, say) has zero harmonic
 * content and would be reproduced by any fit; generation rejects those so the
 * probe is never vacuous. */
static double harmonic_content(const anal_circuit_t *c) {
    int deg = c->q, N = 2 * c->q + 1;
    double f[2 * A_MAX_Q + 1];
    double complex ck[2 * A_MAX_Q + 1];
    for (int j = 0; j < N; j++) {
        double th = 2.0 * M_PI * (double)j / (double)N;
        if (eval_dense(c, th, &f[j]) != 0) return 0.0;
    }
    trig_fit(f, deg, ck);
    double h = 0.0;
    for (int k = 1; k <= deg; k++) h += 2.0 * cabs(ck[k + deg]);
    return h;
}

/* Insert the q shared-parameter occurrences at random depths into the
 * parameter-free base circuit held in `base`. */
static void insert_params(anal_circuit_t *c, const anal_gate_t *base,
                          int base_num, oracle_rng_t *rng) {
    memcpy(c->gates, base, (size_t)base_num * sizeof(anal_gate_t));
    c->num_gates = base_num;
    for (int i = 0; i < c->q; i++) {
        int pos = (int)oracle_rng_below(rng, (uint32_t)(c->num_gates + 1));
        for (int j = c->num_gates; j > pos; j--) c->gates[j] = c->gates[j - 1];
        anal_gate_t *g = &c->gates[pos];
        memset(g, 0, sizeof(*g));
        /* The Clifford-closure family needs the parametric gate to land on a
         * Clifford whenever theta is a multiple of pi/2, so it is rz with an
         * offset that is itself a pi/2 multiple. */
        g->g      = c->clifford_family ? "rz" : ROT_1Q[oracle_rng_below(rng, 3)];
        g->q0     = (int)oracle_rng_below(rng, (uint32_t)c->n);
        g->q1     = -1;
        g->param  = 1;
        g->offset = c->clifford_family
                      ? (double)oracle_rng_below(rng, 4) * (M_PI / 2.0)
                      : (2.0 * oracle_rng_unit(rng) - 1.0) * M_PI;
        c->num_gates++;
    }
}

/**
 * Build a seed-deterministic parameterised circuit.
 *
 * Two-qubit gates are adjacent and forward-direction (control < target) only:
 * a single local SVD keeps the exact MPS cheap, and the reversed-direction
 * tn_mps apply bug quarantined by P1 is deliberately kept out of this pillar so
 * an analyticity failure here means an analyticity failure, not a known
 * transpose defect leaking across lanes.
 */
static void gen_circuit(anal_circuit_t *c, const char *family, int n, int q,
                        int inst, int clifford_family, int noise_k) {
    memset(c, 0, sizeof(*c));
    c->n = n;
    c->q = q;
    c->clifford_family = clifford_family;
    c->noise_k = noise_k;
    if (noise_k > 0)
        snprintf(c->id, sizeof(c->id), "%s_n%d_k%d_s%d", family, n, noise_k, inst);
    else
        snprintf(c->id, sizeof(c->id), "%s_n%d_q%d_s%d", family, n, q, inst);

    oracle_rng_t rng;
    oracle_rng_seed(&rng, BASE_SEED ^ oracle_stable_id_hash(c->id));

    anal_gate_t base[A_MAX_GATES];
    int base_num = 0;
    for (int L = 0; L < A_LAYERS; L++) {
        for (int qb = 0; qb < n; qb++) {
            anal_gate_t *g = &base[base_num++];
            memset(g, 0, sizeof(*g));
            if (clifford_family || oracle_rng_below(&rng, 3) == 0) {
                g->g = CLIFF_1Q[oracle_rng_below(&rng, 6)];
            } else {
                g->g = ROT_1Q[oracle_rng_below(&rng, 3)];
                g->fixed = (2.0 * oracle_rng_unit(&rng) - 1.0) * M_PI;
            }
            g->q0 = qb;
            g->q1 = -1;
        }
        for (int qb = L & 1; qb + 1 < n; qb += 2) {
            anal_gate_t *g = &base[base_num++];
            memset(g, 0, sizeof(*g));
            g->g  = (oracle_rng_below(&rng, 2) == 0) ? "cnot" : "cz";
            g->q0 = qb;
            g->q1 = qb + 1;
        }
    }

    /* Observable: a random weighted Pauli-Z sum, normalised so sum|w| = 1 and
     * therefore |f| <= 1. The Clifford family stays Z-only because the tableau
     * route reads <Z_i> exactly via clone-and-measure determinism. Drawn before
     * the parameter placement so the rejection loop below cannot perturb it. */
    double wsum = 0.0;
    for (int i = 0; i < n; i++) {
        c->wz[i] = 2.0 * oracle_rng_unit(&rng) - 1.0;
        wsum += fabs(c->wz[i]);
    }
    if (!clifford_family) {
        for (int i = 0; i + 1 < n; i++) {
            c->wzz[i] = 2.0 * oracle_rng_unit(&rng) - 1.0;
            wsum += fabs(c->wzz[i]);
        }
    }
    if (wsum <= 0.0) wsum = 1.0;
    for (int i = 0; i < n; i++) c->wz[i] /= wsum;
    for (int i = 0; i + 1 < n; i++) c->wzz[i] /= wsum;

    /* Parameter placement, with deterministic rejection of placements the
     * observable cannot see. The rng stream advances on every attempt, so the
     * accepted circuit is still a pure function of the seed. */
    c->num_gates = base_num;
    if (q > 0) {
        for (int attempt = 0; attempt < 16; attempt++) {
            insert_params(c, base, base_num, &rng);
            if (harmonic_content(c) > 1e-3) break;
        }
    } else {
        memcpy(c->gates, base, (size_t)base_num * sizeof(anal_gate_t));
    }

    for (int i = 0; i < noise_k; i++) {
        c->noise_site[i] = (int)oracle_rng_below(&rng, (uint32_t)n);
        c->noise_at[i]   = (int)oracle_rng_below(&rng, (uint32_t)(c->num_gates + 1));
    }
    for (int i = 1; i < noise_k; i++) {          /* stable ascending by depth */
        int at = c->noise_at[i], st = c->noise_site[i], j = i - 1;
        while (j >= 0 && c->noise_at[j] > at) {
            c->noise_at[j + 1] = c->noise_at[j];
            c->noise_site[j + 1] = c->noise_site[j];
            j--;
        }
        c->noise_at[j + 1] = at;
        c->noise_site[j + 1] = st;
    }
}

static void resolve(const anal_gate_t *g, double theta, oracle_gate_t *out) {
    out->g  = g->g;
    out->q0 = g->q0;
    out->q1 = g->q1;
    out->p  = g->param ? (theta + g->offset) : g->fixed;
}

/* ------------------------------------------------------------------ */
/* Per-backend evaluation of f(theta)                                  */
/* ------------------------------------------------------------------ */

static double observable_dense(const anal_circuit_t *c, const quantum_state_t *s) {
    double v = 0.0;
    for (int i = 0; i < c->n; i++)
        if (c->wz[i] != 0.0) v += c->wz[i] * measurement_expectation_z(s, i);
    for (int i = 0; i + 1 < c->n; i++)
        if (c->wzz[i] != 0.0) v += c->wzz[i] * measurement_correlation_zz(s, i, i + 1);
    return v;
}

static int eval_dense(const anal_circuit_t *c, double theta, double *out) {
    quantum_state_t *s = quantum_state_create(c->n);
    if (!s) return -1;
    oracle_gate_t og;
    for (int i = 0; i < c->num_gates; i++) {
        resolve(&c->gates[i], theta, &og);
        if (oracle_apply_dense(s, &og) != QS_SUCCESS) {
            quantum_state_destroy(s);
            return -1;
        }
    }
    *out = observable_dense(c, s);
    quantum_state_destroy(s);
    return 0;
}

static int eval_mps(const anal_circuit_t *c, double theta, double *out) {
    tn_state_config_t cfg = tn_state_config_default();
    cfg.max_bond_dim = exact_bond_cap(c->n);
    cfg.svd_cutoff   = 1e-15;
    tn_mps_state_t *m = tn_mps_create_zero((uint32_t)c->n, &cfg);
    if (!m) return -1;
    oracle_gate_t og;
    for (int i = 0; i < c->num_gates; i++) {
        resolve(&c->gates[i], theta, &og);
        if (oracle_apply_mps(m, &og) != 0) {
            tn_mps_free(m);
            return -1;
        }
    }
    /* tn_mps normalisation is lazy (the norm lives in log_norm_factor); commit
     * it before reading expectations. */
    tn_mps_normalize(m);
    double v = 0.0;
    for (int i = 0; i < c->n; i++)
        if (c->wz[i] != 0.0) v += c->wz[i] * tn_expectation_z(m, (uint32_t)i);
    for (int i = 0; i + 1 < c->n; i++)
        if (c->wzz[i] != 0.0)
            v += c->wzz[i] * tn_expectation_zz(m, (uint32_t)i, (uint32_t)(i + 1));
    tn_mps_free(m);
    *out = v;
    return 0;
}

/* Tableau <Z_q> from a Clifford state: clone and measure, a deterministic
 * outcome pins <Z_q> = +-1 and a random outcome (anticommuting stabilizer)
 * means <Z_q> = 0. Same route P1 uses. */
static int tableau_z_expect(clifford_tableau_t *t, int q, double *out) {
    clifford_tableau_t *c = clifford_tableau_clone(t);
    if (!c) return -1;
    uint64_t rng = 0x5DEECE66DULL ^ (uint64_t)q;
    int outcome = 0, kind = 0;
    clifford_error_t e = clifford_measure(c, (size_t)q, &rng, &outcome, &kind);
    clifford_tableau_free(c);
    if (e != CLIFFORD_SUCCESS) return -1;
    *out = (kind == 0) ? (1.0 - 2.0 * outcome) : 0.0;
    return 0;
}

/* Evaluate the Clifford-family circuit at theta = m * pi/2 on the tableau.
 * rz(j * pi/2) equals I, S, Z, S-dagger for j = 0,1,2,3 up to a global phase,
 * which expectation values do not see. */
static int eval_clifford(const anal_circuit_t *c, int m, double *out) {
    clifford_tableau_t *t = clifford_tableau_create((size_t)c->n);
    if (!t) return -1;
    for (int i = 0; i < c->num_gates; i++) {
        const anal_gate_t *g = &c->gates[i];
        clifford_error_t rc = CLIFFORD_SUCCESS;
        if (g->param) {
            int j = (m + (int)llround(g->offset / (M_PI / 2.0))) & 3;
            if      (j == 1) rc = clifford_s(t, (size_t)g->q0);
            else if (j == 2) rc = clifford_z(t, (size_t)g->q0);
            else if (j == 3) rc = clifford_s_dag(t, (size_t)g->q0);
        } else {
            oracle_gate_t og;
            resolve(g, 0.0, &og);
            rc = (clifford_error_t)oracle_apply_clifford(t, &og);
        }
        if (rc != CLIFFORD_SUCCESS) { clifford_tableau_free(t); return -1; }
    }
    double v = 0.0;
    for (int qb = 0; qb < c->n; qb++) {
        double z;
        if (tableau_z_expect(t, qb, &z) != 0) { clifford_tableau_free(t); return -1; }
        v += c->wz[qb] * z;
    }
    clifford_tableau_free(t);
    *out = v;
    return 0;
}

typedef int (*anal_eval_fn)(const anal_circuit_t *, double, double *);

/* ------------------------------------------------------------------ */
/* Trigonometric fit / evaluation                                      */
/* ------------------------------------------------------------------ */

/* Exact DFT recovery of c_{-deg..deg} from 2*deg+1 equally spaced samples. */
static void trig_fit(const double *f, int deg, double complex *ck) {
    int N = 2 * deg + 1;
    for (int k = -deg; k <= deg; k++) {
        double complex acc = 0.0;
        for (int j = 0; j < N; j++) {
            double th = 2.0 * M_PI * (double)j / (double)N;
            acc += f[j] * cexp(-I * (double)k * th);
        }
        ck[k + deg] = acc / (double)N;
    }
}

static double trig_eval(const double complex *ck, int deg, double theta) {
    double complex acc = 0.0;
    for (int k = -deg; k <= deg; k++)
        acc += ck[k + deg] * cexp(I * (double)k * theta);
    return creal(acc);
}

static double trig_tol(const anal_circuit_t *c) {
    return TRIG_TOL_BASE * (1.0 + (double)c->num_gates / 16.0);
}

/* ------------------------------------------------------------------ */
/* Polynomial fit in the noise rate                                    */
/* ------------------------------------------------------------------ */

/* Lagrange interpolant through (x_j, y_j) evaluated at t. Also returns the
 * Lebesgue sum A(t) = sum_j |L_j(t)|, which is exactly the factor by which the
 * evaluation amplifies floating-point error in the samples -- the honest
 * conditioning of continuing a fit outside its sampling window. */
static double poly_interp(const double *x, const double *y, int m, double t,
                          double *amp) {
    double s = 0.0, a = 0.0;
    for (int j = 0; j < m; j++) {
        double L = 1.0;
        for (int i = 0; i < m; i++)
            if (i != j) L *= (t - x[i]) / (x[j] - x[i]);
        s += L * y[j];
        a += fabs(L);
    }
    if (amp) *amp = a;
    return s;
}

/* Chebyshev-Gauss nodes on [a, b]: interior, so p = 0 (where the channel is a
 * no-op and the trajectory dispatch short-circuits) is never a fit node. */
static void chebyshev_nodes(double a, double b, int m, double *x) {
    for (int j = 0; j < m; j++) {
        double c = cos(M_PI * (2.0 * (double)j + 1.0) / (2.0 * (double)m));
        x[j] = 0.5 * (a + b) + 0.5 * (b - a) * c;
    }
}

/* ------------------------------------------------------------------ */
/* Noise: exact per-Kraus-branch expectation on the state vector       */
/* ------------------------------------------------------------------ */

/* Uniforms that drive noise_depolarizing_single down a chosen Kraus branch.
 * The production function takes a caller-supplied uniform and dispatches
 *   random_value >= p              -> identity
 *   random_value / p < 1/3         -> X
 *   random_value / p < 2/3         -> Y
 *   otherwise                      -> Z
 * so these select I, X, Y, Z deterministically for any p in (0, 1). */
static double branch_uniform(int branch, double p) {
    switch (branch) {
        case 0:  return 1.0;              /* >= p for every p <= 1: identity */
        case 1:  return p * (1.0 / 6.0);  /* r = 1/6 -> X */
        case 2:  return p * 0.5;          /* r = 1/2 -> Y */
        default: return p * (5.0 / 6.0);  /* r = 5/6 -> Z */
    }
}

static double branch_weight(int branch, double p) {
    return (branch == 0) ? (1.0 - p) : (p / 3.0);
}

/**
 * Exact expectation under k depolarizing channels at rate p, computed by
 * enumerating all 4^k Kraus branches through the production state-vector
 * channel and weighting each by its exact Kraus weight. This is a channel
 * evaluation assembled from the trajectory primitive, not a sample average:
 * the branch values are p-independent by construction, so the whole
 * p-dependence sits in the weights and E(p) is a polynomial of degree <= k.
 */
static int eval_branch_sum(const anal_circuit_t *c, double p, double *out) {
    int k = c->noise_k;
    int nb = 1;
    for (int i = 0; i < k; i++) nb *= 4;
    double total = 0.0;
    for (int b = 0; b < nb; b++) {
        double w = 1.0;
        for (int i = 0; i < k; i++) w *= branch_weight((b >> (2 * i)) & 3, p);
        if (w == 0.0) continue;
        quantum_state_t *s = quantum_state_create(c->n);
        if (!s) return -1;
        int ni = 0;
        oracle_gate_t og;
        for (int gi = 0; gi <= c->num_gates; gi++) {
            while (ni < k && c->noise_at[ni] == gi) {
                int br = (b >> (2 * ni)) & 3;
                noise_depolarizing_single(s, c->noise_site[ni], p,
                                          branch_uniform(br, p));
                ni++;
            }
            if (gi < c->num_gates) {
                resolve(&c->gates[gi], 0.0, &og);
                if (oracle_apply_dense(s, &og) != QS_SUCCESS) {
                    quantum_state_destroy(s);
                    return -1;
                }
            }
        }
        total += w * observable_dense(c, s);
        quantum_state_destroy(s);
    }
    *out = total;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Noise: deterministic MPDO channel path                              */
/* ------------------------------------------------------------------ */

/* A single-qubit chain: 1q rotations (applied to both paths as the identical
 * gate sequence) interleaved with k depolarizing channels on the same qubit,
 * so the observable really does see a degree-k polynomial in p. */
typedef struct {
    char id[128];
    int n;
    int target;
    int k;
    int nrot;                /* rotations before / between / after channels */
    int rot_axis[16];        /* 0 = ry, 1 = rz */
    double rot_angle[16];
    int rot_after[16];       /* channel index this rotation follows (0..k) */
    uint8_t pauli;           /* observable: 1 = X, 2 = Y, 3 = Z */
} mpdo_chain_t;

static void gen_mpdo_chain(mpdo_chain_t *ch, int n, int k, int inst) {
    memset(ch, 0, sizeof(*ch));
    ch->n = n;
    ch->k = k;
    snprintf(ch->id, sizeof(ch->id), "anlp_n%d_k%d_s%d", n, k, inst);
    oracle_rng_t rng;
    oracle_rng_seed(&rng, BASE_SEED ^ 0x4D50444FULL ^ oracle_stable_id_hash(ch->id));
    ch->target = (int)oracle_rng_below(&rng, (uint32_t)n);
    ch->pauli  = (uint8_t)(1 + oracle_rng_below(&rng, 3));
    /* Two rotations in every slot between (and around) the k channels. */
    ch->nrot = 0;
    for (int slot = 0; slot <= k; slot++) {
        for (int r = 0; r < 2; r++) {
            ch->rot_axis[ch->nrot]  = (int)oracle_rng_below(&rng, 2);
            ch->rot_angle[ch->nrot] = (2.0 * oracle_rng_unit(&rng) - 1.0) * M_PI;
            ch->rot_after[ch->nrot] = slot;
            ch->nrot++;
        }
    }
}

/* 2x2 row-major unitary for ry/rz, in the same convention as gate_ry/gate_rz:
 *   RY(t) = [[c, -s], [s, c]],  RZ(t) = diag(e^{-it/2}, e^{+it/2}). */
static void rot_matrix(int axis, double t, mpdo_complex_t *u) {
    if (axis == 0) {
        double c = cos(t / 2.0), s = sin(t / 2.0);
        u[0] = c;  u[1] = -s;
        u[2] = s;  u[3] = c;
    } else {
        u[0] = cexp(-I * t / 2.0); u[1] = 0.0;
        u[2] = 0.0;                u[3] = cexp(I * t / 2.0);
    }
}

/* Deterministic channel path: moonlab_mpdo_*. Only 1q unitaries and 1q Kraus
 * channels are applied, so the bond dimension never leaves 1 and the result is
 * exact -- no truncation enters the polynomial identity. */
static int eval_mpdo(const mpdo_chain_t *ch, double p, double *out) {
    moonlab_mpdo_t *m = moonlab_mpdo_create((uint32_t)ch->n, 4);
    if (!m) return -1;
    for (int slot = 0; slot <= ch->k; slot++) {
        for (int r = 0; r < ch->nrot; r++) {
            if (ch->rot_after[r] != slot) continue;
            mpdo_complex_t u[4];
            rot_matrix(ch->rot_axis[r], ch->rot_angle[r], u);
            if (moonlab_mpdo_apply_kraus_1q(m, (uint32_t)ch->target, u, 1)
                != MPDO_SUCCESS) { moonlab_mpdo_free(m); return -1; }
        }
        if (slot < ch->k) {
            if (moonlab_mpdo_apply_depolarizing_1q(m, (uint32_t)ch->target, p)
                != MPDO_SUCCESS) { moonlab_mpdo_free(m); return -1; }
        }
    }
    double v = 0.0;
    mpdo_error_t e = moonlab_mpdo_expect_pauli_1q(m, (uint32_t)ch->target,
                                                  ch->pauli, &v);
    moonlab_mpdo_free(m);
    if (e != MPDO_SUCCESS) return -1;
    *out = v;
    return 0;
}

/* The same chain on the state vector, as an exact Kraus-branch sum through the
 * production trajectory channel. */
static int eval_chain_branch_sum(const mpdo_chain_t *ch, double p, double *out) {
    int nb = 1;
    for (int i = 0; i < ch->k; i++) nb *= 4;
    double total = 0.0;
    for (int b = 0; b < nb; b++) {
        double w = 1.0;
        for (int i = 0; i < ch->k; i++) w *= branch_weight((b >> (2 * i)) & 3, p);
        if (w == 0.0) continue;
        quantum_state_t *s = quantum_state_create(ch->n);
        if (!s) return -1;
        for (int slot = 0; slot <= ch->k; slot++) {
            for (int r = 0; r < ch->nrot; r++) {
                if (ch->rot_after[r] != slot) continue;
                qs_error_t rc = (ch->rot_axis[r] == 0)
                    ? gate_ry(s, ch->target, ch->rot_angle[r])
                    : gate_rz(s, ch->target, ch->rot_angle[r]);
                if (rc != QS_SUCCESS) { quantum_state_destroy(s); return -1; }
            }
            if (slot < ch->k) {
                int br = (b >> (2 * slot)) & 3;
                noise_depolarizing_single(s, ch->target, p, branch_uniform(br, p));
            }
        }
        double v = (ch->pauli == 1) ? measurement_expectation_x(s, ch->target)
                 : (ch->pauli == 2) ? measurement_expectation_y(s, ch->target)
                                    : measurement_expectation_z(s, ch->target);
        total += w * v;
        quantum_state_destroy(s);
    }
    *out = total;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Probe: trigonometric polynomial identity                            */
/* ------------------------------------------------------------------ */

static void probe_trig(oracle_ctx_t *ctx, const anal_circuit_t *c,
                       const char *backend, anal_eval_fn eval) {
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__trig_fit_%s", c->id, backend);

    int deg = c->q;
    int N = 2 * deg + 1;
    double f[2 * A_MAX_Q + 1];
    double complex ck[2 * A_MAX_Q + 1];

    for (int j = 0; j < N; j++) {
        double th = 2.0 * M_PI * (double)j / (double)N;
        if (eval(c, th, &f[j]) != 0) {
            oracle_probe_fail(ctx, pid, "seed=%llu backend=%s evaluation failed at node %d",
                              (unsigned long long)BASE_SEED, backend, j);
            return;
        }
    }
    trig_fit(f, deg, ck);

    /* Guard against a vacuous probe: a constant f would be reproduced by any
     * fit. Require a non-trivial harmonic content. */
    double harm = 0.0;
    for (int k = 1; k <= deg; k++) harm += 2.0 * cabs(ck[k + deg]);
    if (harm < 1e-6) {
        oracle_probe_fail(ctx, pid,
            "seed=%llu backend=%s n=%d q=%d degenerate probe: harmonic content %.3e",
            (unsigned long long)BASE_SEED, backend, c->n, c->q, harm);
        return;
    }

    oracle_rng_t rng;
    oracle_rng_seed(&rng, BASE_SEED ^ 0x5645524ULL ^ oracle_stable_id_hash(c->id));
    double tol = trig_tol(c);
    double maxd = 0.0, worst_th = 0.0;
    for (int m = 0; m < A_VERIFY_ANGLES; m++) {
        double th = (2.0 * oracle_rng_unit(&rng) - 1.0) * M_PI;
        double actual;
        if (eval(c, th, &actual) != 0) {
            oracle_probe_fail(ctx, pid,
                "seed=%llu backend=%s evaluation failed at verify angle %d",
                (unsigned long long)BASE_SEED, backend, m);
            return;
        }
        double d = fabs(actual - trig_eval(ck, deg, th));
        if (d > maxd) { maxd = d; worst_th = th; }
    }
    if (maxd > tol)
        oracle_probe_fail(ctx, pid,
            "seed=%llu backend=%s n=%d q=%d gates=%d harm=%.3e max|f-fit|=%.3e@theta=%.9f tol=%.3e",
            (unsigned long long)BASE_SEED, backend, c->n, c->q, c->num_gates,
            harm, maxd, worst_th, tol);
    else
        oracle_probe_pass(ctx, pid);
}

/* Over-sampled DFT: with 2(q+1)+1 nodes a degree-q trig polynomial has exactly
 * zero content at |k| = q+1. A backend that leaks a spurious harmonic (a wrong
 * generator, an extra parameter dependence) fails here even when the degree-q
 * fit happens to interpolate its own samples. */
static void probe_degree(oracle_ctx_t *ctx, const anal_circuit_t *c) {
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__trig_degree", c->id);

    int deg = c->q + 1;
    int N = 2 * deg + 1;
    double f[2 * (A_MAX_Q + 1) + 1];
    double complex ck[2 * (A_MAX_Q + 1) + 1];
    for (int j = 0; j < N; j++) {
        double th = 2.0 * M_PI * (double)j / (double)N;
        if (eval_dense(c, th, &f[j]) != 0) {
            oracle_probe_fail(ctx, pid, "seed=%llu evaluation failed at node %d",
                              (unsigned long long)BASE_SEED, j);
            return;
        }
    }
    trig_fit(f, deg, ck);
    double excess = cabs(ck[deg + deg]) + cabs(ck[0]);   /* k = +(q+1) and -(q+1) */
    double tol = DEGREE_TOL_BASE * (1.0 + (double)c->num_gates / 16.0);
    if (excess > tol)
        oracle_probe_fail(ctx, pid,
            "seed=%llu n=%d q=%d gates=%d |c_{+-(q+1)}|=%.3e tol=%.3e",
            (unsigned long long)BASE_SEED, c->n, c->q, c->num_gates, excess, tol);
    else
        oracle_probe_pass(ctx, pid);
}

/* Clifford closure: the polynomial fitted from non-Clifford angles on the dense
 * state vector must predict the Aaronson-Gottesman tableau's exact algebraic
 * value at theta = m pi/2. This is the analytic structure of one backend
 * predicting a completely independent backend's exact answer. */
static void probe_clifford_closure(oracle_ctx_t *ctx, const anal_circuit_t *c) {
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__trig_clifford_closure", c->id);

    int deg = c->q;
    int N = 2 * deg + 1;
    double f[2 * A_MAX_Q + 1];
    double complex ck[2 * A_MAX_Q + 1];
    for (int j = 0; j < N; j++) {
        double th = 2.0 * M_PI * (double)j / (double)N;
        if (eval_dense(c, th, &f[j]) != 0) {
            oracle_probe_fail(ctx, pid, "seed=%llu dense evaluation failed at node %d",
                              (unsigned long long)BASE_SEED, j);
            return;
        }
    }
    trig_fit(f, deg, ck);

    double tol = trig_tol(c);
    double maxd = 0.0;
    int worst_m = 0;
    double spread = 0.0, first = 0.0;
    for (int m = 0; m < 4; m++) {
        double tab;
        if (eval_clifford(c, m, &tab) != 0) {
            oracle_probe_fail(ctx, pid, "seed=%llu tableau path error at m=%d",
                              (unsigned long long)BASE_SEED, m);
            return;
        }
        if (m == 0) first = tab;
        spread = fmax(spread, fabs(tab - first));
        double d = fabs(tab - trig_eval(ck, deg, (double)m * M_PI / 2.0));
        if (d > maxd) { maxd = d; worst_m = m; }
    }
    if (maxd > tol)
        oracle_probe_fail(ctx, pid,
            "seed=%llu n=%d q=%d gates=%d |fit-tableau|=%.3e@theta=%d*pi/2 spread=%.3e tol=%.3e",
            (unsigned long long)BASE_SEED, c->n, c->q, c->num_gates,
            maxd, worst_m, spread, tol);
    else
        oracle_probe_pass(ctx, pid);
}

/* ------------------------------------------------------------------ */
/* Probe: noise continuation                                           */
/* ------------------------------------------------------------------ */

typedef int (*noise_eval_fn)(const void *, double, double *);

static int eval_branch_sum_v(const void *c, double p, double *out) {
    return eval_branch_sum((const anal_circuit_t *)c, p, out);
}
static int eval_mpdo_v(const void *c, double p, double *out) {
    return eval_mpdo((const mpdo_chain_t *)c, p, out);
}
static int eval_chain_branch_v(const void *c, double p, double *out) {
    return eval_chain_branch_sum((const mpdo_chain_t *)c, p, out);
}

static const double NOISE_SAME_REGIME[] = { 0.005, 0.022, 0.048, 0.071, 0.099 };
static const double NOISE_FAR_REGIME[]  = { 0.3, 0.5, 0.75 };

/**
 * Fit E(p) at k+1 Chebyshev nodes in [0, 0.1] and check the two continuations.
 * `far` selects the evaluation set: 0 = same regime (inside the fit window),
 * 1 = different regime (p up to 0.75, where depolarizing saturates the
 * maximally mixed state).
 *
 * The gate is NOISE_TOL_BASE * (1 + A(t)) where A(t) = sum_j |L_j(t)| is the
 * Lagrange conditioning at the evaluation point, computed per point. The
 * identity being tested is exact; A(t) is the exact amplification of the fp
 * error already present in the samples, so scaling by it is arithmetic, not a
 * loosened tolerance.
 */
static void probe_noise_poly(oracle_ctx_t *ctx, const char *id, const char *path,
                             int k, const void *obj, noise_eval_fn eval, int far) {
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__%s_poly_%s", id, path, far ? "continue" : "regime");

    int m = k + 1;
    double x[A_MAX_NOISE_K + 1], y[A_MAX_NOISE_K + 1];
    chebyshev_nodes(0.0, 0.1, m, x);
    for (int j = 0; j < m; j++) {
        if (eval(obj, x[j], &y[j]) != 0) {
            oracle_probe_fail(ctx, pid, "seed=%llu path=%s evaluation failed at node %d",
                              (unsigned long long)BASE_SEED, path, j);
            return;
        }
    }

    /* Guard against a vacuous probe: a p-independent E would be reproduced by
     * any fit. Require the samples to actually move with p. */
    double lo = y[0], hi = y[0];
    for (int j = 1; j < m; j++) { lo = fmin(lo, y[j]); hi = fmax(hi, y[j]); }
    if (hi - lo < 1e-6) {
        oracle_probe_fail(ctx, pid,
            "seed=%llu path=%s k=%d degenerate probe: E(p) varies by only %.3e over the fit window",
            (unsigned long long)BASE_SEED, path, k, hi - lo);
        return;
    }

    const double *pts = far ? NOISE_FAR_REGIME : NOISE_SAME_REGIME;
    int npts = far ? (int)(sizeof(NOISE_FAR_REGIME) / sizeof(double))
                   : (int)(sizeof(NOISE_SAME_REGIME) / sizeof(double));
    double worst_ratio = 0.0, worst_d = 0.0, worst_p = 0.0, worst_amp = 0.0;
    for (int i = 0; i < npts; i++) {
        double actual;
        if (eval(obj, pts[i], &actual) != 0) {
            oracle_probe_fail(ctx, pid, "seed=%llu path=%s evaluation failed at p=%.3f",
                              (unsigned long long)BASE_SEED, path, pts[i]);
            return;
        }
        double amp = 0.0;
        double pred = poly_interp(x, y, m, pts[i], &amp);
        double d = fabs(actual - pred);
        double tol = NOISE_TOL_BASE * (1.0 + amp);
        if (d / tol > worst_ratio) {
            worst_ratio = d / tol;
            worst_d = d; worst_p = pts[i]; worst_amp = amp;
        }
    }
    if (worst_ratio > 1.0)
        oracle_probe_fail(ctx, pid,
            "seed=%llu path=%s k=%d |E-fit|=%.3e@p=%.3f amplification=%.3e tol=%.3e",
            (unsigned long long)BASE_SEED, path, k, worst_d, worst_p, worst_amp,
            NOISE_TOL_BASE * (1.0 + worst_amp));
    else
        oracle_probe_pass(ctx, pid);
}

/* The deterministic MPDO channel and the state-vector Kraus-branch sum are two
 * independent exact routes to the same number wherever both are representable
 * (product states, single-site observables). Pin them against each other. */
static void probe_noise_cross_path(oracle_ctx_t *ctx, const mpdo_chain_t *ch) {
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__noise_mpdo_vs_branch", ch->id);
    double maxd = 0.0, worst_p = 0.0;
    static const double PTS[] = { 0.01, 0.05, 0.1, 0.3, 0.5, 0.75 };
    for (int i = 0; i < (int)(sizeof(PTS) / sizeof(double)); i++) {
        double a, b;
        if (eval_mpdo(ch, PTS[i], &a) != 0 ||
            eval_chain_branch_sum(ch, PTS[i], &b) != 0) {
            oracle_probe_fail(ctx, pid, "seed=%llu evaluation failed at p=%.3f",
                              (unsigned long long)BASE_SEED, PTS[i]);
            return;
        }
        double d = fabs(a - b);
        if (d > maxd) { maxd = d; worst_p = PTS[i]; }
    }
    if (maxd > NOISE_XPATH_TOL)
        oracle_probe_fail(ctx, pid,
            "seed=%llu n=%d k=%d pauli=%u |mpdo-branch|=%.3e@p=%.3f tol=%.3e",
            (unsigned long long)BASE_SEED, ch->n, ch->k, (unsigned)ch->pauli,
            maxd, worst_p, NOISE_XPATH_TOL);
    else
        oracle_probe_pass(ctx, pid);
}

/* ------------------------------------------------------------------ */

int main(void) {
    oracle_ctx_t ctx;
    oracle_ctx_init(&ctx, "analyticity_oracle");
    fprintf(stdout,
            "=== P7 analyticity oracle (trig-polynomial identity + noise continuation) ===\n");

    anal_circuit_t c;

    /* Family 1: trigonometric polynomial identity, dense + tn_mps + degree. */
    for (int n = 4; n <= 8; n++) {
        for (int q = 1; q <= A_MAX_Q; q++) {
            for (int inst = 0; inst < 2; inst++) {
                gen_circuit(&c, "anl", n, q, inst, 0, 0);
                probe_trig(&ctx, &c, "dense", eval_dense);
                probe_trig(&ctx, &c, "mps", eval_mps);
                probe_degree(&ctx, &c);
            }
        }
    }

    /* Family 2: Clifford closure -- the fit predicts the tableau exactly. */
    for (int n = 4; n <= 8; n += 2) {
        for (int q = 1; q <= A_MAX_Q; q++) {
            for (int inst = 0; inst < 2; inst++) {
                gen_circuit(&c, "anlc", n, q, inst, 1, 0);
                probe_clifford_closure(&ctx, &c);
            }
        }
    }

    /* Family 3: noise continuation on the deterministic MPDO channel path,
     * plus the state-vector cross-path check. */
    for (int n = 3; n <= 4; n++) {
        for (int k = 1; k <= A_MAX_NOISE_K; k++) {
            for (int inst = 0; inst < 2; inst++) {
                mpdo_chain_t ch;
                gen_mpdo_chain(&ch, n, k, inst);
                probe_noise_poly(&ctx, ch.id, "mpdo", k, &ch, eval_mpdo_v, 0);
                probe_noise_poly(&ctx, ch.id, "mpdo", k, &ch, eval_mpdo_v, 1);
                probe_noise_poly(&ctx, ch.id, "branch", k, &ch, eval_chain_branch_v, 0);
                probe_noise_poly(&ctx, ch.id, "branch", k, &ch, eval_chain_branch_v, 1);
                probe_noise_cross_path(&ctx, &ch);
            }
        }
    }

    /* Family 4: noise continuation on entangled circuits with a multi-qubit
     * observable -- the regime the MPDO path cannot represent, gated through
     * the exact Kraus-branch sum over the production state-vector channel. */
    for (int n = 4; n <= 6; n += 2) {
        for (int k = 1; k <= A_MAX_NOISE_K; k++) {
            for (int inst = 0; inst < 2; inst++) {
                gen_circuit(&c, "anlb", n, 0, inst, 0, k);
                probe_noise_poly(&ctx, c.id, "branch", k, &c, eval_branch_sum_v, 0);
                probe_noise_poly(&ctx, c.id, "branch", k, &c, eval_branch_sum_v, 1);
            }
        }
    }

    return oracle_finish(&ctx);
}
