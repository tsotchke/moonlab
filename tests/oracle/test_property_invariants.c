/**
 * @file test_property_invariants.c
 * @brief Adversarial pillar P6 -- metamorphic / property-invariant oracle.
 *
 * Over the generated circuit corpus, asserts the universal invariants that
 * must hold on EVERY backend regardless of the specific circuit:
 *   - unitarity / norm preservation (dense to 1e-12, MPS to 1e-10);
 *   - gate reversibility: U then U-dagger returns |0...0> (dense and MPS);
 *   - entanglement-entropy bounds 0 <= S <= log2(dim_A) and invariance of S(A)
 *     under a local unitary acting inside A;
 *   - MPS canonical form is an isometry: canonicalization preserves the
 *     physical state (fidelity 1) and passes structural validation;
 *   - Clifford backend reversibility (C then C-dagger returns the tableau to
 *     |0>) and Clifford-vs-dense agreement on stabilizer circuits.
 *
 * Emits event: property_invariants_oracle
 */
#include "oracle_common.h"
#include "corpus/circuit_corpus.h"

#define NORM_TOL_DENSE 1e-12
/* The MPS norm flows through repeated SVD contractions; 1e-10 is the honest
 * precision floor for exact (untruncated) evolution, and a genuine
 * normalization defect overshoots it by orders of magnitude. */
#define NORM_TOL_MPS   1e-10
#define REV_TOL_DENSE  1e-10
#define REV_TOL_MPS    1e-9
#define ENT_TOL        1e-9
#define ISO_TOL        1e-9
#define CLIFF_TOL      1e-10

static uint32_t exact_bond_cap(int n) { return 1u << ((n + 1) / 2); }

static oracle_gate_t inverse_gate(const oracle_gate_t *g) {
    oracle_gate_t r = *g;
    if      (!strcmp(g->g, "s"))   r.g = "sdg";
    else if (!strcmp(g->g, "sdg")) r.g = "s";
    else if (!strcmp(g->g, "t"))   r.g = "tdg";
    else if (!strcmp(g->g, "tdg")) r.g = "t";
    else if (!strcmp(g->g, "rx") || !strcmp(g->g, "ry") || !strcmp(g->g, "rz"))
        r.p = -g->p;   /* h,x,y,z,cnot,cz,swap are self-inverse */
    return r;
}

static double dense_norm(const quantum_state_t *s) {
    double sum = 0.0;
    for (uint64_t i = 0; i < s->state_dim; i++) {
        double m = cabs(s->amplitudes[i]);
        sum += m * m;
    }
    return sqrt(sum);
}

/* ---- norm preservation ---- */
static void run_norm(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    char pid[256];

    quantum_state_t *s = quantum_state_create(n);
    for (int i = 0; i < c->num_gates; i++) oracle_apply_dense(s, &c->gates[i]);
    snprintf(pid, sizeof(pid), "%s__norm_dense", c->id);
    double nd = fabs(dense_norm(s) - 1.0);
    if (nd > NORM_TOL_DENSE)
        oracle_probe_fail(ctx, pid, "seed=%llu |norm-1|=%.3e",
                          (unsigned long long)oracle_corpus_seed, nd);
    else oracle_probe_pass(ctx, pid);
    quantum_state_destroy(s);

    tn_state_config_t cfg = tn_state_config_default();
    cfg.max_bond_dim = exact_bond_cap(n);
    cfg.svd_cutoff = 1e-15;
    tn_mps_state_t *m = tn_mps_create_zero((uint32_t)n, &cfg);
    for (int i = 0; i < c->num_gates; i++) oracle_apply_mps(m, &c->gates[i]);
    snprintf(pid, sizeof(pid), "%s__norm_mps", c->id);
    /* tn_mps carries its norm lazily in log_norm_factor; the physical norm is
     * tn_mps_true_norm (tn_mps_norm returns only the raw tensor norm). */
    double nm = fabs(tn_mps_true_norm(m) - 1.0);
    if (nm > NORM_TOL_MPS)
        oracle_probe_fail(ctx, pid, "seed=%llu |norm-1|=%.3e",
                          (unsigned long long)oracle_corpus_seed, nm);
    else oracle_probe_pass(ctx, pid);
    tn_mps_free(m);
}

/* ---- reversibility: U then U-dagger == identity ---- */
static void run_reversibility(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    uint64_t dim = (uint64_t)1 << n;
    char pid[256];

    /* dense */
    quantum_state_t *s = quantum_state_create(n);
    for (int i = 0; i < c->num_gates; i++) oracle_apply_dense(s, &c->gates[i]);
    for (int i = c->num_gates - 1; i >= 0; i--) {
        oracle_gate_t inv = inverse_gate(&c->gates[i]);
        oracle_apply_dense(s, &inv);
    }
    double maxd = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        double target = (i == 0) ? 1.0 : 0.0;
        double d = cabs(s->amplitudes[i] - target);
        if (d > maxd) maxd = d;
    }
    snprintf(pid, sizeof(pid), "%s__reversibility_dense", c->id);
    if (maxd > REV_TOL_DENSE)
        oracle_probe_fail(ctx, pid, "seed=%llu max|psi-|0>|=%.3e",
                          (unsigned long long)oracle_corpus_seed, maxd);
    else oracle_probe_pass(ctx, pid);
    quantum_state_destroy(s);

    /* MPS */
    tn_state_config_t cfg = tn_state_config_default();
    cfg.max_bond_dim = exact_bond_cap(n);
    cfg.svd_cutoff = 1e-15;
    tn_mps_state_t *m = tn_mps_create_zero((uint32_t)n, &cfg);
    for (int i = 0; i < c->num_gates; i++) oracle_apply_mps(m, &c->gates[i]);
    for (int i = c->num_gates - 1; i >= 0; i--) {
        oracle_gate_t inv = inverse_gate(&c->gates[i]);
        oracle_apply_mps(m, &inv);
    }
    tn_mps_normalize(m);   /* commit lazy norm before reading amplitudes */
    double p0 = cabs(tn_mps_amplitude(m, 0));
    double maxo = fabs(p0 - 1.0);
    /* A few off-|0> amplitudes should be ~0. */
    for (int k = 0; k < n; k++) {
        double a = cabs(tn_mps_amplitude(m, (uint64_t)1 << k));
        if (a > maxo) maxo = a;
    }
    snprintf(pid, sizeof(pid), "%s__reversibility_mps", c->id);
    if (maxo > REV_TOL_MPS)
        oracle_probe_fail(ctx, pid, "seed=%llu dev=%.3e",
                          (unsigned long long)oracle_corpus_seed, maxo);
    else oracle_probe_pass(ctx, pid);
    tn_mps_free(m);
}

/* ---- entanglement-entropy bounds + local-unitary invariance ---- */
static void run_entropy(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__entropy", c->id);
    int na = n / 2;
    if (na < 1) { oracle_probe_pass(ctx, pid); return; }

    int *qa = (int *)malloc((size_t)na * sizeof(int));
    for (int i = 0; i < na; i++) qa[i] = i;

    quantum_state_t *s = quantum_state_create(n);
    for (int i = 0; i < c->num_gates; i++) oracle_apply_dense(s, &c->gates[i]);
    double sd = quantum_state_entanglement_entropy(s, qa, (size_t)na);

    tn_state_config_t cfg = tn_state_config_default();
    cfg.max_bond_dim = exact_bond_cap(n);
    cfg.svd_cutoff = 1e-15;
    tn_mps_state_t *m = tn_mps_create_zero((uint32_t)n, &cfg);
    for (int i = 0; i < c->num_gates; i++) oracle_apply_mps(m, &c->gates[i]);
    tn_mps_normalize(m);   /* commit lazy norm so the entropy reflects a unit state */
    double sm = tn_mps_entanglement_entropy(m, (uint32_t)(na - 1));

    double bound = (double)na + ENT_TOL;   /* log2(2^na) = na bits */
    int ok = 1; char why[256] = {0};
    if (!(sd >= -ENT_TOL && sd <= bound)) {
        ok = 0; snprintf(why, sizeof(why), "dense S=%.6f out of [0,%d]", sd, na);
    }
    if (ok && !(sm >= -ENT_TOL && sm <= bound)) {
        ok = 0; snprintf(why, sizeof(why), "MPS S=%.6f out of [0,%d]", sm, na);
    }
    if (ok) {
        /* Local unitary on qubit 0 (inside A) must not change S(A). */
        gate_ry(s, 0, 0.7);
        double sd2 = quantum_state_entanglement_entropy(s, qa, (size_t)na);
        if (fabs(sd2 - sd) > ENT_TOL) {
            ok = 0; snprintf(why, sizeof(why),
                "local unitary changed S(A): %.9f -> %.9f", sd, sd2);
        }
    }
    if (ok) oracle_probe_pass(ctx, pid);
    else oracle_probe_fail(ctx, pid, "seed=%llu class=%s n=%d %s",
                           (unsigned long long)oracle_corpus_seed, c->cls, n, why);
    free(qa);
    tn_mps_free(m);
    quantum_state_destroy(s);
}

/* ---- MPS canonical form is an isometry ---- */
static void run_canonical_isometry(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__mps_canonical_isometry", c->id);

    tn_state_config_t cfg = tn_state_config_default();
    cfg.max_bond_dim = exact_bond_cap(n);
    cfg.svd_cutoff = 1e-15;
    tn_mps_state_t *m = tn_mps_create_zero((uint32_t)n, &cfg);
    for (int i = 0; i < c->num_gates; i++) oracle_apply_mps(m, &c->gates[i]);
    tn_mps_normalize(m);
    tn_mps_state_t *m0 = tn_mps_copy(m);
    if (!m0) {
        oracle_probe_fail(ctx, pid, "seed=%llu tn_mps_copy failed",
                          (unsigned long long)oracle_corpus_seed);
        tn_mps_free(m);
        return;
    }
    tn_mps_left_canonicalize(m);
    tn_mps_normalize(m);
    double fid = tn_mps_fidelity(m0, m);
    int valid = (tn_mps_validate(m) == TN_STATE_SUCCESS);
    if (fabs(fid - 1.0) > ISO_TOL || !valid)
        oracle_probe_fail(ctx, pid, "seed=%llu fidelity=%.12f valid=%d",
                          (unsigned long long)oracle_corpus_seed, fid, valid);
    else oracle_probe_pass(ctx, pid);
    tn_mps_free(m0);
    tn_mps_free(m);
}

/* ---- Clifford backend reversibility + agreement with dense ---- */
static int clifford_z_expect(clifford_tableau_t *t, int q, double *out) {
    clifford_tableau_t *c = clifford_tableau_clone(t);
    if (!c) return -1;
    uint64_t rng = 0x9E3779B1ULL ^ (uint64_t)q;
    int outcome = 0, kind = 0;
    clifford_error_t e = clifford_measure(c, (size_t)q, &rng, &outcome, &kind);
    clifford_tableau_free(c);
    if (e != CLIFFORD_SUCCESS) return -1;
    *out = (kind == 0) ? (1.0 - 2.0 * outcome) : 0.0;
    return 0;
}

static void run_clifford_props(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    char pid[256];

    /* Reversibility: C then C-dagger returns the tableau to |0...0>. */
    clifford_tableau_t *t = clifford_tableau_create((size_t)n);
    int err = (t == NULL);
    for (int i = 0; !err && i < c->num_gates; i++)
        if (oracle_apply_clifford(t, &c->gates[i]) != CLIFFORD_SUCCESS) err = 1;
    for (int i = c->num_gates - 1; !err && i >= 0; i--) {
        oracle_gate_t inv = inverse_gate(&c->gates[i]);
        if (oracle_apply_clifford(t, &inv) != CLIFFORD_SUCCESS) err = 1;
    }
    snprintf(pid, sizeof(pid), "%s__clifford_reversibility", c->id);
    int rev_ok = !err;
    for (int q = 0; rev_ok && q < n; q++) {
        double z;
        if (clifford_z_expect(t, q, &z) != 0 || z < 1.0 - CLIFF_TOL) rev_ok = 0;
    }
    if (rev_ok) oracle_probe_pass(ctx, pid);
    else oracle_probe_fail(ctx, pid, "seed=%llu C then C-dagger did not return to |0>",
                           (unsigned long long)oracle_corpus_seed);
    if (t) clifford_tableau_free(t);

    /* Agreement: tableau <Z_q> equals the dense <Z_q>. */
    clifford_tableau_t *t2 = clifford_tableau_create((size_t)n);
    quantum_state_t *s = quantum_state_create(n);
    int err2 = (t2 == NULL || s == NULL);
    for (int i = 0; !err2 && i < c->num_gates; i++) {
        if (oracle_apply_clifford(t2, &c->gates[i]) != CLIFFORD_SUCCESS) err2 = 1;
        oracle_apply_dense(s, &c->gates[i]);
    }
    snprintf(pid, sizeof(pid), "%s__clifford_agreement", c->id);
    double maxd = 0.0; int wq = 0;
    for (int q = 0; !err2 && q < n; q++) {
        double zc;
        if (clifford_z_expect(t2, q, &zc) != 0) { err2 = 1; break; }
        double d = fabs(zc - measurement_expectation_z(s, q));
        if (d > maxd) { maxd = d; wq = q; }
    }
    if (err2)
        oracle_probe_fail(ctx, pid, "seed=%llu tableau path error",
                          (unsigned long long)oracle_corpus_seed);
    else if (maxd > CLIFF_TOL)
        oracle_probe_fail(ctx, pid, "seed=%llu max|dZ|=%.3e@q%d",
                          (unsigned long long)oracle_corpus_seed, maxd, wq);
    else oracle_probe_pass(ctx, pid);
    if (t2) clifford_tableau_free(t2);
    if (s) quantum_state_destroy(s);
}

int main(void) {
    oracle_ctx_t ctx;
    oracle_ctx_init(&ctx, "property_invariants_oracle");
    fprintf(stdout, "=== P6 metamorphic / property-invariant oracle ===\n");
    for (int i = 0; i < oracle_corpus_count; i++) {
        const oracle_circuit_t *c = &oracle_corpus[i];
        if (c->num_qubits > 8) continue;   /* fast probes; P1 covers the large end */
        run_norm(&ctx, c);
        run_reversibility(&ctx, c);
        run_entropy(&ctx, c);
        run_canonical_isometry(&ctx, c);
        if (oracle_class_is_clifford(c)) run_clifford_props(&ctx, c);
    }
    return oracle_finish(&ctx);
}
