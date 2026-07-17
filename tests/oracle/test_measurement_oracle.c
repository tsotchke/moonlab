/**
 * @file test_measurement_oracle.c
 * @brief Adversarial pillar P3 -- measurement statistics oracle.
 *
 * For corpus circuits (n <= 6, so the bin count stays statistically healthy),
 * compares exact Born probabilities against empirical sampling frequencies via
 * a chi-square goodness-of-fit test drawn from a deterministic splitmix64
 * entropy source (reproducible at the fixed seed and shot count). Also checks
 * collapse consistency: after a projective single-qubit measurement the state
 * is renormalized and a repeated measurement of the same qubit is idempotent.
 *
 * Emits event: measurement_statistics_oracle
 */
#include "oracle_common.h"
#include "corpus/circuit_corpus.h"

#define N_SHOTS 40000
/* Standard-normal quantile for upper-tail probability 1e-3. */
#define Z_999 3.0902323062

/* Wilson-Hilferty upper-tail chi-square critical value at significance 1e-3. */
static double chi2_critical(int dof) {
    double k = (double)dof;
    double t = 1.0 - 2.0 / (9.0 * k) + Z_999 * sqrt(2.0 / (9.0 * k));
    return k * t * t * t;
}

static void run_chi2(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    uint64_t dim = (uint64_t)1 << n;
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__chi2", c->id);

    quantum_state_t *s = quantum_state_create(n);
    double *rv = (double *)malloc(N_SHOTS * sizeof(double));
    uint64_t *outc = (uint64_t *)malloc(N_SHOTS * sizeof(uint64_t));
    uint64_t *counts = (uint64_t *)calloc((size_t)dim, sizeof(uint64_t));
    if (!s || !rv || !outc || !counts) {
        oracle_probe_fail(ctx, pid, "seed=%llu alloc failure",
                          (unsigned long long)oracle_corpus_seed);
        goto cleanup;
    }
    for (int i = 0; i < c->num_gates; i++) oracle_apply_dense(s, &c->gates[i]);

    /* Deterministic uniform draws for the sampler. */
    oracle_rng_t rng;
    oracle_rng_seed(&rng, oracle_corpus_seed ^ 0x5253u ^ (uint64_t)(uintptr_t)c->id);
    for (int i = 0; i < N_SHOTS; i++) rv[i] = oracle_rng_unit(&rng);

    measurement_sample(s, outc, N_SHOTS, rv);
    for (int i = 0; i < N_SHOTS; i++) if (outc[i] < dim) counts[outc[i]]++;

    double chi2 = 0.0;
    int usable = 0;
    for (uint64_t b = 0; b < dim; b++) {
        double e = (double)N_SHOTS * quantum_state_get_probability(s, b);
        if (e >= 5.0) {                 /* Cochran's rule for the approximation. */
            double d = (double)counts[b] - e;
            chi2 += d * d / e;
            usable++;
        }
    }
    if (usable < 2) {
        /* Distribution too concentrated to form a chi-square; the collapse
         * probe covers this circuit's measurement correctness instead. */
        oracle_probe_pass(ctx, pid);
    } else {
        int dof = usable - 1;
        double crit = chi2_critical(dof);
        if (chi2 > crit)
            oracle_probe_fail(ctx, pid,
                "seed=%llu class=%s n=%d depth=%d chi2=%.2f > crit(dof=%d)=%.2f",
                (unsigned long long)oracle_corpus_seed, c->cls, n, c->depth,
                chi2, dof, crit);
        else
            oracle_probe_pass(ctx, pid);
    }

cleanup:
    free(counts); free(outc); free(rv);
    if (s) quantum_state_destroy(s);
}

static void run_collapse(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__collapse", c->id);

    quantum_state_t *s = quantum_state_create(n);
    if (!s) {
        oracle_probe_fail(ctx, pid, "seed=%llu alloc failure",
                          (unsigned long long)oracle_corpus_seed);
        return;
    }
    for (int i = 0; i < c->num_gates; i++) oracle_apply_dense(s, &c->gates[i]);

    oracle_rng_t rng;
    oracle_rng_seed(&rng, oracle_corpus_seed ^ 0xC0115Eu ^ (uint64_t)(uintptr_t)c->id);

    int q = n / 2;
    double r1 = oracle_rng_unit(&rng);
    int o1 = measurement_single_qubit(s, q, r1);

    int ok = 1;
    char why[256] = {0};
    if (o1 < 0) { ok = 0; snprintf(why, sizeof(why), "measurement returned %d", o1); }
    if (ok && !quantum_state_is_normalized(s, 1e-10)) {
        ok = 0; snprintf(why, sizeof(why), "post-measurement state not normalized");
    }
    if (ok) {
        double pkeep = (o1 == 1) ? measurement_probability_one(s, q)
                                 : measurement_probability_zero(s, q);
        if (pkeep < 1.0 - 1e-9) {
            ok = 0; snprintf(why, sizeof(why),
                             "P(outcome %d)=%.12f != 1 after collapse", o1, pkeep);
        }
    }
    if (ok) {
        double r2 = oracle_rng_unit(&rng);
        int o2 = measurement_single_qubit(s, q, r2);
        if (o2 != o1) {
            ok = 0; snprintf(why, sizeof(why),
                             "repeated measurement not idempotent: %d then %d", o1, o2);
        }
    }
    if (ok) oracle_probe_pass(ctx, pid);
    else oracle_probe_fail(ctx, pid, "seed=%llu class=%s n=%d q=%d %s",
                           (unsigned long long)oracle_corpus_seed, c->cls, n, q, why);
    quantum_state_destroy(s);
}

int main(void) {
    oracle_ctx_t ctx;
    oracle_ctx_init(&ctx, "measurement_statistics_oracle");
    fprintf(stdout, "=== P3 measurement statistics oracle (N=%d shots) ===\n", N_SHOTS);
    for (int i = 0; i < oracle_corpus_count; i++) {
        const oracle_circuit_t *c = &oracle_corpus[i];
        if (c->num_qubits > 6) continue;   /* keep the chi-square well-conditioned */
        run_chi2(&ctx, c);
        run_collapse(&ctx, c);
    }
    return oracle_finish(&ctx);
}
