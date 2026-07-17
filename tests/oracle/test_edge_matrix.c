/**
 * @file test_edge_matrix.c
 * @brief Adversarial pillar P5 -- edge composition matrix.
 *
 * Sweeps feature PAIRS that single-feature unit tests miss:
 *   - fusion on/off parity (fuse_execute raw vs on the fuse_compile output);
 *   - MPS canonical-form transitions (an expectation is invariant under
 *     left/right/mixed canonicalization and under a lossless truncate);
 *   - gate then measure then gate (collapse followed by further evolution
 *     stays normalized and respects the collapsed subspace);
 *   - noise-channel + measurement composition (p=0 no-op, p=1 deterministic
 *     flip, trace preservation under a stochastic channel).
 *
 * Emits event: edge_matrix_oracle
 */
#include "oracle_common.h"
#include "corpus/circuit_corpus.h"

#include "../../src/quantum/noise.h"
#include "../../src/optimization/fusion/fusion.h"

#define P5_TOL 1e-10

static uint32_t exact_bond_cap(int n) { return 1u << ((n + 1) / 2); }

/* Translate one corpus gate into the fusion circuit builder. */
static int fuse_append_gate(fuse_circuit_t *fc, const oracle_gate_t *g) {
    const char *n = g->g;
    if (!strcmp(n, "h"))    return fuse_append_h(fc, g->q0);
    if (!strcmp(n, "x"))    return fuse_append_x(fc, g->q0);
    if (!strcmp(n, "y"))    return fuse_append_y(fc, g->q0);
    if (!strcmp(n, "z"))    return fuse_append_z(fc, g->q0);
    if (!strcmp(n, "s"))    return fuse_append_s(fc, g->q0);
    if (!strcmp(n, "sdg"))  return fuse_append_sdg(fc, g->q0);
    if (!strcmp(n, "t"))    return fuse_append_t(fc, g->q0);
    if (!strcmp(n, "tdg"))  return fuse_append_tdg(fc, g->q0);
    if (!strcmp(n, "rx"))   return fuse_append_rx(fc, g->q0, g->p);
    if (!strcmp(n, "ry"))   return fuse_append_ry(fc, g->q0, g->p);
    if (!strcmp(n, "rz"))   return fuse_append_rz(fc, g->q0, g->p);
    if (!strcmp(n, "cnot")) return fuse_append_cnot(fc, g->q0, g->q1);
    if (!strcmp(n, "cz"))   return fuse_append_cz(fc, g->q0, g->q1);
    if (!strcmp(n, "swap")) return fuse_append_swap(fc, g->q0, g->q1);
    return -1;
}

static void run_fusion_parity(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    uint64_t dim = (uint64_t)1 << n;
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__fusion_parity", c->id);

    fuse_circuit_t *fc = fuse_circuit_create((size_t)n);
    quantum_state_t *raw = quantum_state_create(n);
    quantum_state_t *fused = quantum_state_create(n);
    if (!fc || !raw || !fused) {
        oracle_probe_fail(ctx, pid, "seed=%llu alloc failure",
                          (unsigned long long)oracle_corpus_seed);
        goto cleanup;
    }
    int build_err = 0;
    for (int i = 0; i < c->num_gates; i++)
        if (fuse_append_gate(fc, &c->gates[i]) != 0) build_err = 1;
    if (build_err) {
        oracle_probe_fail(ctx, pid, "seed=%llu fusion circuit build error",
                          (unsigned long long)oracle_corpus_seed);
        goto cleanup;
    }
    fuse_stats_t st;
    fuse_circuit_t *compiled = fuse_compile(fc, &st);
    if (!compiled) {
        oracle_probe_fail(ctx, pid, "seed=%llu fuse_compile failed",
                          (unsigned long long)oracle_corpus_seed);
        goto cleanup;
    }
    fuse_execute(fc, raw);
    fuse_execute(compiled, fused);
    double maxd = 0.0; uint64_t wi = 0;
    for (uint64_t i = 0; i < dim; i++) {
        double d = cabs(raw->amplitudes[i] - fused->amplitudes[i]);
        if (d > maxd) { maxd = d; wi = i; }
    }
    fuse_circuit_free(compiled);
    if (maxd > P5_TOL)
        oracle_probe_fail(ctx, pid,
            "seed=%llu class=%s n=%d depth=%d max|raw-fused|=%.3e@|%llu> (orig=%zu fused=%zu)",
            (unsigned long long)oracle_corpus_seed, c->cls, n, c->depth,
            maxd, (unsigned long long)wi, st.original_gates, st.fused_gates);
    else
        oracle_probe_pass(ctx, pid);

cleanup:
    if (fc) fuse_circuit_free(fc);
    if (raw) quantum_state_destroy(raw);
    if (fused) quantum_state_destroy(fused);
}

static double max_z_diff(tn_mps_state_t *m, const double *base, int n) {
    double maxd = 0.0;
    for (int q = 0; q < n; q++) {
        double d = fabs(base[q] - tn_expectation_z(m, (uint32_t)q));
        if (d > maxd) maxd = d;
    }
    return maxd;
}

static void run_canonical_invariance(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    char pid[256];
    snprintf(pid, sizeof(pid), "%s__canonical_invariance", c->id);

    tn_state_config_t cfg = tn_state_config_default();
    cfg.max_bond_dim = exact_bond_cap(n);
    cfg.svd_cutoff = 1e-15;
    tn_mps_state_t *m = tn_mps_create_zero((uint32_t)n, &cfg);
    double *base = (double *)malloc((size_t)n * sizeof(double));
    if (!m || !base) {
        oracle_probe_fail(ctx, pid, "seed=%llu alloc failure",
                          (unsigned long long)oracle_corpus_seed);
        goto cleanup;
    }
    for (int i = 0; i < c->num_gates; i++) oracle_apply_mps(m, &c->gates[i]);
    for (int q = 0; q < n; q++) base[q] = tn_expectation_z(m, (uint32_t)q);

    double worst = 0.0;
    const char *stage = "left";
    tn_mps_left_canonicalize(m);
    double d = max_z_diff(m, base, n);
    if (d > worst) { worst = d; stage = "left"; }
    tn_mps_right_canonicalize(m);
    d = max_z_diff(m, base, n);
    if (d > worst) { worst = d; stage = "right"; }
    tn_mps_mixed_canonicalize(m, (uint32_t)(n / 2));
    d = max_z_diff(m, base, n);
    if (d > worst) { worst = d; stage = "mixed"; }
    double terr = 0.0;
    tn_mps_truncate(m, exact_bond_cap(n), &terr);   /* lossless at the exact cap */
    d = max_z_diff(m, base, n);
    if (d > worst) { worst = d; stage = "truncate"; }

    if (worst > P5_TOL)
        oracle_probe_fail(ctx, pid,
            "seed=%llu class=%s n=%d depth=%d worst dZ=%.3e at stage=%s",
            (unsigned long long)oracle_corpus_seed, c->cls, n, c->depth, worst, stage);
    else
        oracle_probe_pass(ctx, pid);

cleanup:
    free(base);
    if (m) tn_mps_free(m);
}

/* gate then measure then gate: entangle, collapse, evolve, re-measure. */
static void run_gate_measure_gate(oracle_ctx_t *ctx) {
    const char *pid = "compose__gate_measure_gate";
    int n = 4;
    quantum_state_t *s = quantum_state_create(n);
    if (!s) { oracle_probe_fail(ctx, pid, "alloc failure"); return; }
    gate_hadamard(s, 0);
    gate_cnot(s, 0, 1);
    gate_cnot(s, 1, 2);
    gate_cnot(s, 2, 3);   /* GHZ_4 */

    oracle_rng_t rng; oracle_rng_seed(&rng, 0xED6E1234ULL);
    int q = 0;
    int o1 = measurement_single_qubit(s, q, oracle_rng_unit(&rng));
    int ok = 1; char why[256] = {0};
    if (o1 < 0) { ok = 0; snprintf(why, sizeof(why), "measure failed"); }
    if (ok && !quantum_state_is_normalized(s, 1e-10)) {
        ok = 0; snprintf(why, sizeof(why), "not normalized after collapse");
    }
    if (ok) {
        /* GHZ: measuring q0 forces the whole register; qubit 3 must agree. */
        double p_agree = (o1 == 1) ? measurement_probability_one(s, 3)
                                   : measurement_probability_zero(s, 3);
        if (p_agree < 1.0 - 1e-9) {
            ok = 0; snprintf(why, sizeof(why), "GHZ correlation broken: P=%.12f", p_agree);
        }
    }
    if (ok) {
        gate_pauli_x(s, q);   /* evolve the collapsed qubit */
        double pflip = (o1 == 1) ? measurement_probability_zero(s, q)
                                 : measurement_probability_one(s, q);
        if (pflip < 1.0 - 1e-9) {
            ok = 0; snprintf(why, sizeof(why), "post-collapse X did not flip: P=%.12f", pflip);
        }
    }
    if (ok) {
        int o2 = measurement_single_qubit(s, q, oracle_rng_unit(&rng));
        if (o2 == o1) { ok = 0; snprintf(why, sizeof(why), "flip not observed on remeasure"); }
    }
    if (ok) oracle_probe_pass(ctx, pid);
    else oracle_probe_fail(ctx, pid, "%s", why);
    quantum_state_destroy(s);
}

static double p1(const quantum_state_t *s, int q) { return measurement_probability_one(s, q); }

static void run_noise_composition(oracle_ctx_t *ctx) {
    oracle_rng_t rng; oracle_rng_seed(&rng, 0x0155EEDULL);

    /* p=0 bit flip is a measurement no-op. */
    {
        const char *pid = "compose__noise_p0_noop";
        quantum_state_t *s = quantum_state_create(3);
        gate_ry(s, 1, 0.9);
        double before = p1(s, 1);
        noise_bit_flip(s, 1, 0.0, oracle_rng_unit(&rng));
        double after = p1(s, 1);
        if (fabs(before - after) > 1e-12)
            oracle_probe_fail(ctx, pid, "p=0 channel changed P(1): %.12f -> %.12f", before, after);
        else
            oracle_probe_pass(ctx, pid);
        quantum_state_destroy(s);
    }

    /* p=1 bit flip is a deterministic X; composed with measurement it flips. */
    {
        const char *pid = "compose__noise_p1_flip";
        quantum_state_t *s = quantum_state_create(3);           /* |000> */
        noise_bit_flip(s, 2, 1.0, oracle_rng_unit(&rng));       /* -> |001..> on q2 */
        double pafter = p1(s, 2);
        int ok = (pafter > 1.0 - 1e-12);
        int o = ok ? measurement_single_qubit(s, 2, oracle_rng_unit(&rng)) : -1;
        if (ok && o == 1) oracle_probe_pass(ctx, pid);
        else oracle_probe_fail(ctx, pid, "p=1 flip P(1)=%.12f outcome=%d", pafter, o);
        quantum_state_destroy(s);
    }

    /* A stochastic depolarizing channel preserves the pure-state norm. */
    {
        const char *pid = "compose__noise_trace_preserving";
        quantum_state_t *s = quantum_state_create(3);
        gate_hadamard(s, 0); gate_cnot(s, 0, 1); gate_ry(s, 2, 1.3);
        int ok = 1;
        for (int k = 0; k < 8 && ok; k++) {
            noise_depolarizing_single(s, k % 3, 0.3, oracle_rng_unit(&rng));
            if (!quantum_state_is_normalized(s, 1e-10)) ok = 0;
        }
        if (ok) oracle_probe_pass(ctx, pid);
        else oracle_probe_fail(ctx, pid, "norm not preserved under depolarizing channel");
        quantum_state_destroy(s);
    }
}

int main(void) {
    oracle_ctx_t ctx;
    oracle_ctx_init(&ctx, "edge_matrix_oracle");
    fprintf(stdout, "=== P5 edge composition matrix ===\n");
    for (int i = 0; i < oracle_corpus_count; i++) {
        const oracle_circuit_t *c = &oracle_corpus[i];
        if (c->num_qubits > 6) continue;   /* fast probes; P1 covers the large end */
        run_fusion_parity(&ctx, c);
        run_canonical_invariance(&ctx, c);
    }
    run_gate_measure_gate(&ctx);
    run_noise_composition(&ctx);
    return oracle_finish(&ctx);
}
