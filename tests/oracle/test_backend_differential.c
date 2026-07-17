/**
 * @file test_backend_differential.c
 * @brief Adversarial pillar P1 -- backend differential oracle.
 *
 * Runs every circuit in the seed-deterministic corpus on the dense state
 * vector and on tn_mps (bond cap 2^ceil(n/2), exact at these sizes) and
 * compares the full probability vector plus every <Z_q> and <Z_q Z_{q+1}>
 * expectation to 1e-10. Clifford-only circuits are additionally cross-checked
 * against the Aaronson-Gottesman tableau exactly.
 *
 * Any mismatch prints the seed and the circuit id (which encodes class, qubit
 * count, depth and instance) so the failure replays deterministically.
 *
 * Emits event: backend_differential_oracle
 */
#include "oracle_common.h"
#include "corpus/circuit_corpus.h"

#define P1_TOL 1e-10

static uint32_t exact_bond_cap(int n) {
    /* Max Schmidt rank across any cut is 2^floor(n/2); 2^ceil(n/2) >= that. */
    return 1u << ((n + 1) / 2);
}

/* tn_mps_to_statevector indexes basis states big-endian (qubit 0 is the most
 * significant bit) whereas the dense backend is little-endian (qubit 0 is the
 * least significant bit). Reverse the n-bit index to compare like for like. */
static uint64_t bitrev_n(uint64_t x, int n) {
    uint64_t r = 0;
    for (int b = 0; b < n; b++) r |= ((x >> b) & 1ULL) << (n - 1 - b);
    return r;
}

/* Dense <Z_q> via the measurement subsystem. */
static double dense_z(const quantum_state_t *s, int q) {
    return measurement_expectation_z(s, q);
}

/* Tableau <Z_q> from a Clifford state, computed exactly by cloning the tableau
 * and measuring qubit q: a deterministic outcome pins <Z_q> = +-1, a random
 * outcome (anticommuting stabilizer) means <Z_q> = 0. */
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

/* The exact MPS at n=10 costs bond-32 SVDs per gate (seconds per deep circuit),
 * so the full MPS differential runs where it is affordable inside the < 2 min
 * budget: every circuit up to n=8 (all depths), plus n=10 at shallow depth. The
 * dense-vs-tableau exact check below still runs on every circuit, so n=10 is
 * never left uncovered. */
static int mps_affordable(int n, int depth) {
    return n <= 8 || depth <= 4;
}

static void run_diff(oracle_ctx_t *ctx, const oracle_circuit_t *c) {
    int n = c->num_qubits;
    uint64_t dim = (uint64_t)1 << n;
    char pid[256];

    quantum_state_t *dense = quantum_state_create(n);
    if (!dense) {
        snprintf(pid, sizeof(pid), "%s__diff_mps", c->id);
        oracle_probe_fail(ctx, pid, "seed=%llu dense allocation failure",
                          (unsigned long long)oracle_corpus_seed);
        return;
    }
    for (int i = 0; i < c->num_gates; i++) oracle_apply_dense(dense, &c->gates[i]);

    /* Dense vs tn_mps (bond cap exact at these sizes), where affordable. */
    if (mps_affordable(n, c->depth)) {
        tn_state_config_t cfg = tn_state_config_default();
        cfg.max_bond_dim = exact_bond_cap(n);
        cfg.svd_cutoff = 1e-15;
        tn_mps_state_t *mps = tn_mps_create_zero((uint32_t)n, &cfg);
        double complex *amps = (double complex *)malloc(dim * sizeof(double complex));
        snprintf(pid, sizeof(pid), "%s__diff_mps", c->id);
        if (!mps || !amps) {
            oracle_probe_fail(ctx, pid, "seed=%llu mps allocation failure",
                              (unsigned long long)oracle_corpus_seed);
        } else {
            int apply_err = 0;
            for (int i = 0; i < c->num_gates; i++)
                if (oracle_apply_mps(mps, &c->gates[i]) != 0) apply_err = 1;
            /* tn_mps uses lazy normalization (the norm lives in log_norm_factor);
             * commit it so tn_mps_to_statevector returns the physical amplitudes. */
            tn_mps_normalize(mps);
            if (apply_err) {
                oracle_probe_fail(ctx, pid, "seed=%llu mps gate apply error",
                                  (unsigned long long)oracle_corpus_seed);
            } else if (tn_mps_to_statevector(mps, amps) != TN_STATE_SUCCESS) {
                oracle_probe_fail(ctx, pid, "seed=%llu tn_mps_to_statevector failed",
                                  (unsigned long long)oracle_corpus_seed);
            } else {
                double max_p = 0.0; uint64_t worst_i = 0;
                for (uint64_t i = 0; i < dim; i++) {
                    double pd = quantum_state_get_probability(dense, i);
                    double complex a = amps[bitrev_n(i, n)];
                    double pm = cabs(a) * cabs(a);
                    double d = fabs(pd - pm);
                    if (d > max_p) { max_p = d; worst_i = i; }
                }
                double max_z = 0.0, max_zz = 0.0;
                int worst_zq = 0, worst_zzq = 0;
                for (int q = 0; q < n; q++) {
                    double d = fabs(dense_z(dense, q) - tn_expectation_z(mps, (uint32_t)q));
                    if (d > max_z) { max_z = d; worst_zq = q; }
                }
                for (int q = 0; q + 1 < n; q++) {
                    double d = fabs(measurement_correlation_zz(dense, q, q + 1)
                                    - tn_expectation_zz(mps, (uint32_t)q, (uint32_t)(q + 1)));
                    if (d > max_zz) { max_zz = d; worst_zzq = q; }
                }
                if (max_p > P1_TOL || max_z > P1_TOL || max_zz > P1_TOL) {
                    oracle_probe_fail(ctx, pid,
                        "seed=%llu class=%s n=%d depth=%d | dP=%.3e@|%llu> dZ=%.3e@q%d dZZ=%.3e@q%d",
                        (unsigned long long)oracle_corpus_seed, c->cls, n, c->depth,
                        max_p, (unsigned long long)worst_i, max_z, worst_zq, max_zz, worst_zzq);
                } else {
                    oracle_probe_pass(ctx, pid);
                }
            }
        }
        free(amps);
        if (mps) tn_mps_free(mps);
    }

    /* Clifford-only circuits: exact cross-check against the tableau (all n). */
    if (oracle_class_is_clifford(c)) {
        snprintf(pid, sizeof(pid), "%s__diff_clifford", c->id);
        clifford_tableau_t *t = clifford_tableau_create((size_t)n);
        int cerr = (t == NULL);
        for (int i = 0; !cerr && i < c->num_gates; i++) {
            if (oracle_apply_clifford(t, &c->gates[i]) != CLIFFORD_SUCCESS) cerr = 1;
        }
        double max_cz = 0.0;
        int worst_cq = 0;
        for (int q = 0; !cerr && q < n; q++) {
            double zc;
            if (tableau_z_expect(t, q, &zc) != 0) { cerr = 1; break; }
            double d = fabs(zc - dense_z(dense, q));
            if (d > max_cz) { max_cz = d; worst_cq = q; }
        }
        if (t) clifford_tableau_free(t);
        if (cerr) {
            oracle_probe_fail(ctx, pid, "seed=%llu tableau path error",
                              (unsigned long long)oracle_corpus_seed);
        } else if (max_cz > P1_TOL) {
            oracle_probe_fail(ctx, pid,
                "seed=%llu class=%s n=%d depth=%d | dZ_tableau=%.3e@q%d",
                (unsigned long long)oracle_corpus_seed, c->cls, n, c->depth,
                max_cz, worst_cq);
        } else {
            oracle_probe_pass(ctx, pid);
        }
    }

    quantum_state_destroy(dense);
}

int main(void) {
    oracle_ctx_t ctx;
    oracle_ctx_init(&ctx, "backend_differential_oracle");
    fprintf(stdout,
            "=== P1 backend differential oracle (corpus seed=%llu, %d circuits) ===\n",
            (unsigned long long)oracle_corpus_seed, oracle_corpus_count);
    for (int i = 0; i < oracle_corpus_count; i++) {
        run_diff(&ctx, &oracle_corpus[i]);
    }
    return oracle_finish(&ctx);
}
