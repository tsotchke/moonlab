/**
 * @file test_gradient_oracle.c
 * @brief Adversarial pillar P2 -- gradient oracle.
 *
 * Seeded hardware-efficient ansaetze (2-6 qubits, 1-3 layers) over random
 * Pauli-sum Hamiltonians. Cross-checks three independent gradient routes:
 *   adjoint autograd (vqe_compute_gradient) vs analytic parameter-shift  (1e-7)
 *   adjoint autograd vs central finite differences                       (1e-5)
 * Plus the quantum geometric tensor (vqe_compute_qgt) symmetry and PSD checks.
 *
 * Emits event: gradient_oracle
 */
#include "oracle_common.h"
#include "../../src/algorithms/vqe.h"

#define G_TOL_PSR 1e-7
#define G_TOL_FD  1e-5
#define FD_EPS    1e-4
#define QGT_SYM_TOL 1e-6
#define QGT_PSD_JITTER 1e-6

static const uint64_t BASE_SEED = 0x9E3779B97F4A7C15ULL;

/* Build a random Hermitian Pauli-sum Hamiltonian on n qubits. */
static pauli_hamiltonian_t *random_hamiltonian(oracle_rng_t *rng, int n, int num_terms) {
    pauli_hamiltonian_t *h = pauli_hamiltonian_create((size_t)n, (size_t)num_terms);
    if (!h) return NULL;
    char *ps = (char *)malloc((size_t)n + 1);
    for (int t = 0; t < num_terms; t++) {
        for (int q = 0; q < n; q++) ps[q] = "IXYZ"[oracle_rng_below(rng, 4)];
        ps[n] = '\0';
        double coeff = 2.0 * oracle_rng_unit(rng) - 1.0;   /* [-1, 1] */
        pauli_hamiltonian_add_term(h, coeff, ps, (size_t)t);
    }
    free(ps);
    return h;
}

/* Cholesky-based PSD test: returns 1 iff (G + jitter*I) is positive definite,
 * i.e. every eigenvalue of the symmetric matrix G exceeds -jitter. */
static int is_psd(const double *g, int m, double jitter) {
    double *L = (double *)calloc((size_t)m * m, sizeof(double));
    if (!L) return 0;
    int ok = 1;
    for (int i = 0; i < m && ok; i++) {
        for (int j = 0; j <= i && ok; j++) {
            double sum = g[i * m + j];
            if (i == j) sum += jitter;
            for (int k = 0; k < j; k++) sum -= L[i * m + k] * L[j * m + k];
            if (i == j) {
                if (sum <= 0.0) { ok = 0; break; }
                L[i * m + i] = sqrt(sum);
            } else {
                L[i * m + j] = sum / L[j * m + j];
            }
        }
    }
    free(L);
    return ok;
}

static void run_instance(oracle_ctx_t *ctx, int n, int layers, int inst) {
    char base[128];
    snprintf(base, sizeof(base), "hea_n%d_L%d_s%d", n, layers, inst);
    char pid[192];

    oracle_rng_t rng;
    oracle_rng_seed(&rng, BASE_SEED ^ ((uint64_t)n << 40) ^ ((uint64_t)layers << 20)
                          ^ ((uint64_t)inst << 3));
    oracle_rng_t erng;
    quantum_entropy_ctx_t entropy;
    oracle_entropy_bind(&entropy, &erng, BASE_SEED ^ 0xABCDEFULL ^ (uint64_t)inst);

    pauli_hamiltonian_t *h = random_hamiltonian(&rng, n, 3 + n);
    vqe_ansatz_t *ansatz = vqe_create_hardware_efficient_ansatz((size_t)n, (size_t)layers);
    vqe_optimizer_t *opt = vqe_optimizer_create(VQE_OPTIMIZER_GRADIENT_DESCENT);
    if (!h || !ansatz || !opt) {
        snprintf(pid, sizeof(pid), "%s__adjoint_vs_psr", base);
        oracle_probe_fail(ctx, pid, "setup failure");
        goto cleanup;
    }
    vqe_solver_t *solver = vqe_solver_create(h, ansatz, opt, &entropy);
    if (!solver) {
        snprintf(pid, sizeof(pid), "%s__adjoint_vs_psr", base);
        oracle_probe_fail(ctx, pid, "solver create failure");
        goto cleanup;
    }

    size_t np = ansatz->num_parameters;
    double *theta = (double *)malloc(np * sizeof(double));
    double *g_adj = (double *)malloc(np * sizeof(double));
    double *g_psr = (double *)malloc(np * sizeof(double));
    double *g_fd  = (double *)malloc(np * sizeof(double));
    double *pert  = (double *)malloc(np * sizeof(double));
    double *qgt   = (double *)malloc(np * np * sizeof(double));
    if (!theta || !g_adj || !g_psr || !g_fd || !pert || !qgt) {
        snprintf(pid, sizeof(pid), "%s__adjoint_vs_psr", base);
        oracle_probe_fail(ctx, pid, "gradient buffer alloc failure");
        goto free_solver;
    }

    for (size_t i = 0; i < np; i++) {
        theta[i] = (2.0 * oracle_rng_unit(&rng) - 1.0) * M_PI;
        ansatz->parameters[i] = theta[i];
    }

    /* Adjoint autograd (fast path for noise-free HEA). */
    int adj_ok = (vqe_compute_gradient(solver, theta, g_adj) == 0);

    /* Analytic parameter shift and central finite difference. */
    for (size_t i = 0; i < np; i++) {
        memcpy(pert, theta, np * sizeof(double));
        pert[i] = theta[i] + M_PI / 2.0;
        double ep = vqe_compute_energy(solver, pert);
        pert[i] = theta[i] - M_PI / 2.0;
        double em = vqe_compute_energy(solver, pert);
        g_psr[i] = 0.5 * (ep - em);

        memcpy(pert, theta, np * sizeof(double));
        pert[i] = theta[i] + FD_EPS;
        double fp = vqe_compute_energy(solver, pert);
        pert[i] = theta[i] - FD_EPS;
        double fm = vqe_compute_energy(solver, pert);
        g_fd[i] = (fp - fm) / (2.0 * FD_EPS);
    }

    /* adjoint vs parameter-shift */
    snprintf(pid, sizeof(pid), "%s__adjoint_vs_psr", base);
    if (!adj_ok) {
        oracle_probe_fail(ctx, pid, "seed=%llu vqe_compute_gradient returned error",
                          (unsigned long long)BASE_SEED);
    } else {
        double maxd = 0.0; size_t wi = 0;
        for (size_t i = 0; i < np; i++) {
            double d = fabs(g_adj[i] - g_psr[i]);
            if (d > maxd) { maxd = d; wi = i; }
        }
        if (maxd > G_TOL_PSR)
            oracle_probe_fail(ctx, pid, "seed=%llu n=%d L=%d np=%zu max|adj-psr|=%.3e@%zu",
                              (unsigned long long)BASE_SEED, n, layers, np, maxd, wi);
        else
            oracle_probe_pass(ctx, pid);
    }

    /* adjoint vs finite difference */
    snprintf(pid, sizeof(pid), "%s__adjoint_vs_fd", base);
    if (!adj_ok) {
        oracle_probe_fail(ctx, pid, "seed=%llu vqe_compute_gradient returned error",
                          (unsigned long long)BASE_SEED);
    } else {
        double maxd = 0.0; size_t wi = 0;
        for (size_t i = 0; i < np; i++) {
            double d = fabs(g_adj[i] - g_fd[i]);
            if (d > maxd) { maxd = d; wi = i; }
        }
        if (maxd > G_TOL_FD)
            oracle_probe_fail(ctx, pid, "seed=%llu n=%d L=%d np=%zu max|adj-fd|=%.3e@%zu",
                              (unsigned long long)BASE_SEED, n, layers, np, maxd, wi);
        else
            oracle_probe_pass(ctx, pid);
    }

    /* QGT symmetry + PSD */
    int qgt_ok = (vqe_compute_qgt(solver, theta, qgt) == 0);
    snprintf(pid, sizeof(pid), "%s__qgt_symmetry", base);
    if (!qgt_ok) {
        oracle_probe_fail(ctx, pid, "seed=%llu vqe_compute_qgt returned error",
                          (unsigned long long)BASE_SEED);
    } else {
        double maxa = 0.0;
        for (size_t i = 0; i < np; i++)
            for (size_t j = i + 1; j < np; j++) {
                double d = fabs(qgt[i * np + j] - qgt[j * np + i]);
                if (d > maxa) maxa = d;
            }
        if (maxa > QGT_SYM_TOL)
            oracle_probe_fail(ctx, pid, "seed=%llu asymmetry=%.3e",
                              (unsigned long long)BASE_SEED, maxa);
        else
            oracle_probe_pass(ctx, pid);
    }

    snprintf(pid, sizeof(pid), "%s__qgt_psd", base);
    if (!qgt_ok) {
        oracle_probe_fail(ctx, pid, "seed=%llu vqe_compute_qgt returned error",
                          (unsigned long long)BASE_SEED);
    } else {
        /* Symmetrize before the PSD test so an asymmetry finding is isolated
         * to the symmetry probe rather than doubly failing here. */
        for (size_t i = 0; i < np; i++)
            for (size_t j = i + 1; j < np; j++) {
                double avg = 0.5 * (qgt[i * np + j] + qgt[j * np + i]);
                qgt[i * np + j] = qgt[j * np + i] = avg;
            }
        if (!is_psd(qgt, (int)np, QGT_PSD_JITTER))
            oracle_probe_fail(ctx, pid,
                "seed=%llu QGT not PSD (min eig < -%.0e)",
                (unsigned long long)BASE_SEED, QGT_PSD_JITTER);
        else
            oracle_probe_pass(ctx, pid);
    }

    free(theta); free(g_adj); free(g_psr); free(g_fd); free(pert); free(qgt);
free_solver:
    vqe_solver_free(solver);
cleanup:
    if (opt) vqe_optimizer_free(opt);
    if (ansatz) vqe_ansatz_free(ansatz);
    if (h) pauli_hamiltonian_free(h);
}

int main(void) {
    oracle_ctx_t ctx;
    oracle_ctx_init(&ctx, "gradient_oracle");
    fprintf(stdout, "=== P2 gradient oracle (adjoint vs PSR vs FD, + QGT) ===\n");
    int inst = 0;
    for (int n = 2; n <= 6; n += 2) {
        for (int layers = 1; layers <= 3; layers++) {
            run_instance(&ctx, n, layers, inst++);
        }
    }
    return oracle_finish(&ctx);
}
