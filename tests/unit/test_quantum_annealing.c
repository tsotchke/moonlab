/**
 * @file test_quantum_annealing.c
 * @brief Independent and contract tests for transverse-field annealing.
 */

#include "../../src/algorithms/quantum_annealing.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); failures++; } \
} while (0)

static double ising_energy_arrays(size_t n, const double *h, const double *J,
                                  double offset, uint64_t bits)
{
    double e = offset;
    for (size_t i = 0; i < n; i++) {
        const double zi = ((bits >> i) & 1u) ? -1.0 : 1.0;
        e += h[i] * zi;
        for (size_t j = i + 1; j < n; j++) {
            const double zj = ((bits >> j) & 1u) ? -1.0 : 1.0;
            e += J[i * n + j] * zi * zj;
        }
    }
    return e;
}

/* Independent RK4 oracle for one qubit under
 * H(t) = -cos^2(pi s/2) X + sin^2(pi s/2) h Z. */
static void oneq_rhs(double t, double T, double h,
                     double complex a, double complex b,
                     double complex *da, double complex *db)
{
    const double s = t / T;
    const double sn = sin(0.5 * M_PI * s);
    const double cs = cos(0.5 * M_PI * s);
    const double A = cs * cs, B = sn * sn;
    *da = -I * (B * h * a - A * b);
    *db = -I * (-A * a - B * h * b);
}

static double oneq_rk4_ground_probability(double T, double h)
{
    const size_t steps = 100000;
    const double dt = T / (double)steps;
    double complex a = M_SQRT1_2, b = M_SQRT1_2;
    for (size_t k = 0; k < steps; k++) {
        const double t = (double)k * dt;
        double complex a1, b1, a2, b2, a3, b3, a4, b4;
        oneq_rhs(t, T, h, a, b, &a1, &b1);
        oneq_rhs(t + 0.5 * dt, T, h,
                 a + 0.5 * dt * a1, b + 0.5 * dt * b1, &a2, &b2);
        oneq_rhs(t + 0.5 * dt, T, h,
                 a + 0.5 * dt * a2, b + 0.5 * dt * b2, &a3, &b3);
        oneq_rhs(t + dt, T, h,
                 a + dt * a3, b + dt * b3, &a4, &b4);
        a += dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0;
        b += dt * (b1 + 2.0 * b2 + 2.0 * b3 + b4) / 6.0;
    }
    return h < 0.0 ? pow(cabs(a), 2.0) : pow(cabs(b), 2.0);
}

static void test_schedules(void)
{
    for (int kind = MOONLAB_ANNEAL_SCHEDULE_LINEAR;
         kind <= MOONLAB_ANNEAL_SCHEDULE_COSINE; kind++) {
        double A0, B0, A1, B1, prevA = 2.0, prevB = -1.0;
        CHECK(moonlab_anneal_schedule_values(kind, 0.0, &A0, &B0) == 0,
              "schedule %d accepts s=0", kind);
        CHECK(moonlab_anneal_schedule_values(kind, 1.0, &A1, &B1) == 0,
              "schedule %d accepts s=1", kind);
        CHECK(fabs(A0 - 1.0) < 1e-15 && fabs(B0) < 1e-15,
              "schedule %d starts at driver", kind);
        CHECK(fabs(A1) < 1e-15 && fabs(B1 - 1.0) < 1e-15,
              "schedule %d ends at problem", kind);
        for (int i = 0; i <= 100; i++) {
            double A, B;
            const double s = (double)i / 100.0;
            CHECK(moonlab_anneal_schedule_values(kind, s, &A, &B) == 0,
                  "schedule %d point %d", kind, i);
            CHECK(A <= prevA + 1e-15 && B >= prevB - 1e-15,
                  "schedule %d monotone at %d", kind, i);
            CHECK(fabs(A + B - 1.0) < 2e-15,
                  "schedule %d partitions unity at %d", kind, i);
            prevA = A; prevB = B;
        }
    }
    double A, B;
    CHECK(moonlab_anneal_schedule_values(99, 0.5, &A, &B) ==
          MOONLAB_ANNEAL_SCHEDULE_ERROR, "unknown schedule rejected");
    CHECK(moonlab_anneal_schedule_values(0, -0.1, &A, &B) ==
          MOONLAB_ANNEAL_BAD_ARG, "out-of-domain schedule point rejected");
}

static void test_qubo_conversion(void)
{
    const size_t n = 3;
    const double Q[9] = {
        -1.5, 0.7, -0.2,
         0.3, 2.0,  0.9,
        -0.4, 0.1, -0.5,
    };
    double h[3], J[9], offset;
    CHECK(moonlab_qubo_to_ising(n, Q, 0.25, h, J, &offset) == 0,
          "QUBO converts to Ising");
    for (uint64_t bits = 0; bits < 8; bits++) {
        const double q = moonlab_qubo_evaluate(n, Q, 0.25, bits);
        const double z = ising_energy_arrays(n, h, J, offset, bits);
        CHECK(fabs(q - z) < 2e-14,
              "QUBO/Ising energy parity bits=%llu q=%.17g z=%.17g",
              (unsigned long long)bits, q, z);
    }
}

static moonlab_anneal_config_t test_config(uint64_t seed)
{
    moonlab_anneal_config_t c = moonlab_anneal_config_default();
    c.total_time = 20.0;
    c.num_steps = 4000;
    c.num_samples = 512;
    c.seed = seed;
    c.schedule = MOONLAB_ANNEAL_SCHEDULE_COSINE;
    c.second_order = 1;
    return c;
}

static void test_one_qubit_oracle(void)
{
    const double h[1] = {-1.0};
    const double J[1] = {0.0};
    moonlab_anneal_config_t c = test_config(UINT64_C(0x123456789abcdef0));
    moonlab_anneal_result_t *r = NULL;
    CHECK(moonlab_quantum_anneal_ising(1, h, J, 0.0, &c, &r) == 0 && r,
          "one-qubit anneal succeeds");
    if (!r) return;
    const double oracle = oneq_rk4_ground_probability(c.total_time, h[0]);
    const double got = moonlab_anneal_result_success_probability(r);
    CHECK(fabs(got - oracle) < 2e-5,
          "Strang anneal agrees with independent RK4 (got %.12g oracle %.12g)",
          got, oracle);
    CHECK(got > 0.99, "slow one-qubit anneal reaches ground state (%.12g)", got);
    CHECK(moonlab_anneal_result_ground_bitstring(r) == 0,
          "h=-1 ground state is bit 0");
    CHECK(moonlab_anneal_result_ground_degeneracy(r) == 1,
          "one-qubit ground state is unique");
    CHECK(fabs(moonlab_anneal_result_ground_energy(r) + 1.0) < 1e-14,
          "one-qubit ground energy exact");
    CHECK(fabs(moonlab_anneal_result_problem_gap(r) - 2.0) < 1e-14,
          "one-qubit final problem gap exact");
    CHECK(fabs(moonlab_anneal_result_final_norm(r) - 1.0) < 1e-11,
          "anneal preserves norm");
    moonlab_anneal_result_free(r);
}

static void test_qubo_and_seed_replay(void)
{
    /* (x0+x1-1)^2 = 1 - x0 - x1 + 2 x0 x1.
     * Full x^T Q x uses Q01=Q10=1. */
    const double Q[4] = {-1.0, 1.0, 1.0, -1.0};
    moonlab_anneal_config_t c = test_config(UINT64_C(0xdeadbeef12345678));
    moonlab_anneal_result_t *a = NULL, *b = NULL;
    CHECK(moonlab_quantum_anneal_qubo(2, Q, 1.0, &c, &a) == 0 && a,
          "two-variable QUBO anneal succeeds");
    CHECK(moonlab_quantum_anneal_qubo(2, Q, 1.0, &c, &b) == 0 && b,
          "two-variable QUBO replay succeeds");
    if (!a || !b) { moonlab_anneal_result_free(a); moonlab_anneal_result_free(b); return; }
    CHECK(moonlab_anneal_result_ground_degeneracy(a) == 2,
          "QUBO has two ground states");
    CHECK(fabs(moonlab_anneal_result_ground_energy(a)) < 1e-14,
          "QUBO ground energy is zero");
    CHECK(fabs(moonlab_anneal_result_problem_gap(a) - 1.0) < 1e-14,
          "QUBO problem gap is one");
    CHECK(moonlab_anneal_result_best_energy(a) < 1e-14,
          "sampled QUBO finds a ground state");
    CHECK(moonlab_anneal_result_effective_seed(a) == c.seed,
          "explicit seed is echoed");
    CHECK(moonlab_anneal_result_num_samples(a) == c.num_samples,
          "sample count retained");
    for (size_t i = 0; i < c.num_samples; i++) {
        uint64_t abits = 0, bbits = 0;
        double ae = 0.0, be = 0.0;
        CHECK(moonlab_anneal_result_sample(a, i, &abits, &ae) == 0,
              "sample A %zu readable", i);
        CHECK(moonlab_anneal_result_sample(b, i, &bbits, &be) == 0,
              "sample B %zu readable", i);
        CHECK(abits == bbits && ae == be,
              "same seed replays sample %zu byte-identically", i);
    }
    CHECK(moonlab_anneal_result_sample(a, c.num_samples, NULL, NULL) ==
          MOONLAB_ANNEAL_BAD_ARG, "out-of-range sample rejected");
    moonlab_anneal_result_free(a);
    moonlab_anneal_result_free(b);
}

static void test_all_schedule_evolution_and_assigned_seed(void)
{
    const double h[1] = {-1.0}, J[1] = {0.0};
    for (int schedule = MOONLAB_ANNEAL_SCHEDULE_LINEAR;
         schedule <= MOONLAB_ANNEAL_SCHEDULE_COSINE; schedule++) {
        for (int second = 0; second <= 1; second++) {
            moonlab_anneal_config_t c = test_config(
                UINT64_C(0x900d000000000000) + (uint64_t)(2 * schedule + second));
            c.total_time = 12.0;
            c.num_steps = 1200;
            c.num_samples = 32;
            c.schedule = (moonlab_anneal_schedule_t)schedule;
            c.second_order = second;
            moonlab_anneal_result_t *r = NULL;
            CHECK(moonlab_quantum_anneal_ising(1, h, J, 0.0, &c, &r) == 0 && r,
                  "schedule=%d second=%d evolves", schedule, second);
            if (r) {
                CHECK(moonlab_anneal_result_success_probability(r) > 0.95,
                      "schedule=%d second=%d reaches ground manifold",
                      schedule, second);
                CHECK(fabs(moonlab_anneal_result_final_norm(r) - 1.0) < 1e-11,
                      "schedule=%d second=%d preserves norm", schedule, second);
            }
            moonlab_anneal_result_free(r);
        }
    }

    moonlab_anneal_config_t assigned_config = test_config(0);
    assigned_config.num_samples = 64;
    moonlab_anneal_result_t *assigned = NULL, *replay = NULL;
    CHECK(moonlab_quantum_anneal_ising(
              1, h, J, 0.0, &assigned_config, &assigned) == 0 && assigned,
          "zero seed requests assignment");
    if (assigned) {
        const uint64_t effective =
            moonlab_anneal_result_effective_seed(assigned);
        CHECK(effective != 0, "assigned seed is non-zero");
        assigned_config.seed = effective;
        CHECK(moonlab_quantum_anneal_ising(
                  1, h, J, 0.0, &assigned_config, &replay) == 0 && replay,
              "assigned seed can be replayed");
        if (replay) {
            for (size_t i = 0; i < assigned_config.num_samples; i++) {
                uint64_t a = 0, b = 0;
                moonlab_anneal_result_sample(assigned, i, &a, NULL);
                moonlab_anneal_result_sample(replay, i, &b, NULL);
                CHECK(a == b, "assigned seed replays sample %zu", i);
            }
        }
    }
    moonlab_anneal_result_free(assigned);
    moonlab_anneal_result_free(replay);
}

static void test_validation(void)
{
    CHECK(strcmp(moonlab_anneal_status_string(MOONLAB_ANNEAL_OK),
                 "MOONLAB_ANNEAL_OK") == 0, "status string for OK");
    CHECK(strcmp(moonlab_anneal_status_string(-999),
                 "MOONLAB_ANNEAL_UNKNOWN") == 0, "unknown status string");
    moonlab_anneal_config_t c = moonlab_anneal_config_default();
    const double h[2] = {0.0, 0.0};
    const double asymmetric_J[4] = {0.0, 1.0, 0.0, 0.0};
    moonlab_anneal_result_t *r = (moonlab_anneal_result_t *)(uintptr_t)1;
    CHECK(moonlab_quantum_anneal_ising(2, h, asymmetric_J, 0.0, &c, &r) ==
          MOONLAB_ANNEAL_BAD_ARG && r == NULL,
          "asymmetric Ising coupling rejected and output cleared");
    c.num_steps = 0;
    r = NULL;
    CHECK(moonlab_quantum_anneal_qubo(2, asymmetric_J, 0.0, &c, &r) ==
          MOONLAB_ANNEAL_BAD_ARG, "zero steps rejected");
    c = moonlab_anneal_config_default();
    c.schedule = (moonlab_anneal_schedule_t)99;
    r = NULL;
    CHECK(moonlab_quantum_anneal_qubo(2, asymmetric_J, 0.0, &c, &r) ==
          MOONLAB_ANNEAL_SCHEDULE_ERROR, "invalid schedule has distinct status");
}

static void test_piecewise_pause_and_quench_schedule(void)
{
    /* 1. Pause schedule helper validation */
    moonlab_anneal_schedule_point_t pause_pts[4];
    CHECK(moonlab_anneal_schedule_make_pause(0.5, 0.3, 0.2, pause_pts) == MOONLAB_ANNEAL_OK,
          "pause schedule created");
    CHECK(fabs(pause_pts[0].t) < 1e-12 && fabs(pause_pts[0].s) < 1e-12, "pause pt 0");
    CHECK(fabs(pause_pts[1].t - 0.3) < 1e-12 && fabs(pause_pts[1].s - 0.5) < 1e-12, "pause pt 1");
    CHECK(fabs(pause_pts[2].t - 0.5) < 1e-12 && fabs(pause_pts[2].s - 0.5) < 1e-12, "pause pt 2");
    CHECK(fabs(pause_pts[3].t - 1.0) < 1e-12 && fabs(pause_pts[3].s - 1.0) < 1e-12, "pause pt 3");

    /* Out of bounds pause parameters */
    CHECK(moonlab_anneal_schedule_make_pause(0.5, 0.8, 0.3, pause_pts) == MOONLAB_ANNEAL_BAD_ARG,
          "pause schedule rejects sum > 1.0");

    /* 2. Quench schedule helper validation */
    moonlab_anneal_schedule_point_t quench_pts[3];
    CHECK(moonlab_anneal_schedule_make_quench(0.7, 0.6, quench_pts) == MOONLAB_ANNEAL_OK,
          "quench schedule created");
    CHECK(fabs(quench_pts[0].t) < 1e-12 && fabs(quench_pts[0].s) < 1e-12, "quench pt 0");
    CHECK(fabs(quench_pts[1].t - 0.6) < 1e-12 &&
          fabs(quench_pts[1].s - 0.7) < 1e-12, "quench pt 1");
    CHECK(fabs(quench_pts[2].t - 1.0) < 1e-12 &&
          fabs(quench_pts[2].s - 1.0) < 1e-12, "quench pt 2");

    CHECK(moonlab_anneal_schedule_make_quench(0.7, 1.0, quench_pts) == MOONLAB_ANNEAL_BAD_ARG,
          "quench rejects start >= 1.0");

    /* 3. Execute evolution under piecewise schedule */
    const double h[1] = {-1.0}, J[1] = {0.0};
    moonlab_anneal_config_t c = test_config(UINT64_C(0x1234567890abcdef));
    c.schedule = MOONLAB_ANNEAL_SCHEDULE_PIECEWISE;
    c.schedule_points = pause_pts;
    c.num_schedule_points = 4;
    moonlab_anneal_result_t *r = NULL;
    CHECK(moonlab_quantum_anneal_ising(1, h, J, 0.0, &c, &r) == MOONLAB_ANNEAL_OK && r,
          "piecewise pause anneal succeeds");
    if (r) {
        CHECK(moonlab_anneal_result_success_probability(r) > 0.90,
              "piecewise pause reaches ground state");
        CHECK(fabs(moonlab_anneal_result_final_norm(r) - 1.0) < 1e-10,
              "piecewise pause preserves norm");
    }
    moonlab_anneal_result_free(r);
}

static void test_reverse_annealing_ising(void)
{
    /* 2 qubits with ferromagnetic coupling J_01 = -1.0 */
    const double h[2] = {0.0, 0.0};
    const double J[4] = {0.0, -1.0, -1.0, 0.0};
    moonlab_anneal_config_t c = test_config(UINT64_C(0xfeedfacecafebeef));
    c.total_time = 15.0;
    c.num_steps = 1500;
    moonlab_anneal_result_t *r = NULL;

    /* Start from state |00> (initial_bitstring = 0) */
    CHECK(moonlab_quantum_reverse_anneal_ising(
              2, h, J, 0.0, 0, 0.35, 0.2, &c, &r) == MOONLAB_ANNEAL_OK && r,
          "reverse annealing ising succeeds");
    if (r) {
        CHECK(fabs(moonlab_anneal_result_best_energy(r) - (-1.0)) < 1e-10,
              "reverse annealing finds ferromagnetic ground state");
        CHECK(fabs(moonlab_anneal_result_final_norm(r) - 1.0) < 1e-10,
              "reverse annealing preserves state norm");
    }
    moonlab_anneal_result_free(r);

    /* Test QUBO reverse annealing */
    const double Q[4] = {-1.0, 0.0, 0.0, -1.0};
    r = NULL;
    CHECK(moonlab_quantum_reverse_anneal_qubo(
              2, Q, 0.0, 0b11, 0.4, 0.1, &c, &r) == MOONLAB_ANNEAL_OK && r,
          "reverse annealing qubo succeeds");
    if (r) {
        CHECK(fabs(moonlab_anneal_result_final_norm(r) - 1.0) < 1e-10,
              "reverse annealing qubo preserves norm");
    }
    moonlab_anneal_result_free(r);

    /* Test validation */
    r = NULL;
    CHECK(moonlab_quantum_reverse_anneal_ising(
              2, h, J, 0.0, 0, 1.5, 0.2, &c, &r) == MOONLAB_ANNEAL_BAD_ARG && !r,
          "reverse annealing rejects invalid s_target");
}

static void test_per_qubit_anneal_offsets(void)
{
    const double h[2] = {-0.5, 0.5};
    const double J[4] = {0.0, -0.2, -0.2, 0.0};
    const double offsets[2] = {0.05, -0.05};
    moonlab_anneal_config_t c = test_config(UINT64_C(0x1122334455667788));
    c.anneal_offsets = offsets;
    moonlab_anneal_result_t *r = NULL;

    CHECK(moonlab_quantum_anneal_ising(2, h, J, 0.0, &c, &r) == MOONLAB_ANNEAL_OK && r,
          "per-qubit offsets anneal succeeds");
    if (r) {
        CHECK(fabs(moonlab_anneal_result_final_norm(r) - 1.0) < 1e-10,
              "offsets anneal preserves norm");
    }
    moonlab_anneal_result_free(r);

    /* Invalid offset out of [-1, 1] */
    const double bad_offsets[2] = {1.5, 0.0};
    c.anneal_offsets = bad_offsets;
    r = NULL;
    CHECK(moonlab_quantum_anneal_ising(2, h, J, 0.0, &c, &r) == MOONLAB_ANNEAL_BAD_ARG && !r,
          "invalid offset rejected");
}

static void test_zephyr_graph_and_clique_embedding(void)
{
    /* 1. Zephyr graph creation and properties */
    moonlab_zephyr_graph_t *z1 = moonlab_zephyr_graph_create(1);
    CHECK(z1 != NULL, "zephyr graph m=1 created");
    if (z1) {
        CHECK(z1->num_qubits == 48, "zephyr m=1 qubit count N=48");
        CHECK(z1->num_couplers == 280, "zephyr m=1 has exactly 280 couplers");
        CHECK(moonlab_zephyr_has_coupler(z1, 999, 999) == 0,
              "out of bounds coupler query returns 0");
    }

    moonlab_zephyr_graph_t *z2 = moonlab_zephyr_graph_create(2);
    CHECK(z2 != NULL, "zephyr graph m=2 created");
    if (z2) {
        CHECK(z2->num_qubits == 160, "zephyr m=2 qubit count N=160");
        CHECK(z2->num_couplers == 1224, "zephyr m=2 has exactly 1224 couplers");
    }

    /* 2. Clique embedding into Zephyr Z_1 */
    moonlab_zephyr_embedding_t *emb = moonlab_zephyr_find_clique_embedding(4, 1);
    CHECK(emb != NULL, "K_4 clique embedding created");
    if (emb && z1) {
        CHECK(emb->num_logical == 4, "embedding logical count");
        for (size_t i = 0; i < 4; i++) {
            CHECK(emb->chain_lengths[i] == 2, "chain length is 2 for K_4");
        }

        /* 3. Embed Ising problem */
        const double log_h[4] = {0.2, -0.3, 0.1, -0.4};
        const double log_J[16] = {
             0.0, -0.5, -0.3, -0.2,
            -0.5,  0.0, -0.4, -0.1,
            -0.3, -0.4,  0.0, -0.6,
            -0.2, -0.1, -0.6,  0.0
        };
        double *phys_h = calloc(z1->num_qubits, sizeof(double));
        double *phys_J = calloc(z1->num_qubits * z1->num_qubits, sizeof(double));
        CHECK(phys_h && phys_J, "physical arrays allocated");

        int rc = moonlab_zephyr_embed_ising(z1, emb, log_h, log_J, 2.5, phys_h, phys_J);
        CHECK(rc == MOONLAB_ANNEAL_OK, "zephyr embed ising succeeds");

        /* 4. Unembed samples */
        uint64_t phys_samples[3] = {0, 0, 0};
        /* Sample 0: all physical qubits 0 -> logical 0 */
        phys_samples[0] = 0;
        /* Sample 1: all chain qubits 1 -> logical 0b1111 = 15 */
        for (size_t i = 0; i < 4; i++) {
            for (size_t c = 0; c < emb->chain_lengths[i]; c++) {
                phys_samples[1] |= (UINT64_C(1) << emb->chains[i][c]);
            }
        }
        /* Sample 2: break chain 0 (set first qubit to 1, second to 0) */
        phys_samples[2] = (UINT64_C(1) << emb->chains[0][0]);

        uint64_t log_samples[3] = {0};
        double break_fracs[3] = {0.0};
        rc = moonlab_zephyr_unembed_samples(emb, 3, phys_samples, log_samples, break_fracs);
        CHECK(rc == MOONLAB_ANNEAL_OK, "zephyr unembed samples succeeds");
        CHECK(log_samples[0] == 0, "sample 0 decoded to 0");
        CHECK(fabs(break_fracs[0]) < 1e-12, "sample 0 zero chain breaks");
        CHECK(log_samples[1] == 15, "sample 1 decoded to 15");
        CHECK(fabs(break_fracs[1]) < 1e-12, "sample 1 zero chain breaks");
        CHECK(fabs(break_fracs[2] - 0.25) < 1e-12, "sample 2 detected broken chain fraction 0.25");

        free(phys_h);
        free(phys_J);
    }

    /* 5. Multi-tile clique embeddings into Zephyr Z_2 */
    if (z2) {
        moonlab_zephyr_embedding_t *emb_k8 = moonlab_zephyr_find_clique_embedding(8, 2);
        CHECK(emb_k8 != NULL, "K_8 clique embedding on Z_2 created");
        if (emb_k8) {
            CHECK(emb_k8->num_logical == 8, "K_8 logical count is 8");
            double *phys_h8 = calloc(z2->num_qubits, sizeof(double));
            double *phys_J8 = calloc(z2->num_qubits * z2->num_qubits, sizeof(double));
            double log_h8[8] = {0};
            double log_J8[64] = {0};
            for (size_t i = 0; i < 8; i++) {
                for (size_t j = i + 1; j < 8; j++) {
                    log_J8[i * 8 + j] = -0.5;
                    log_J8[j * 8 + i] = -0.5;
                }
            }
            int rc8 = moonlab_zephyr_embed_ising(z2, emb_k8, log_h8, log_J8, 2.0, phys_h8, phys_J8);
            CHECK(rc8 == MOONLAB_ANNEAL_OK, "K_8 embedding into Z_2 physical couplers succeeds");
            free(phys_h8);
            free(phys_J8);
            moonlab_zephyr_embedding_free(emb_k8);
        }

        moonlab_zephyr_embedding_t *emb_k12 = moonlab_zephyr_find_clique_embedding(12, 2);
        CHECK(emb_k12 != NULL, "K_12 multi-tile clique embedding on Z_2 created");
        if (emb_k12) {
            CHECK(emb_k12->num_logical == 12, "K_12 logical count is 12");
            double *phys_h12 = calloc(z2->num_qubits, sizeof(double));
            double *phys_J12 = calloc(z2->num_qubits * z2->num_qubits, sizeof(double));
            double log_h12[12] = {0};
            double log_J12[144] = {0};
            for (size_t i = 0; i < 12; i++) {
                for (size_t j = i + 1; j < 12; j++) {
                    log_J12[i * 12 + j] = -0.25;
                    log_J12[j * 12 + i] = -0.25;
                }
            }
            int rc12 = moonlab_zephyr_embed_ising(
                z2, emb_k12, log_h12, log_J12, 2.0, phys_h12, phys_J12);
            CHECK(rc12 == MOONLAB_ANNEAL_OK, "K_12 embedding into Z_2 physical couplers succeeds");
            free(phys_h12);
            free(phys_J12);
            moonlab_zephyr_embedding_free(emb_k12);
        }
    }

    moonlab_zephyr_embedding_free(emb);
    moonlab_zephyr_graph_free(z1);
    moonlab_zephyr_graph_free(z2);
}

int main(void)
{
    fprintf(stdout, "=== Quantum Annealing Trotter & Zephyr Tests ===\n");
    test_schedules();
    test_qubo_conversion();
    test_one_qubit_oracle();
    test_qubo_and_seed_replay();
    test_all_schedule_evolution_and_assigned_seed();
    test_piecewise_pause_and_quench_schedule();
    test_reverse_annealing_ising();
    test_per_qubit_anneal_offsets();
    test_zephyr_graph_and_clique_embedding();
    test_validation();
    fprintf(stdout, "=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
