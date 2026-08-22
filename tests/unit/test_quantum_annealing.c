/**
 * @file test_quantum_annealing.c
 * @brief Independent and contract tests for transverse-field annealing.
 */

#include "../../src/algorithms/quantum_annealing.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
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

int main(void)
{
    test_schedules();
    test_qubo_conversion();
    test_one_qubit_oracle();
    test_qubo_and_seed_replay();
    test_all_schedule_evolution_and_assigned_seed();
    test_validation();
    if (failures) fprintf(stderr, "%d quantum-annealing failure(s)\n", failures);
    return failures ? 1 : 0;
}
