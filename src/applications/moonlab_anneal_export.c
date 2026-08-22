/** @file moonlab_anneal_export.c Stable ABI 0.8.0 annealing one-shots. */

#include "moonlab_export.h"
#include "../algorithms/quantum_annealing.h"

#include <string.h>

static void fill_summary(const moonlab_anneal_result_t *result,
                         moonlab_anneal_summary_v1 *out)
{
    memset(out, 0, sizeof(*out));
    out->effective_seed = moonlab_anneal_result_effective_seed(result);
    out->best_bitstring = moonlab_anneal_result_best_bitstring(result);
    out->most_likely_bitstring = moonlab_anneal_result_most_likely_bitstring(result);
    out->ground_bitstring = moonlab_anneal_result_ground_bitstring(result);
    out->ground_degeneracy = moonlab_anneal_result_ground_degeneracy(result);
    out->best_energy = moonlab_anneal_result_best_energy(result);
    out->ground_energy = moonlab_anneal_result_ground_energy(result);
    out->expected_energy = moonlab_anneal_result_expected_energy(result);
    out->success_probability = moonlab_anneal_result_success_probability(result);
    out->residual_energy = moonlab_anneal_result_residual_energy(result);
    out->problem_gap = moonlab_anneal_result_problem_gap(result);
    out->final_norm = moonlab_anneal_result_final_norm(result);
}

static moonlab_anneal_config_t stable_config(double total_time,
                                              size_t num_steps,
                                              size_t num_samples,
                                              uint64_t seed,
                                              int schedule,
                                              double driver_strength,
                                              double problem_strength,
                                              int second_order)
{
    moonlab_anneal_config_t config = moonlab_anneal_config_default();
    config.total_time = total_time;
    config.num_steps = num_steps;
    config.num_samples = num_samples;
    config.seed = seed;
    config.schedule = (moonlab_anneal_schedule_t)schedule;
    config.driver_strength = driver_strength;
    config.problem_strength = problem_strength;
    config.second_order = second_order;
    return config;
}

static int copy_samples(const moonlab_anneal_result_t *result,
                        size_t count, uint64_t *samples_out,
                        double *energies_out)
{
    if (!samples_out && !energies_out) return MOONLAB_ANNEAL_OK;
    for (size_t i = 0; i < count; i++) {
        int rc = moonlab_anneal_result_sample(
            result, i,
            samples_out ? &samples_out[i] : NULL,
            energies_out ? &energies_out[i] : NULL);
        if (rc != MOONLAB_ANNEAL_OK) return rc;
    }
    return MOONLAB_ANNEAL_OK;
}

int moonlab_anneal_ising_v1(size_t n, const double *h, const double *J,
                            double offset, double total_time,
                            size_t num_steps, size_t num_samples,
                            uint64_t seed, int schedule,
                            double driver_strength, double problem_strength,
                            int second_order,
                            moonlab_anneal_summary_v1 *summary_out,
                            uint64_t *samples_out, double *sample_energies_out)
{
    if (!summary_out) return MOONLAB_ANNEAL_BAD_ARG;
    memset(summary_out, 0, sizeof(*summary_out));
    moonlab_anneal_config_t config = stable_config(
        total_time, num_steps, num_samples, seed, schedule,
        driver_strength, problem_strength, second_order);
    moonlab_anneal_result_t *result = NULL;
    int rc = moonlab_quantum_anneal_ising(
        n, h, J, offset, &config, &result);
    if (rc == MOONLAB_ANNEAL_OK) {
        fill_summary(result, summary_out);
        rc = copy_samples(result, num_samples, samples_out, sample_energies_out);
    }
    moonlab_anneal_result_free(result);
    return rc;
}

int moonlab_anneal_qubo_v1(size_t n, const double *Q, double offset,
                           double total_time, size_t num_steps,
                           size_t num_samples, uint64_t seed, int schedule,
                           double driver_strength, double problem_strength,
                           int second_order,
                           moonlab_anneal_summary_v1 *summary_out,
                           uint64_t *samples_out, double *sample_energies_out)
{
    if (!summary_out) return MOONLAB_ANNEAL_BAD_ARG;
    memset(summary_out, 0, sizeof(*summary_out));
    moonlab_anneal_config_t config = stable_config(
        total_time, num_steps, num_samples, seed, schedule,
        driver_strength, problem_strength, second_order);
    moonlab_anneal_result_t *result = NULL;
    int rc = moonlab_quantum_anneal_qubo(
        n, Q, offset, &config, &result);
    if (rc == MOONLAB_ANNEAL_OK) {
        fill_summary(result, summary_out);
        rc = copy_samples(result, num_samples, samples_out, sample_energies_out);
    }
    moonlab_anneal_result_free(result);
    return rc;
}
