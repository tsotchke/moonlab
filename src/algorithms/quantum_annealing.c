#include "quantum_annealing.h"

#include "../quantum/gates.h"

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ANNEAL_MAX_STEPS   ((size_t)10000000)
#define ANNEAL_MAX_SAMPLES ((size_t)10000000)
#define ANNEAL_ENERGY_TOL  1e-12

struct moonlab_anneal_result {
    size_t num_qubits;
    size_t num_samples;
    uint64_t effective_seed;
    uint64_t best_bitstring;
    uint64_t most_likely_bitstring;
    uint64_t ground_bitstring;
    size_t ground_degeneracy;
    double best_energy;
    double ground_energy;
    double expected_energy;
    double success_probability;
    double residual_energy;
    double problem_gap;
    double final_norm;
    uint64_t *samples;
    double *sample_energies;
};

static int finite_array(const double *values, size_t count)
{
    if (!values) return 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static uint64_t assigned_seed(void)
{
    static _Atomic uint64_t sequence = 0;
    struct timespec ts = {0};
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) ts.tv_sec = time(NULL);
    uint64_t x = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec;
    x ^= atomic_fetch_add_explicit(&sequence, 1, memory_order_relaxed)
         + UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x ? x : UINT64_C(0x6a09e667f3bcc909);
}

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static double rng_unit(uint64_t *state)
{
    return (double)(xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
}

static double ising_energy(const ising_model_t *model, uint64_t bits)
{
    double energy = model->offset;
    for (size_t i = 0; i < model->num_qubits; i++) {
        const double zi = ((bits >> i) & 1u) ? -1.0 : 1.0;
        energy += model->h[i] * zi;
        for (size_t j = i + 1; j < model->num_qubits; j++) {
            const double zj = ((bits >> j) & 1u) ? -1.0 : 1.0;
            energy += model->J[i][j] * zi * zj;
        }
    }
    return energy;
}

static int energy_equal(double a, double b)
{
    const double scale = 1.0 + fmax(fabs(a), fabs(b));
    return fabs(a - b) <= ANNEAL_ENERGY_TOL * scale;
}

moonlab_anneal_config_t moonlab_anneal_config_default(void)
{
    moonlab_anneal_config_t config;
    config.total_time = 10.0;
    config.num_steps = 1000;
    config.num_samples = 1024;
    config.seed = 0;
    config.schedule = MOONLAB_ANNEAL_SCHEDULE_COSINE;
    config.driver_strength = 1.0;
    config.problem_strength = 1.0;
    config.second_order = 1;
    return config;
}

const char *moonlab_anneal_status_string(int status)
{
    switch (status) {
    case MOONLAB_ANNEAL_OK: return "MOONLAB_ANNEAL_OK";
    case MOONLAB_ANNEAL_BAD_ARG: return "MOONLAB_ANNEAL_BAD_ARG";
    case MOONLAB_ANNEAL_OOM: return "MOONLAB_ANNEAL_OOM";
    case MOONLAB_ANNEAL_STATE_ERROR: return "MOONLAB_ANNEAL_STATE_ERROR";
    case MOONLAB_ANNEAL_SCHEDULE_ERROR: return "MOONLAB_ANNEAL_SCHEDULE_ERROR";
    default: return "MOONLAB_ANNEAL_UNKNOWN";
    }
}

int moonlab_anneal_schedule_values(moonlab_anneal_schedule_t schedule,
                                   double s, double *driver_out,
                                   double *problem_out)
{
    if (!driver_out || !problem_out || !isfinite(s) || s < 0.0 || s > 1.0) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    switch (schedule) {
    case MOONLAB_ANNEAL_SCHEDULE_LINEAR:
        *driver_out = 1.0 - s;
        *problem_out = s;
        break;
    case MOONLAB_ANNEAL_SCHEDULE_QUADRATIC:
        *problem_out = s * s;
        *driver_out = 1.0 - *problem_out;
        break;
    case MOONLAB_ANNEAL_SCHEDULE_COSINE: {
        const double sn = sin(0.5 * M_PI * s);
        const double cs = cos(0.5 * M_PI * s);
        *driver_out = cs * cs;
        *problem_out = sn * sn;
        break;
    }
    default:
        return MOONLAB_ANNEAL_SCHEDULE_ERROR;
    }
    return MOONLAB_ANNEAL_OK;
}

double moonlab_qubo_evaluate(size_t n, const double *Q, double offset,
                             uint64_t bits)
{
    if (n == 0 || n > 32 || !Q || !isfinite(offset) ||
        !finite_array(Q, n * n)) {
        return DBL_MAX;
    }
    double energy = offset;
    for (size_t i = 0; i < n; i++) {
        const double xi = (double)((bits >> i) & 1u);
        if (xi == 0.0) continue;
        for (size_t j = 0; j < n; j++) {
            const double xj = (double)((bits >> j) & 1u);
            energy += xi * Q[i * n + j] * xj;
        }
    }
    return energy;
}

int moonlab_qubo_to_ising(size_t n, const double *Q, double qubo_offset,
                          double *out_h, double *out_J,
                          double *out_ising_offset)
{
    if (n == 0 || n > 32 || !Q || !out_h || !out_J ||
        !out_ising_offset || !isfinite(qubo_offset) ||
        !finite_array(Q, n * n)) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    memset(out_h, 0, n * sizeof(double));
    memset(out_J, 0, n * n * sizeof(double));
    double constant = qubo_offset;
    for (size_t i = 0; i < n; i++) {
        const double diag = Q[i * n + i];
        constant += 0.5 * diag;
        out_h[i] -= 0.5 * diag;
        for (size_t j = i + 1; j < n; j++) {
            const double pair = Q[i * n + j] + Q[j * n + i];
            const double coupling = 0.25 * pair;
            constant += coupling;
            out_h[i] -= coupling;
            out_h[j] -= coupling;
            out_J[i * n + j] = coupling;
            out_J[j * n + i] = coupling;
        }
    }
    *out_ising_offset = constant;
    return MOONLAB_ANNEAL_OK;
}

static int validate_config(const moonlab_anneal_config_t *config)
{
    if (!config || !isfinite(config->total_time) || config->total_time <= 0.0 ||
        config->num_steps < 1 || config->num_steps > ANNEAL_MAX_STEPS ||
        config->num_samples < 1 || config->num_samples > ANNEAL_MAX_SAMPLES ||
        !isfinite(config->driver_strength) || config->driver_strength <= 0.0 ||
        !isfinite(config->problem_strength) || config->problem_strength <= 0.0) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    if (config->schedule < MOONLAB_ANNEAL_SCHEDULE_LINEAR ||
        config->schedule > MOONLAB_ANNEAL_SCHEDULE_COSINE) {
        return MOONLAB_ANNEAL_SCHEDULE_ERROR;
    }
    return MOONLAB_ANNEAL_OK;
}

static int validate_model_arrays(size_t n, const double *h,
                                 const double *J, double offset)
{
    if (n == 0 || n > 32 || !h || !J || !isfinite(offset) ||
        !finite_array(h, n) || !finite_array(J, n * n)) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        if (fabs(J[i * n + i]) > ANNEAL_ENERGY_TOL) return 0;
        for (size_t j = i + 1; j < n; j++) {
            const double a = J[i * n + j], b = J[j * n + i];
            const double scale = 1.0 + fmax(fabs(a), fabs(b));
            if (fabs(a - b) > ANNEAL_ENERGY_TOL * scale) return 0;
        }
    }
    return 1;
}

static ising_model_t *model_from_arrays(size_t n, const double *h,
                                        const double *J, double offset)
{
    ising_model_t *model = ising_model_create(n);
    if (!model) return NULL;
    model->offset = offset;
    for (size_t i = 0; i < n; i++) {
        model->h[i] = h[i];
        for (size_t j = i + 1; j < n; j++) {
            model->J[i][j] = J[i * n + j];
            model->J[j][i] = J[i * n + j];
        }
    }
    return model;
}

static int evolve(quantum_state_t *state, const ising_model_t *model,
                  const moonlab_anneal_config_t *config)
{
    const double dt = config->total_time / (double)config->num_steps;
    for (size_t step = 0; step < config->num_steps; step++) {
        const double s = ((double)step + 0.5) / (double)config->num_steps;
        double driver = 0.0, problem = 0.0;
        if (moonlab_anneal_schedule_values(config->schedule, s,
                                           &driver, &problem) != 0) {
            return MOONLAB_ANNEAL_SCHEDULE_ERROR;
        }
        const double beta = -dt * config->driver_strength * driver;
        const double gamma = dt * config->problem_strength * problem;
        if (config->second_order) {
            if (qaoa_apply_mixer_hamiltonian(state, 0.5 * beta) != QS_SUCCESS ||
                qaoa_apply_cost_hamiltonian(state, model, gamma) != QS_SUCCESS ||
                qaoa_apply_mixer_hamiltonian(state, 0.5 * beta) != QS_SUCCESS) {
                return MOONLAB_ANNEAL_STATE_ERROR;
            }
        } else if (qaoa_apply_mixer_hamiltonian(state, beta) != QS_SUCCESS ||
                   qaoa_apply_cost_hamiltonian(state, model, gamma) != QS_SUCCESS) {
            return MOONLAB_ANNEAL_STATE_ERROR;
        }
    }
    return MOONLAB_ANNEAL_OK;
}

static uint64_t sample_cdf(const double *cdf, uint64_t dim, double u)
{
    uint64_t lo = 0, hi = dim;
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (u < cdf[mid]) hi = mid;
        else lo = mid + 1;
    }
    return lo < dim ? lo : dim - 1;
}

static int run_model(ising_model_t *model,
                     const moonlab_anneal_config_t *config,
                     moonlab_anneal_result_t **out_result)
{
    *out_result = NULL;
    quantum_state_t state = {0};
    if (quantum_state_init(&state, model->num_qubits) != QS_SUCCESS) {
        return MOONLAB_ANNEAL_STATE_ERROR;
    }
    for (size_t q = 0; q < model->num_qubits; q++) {
        if (gate_hadamard(&state, (int)q) != QS_SUCCESS) {
            quantum_state_free(&state);
            return MOONLAB_ANNEAL_STATE_ERROR;
        }
    }
    int rc = evolve(&state, model, config);
    if (rc != MOONLAB_ANNEAL_OK) {
        quantum_state_free(&state);
        return rc;
    }

    const uint64_t dim = UINT64_C(1) << model->num_qubits;
    moonlab_anneal_result_t *result = calloc(1, sizeof(*result));
    double *cdf = malloc((size_t)dim * sizeof(double));
    if (!result || !cdf) {
        free(result); free(cdf); quantum_state_free(&state);
        return MOONLAB_ANNEAL_OOM;
    }
    result->samples = malloc(config->num_samples * sizeof(uint64_t));
    result->sample_energies = malloc(config->num_samples * sizeof(double));
    if (!result->samples || !result->sample_energies) {
        moonlab_anneal_result_free(result);
        free(cdf); quantum_state_free(&state);
        return MOONLAB_ANNEAL_OOM;
    }

    double ground = DBL_MAX, first_excited = DBL_MAX;
    uint64_t ground_bits = 0;
    size_t degeneracy = 0;
    for (uint64_t bits = 0; bits < dim; bits++) {
        const double energy = ising_energy(model, bits);
        if (energy < ground && !energy_equal(energy, ground)) {
            first_excited = ground;
            ground = energy;
            ground_bits = bits;
            degeneracy = 1;
        } else if (energy_equal(energy, ground)) {
            degeneracy++;
        } else if (energy < first_excited) {
            first_excited = energy;
        }
    }

    double cumulative = 0.0, expected = 0.0, success = 0.0;
    double norm = 0.0, max_probability = -1.0;
    uint64_t most_likely = 0;
    for (uint64_t bits = 0; bits < dim; bits++) {
        const double probability = quantum_state_get_probability(&state, bits);
        const double energy = ising_energy(model, bits);
        cumulative += probability;
        cdf[bits] = cumulative;
        norm += probability;
        expected += probability * energy;
        if (energy_equal(energy, ground)) success += probability;
        if (probability > max_probability) {
            max_probability = probability;
            most_likely = bits;
        }
    }
    if (!(norm > 0.0) || !isfinite(norm)) {
        moonlab_anneal_result_free(result);
        free(cdf); quantum_state_free(&state);
        return MOONLAB_ANNEAL_STATE_ERROR;
    }
    for (uint64_t bits = 0; bits < dim; bits++) cdf[bits] /= norm;
    cdf[dim - 1] = 1.0;
    expected /= norm;
    success /= norm;

    result->num_qubits = model->num_qubits;
    result->num_samples = config->num_samples;
    result->effective_seed = config->seed ? config->seed : assigned_seed();
    result->ground_energy = ground;
    result->ground_bitstring = ground_bits;
    result->ground_degeneracy = degeneracy;
    result->problem_gap = first_excited < DBL_MAX ? first_excited - ground : 0.0;
    result->expected_energy = expected;
    result->success_probability = success;
    result->residual_energy = expected - ground;
    result->final_norm = norm;
    result->most_likely_bitstring = most_likely;
    result->best_energy = DBL_MAX;

    uint64_t rng = result->effective_seed;
    for (size_t sample = 0; sample < config->num_samples; sample++) {
        const uint64_t bits = sample_cdf(cdf, dim, rng_unit(&rng));
        const double energy = ising_energy(model, bits);
        result->samples[sample] = bits;
        result->sample_energies[sample] = energy;
        if (energy < result->best_energy) {
            result->best_energy = energy;
            result->best_bitstring = bits;
        }
    }

    free(cdf);
    quantum_state_free(&state);
    *out_result = result;
    return MOONLAB_ANNEAL_OK;
}

int moonlab_quantum_anneal_ising(size_t n, const double *h,
                                 const double *J, double offset,
                                 const moonlab_anneal_config_t *config,
                                 moonlab_anneal_result_t **out_result)
{
    if (!out_result) return MOONLAB_ANNEAL_BAD_ARG;
    *out_result = NULL;
    const int config_rc = validate_config(config);
    if (config_rc != MOONLAB_ANNEAL_OK) return config_rc;
    if (!validate_model_arrays(n, h, J, offset)) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    ising_model_t *model = model_from_arrays(n, h, J, offset);
    if (!model) return MOONLAB_ANNEAL_OOM;
    const int rc = run_model(model, config, out_result);
    ising_model_free(model);
    return rc;
}

int moonlab_quantum_anneal_qubo(size_t n, const double *Q, double offset,
                                const moonlab_anneal_config_t *config,
                                moonlab_anneal_result_t **out_result)
{
    if (!out_result) return MOONLAB_ANNEAL_BAD_ARG;
    *out_result = NULL;
    const int config_rc = validate_config(config);
    if (config_rc != MOONLAB_ANNEAL_OK) return config_rc;
    if (n == 0 || n > 32 || !Q) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    double *h = calloc(n, sizeof(double));
    double *J = calloc(n * n, sizeof(double));
    double ising_offset = 0.0;
    if (!h || !J) { free(h); free(J); return MOONLAB_ANNEAL_OOM; }
    int rc = moonlab_qubo_to_ising(n, Q, offset, h, J, &ising_offset);
    if (rc == MOONLAB_ANNEAL_OK) {
        rc = moonlab_quantum_anneal_ising(n, h, J, ising_offset,
                                          config, out_result);
    }
    free(h); free(J);
    return rc;
}

void moonlab_anneal_result_free(moonlab_anneal_result_t *result)
{
    if (!result) return;
    free(result->samples);
    free(result->sample_energies);
    free(result);
}

size_t moonlab_anneal_result_num_qubits(const moonlab_anneal_result_t *r) {
    return r ? r->num_qubits : 0;
}

size_t moonlab_anneal_result_num_samples(const moonlab_anneal_result_t *r) {
    return r ? r->num_samples : 0;
}

uint64_t moonlab_anneal_result_effective_seed(const moonlab_anneal_result_t *r) {
    return r ? r->effective_seed : 0;
}

uint64_t moonlab_anneal_result_best_bitstring(const moonlab_anneal_result_t *r) {
    return r ? r->best_bitstring : 0;
}

uint64_t moonlab_anneal_result_most_likely_bitstring(
    const moonlab_anneal_result_t *r
) {
    return r ? r->most_likely_bitstring : 0;
}

uint64_t moonlab_anneal_result_ground_bitstring(const moonlab_anneal_result_t *r) {
    return r ? r->ground_bitstring : 0;
}

size_t moonlab_anneal_result_ground_degeneracy(const moonlab_anneal_result_t *r) {
    return r ? r->ground_degeneracy : 0;
}

double moonlab_anneal_result_best_energy(const moonlab_anneal_result_t *r) {
    return r ? r->best_energy : DBL_MAX;
}

double moonlab_anneal_result_ground_energy(const moonlab_anneal_result_t *r) {
    return r ? r->ground_energy : DBL_MAX;
}

double moonlab_anneal_result_expected_energy(const moonlab_anneal_result_t *r) {
    return r ? r->expected_energy : DBL_MAX;
}

double moonlab_anneal_result_success_probability(const moonlab_anneal_result_t *r) {
    return r ? r->success_probability : 0.0;
}

double moonlab_anneal_result_residual_energy(const moonlab_anneal_result_t *r) {
    return r ? r->residual_energy : DBL_MAX;
}

double moonlab_anneal_result_problem_gap(const moonlab_anneal_result_t *r) {
    return r ? r->problem_gap : 0.0;
}

double moonlab_anneal_result_final_norm(const moonlab_anneal_result_t *r) {
    return r ? r->final_norm : 0.0;
}

int moonlab_anneal_result_sample(const moonlab_anneal_result_t *result,
                                 size_t index, uint64_t *bitstring_out,
                                 double *energy_out)
{
    if (!result || index >= result->num_samples ||
        (!bitstring_out && !energy_out)) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    if (bitstring_out) *bitstring_out = result->samples[index];
    if (energy_out) *energy_out = result->sample_energies[index];
    return MOONLAB_ANNEAL_OK;
}
