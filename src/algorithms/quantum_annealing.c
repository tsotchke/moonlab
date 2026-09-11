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


static int energy_equal(double a, double b)
{
    const double scale = 1.0 + fmax(fabs(a), fabs(b));
    return fabs(a - b) <= ANNEAL_ENERGY_TOL * scale;
}

moonlab_anneal_config_t moonlab_anneal_config_default(void)
{
    moonlab_anneal_config_t config;
    memset(&config, 0, sizeof(config));
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
    case MOONLAB_ANNEAL_EMBEDDING_ERROR: return "MOONLAB_ANNEAL_EMBEDDING_ERROR";
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
    case MOONLAB_ANNEAL_SCHEDULE_PIECEWISE:
        *driver_out = 1.0 - s;
        *problem_out = s;
        break;
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

static double evaluate_piecewise_s(size_t num_points,
                                   const moonlab_anneal_schedule_point_t *points,
                                   double t)
{
    if (t <= points[0].t) return points[0].s;
    if (t >= points[num_points - 1].t) return points[num_points - 1].s;

    for (size_t i = 0; i + 1 < num_points; i++) {
        if (t >= points[i].t && t <= points[i + 1].t) {
            double dt = points[i + 1].t - points[i].t;
            if (dt <= 1e-15) return points[i + 1].s;
            double u = (t - points[i].t) / dt;
            return points[i].s + u * (points[i + 1].s - points[i].s);
        }
    }
    return points[num_points - 1].s;
}

static double evaluate_reverse_s(double t, double s_target, double hold_fraction)
{
    if (hold_fraction < 0.0) hold_fraction = 0.0;
    if (hold_fraction >= 1.0) hold_fraction = 0.999;
    const double ramp_frac = 0.5 * (1.0 - hold_fraction);
    if (t <= ramp_frac) {
        const double u = ramp_frac > 0.0 ? t / ramp_frac : 1.0;
        return 1.0 - u * (1.0 - s_target);
    } else if (t <= ramp_frac + hold_fraction) {
        return s_target;
    } else {
        const double u = (t - (ramp_frac + hold_fraction)) / ramp_frac;
        return s_target + u * (1.0 - s_target);
    }
}

int moonlab_anneal_schedule_make_pause(
    double s_pause, double pause_start_frac, double pause_duration_frac,
    moonlab_anneal_schedule_point_t points[4])
{
    if (!points || !isfinite(s_pause) || s_pause < 0.0 || s_pause > 1.0 ||
        !isfinite(pause_start_frac) || pause_start_frac < 0.0 ||
        !isfinite(pause_duration_frac) || pause_duration_frac < 0.0 ||
        pause_start_frac + pause_duration_frac > 1.0) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    points[0].t = 0.0;
    points[0].s = 0.0;
    points[1].t = pause_start_frac;
    points[1].s = s_pause;
    points[2].t = pause_start_frac + pause_duration_frac;
    points[2].s = s_pause;
    points[3].t = 1.0;
    points[3].s = 1.0;
    return MOONLAB_ANNEAL_OK;
}

int moonlab_anneal_schedule_make_quench(
    double s_quench, double quench_start_frac,
    moonlab_anneal_schedule_point_t points[3])
{
    if (!points || !isfinite(s_quench) || s_quench < 0.0 || s_quench > 1.0 ||
        !isfinite(quench_start_frac) || quench_start_frac < 0.0 || quench_start_frac >= 1.0) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    points[0].t = 0.0;
    points[0].s = 0.0;
    points[1].t = quench_start_frac;
    points[1].s = s_quench;
    points[2].t = 1.0;
    points[2].s = 1.0;
    return MOONLAB_ANNEAL_OK;
}

static int validate_config(const moonlab_anneal_config_t *config, size_t num_qubits)
{
    if (!config || !isfinite(config->total_time) || config->total_time <= 0.0 ||
        config->num_steps < 1 || config->num_steps > ANNEAL_MAX_STEPS ||
        config->num_samples < 1 || config->num_samples > ANNEAL_MAX_SAMPLES ||
        !isfinite(config->driver_strength) || config->driver_strength <= 0.0 ||
        !isfinite(config->problem_strength) || config->problem_strength <= 0.0) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    if (config->schedule < MOONLAB_ANNEAL_SCHEDULE_LINEAR ||
        config->schedule > MOONLAB_ANNEAL_SCHEDULE_PIECEWISE) {
        return MOONLAB_ANNEAL_SCHEDULE_ERROR;
    }
    if (config->schedule == MOONLAB_ANNEAL_SCHEDULE_PIECEWISE) {
        if (!config->schedule_points || config->num_schedule_points < 2) {
            return MOONLAB_ANNEAL_SCHEDULE_ERROR;
        }
        if (fabs(config->schedule_points[0].t) > 1e-12 ||
            fabs(config->schedule_points[config->num_schedule_points - 1].t - 1.0) > 1e-12) {
            return MOONLAB_ANNEAL_SCHEDULE_ERROR;
        }
        for (size_t i = 0; i < config->num_schedule_points; i++) {
            const double t = config->schedule_points[i].t;
            const double s = config->schedule_points[i].s;
            if (!isfinite(t) || !isfinite(s) || t < 0.0 || t > 1.0 || s < 0.0 || s > 1.0) {
                return MOONLAB_ANNEAL_SCHEDULE_ERROR;
            }
            if (i > 0 && t < config->schedule_points[i - 1].t) {
                return MOONLAB_ANNEAL_SCHEDULE_ERROR;
            }
        }
    }
    if (config->reverse_anneal) {
        if (!isfinite(config->reverse_s_target) ||
            config->reverse_s_target < 0.0 || config->reverse_s_target > 1.0 ||
            !isfinite(config->reverse_hold_fraction) ||
            config->reverse_hold_fraction < 0.0 || config->reverse_hold_fraction >= 1.0) {
            return MOONLAB_ANNEAL_BAD_ARG;
        }
    }
    if (config->anneal_offsets) {
        for (size_t i = 0; i < num_qubits; i++) {
            const double off = config->anneal_offsets[i];
            if (!isfinite(off) || off < -1.0 || off > 1.0) {
                return MOONLAB_ANNEAL_BAD_ARG;
            }
        }
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

static void compute_diagonal_energies(const ising_model_t *model,
                                     double *diag_energies)
{
    const size_t n = model->num_qubits;
    const uint64_t dim = UINT64_C(1) << n;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t b = 0; b < (int64_t)dim; b++) {
        const uint64_t bits = (uint64_t)b;
        double e = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double zi = ((bits >> i) & 1u) ? -1.0 : 1.0;
            e += model->h[i] * zi;
            for (size_t j = i + 1; j < n; j++) {
                const double zj = ((bits >> j) & 1u) ? -1.0 : 1.0;
                e += model->J[i][j] * zi * zj;
            }
        }
        diag_energies[bits] = e;
    }
}

static inline void apply_diagonal_cost_phases(quantum_state_t *state,
                                              const double *diag_energies,
                                              uint64_t dim,
                                              double gamma)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t b = 0; b < (int64_t)dim; b++) {
        const uint64_t bits = (uint64_t)b;
        const double theta = -gamma * diag_energies[bits];
        const double c = cos(theta);
        const double s = sin(theta);
        const complex_t amp = state->amplitudes[bits];
        const double re = creal(amp);
        const double im = cimag(amp);
        state->amplitudes[bits] = (re * c - im * s) + I * (re * s + im * c);
    }
}

static void apply_offset_cost_phases(quantum_state_t *state,
                                     const ising_model_t *model,
                                     const double *offsets,
                                     double s_base,
                                     moonlab_anneal_schedule_t sched,
                                     double dt,
                                     double problem_strength)
{
    const size_t n = model->num_qubits;
    const uint64_t dim = UINT64_C(1) << n;
    double h_eff[32];
    double J_eff[32][32];

    for (size_t i = 0; i < n; i++) {
        double s_i = fmin(1.0, fmax(0.0, s_base + offsets[i]));
        double Adummy, B_i;
        moonlab_anneal_schedule_values(sched, s_i, &Adummy, &B_i);
        h_eff[i] = model->h[i] * B_i;
        for (size_t j = i + 1; j < n; j++) {
            double s_j = fmin(1.0, fmax(0.0, s_base + offsets[j]));
            double s_ij = 0.5 * (s_i + s_j);
            double B_ij;
            moonlab_anneal_schedule_values(sched, s_ij, &Adummy, &B_ij);
            J_eff[i][j] = model->J[i][j] * B_ij;
        }
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t b = 0; b < (int64_t)dim; b++) {
        const uint64_t bits = (uint64_t)b;
        double e = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double zi = ((bits >> i) & 1u) ? -1.0 : 1.0;
            e += h_eff[i] * zi;
            for (size_t j = i + 1; j < n; j++) {
                const double zj = ((bits >> j) & 1u) ? -1.0 : 1.0;
                e += J_eff[i][j] * zi * zj;
            }
        }
        const double theta = -dt * problem_strength * e;
        const double c = cos(theta);
        const double s = sin(theta);
        const complex_t amp = state->amplitudes[bits];
        const double re = creal(amp);
        const double im = cimag(amp);
        state->amplitudes[bits] = (re * c - im * s) + I * (re * s + im * c);
    }
}

static void apply_mixer_hamiltonian(quantum_state_t *state,
                                    size_t num_qubits,
                                    double beta_uniform,
                                    const double *offsets,
                                    double s_base,
                                    moonlab_anneal_schedule_t sched,
                                    double dt,
                                    double driver_strength,
                                    double factor)
{
    if (!offsets) {
        for (size_t q = 0; q < num_qubits; q++) {
            gate_rx(state, (int)q, 2.0 * beta_uniform);
        }
    } else {
        for (size_t q = 0; q < num_qubits; q++) {
            double sq = fmin(1.0, fmax(0.0, s_base + offsets[q]));
            double Aq, Bq;
            moonlab_anneal_schedule_values(sched, sq, &Aq, &Bq);
            double beta_q = -dt * driver_strength * Aq * factor;
            gate_rx(state, (int)q, 2.0 * beta_q);
        }
    }
}

static int evolve(quantum_state_t *state, const ising_model_t *model,
                  const moonlab_anneal_config_t *config,
                  const double *diag_energies)
{
    const size_t n = model->num_qubits;
    const uint64_t dim = UINT64_C(1) << n;
    const double dt = config->total_time / (double)config->num_steps;

    for (size_t step = 0; step < config->num_steps; step++) {
        const double t = ((double)step + 0.5) / (double)config->num_steps;
        double s = t;

        if (config->reverse_anneal) {
            s = evaluate_reverse_s(t, config->reverse_s_target,
                                   config->reverse_hold_fraction);
        } else if (config->schedule == MOONLAB_ANNEAL_SCHEDULE_PIECEWISE) {
            s = evaluate_piecewise_s(config->num_schedule_points,
                                     config->schedule_points, t);
        }

        double driver = 0.0, problem = 0.0;
        if (moonlab_anneal_schedule_values(config->schedule, s,
                                           &driver, &problem) != 0) {
            return MOONLAB_ANNEAL_SCHEDULE_ERROR;
        }

        const double beta = -dt * config->driver_strength * driver;
        const double gamma = dt * config->problem_strength * problem;

        if (config->second_order) {
            apply_mixer_hamiltonian(state, n, 0.5 * beta, config->anneal_offsets,
                                    s, config->schedule, dt, config->driver_strength, 0.5);
            if (!config->anneal_offsets) {
                apply_diagonal_cost_phases(state, diag_energies, dim, gamma);
            } else {
                apply_offset_cost_phases(state, model, config->anneal_offsets,
                                         s, config->schedule, dt, config->problem_strength);
            }
            apply_mixer_hamiltonian(state, n, 0.5 * beta, config->anneal_offsets,
                                    s, config->schedule, dt, config->driver_strength, 0.5);
        } else {
            apply_mixer_hamiltonian(state, n, beta, config->anneal_offsets,
                                    s, config->schedule, dt, config->driver_strength, 1.0);
            if (!config->anneal_offsets) {
                apply_diagonal_cost_phases(state, diag_energies, dim, gamma);
            } else {
                apply_offset_cost_phases(state, model, config->anneal_offsets,
                                         s, config->schedule, dt, config->problem_strength);
            }
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

    if (config->reverse_anneal) {
        for (size_t q = 0; q < model->num_qubits; q++) {
            if ((config->initial_bitstring >> q) & 1u) {
                gate_pauli_x(&state, (int)q);
            }
        }
    } else {
        for (size_t q = 0; q < model->num_qubits; q++) {
            if (gate_hadamard(&state, (int)q) != QS_SUCCESS) {
                quantum_state_free(&state);
                return MOONLAB_ANNEAL_STATE_ERROR;
            }
        }
    }

    const uint64_t dim = UINT64_C(1) << model->num_qubits;
    double *diag_energies = malloc((size_t)dim * sizeof(double));
    double *cdf = malloc((size_t)dim * sizeof(double));
    moonlab_anneal_result_t *result = calloc(1, sizeof(*result));
    if (!diag_energies || !cdf || !result) {
        free(diag_energies); free(cdf); free(result);
        quantum_state_free(&state);
        return MOONLAB_ANNEAL_OOM;
    }

    compute_diagonal_energies(model, diag_energies);

    int rc = evolve(&state, model, config, diag_energies);
    if (rc != MOONLAB_ANNEAL_OK) {
        free(diag_energies); free(cdf); free(result);
        quantum_state_free(&state);
        return rc;
    }

    result->samples = malloc(config->num_samples * sizeof(uint64_t));
    result->sample_energies = malloc(config->num_samples * sizeof(double));
    if (!result->samples || !result->sample_energies) {
        moonlab_anneal_result_free(result);
        free(diag_energies); free(cdf);
        quantum_state_free(&state);
        return MOONLAB_ANNEAL_OOM;
    }

    /* Fused diagnostic pass:
     * - Single loop over 2^n bitstrings.
     * - Direct probability from real/imag squares without cabs()/hypot().
     * - Reuses precomputed diag_energies without redundant ising_energy calls. */
    double ground = DBL_MAX, first_excited = DBL_MAX;
    uint64_t ground_bits = 0;
    size_t degeneracy = 0;
    double cumulative = 0.0, expected = 0.0, success = 0.0;
    double norm = 0.0, max_probability = -1.0;
    uint64_t most_likely = 0;

    for (uint64_t bits = 0; bits < dim; bits++) {
        const complex_t amp = state.amplitudes[bits];
        const double re = creal(amp);
        const double im = cimag(amp);
        const double probability = re * re + im * im;
        const double energy = model->offset + diag_energies[bits];

        cumulative += probability;
        cdf[bits] = cumulative;
        norm += probability;
        expected += probability * energy;

        if (probability > max_probability) {
            max_probability = probability;
            most_likely = bits;
        }

        if (energy < ground && !energy_equal(energy, ground)) {
            first_excited = ground;
            ground = energy;
            ground_bits = bits;
            degeneracy = 1;
            success = probability;
        } else if (energy_equal(energy, ground)) {
            degeneracy++;
            success += probability;
        } else if (energy < first_excited) {
            first_excited = energy;
        }
    }

    if (!(norm > 0.0) || !isfinite(norm)) {
        moonlab_anneal_result_free(result);
        free(diag_energies); free(cdf);
        quantum_state_free(&state);
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
        const double energy = model->offset + diag_energies[bits];
        result->samples[sample] = bits;
        result->sample_energies[sample] = energy;
        if (energy < result->best_energy) {
            result->best_energy = energy;
            result->best_bitstring = bits;
        }
    }

    free(diag_energies);
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
    const int config_rc = validate_config(config, n);
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
    const int config_rc = validate_config(config, n);
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

int moonlab_quantum_reverse_anneal_ising(
    size_t num_qubits, const double *h, const double *J, double offset,
    uint64_t initial_bitstring,
    double s_target,
    double hold_fraction,
    const moonlab_anneal_config_t *config,
    moonlab_anneal_result_t **out_result)
{
    if (!out_result) return MOONLAB_ANNEAL_BAD_ARG;
    *out_result = NULL;
    if (num_qubits == 0 || num_qubits > 32 || !h || !J) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    if (!isfinite(s_target) || s_target < 0.0 || s_target > 1.0 ||
        !isfinite(hold_fraction) || hold_fraction < 0.0 || hold_fraction >= 1.0) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }

    moonlab_anneal_config_t cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = moonlab_anneal_config_default();
    }
    cfg.reverse_anneal = 1;
    cfg.initial_bitstring = initial_bitstring;
    cfg.reverse_s_target = s_target;
    cfg.reverse_hold_fraction = hold_fraction;

    return moonlab_quantum_anneal_ising(num_qubits, h, J, offset, &cfg, out_result);
}

int moonlab_quantum_reverse_anneal_qubo(
    size_t num_variables, const double *Q, double offset,
    uint64_t initial_bitstring,
    double s_target,
    double hold_fraction,
    const moonlab_anneal_config_t *config,
    moonlab_anneal_result_t **out_result)
{
    if (!out_result) return MOONLAB_ANNEAL_BAD_ARG;
    *out_result = NULL;
    if (num_variables == 0 || num_variables > 32 || !Q) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    if (!isfinite(s_target) || s_target < 0.0 || s_target > 1.0 ||
        !isfinite(hold_fraction) || hold_fraction < 0.0 || hold_fraction >= 1.0) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }

    double *h = calloc(num_variables, sizeof(double));
    double *J = calloc(num_variables * num_variables, sizeof(double));
    double ising_offset = 0.0;
    if (!h || !J) {
        free(h); free(J);
        return MOONLAB_ANNEAL_OOM;
    }
    int rc = moonlab_qubo_to_ising(num_variables, Q, offset, h, J, &ising_offset);
    if (rc == MOONLAB_ANNEAL_OK) {
        rc = moonlab_quantum_reverse_anneal_ising(num_variables, h, J, ising_offset,
                                                  initial_bitstring, s_target,
                                                  hold_fraction, config, out_result);
    }
    free(h); free(J);
    return rc;
}

static inline size_t zephyr_linear_index(
    size_t u, size_t w, size_t k, size_t j, size_t z, size_t m, size_t t)
{
    return (((u * (2 * m + 1) + w) * t + k) * 2 + j) * m + z;
}

static void zephyr_add_coupler(uint8_t *adj, size_t n, size_t u, size_t v)
{
    if (u == v || u >= n || v >= n) return;
    adj[u * n + v] = 1;
    adj[v * n + u] = 1;
}

moonlab_zephyr_graph_t *moonlab_zephyr_graph_create(size_t m)
{
    if (m < 1 || m > 64) return NULL;
    const size_t t = 4;
    const size_t M = 2 * m + 1;
    const size_t num_qubits = 16 * m * (2 * m + 1);

    moonlab_zephyr_graph_t *graph = calloc(1, sizeof(*graph));
    if (!graph) return NULL;

    graph->m = m;
    graph->t = t;
    graph->num_qubits = num_qubits;
    graph->adjacency_matrix = calloc(num_qubits * num_qubits, sizeof(uint8_t));
    if (!graph->adjacency_matrix) {
        free(graph);
        return NULL;
    }

    /* 1. External couplers: collinear along parallel offset z */
    for (size_t u = 0; u < 2; u++) {
        for (size_t w = 0; w < M; w++) {
            for (size_t k = 0; k < t; k++) {
                for (size_t j = 0; j < 2; j++) {
                    for (size_t z = 0; z + 1 < m; z++) {
                        const size_t u_node = zephyr_linear_index(u, w, k, j, z, m, t);
                        const size_t v_node = zephyr_linear_index(u, w, k, j, z + 1, m, t);
                        zephyr_add_coupler(graph->adjacency_matrix, num_qubits, u_node, v_node);
                    }
                }
            }
        }
    }

    /* 2. Odd couplers: intra-unit-tile pairings between j=0 and j=1 */
    for (size_t u = 0; u < 2; u++) {
        for (size_t w = 0; w < M; w++) {
            for (size_t k = 0; k < t; k++) {
                for (size_t a = 0; a < 2; a++) {
                    for (size_t z = a; z < m; z++) {
                        const size_t u_node = zephyr_linear_index(u, w, k, 0, z, m, t);
                        const size_t v_node = zephyr_linear_index(u, w, k, 1, z - a, m, t);
                        zephyr_add_coupler(graph->adjacency_matrix, num_qubits, u_node, v_node);
                    }
                }
            }
        }
    }

    /* 3. Internal couplers: orthogonal crossings between vertical (u=0) and horizontal (u=1) */
    for (size_t w = 0; w < m; w++) {
        for (size_t z = 0; z < m; z++) {
            for (size_t h = 0; h < t; h++) {
                for (size_t k = 0; k < t; k++) {
                    for (size_t i = 0; i < 2; i++) {
                        for (size_t j = 0; j < 2; j++) {
                            for (size_t a = 0; a < 2; a++) {
                                for (size_t b = 0; b < 2; b++) {
                                    const size_t w_v = 2 * w + 1 + a * (2 * i - 1);
                                    const size_t w_h = 2 * z + 1 + b * (2 * j - 1);
                                    const size_t u_node =
                                        zephyr_linear_index(0, w_v, k, j, z, m, t);
                                    const size_t v_node =
                                        zephyr_linear_index(1, w_h, h, i, w, m, t);
                                    zephyr_add_coupler(
                                        graph->adjacency_matrix, num_qubits, u_node, v_node);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* Count unique bidirectional couplers (u < v) */
    size_t count = 0;
    for (size_t i = 0; i < num_qubits; i++) {
        for (size_t j = i + 1; j < num_qubits; j++) {
            if (graph->adjacency_matrix[i * num_qubits + j]) count++;
        }
    }

    graph->num_couplers = count;
    graph->coupler_u = calloc(count, sizeof(uint32_t));
    graph->coupler_v = calloc(count, sizeof(uint32_t));
    if (!graph->coupler_u || !graph->coupler_v) {
        moonlab_zephyr_graph_free(graph);
        return NULL;
    }

    size_t edge_idx = 0;
    for (size_t i = 0; i < num_qubits; i++) {
        for (size_t j = i + 1; j < num_qubits; j++) {
            if (graph->adjacency_matrix[i * num_qubits + j]) {
                graph->coupler_u[edge_idx] = (uint32_t)i;
                graph->coupler_v[edge_idx] = (uint32_t)j;
                edge_idx++;
            }
        }
    }

    return graph;
}

void moonlab_zephyr_graph_free(moonlab_zephyr_graph_t *graph)
{
    if (!graph) return;
    free(graph->coupler_u);
    free(graph->coupler_v);
    free(graph->adjacency_matrix);
    free(graph);
}

int moonlab_zephyr_has_coupler(const moonlab_zephyr_graph_t *graph, size_t u, size_t v)
{
    if (!graph || !graph->adjacency_matrix ||
        u >= graph->num_qubits || v >= graph->num_qubits) {
        return 0;
    }
    return graph->adjacency_matrix[u * graph->num_qubits + v] ? 1 : 0;
}

void moonlab_zephyr_embedding_free(moonlab_zephyr_embedding_t *emb)
{
    if (!emb) return;
    if (emb->chains) {
        for (size_t i = 0; i < emb->num_logical; i++) {
            free(emb->chains[i]);
        }
        free(emb->chains);
    }
    free(emb->chain_lengths);
    free(emb);
}

moonlab_zephyr_embedding_t *moonlab_zephyr_find_clique_embedding(size_t num_logical, size_t m)
{
    if (num_logical == 0 || m == 0) return NULL;
    const size_t t = 4;
    const size_t max_capacity = 8 * m;
    if (num_logical > max_capacity) return NULL;

    moonlab_zephyr_embedding_t *emb = calloc(1, sizeof(*emb));
    if (!emb) return NULL;

    emb->num_logical = num_logical;
    emb->chain_lengths = calloc(num_logical, sizeof(size_t));
    emb->chains = calloc(num_logical, sizeof(size_t *));
    if (!emb->chain_lengths || !emb->chains) {
        moonlab_zephyr_embedding_free(emb);
        return NULL;
    }

    if (num_logical <= 2 * t) {
        /* Optimal chain length 2 within a single unit cell (tile w=1) */
        for (size_t i = 0; i < num_logical; i++) {
            emb->chain_lengths[i] = 2;
            emb->chains[i] = malloc(2 * sizeof(size_t));
            if (!emb->chains[i]) {
                moonlab_zephyr_embedding_free(emb);
                return NULL;
            }
            const size_t k = i / 2;
            const size_t j = i % 2;
            emb->chains[i][0] = zephyr_linear_index(0, 1, k, j, 0, m, t);
            emb->chains[i][1] = zephyr_linear_index(1, 1, k, j, 0, m, t);
        }
    } else {
        /* Contiguous multi-tile clique embedding with chain length 3 */
        for (size_t i = 0; i < num_logical; i++) {
            emb->chain_lengths[i] = 3;
            emb->chains[i] = malloc(3 * sizeof(size_t));
            if (!emb->chains[i]) {
                moonlab_zephyr_embedding_free(emb);
                return NULL;
            }
            const size_t tile = i / (2 * t);
            const size_t rem = i % (2 * t);
            const size_t k = rem / 2;
            const size_t j = rem % 2;
            const size_t w = tile + 1;
            if (tile == 0) {
                emb->chains[i][0] = zephyr_linear_index(0, 1, k, j, 1, m, t);
                emb->chains[i][1] = zephyr_linear_index(0, 1, k, j, 0, m, t);
                emb->chains[i][2] = zephyr_linear_index(1, 1, k, j, 0, m, t);
            } else if (tile == 1) {
                if (j == 0) {
                    emb->chains[i][0] = zephyr_linear_index(0, 2, k, 0, 1, m, t);
                    emb->chains[i][1] = zephyr_linear_index(1, 2, k, 0, 1, m, t);
                    emb->chains[i][2] = zephyr_linear_index(1, 2, k, 0, 0, m, t);
                } else {
                    emb->chains[i][0] = zephyr_linear_index(0, 2, k, 1, 1, m, t);
                    emb->chains[i][1] = zephyr_linear_index(0, 2, k, 1, 0, m, t);
                    emb->chains[i][2] = zephyr_linear_index(1, 2, k, 1, 0, m, t);
                }
            } else {
                emb->chains[i][0] = zephyr_linear_index(0, w, k, j, 1, m, t);
                emb->chains[i][1] = zephyr_linear_index(1, w, k, j, 1, m, t);
                emb->chains[i][2] = zephyr_linear_index(1, w, k, j, 0, m, t);
            }
        }
    }

    return emb;
}

int moonlab_zephyr_embed_ising(
    const moonlab_zephyr_graph_t *graph,
    const moonlab_zephyr_embedding_t *emb,
    const double *logical_h,
    const double *logical_J,
    double chain_strength,
    double *out_physical_h,
    double *out_physical_J)
{
    if (!graph || !emb || !logical_h || !logical_J ||
        !out_physical_h || !out_physical_J ||
        !isfinite(chain_strength) || chain_strength <= 0.0) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }
    const size_t n_phys = graph->num_qubits;
    const size_t n_log = emb->num_logical;

    memset(out_physical_h, 0, n_phys * sizeof(double));
    memset(out_physical_J, 0, n_phys * n_phys * sizeof(double));

    /* 1. Distribute local fields h evenly across each logical variable's chain */
    for (size_t i = 0; i < n_log; i++) {
        const size_t len = emb->chain_lengths[i];
        if (len == 0) return MOONLAB_ANNEAL_EMBEDDING_ERROR;
        const double h_part = logical_h[i] / (double)len;
        for (size_t c = 0; c < len; c++) {
            const size_t q = emb->chains[i][c];
            if (q >= n_phys) return MOONLAB_ANNEAL_EMBEDDING_ERROR;
            out_physical_h[q] += h_part;
        }

        /* Ferromagnetic intra-chain coupling between adjacent qubits in the chain */
        for (size_t c = 0; c + 1 < len; c++) {
            const size_t q1 = emb->chains[i][c];
            const size_t q2 = emb->chains[i][c + 1];
            if (!moonlab_zephyr_has_coupler(graph, q1, q2)) {
                return MOONLAB_ANNEAL_EMBEDDING_ERROR;
            }
            out_physical_J[q1 * n_phys + q2] -= chain_strength;
            out_physical_J[q2 * n_phys + q1] -= chain_strength;
        }
    }

    /* 2. Distribute inter-variable couplings J_ij across physical couplers
     * connecting chain i and chain j */
    for (size_t i = 0; i < n_log; i++) {
        for (size_t j = i + 1; j < n_log; j++) {
            const double jij = logical_J[i * n_log + j];
            if (fabs(jij) <= 1e-15) continue;

            /* Count available couplers between chain i and chain j */
            size_t coupler_count = 0;
            const size_t len_i = emb->chain_lengths[i];
            const size_t len_j = emb->chain_lengths[j];
            for (size_t ci = 0; ci < len_i; ci++) {
                const size_t qi = emb->chains[i][ci];
                for (size_t cj = 0; cj < len_j; cj++) {
                    const size_t qj = emb->chains[j][cj];
                    if (moonlab_zephyr_has_coupler(graph, qi, qj)) {
                        coupler_count++;
                    }
                }
            }

            if (coupler_count == 0) {
                return MOONLAB_ANNEAL_EMBEDDING_ERROR;
            }

            const double j_part = jij / (double)coupler_count;
            for (size_t ci = 0; ci < len_i; ci++) {
                const size_t qi = emb->chains[i][ci];
                for (size_t cj = 0; cj < len_j; cj++) {
                    const size_t qj = emb->chains[j][cj];
                    if (moonlab_zephyr_has_coupler(graph, qi, qj)) {
                        out_physical_J[qi * n_phys + qj] += j_part;
                        out_physical_J[qj * n_phys + qi] += j_part;
                    }
                }
            }
        }
    }

    return MOONLAB_ANNEAL_OK;
}

int moonlab_zephyr_unembed_samples(
    const moonlab_zephyr_embedding_t *emb,
    size_t num_samples,
    const uint64_t *physical_samples,
    uint64_t *out_logical_samples,
    double *out_chain_break_fractions)
{
    if (!emb || !physical_samples || !out_logical_samples ||
        emb->num_logical == 0 || emb->num_logical > 64) {
        return MOONLAB_ANNEAL_BAD_ARG;
    }

    for (size_t s = 0; s < num_samples; s++) {
        const uint64_t phys = physical_samples[s];
        uint64_t log_bits = 0;
        size_t broken_chains = 0;

        for (size_t i = 0; i < emb->num_logical; i++) {
            const size_t len = emb->chain_lengths[i];
            size_t ones = 0;
            for (size_t c = 0; c < len; c++) {
                const size_t q = emb->chains[i][c];
                if (q < 64 && ((phys >> q) & UINT64_C(1))) {
                    ones++;
                }
            }
            const size_t zeros = len - ones;
            if (ones > zeros) {
                log_bits |= (UINT64_C(1) << i);
            }
            if (ones > 0 && ones < len) {
                broken_chains++;
            }
        }

        out_logical_samples[s] = log_bits;
        if (out_chain_break_fractions) {
            out_chain_break_fractions[s] = (double)broken_chains / (double)emb->num_logical;
        }
    }

    return MOONLAB_ANNEAL_OK;
}

