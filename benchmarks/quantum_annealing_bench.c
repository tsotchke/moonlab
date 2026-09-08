/** Reproducible dense annealing scaling benchmark; emits CSV to stdout. */

#include "../src/algorithms/quantum_annealing.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec ts = {0};
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    const size_t max_n = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 16;
    const size_t steps = argc > 2 ? (size_t)strtoul(argv[2], NULL, 10) : 200;
    if (max_n < 2 || max_n > 24 || steps < 1) return 2;
    puts("qubits,steps,seconds,success_probability,residual_energy,final_norm");
    for (size_t n = 2; n <= max_n; n += 2) {
        double *h = calloc(n, sizeof(double));
        double *J = calloc(n * n, sizeof(double));
        if (!h || !J) { free(h); free(J); return 2; }
        for (size_t i = 0; i + 1 < n; i++) {
            J[i * n + i + 1] = J[(i + 1) * n + i] = -1.0;
        }
        moonlab_anneal_config_t config = moonlab_anneal_config_default();
        config.total_time = 8.0;
        config.num_steps = steps;
        config.num_samples = 1;
        config.seed = UINT64_C(0x414e4e45414c0000) + n;
        moonlab_anneal_result_t *result = NULL;
        const double t0 = now_seconds();
        const int rc = moonlab_quantum_anneal_ising(
            n, h, J, 0.0, &config, &result);
        const double elapsed = now_seconds() - t0;
        free(h); free(J);
        if (rc != 0 || !result) return 1;
        printf("%zu,%zu,%.9f,%.12g,%.12g,%.12g\n",
               n, steps, elapsed,
               moonlab_anneal_result_success_probability(result),
               moonlab_anneal_result_residual_energy(result),
               moonlab_anneal_result_final_norm(result));
        moonlab_anneal_result_free(result);
    }
    return 0;
}
