/** Solve (x0+x1-1)^2 by closed-system quantum annealing. */

#include "../../src/applications/moonlab_export.h"

#include <math.h>
#include <stdio.h>

int main(void)
{
    const double Q[4] = {-1.0, 1.0, 1.0, -1.0};
    moonlab_anneal_summary_v1 result;
    uint64_t samples[128];
    double energies[128];
    int rc = moonlab_anneal_qubo_v1(
        2, Q, 1.0,
        12.0, 1200, 128, UINT64_C(0x5155414e54554d),
        /*cosine=*/2, 1.0, 1.0, /*second_order=*/1,
        &result, samples, energies);
    if (rc != 0) {
        fprintf(stderr, "quantum annealing failed: %d\n", rc);
        return 1;
    }
    printf("best=%llu energy=%.6f success=%.6f seed=%016llx\n",
           (unsigned long long)result.best_bitstring,
           result.best_energy, result.success_probability,
           (unsigned long long)result.effective_seed);
    return fabs(result.best_energy) < 1e-12 &&
           result.success_probability > 0.95 ? 0 : 1;
}
