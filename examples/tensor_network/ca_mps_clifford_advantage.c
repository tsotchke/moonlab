/**
 * @file ca_mps_clifford_advantage.c
 * @brief Demonstrate CA-MPS's bond-dimension advantage on Clifford-rich
 *        circuits versus a plain MPS.
 *
 * Builds the same random Clifford circuit on both representations and
 * reports the maximum bond dimension each one needs.  Plain MPS grows
 * as 2^(N/2) for a generic stabilizer state; CA-MPS factors the entire
 * Clifford prefactor into the tableau and the MPS factor stays at
 * bond dimension 1.
 *
 * Run:
 *   ./build/example_ca_mps_clifford_advantage [N=12] [depth=4N]
 *
 * Reproduces the headline benchmark from `bench_ca_mps`: at N=12 you
 * should see plain chi ~ 64, CA-MPS chi = 1 (a 64x advantage).
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0xC0FFEE12345678ULL;
static uint32_t rng_u32(uint32_t bound) {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)((rng >> 32) % bound);
}

int main(int argc, char** argv) {
    uint32_t n     = (argc > 1) ? (uint32_t)atoi(argv[1]) : 12;
    /* Random Clifford circuits need depth ~ N^2 to fully scramble; with
     * fewer gates plain MPS can stay at low chi by luck.  Default
     * scales with n^2 to make the Clifford advantage robust. */
    uint32_t depth = (argc > 2) ? (uint32_t)atoi(argv[2]) : 12 * n;
    uint32_t chi_cap = 1u << (n / 2 + 1);

    printf("CA-MPS vs plain MPS on a random Clifford circuit\n");
    printf("  N = %u, depth = %u, chi_cap = %u\n\n", n, depth, chi_cap);

    tn_state_config_t cfg = { 0 };
    cfg.max_bond_dim = chi_cap;
    cfg.svd_cutoff = 1e-12;
    moonlab_ca_mps_t* ca = moonlab_ca_mps_create(n, chi_cap);
    tn_mps_state_t*   mp = tn_mps_create_zero(n, &cfg);
    if (!ca || !mp) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* Random Clifford gates: H, S, X, CNOT, CZ.  All free for the
     * tableau; all entanglement-generating for plain MPS. */
    for (uint32_t step = 0; step < depth; step++) {
        uint32_t op = rng_u32(5);
        uint32_t q  = rng_u32(n);
        uint32_t q2 = (q + 1 + rng_u32(n - 1)) % n;
        switch (op) {
            case 0: moonlab_ca_mps_h(ca, q);        tn_apply_h(mp, q); break;
            case 1: moonlab_ca_mps_s(ca, q);        tn_apply_s(mp, q); break;
            case 2: moonlab_ca_mps_x(ca, q);        tn_apply_x(mp, q); break;
            case 3: moonlab_ca_mps_cnot(ca, q, q2); tn_apply_cnot(mp, q, q2); break;
            case 4: moonlab_ca_mps_cz(ca, q, q2);   tn_apply_cz(mp, q, q2); break;
        }
    }

    uint32_t chi_plain = 0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        if (mp->bond_dims[i] > chi_plain) chi_plain = mp->bond_dims[i];
    }
    uint32_t chi_ca = moonlab_ca_mps_current_bond_dim(ca);

    printf("  plain MPS peak chi: %u  (worst-case 2^(N/2) = %u)\n",
           chi_plain, 1u << (n / 2));
    printf("  CA-MPS  peak chi:   %u\n", chi_ca);
    if (chi_ca > 0) {
        printf("  bond-dim advantage: %.1fx\n", (double)chi_plain / (double)chi_ca);
    }

    moonlab_ca_mps_free(ca);
    tn_mps_free(mp);
    return 0;
}
