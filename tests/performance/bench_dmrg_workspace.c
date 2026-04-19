/**
 * @file bench_dmrg_workspace.c
 * @brief Measure the allocation overhead saved by the
 *        effective_hamiltonian_apply_ws persistent-scratch path.
 *
 * Both variants apply the same local effective Hamiltonian to the
 * same input many times.  The difference is whether each call
 * calloc/free-s three chi^2 * bond^2 -sized buffers (legacy) or
 * reuses a single workspace (new).  We do NOT diff correctness -- the
 * unit tests in test_dmrg handle that; this file is a stopwatch.
 *
 * Output columns:
 *   (chi_l, chi_r, iters, legacy us/iter, ws us/iter, speedup)
 * All times are per effective_hamiltonian_apply invocation; one
 * Lanczos typically does 30-100 of these per bond, one DMRG sweep
 * does ~L of those per sweep, a full run ~(10-50) sweeps, so divide
 * the saved us/iter by ~1 us to estimate saved microseconds per
 * sweep and multiply by (sweeps * bonds) for the run.
 */

#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tensor.h"
#include "../../src/utils/manifest.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static char*  g_metrics = NULL;
static size_t g_metrics_cap = 0;

/* Build a TFIM MPO and one bulk site's effective Hamiltonian at a
 * given bond dimension.  Environments are initialised to zero-pattern
 * identity-like blocks (we only care about runtime, not the numerical
 * value of H_eff @ x). */
static void run_case(uint32_t chi, uint32_t iters) {
    const uint32_t L = 8;
    mpo_t* mpo = mpo_tfim_create(L, 1.0, 0.8);
    if (!mpo) { fprintf(stderr, "mpo build failed\n"); return; }

    const uint32_t b_l = mpo->tensors[3].bond_dim_left;
    const uint32_t b_r = mpo->tensors[4].bond_dim_right;
    const uint32_t d   = 2;

    uint32_t L_dims[3] = {chi, b_l, chi};
    tensor_t* Lenv = tensor_create(3, L_dims);
    uint32_t R_dims[3] = {chi, b_r, chi};
    tensor_t* Renv = tensor_create(3, R_dims);
    /* Fill with non-trivial numbers so the compiler can't optimise
     * the inner loops away. */
    for (uint64_t i = 0; i < Lenv->total_size; i++) Lenv->data[i] = 1.0 / (double)(i + 1);
    for (uint64_t i = 0; i < Renv->total_size; i++) Renv->data[i] = 1.0 / (double)(i + 2);

    effective_hamiltonian_t H = {
        .L = Lenv, .R = Renv,
        .W_left  = &mpo->tensors[3],
        .W_right = &mpo->tensors[4],
        .chi_l = chi, .chi_r = chi,
        .phys_dim = d, .two_site = true
    };

    uint32_t dims[4] = {chi, d, d, chi};
    tensor_t* x = tensor_create(4, dims);
    tensor_t* y = tensor_create(4, dims);
    for (uint64_t i = 0; i < x->total_size; i++) x->data[i] = 0.1 + 0.01 * (double)i;

    /* Warmup. */
    for (uint32_t w = 0; w < 3; w++) effective_hamiltonian_apply(&H, x, y);

    const double t0_legacy = now_us();
    for (uint32_t i = 0; i < iters; i++) {
        effective_hamiltonian_apply(&H, x, y);
    }
    const double legacy = (now_us() - t0_legacy) / iters;

    effective_hamiltonian_workspace_t ws;
    effective_hamiltonian_workspace_init(&ws);
    /* Warmup WS to amortise first-call realloc. */
    for (uint32_t w = 0; w < 3; w++) effective_hamiltonian_apply_ws(&H, x, y, &ws);

    const double t0_ws = now_us();
    for (uint32_t i = 0; i < iters; i++) {
        effective_hamiltonian_apply_ws(&H, x, y, &ws);
    }
    const double ws_us = (now_us() - t0_ws) / iters;

    effective_hamiltonian_workspace_free(&ws);

    printf("  chi=%-4u iters=%-5u   legacy %8.2f us   ws %8.2f us   %5.2fx\n",
           chi, iters, legacy, ws_us, legacy / ws_us);

    if (g_metrics && g_metrics_cap) {
        size_t cur = strlen(g_metrics);
        snprintf(g_metrics + cur, g_metrics_cap - cur,
                 "%s{\"chi\":%u,\"iters\":%u,\"legacy_us\":%.3f,"
                 "\"ws_us\":%.3f,\"speedup\":%.3f}",
                 cur == 0 ? "" : ",",
                 chi, iters, legacy, ws_us, legacy / ws_us);
    }

    tensor_free(Lenv); tensor_free(Renv);
    tensor_free(x); tensor_free(y);
    mpo_free(mpo);
}

int main(void) {
    printf("=== DMRG H_eff apply: legacy vs workspace ===\n");
    printf("  per-call times; lower is better; speedup = legacy/ws.\n\n");

    moonlab_manifest_t manifest;
    moonlab_manifest_capture(&manifest, "bench_dmrg_workspace", 0);

    char buf[4096] = "[";
    g_metrics = buf + 1;
    g_metrics_cap = sizeof buf - 2;

    run_case(8,  2000);
    run_case(16, 1000);
    run_case(32,  500);
    run_case(64,  200);
    run_case(128, 100);

    size_t mlen = strlen(buf);
    buf[mlen] = ']'; buf[mlen + 1] = '\0';

    char metrics_obj[5120];
    snprintf(metrics_obj, sizeof metrics_obj, "{\"rows\":%s}", buf);
    manifest.metrics_json = metrics_obj;

    moonlab_manifest_stamp_finish(&manifest);

    const char* out_path = getenv("MOONLAB_MANIFEST_OUT");
    if (out_path && *out_path) {
        FILE* f = fopen(out_path, "w");
        if (f) {
            moonlab_manifest_write_json_pretty(&manifest, f);
            fclose(f);
            fprintf(stderr, "[manifest] written to %s\n", out_path);
        }
    }
    moonlab_manifest_release(&manifest);
    return 0;
}
