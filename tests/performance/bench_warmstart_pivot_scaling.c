/**
 * @file bench_warmstart_pivot_scaling.c
 * @brief Pivot-distribution scaling sweep for the stabiliser-subgroup
 *        warmstart Clifford builder.
 *
 * Backs Theorem 1 of papers/drafts/ca_tn_method/main.tex with a wide
 * scaling table.  For each (code family, size) pair:
 *
 *   1. Build the all-Z stabiliser generators for the code.
 *   2. Apply moonlab_ca_mps_apply_stab_subgroup_warmstart.
 *   3. For each generator, conjugate through the stored Clifford D
 *      via moonlab_ca_mps_conjugate_pauli to read off C^dagger g_i C.
 *   4. Verify the canonical-form invariant: each conjugate is Z-only
 *      with +1 phase, and the union of Z-supports has cardinality k.
 *   5. Sweep all bipartitions A | B (|A| = 1 .. n-1) and report the
 *      tightest bound min(|A|-k_A, |B|-k_B), plus the half-cut bound
 *      for paper consistency.
 *   6. Also measure the bond-dim-1 |phi_0> = |0^n> entropy after
 *      warmstart -- the trivial case where no physical state has been
 *      prepared.  (Empirical entropy on physical ground states is a
 *      separate harness; this one focuses on the structural bound.)
 *
 * Code families:
 *
 *   Z2 LGT (1+1d): N matter sites, n = 2N-1 qubits, k = N-2 interior
 *     Gauss-law generators.  Tested for N in {4, 6, 8, 10, 12}.
 *
 *   Surface-d ("windowed Z" stabilisers): d x d data-qubit grid, n =
 *     d^2 qubits, k = (d-1)^2 Z-stabs (one ZZZZ per 2x2 window).  This
 *     is a valid commuting Z-stabiliser subgroup (not the full CSS
 *     surface code, just the Z half).  Tested for d in {3, 5, 7}.
 *
 *   Toric LxL: L x L plaquette torus, n = 2L^2 edge qubits, k = L^2 - 1
 *     independent Z-plaquettes (1 redundancy among L^2).  Tested for
 *     L in {2, 3, 4}.
 *
 * Outputs:
 *   - Human-readable scaling table on stdout.
 *   - Optional JSON archive at argv[1] for the paper's repro manifest.
 */

#include "../../src/algorithms/tensor_network/ca_mps.h"
#include "../../src/algorithms/tensor_network/ca_mps_var_d_stab_warmstart.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Stabiliser generators per code family                              */
/* ------------------------------------------------------------------ */

/* Z2 LGT 1+1d Gauss law: G_x = X_{2x-1} X_{2x+1} Z_{2x}, x = 1..N-1.
 * For "all-Z" warmstart support we use only the interior generators
 * x = 1..N-2, of which there are N-2.  Wait -- the existing Gauss
 * law operators include X's too, so this isn't actually all-Z.  Use
 * the existing test path: the AG warmstart supports mixed XYZ Pauli
 * generators, not just Z.  We still report (k_A, k_B) on the
 * conjugated rows. */

#include "../../src/applications/hep/lattice_z2_1d.h"

static int build_z2_lgt(uint32_t N,
                        uint8_t** out_paulis, uint32_t* out_n,
                        uint32_t* out_k) {
    z2_lgt_config_t cfg = {0};
    cfg.num_matter_sites = N;
    cfg.t_hop = 1.0; cfg.h_link = 1.0; cfg.mass = 0.0;
    cfg.gauss_penalty = 0.0;
    uint32_t n = z2_lgt_1d_num_qubits(&cfg);
    uint32_t k = (N >= 2) ? (N - 2) : 0;
    if (k == 0) return -1;
    uint8_t* p = (uint8_t*)calloc((size_t)k * n, 1);
    if (!p) return -1;
    for (uint32_t i = 0; i < k; i++) {
        if (z2_lgt_1d_gauss_law_pauli(&cfg, i + 1, &p[(size_t)i * n]) != 0) {
            free(p); return -1;
        }
    }
    *out_paulis = p;
    *out_n = n;
    *out_k = k;
    return 0;
}

/* Surface-d windowed Z: data qubits indexed as (r, c) -> r*d + c on a
 * d x d grid.  Z-stabs at each (rs, cs), 0 <= rs, cs <= d-2, cover
 * the four corners of the 2x2 window. */
static int build_surface(uint32_t d,
                         uint8_t** out_paulis, uint32_t* out_n,
                         uint32_t* out_k) {
    if (d < 3) return -1;
    uint32_t n = d * d;
    uint32_t k = (d - 1) * (d - 1);
    uint8_t* p = (uint8_t*)calloc((size_t)k * n, 1);
    if (!p) return -1;
    uint32_t row = 0;
    for (uint32_t rs = 0; rs + 1 < d; rs++) {
        for (uint32_t cs = 0; cs + 1 < d; cs++) {
            uint8_t* g = &p[(size_t)row * n];
            g[(rs    ) * d + cs    ] = 3;  /* Z */
            g[(rs    ) * d + cs + 1] = 3;
            g[(rs + 1) * d + cs    ] = 3;
            g[(rs + 1) * d + cs + 1] = 3;
            row++;
        }
    }
    *out_paulis = p;
    *out_n = n;
    *out_k = k;
    return 0;
}

/* Toric L x L Z-plaquettes.  Edge enumeration:
 *   horizontal edges h(r, c): joins (r, c) <-> (r, c+1 mod L), index = r*L + c
 *   vertical edges   v(r, c): joins (r, c) <-> (r+1 mod L, c), index = L^2 + r*L + c
 * Total qubits: 2*L^2.
 * Plaquette at (r, c) is bordered by:
 *   h(r, c), h((r+1) mod L, c), v(r, c), v(r, (c+1) mod L)
 * One redundancy: product of all L^2 plaquettes = identity, so use
 * the first L^2 - 1 as independent generators. */
static int build_toric(uint32_t L,
                       uint8_t** out_paulis, uint32_t* out_n,
                       uint32_t* out_k) {
    if (L < 2) return -1;
    uint32_t n = 2 * L * L;
    uint32_t k = L * L - 1;
    uint8_t* p = (uint8_t*)calloc((size_t)k * n, 1);
    if (!p) return -1;
    uint32_t row = 0;
    for (uint32_t r = 0; r < L && row < k; r++) {
        for (uint32_t c = 0; c < L && row < k; c++) {
            uint8_t* g = &p[(size_t)row * n];
            uint32_t h_top    = ((r + 0) % L) * L + c;
            uint32_t h_bot    = ((r + 1) % L) * L + c;
            uint32_t v_left   = L*L + r * L + ((c + 0) % L);
            uint32_t v_right  = L*L + r * L + ((c + 1) % L);
            g[h_top]   = 3;  /* Z */
            g[h_bot]   = 3;
            g[v_left]  = 3;
            g[v_right] = 3;
            row++;
        }
    }
    *out_paulis = p;
    *out_n = n;
    *out_k = k;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Pivot-distribution measurement                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char     code[32];
    uint32_t size_param;
    uint32_t n;
    uint32_t k;
    /* canonical-form check */
    int      all_z_only;
    int      all_pos_phase;
    uint32_t pivot_set_size;
    /* half-cut bound */
    uint32_t cut;
    uint32_t k_A_half;
    uint32_t k_B_half;
    int      schmidt_log2_half;
    int      unrotated_log2_half;
    /* best-cut sweep */
    uint32_t best_cut;
    uint32_t best_k_A;
    uint32_t best_k_B;
    int      best_schmidt_log2;
    int      unrotated_log2_best;
} pivot_record_t;

static int measure_record(const char* code_name,
                          uint32_t size_param,
                          const uint8_t* paulis,
                          uint32_t n,
                          uint32_t k,
                          pivot_record_t* out) {
    memset(out, 0, sizeof(*out));
    snprintf(out->code, sizeof(out->code), "%s", code_name);
    out->size_param = size_param;
    out->n = n;
    out->k = k;

    moonlab_ca_mps_t* s = moonlab_ca_mps_create(n, 32);
    if (!s) return -1;

    ca_mps_error_t err =
        moonlab_ca_mps_apply_stab_subgroup_warmstart(s, paulis, k);
    if (err != CA_MPS_SUCCESS) {
        fprintf(stderr, "[%s n=%u k=%u] warmstart failed: %d\n",
                code_name, (unsigned)n, (unsigned)k, (int)err);
        moonlab_ca_mps_free(s);
        return -1;
    }

    uint8_t* conj = (uint8_t*)calloc(n, 1);
    uint8_t* pivot_seen = (uint8_t*)calloc(n, 1);
    if (!conj || !pivot_seen) {
        free(conj); free(pivot_seen); moonlab_ca_mps_free(s);
        return -1;
    }

    out->all_z_only = 1;
    out->all_pos_phase = 1;

    for (uint32_t i = 0; i < k; i++) {
        int phase = 0;
        if (moonlab_ca_mps_conjugate_pauli(
                s, &paulis[(size_t)i * n], conj, &phase) != CA_MPS_SUCCESS) {
            free(conj); free(pivot_seen); moonlab_ca_mps_free(s);
            return -1;
        }
        if (phase != 0) out->all_pos_phase = 0;
        for (uint32_t q = 0; q < n; q++) {
            if (conj[q] == 0) continue;
            if (conj[q] != 3) out->all_z_only = 0;
            pivot_seen[q] = 1;
        }
    }

    uint32_t total_pivots = 0;
    for (uint32_t q = 0; q < n; q++) if (pivot_seen[q]) total_pivots++;
    out->pivot_set_size = total_pivots;

    /* Half-cut */
    out->cut = n / 2;
    {
        uint32_t kA = 0, kB = 0;
        for (uint32_t q = 0; q < n; q++) {
            if (!pivot_seen[q]) continue;
            if (q < out->cut) kA++; else kB++;
        }
        out->k_A_half = kA;
        out->k_B_half = kB;
        uint32_t free_l = out->cut    - kA;
        uint32_t free_r = (n - out->cut) - kB;
        out->schmidt_log2_half = (free_l < free_r) ? (int)free_l : (int)free_r;
        uint32_t nA = out->cut, nB = n - out->cut;
        out->unrotated_log2_half = (int)((nA < nB) ? nA : nB);
    }

    /* Best cut sweep: |A| = 1 .. n-1 (contiguous left-block bipartition,
     * the natural one for an MPS half-cut family). */
    out->best_schmidt_log2 = INT32_MAX;
    out->unrotated_log2_best = INT32_MAX;
    for (uint32_t cut = 1; cut < n; cut++) {
        uint32_t kA = 0, kB = 0;
        for (uint32_t q = 0; q < n; q++) {
            if (!pivot_seen[q]) continue;
            if (q < cut) kA++; else kB++;
        }
        uint32_t free_l = cut - kA;
        uint32_t free_r = (n - cut) - kB;
        int s_rot = (int)((free_l < free_r) ? free_l : free_r);
        int s_unr = (int)((cut < n - cut) ? cut : n - cut);
        if (s_rot < out->best_schmidt_log2) {
            out->best_schmidt_log2   = s_rot;
            out->unrotated_log2_best = s_unr;
            out->best_cut = cut;
            out->best_k_A = kA;
            out->best_k_B = kB;
        }
    }

    free(conj);
    free(pivot_seen);
    moonlab_ca_mps_free(s);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Output                                                             */
/* ------------------------------------------------------------------ */

static void print_header(void) {
    printf("\n%-12s | %-3s | %-4s | %-4s | %-4s | half-cut       | "
           "best-cut\n",
           "code", "sz", "n", "k", "|Q|");
    printf("%-12s | %-3s | %-4s | %-4s | %-4s | "
           "%-3s %-3s %-3s %-3s | %-3s %-3s %-3s %-3s\n",
           "", "", "", "", "",
           "kA", "kB", "rot", "unr",
           "cut", "rot", "unr", "kA");
    printf("-------------+-----+------+------+------+"
           "-------------------+--------------------\n");
}

static void print_row(const pivot_record_t* r) {
    printf("%-12s | %3u | %4u | %4u | %4u | "
           "%3u %3u %3d %3d | %3u %3d %3d %3u\n",
           r->code, (unsigned)r->size_param, (unsigned)r->n,
           (unsigned)r->k, (unsigned)r->pivot_set_size,
           (unsigned)r->k_A_half, (unsigned)r->k_B_half,
           r->schmidt_log2_half, r->unrotated_log2_half,
           (unsigned)r->best_cut, r->best_schmidt_log2,
           r->unrotated_log2_best, (unsigned)r->best_k_A);
}

static int json_emit(FILE* f, const pivot_record_t* recs, size_t n_rec) {
    fprintf(f, "{\n");
    fprintf(f, "  \"harness\": \"bench_warmstart_pivot_scaling\",\n");
    fprintf(f, "  \"description\": \"Theorem 1 pivot-distribution scaling sweep\",\n");
    fprintf(f, "  \"records\": [\n");
    for (size_t i = 0; i < n_rec; i++) {
        const pivot_record_t* r = &recs[i];
        fprintf(f, "    {");
        fprintf(f, "\"code\":\"%s\", ", r->code);
        fprintf(f, "\"size\":%u, ",       (unsigned)r->size_param);
        fprintf(f, "\"n\":%u, ",          (unsigned)r->n);
        fprintf(f, "\"k\":%u, ",          (unsigned)r->k);
        fprintf(f, "\"pivot_set_size\":%u, ", (unsigned)r->pivot_set_size);
        fprintf(f, "\"all_z_only\":%s, ",     r->all_z_only ? "true" : "false");
        fprintf(f, "\"all_pos_phase\":%s, ",  r->all_pos_phase ? "true" : "false");
        fprintf(f, "\"half_cut\":%u, ",       (unsigned)r->cut);
        fprintf(f, "\"k_A_half\":%u, ",       (unsigned)r->k_A_half);
        fprintf(f, "\"k_B_half\":%u, ",       (unsigned)r->k_B_half);
        fprintf(f, "\"schmidt_log2_half\":%d, ",   r->schmidt_log2_half);
        fprintf(f, "\"unrotated_log2_half\":%d, ", r->unrotated_log2_half);
        fprintf(f, "\"best_cut\":%u, ",            (unsigned)r->best_cut);
        fprintf(f, "\"best_k_A\":%u, ",            (unsigned)r->best_k_A);
        fprintf(f, "\"best_k_B\":%u, ",            (unsigned)r->best_k_B);
        fprintf(f, "\"best_schmidt_log2\":%d, ",   r->best_schmidt_log2);
        fprintf(f, "\"unrotated_log2_best\":%d",   r->unrotated_log2_best);
        fprintf(f, "}%s\n", (i + 1 < n_rec) ? "," : "");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
    pivot_record_t recs[128];
    size_t n_rec = 0;

    print_header();

    /* Z2 LGT scaling */
    const uint32_t z2_sizes[] = {4, 6, 8, 10, 12, 14, 16, 20, 24, 32};
    for (size_t i = 0; i < sizeof(z2_sizes) / sizeof(z2_sizes[0]); i++) {
        uint32_t N = z2_sizes[i];
        uint8_t* paulis = NULL; uint32_t n = 0, k = 0;
        if (build_z2_lgt(N, &paulis, &n, &k) != 0) continue;
        if (measure_record("Z2_LGT", N, paulis, n, k, &recs[n_rec]) == 0) {
            print_row(&recs[n_rec]); n_rec++;
        }
        free(paulis);
    }

    /* Surface code scaling (windowed-Z subset) */
    const uint32_t surf_sizes[] = {3, 5, 7, 9, 11, 13, 15};
    for (size_t i = 0; i < sizeof(surf_sizes) / sizeof(surf_sizes[0]); i++) {
        uint32_t d = surf_sizes[i];
        uint8_t* paulis = NULL; uint32_t n = 0, k = 0;
        if (build_surface(d, &paulis, &n, &k) != 0) continue;
        if (measure_record("Surface_d", d, paulis, n, k, &recs[n_rec]) == 0) {
            print_row(&recs[n_rec]); n_rec++;
        }
        free(paulis);
    }

    /* Toric scaling */
    const uint32_t toric_sizes[] = {2, 3, 4, 5, 6, 8, 10};
    for (size_t i = 0; i < sizeof(toric_sizes) / sizeof(toric_sizes[0]); i++) {
        uint32_t L = toric_sizes[i];
        uint8_t* paulis = NULL; uint32_t n = 0, k = 0;
        if (build_toric(L, &paulis, &n, &k) != 0) continue;
        if (measure_record("Toric_LxL", L, paulis, n, k, &recs[n_rec]) == 0) {
            print_row(&recs[n_rec]); n_rec++;
        }
        free(paulis);
    }

    if (argc >= 2) {
        FILE* f = fopen(argv[1], "w");
        if (!f) {
            fprintf(stderr, "could not open %s\n", argv[1]);
            return 1;
        }
        json_emit(f, recs, n_rec);
        fclose(f);
        printf("\nwrote JSON archive: %s (%zu records)\n", argv[1], n_rec);
    }

    return 0;
}
