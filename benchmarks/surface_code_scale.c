/**
 * @file surface_code_scale.c
 * @brief Distance-d rotated surface-code memory benchmark (task #59).
 *
 * Builds the standard rotated surface-code Z-memory experiment at distance
 * d with d rounds of syndrome extraction and circuit-level depolarising
 * noise, then measures the two phases that Stim's 2021 paper separates:
 *
 *   ANALYSIS  -- the one-off reference/deterministic pass that fixes, per
 *                measurement, the noiseless outcome and whether it is
 *                random.  In Moonlab this is pf_compute_reference(), an
 *                Aaronson-Gottesman tableau run; it is replicated here
 *                through the public clifford_* API so it can be timed and
 *                bounded independently of the sampler.
 *   SAMPLING   -- the per-shot Pauli-frame pass, batched 64 shots to a
 *                machine word, which is what runs at kHz+ once analysis
 *                is done.
 *
 * Stim's headline claim (arXiv:2103.02202) is d = 100 (~20k qubits, ~8M
 * gates, ~1M measurements) analysed in ~15 s, then sampled at kHz rates.
 * This harness reproduces exactly that circuit family so the comparison is
 * like-for-like, and can dump the circuit in Stim's text format so both
 * engines run the *same* circuit rather than two independent constructions.
 *
 * LAYOUT (standard rotated surface code, odd d)
 * ---------------------------------------------
 *   data qubits    : d^2, at integer grid points (c, r), c,r in [0, d)
 *   ancilla qubits : d^2 - 1, at plaquette centres (ac + 0.5, ar + 0.5)
 *                    for ac, ar in [-1, d-1].  A plaquette's four corners
 *                    are (ac, ar), (ac+1, ar), (ac, ar+1), (ac+1, ar+1).
 *                    Type is the checkerboard parity: X if (ac+ar) even,
 *                    else Z.  Interior plaquettes (4 corners) are all
 *                    kept; edge plaquettes (2 corners) are kept only when
 *                    X sits on the top/bottom edge and Z on the left/right
 *                    edge.  That yields (d-1)^2 + 2(d-1) = d^2 - 1
 *                    ancillas, half X and half Z.
 *   total qubits   : 2d^2 - 1  (19999 at d = 100)
 *
 * SCHEDULE
 * --------
 * The hook-error-avoiding CNOT order: X ancillas take corners in the order
 * NW, NE, SW, SE ("Z" shape) and Z ancillas take NW, SW, NE, SE ("N"
 * shape).  The two orders are transposes of one another, which is what
 * makes each of the four layers collision-free (proved by the checkerboard
 * parity: a collision in layer 1 or 2 would force an X plaquette and a Z
 * plaquette to share a parity).
 *
 * NOISE (matches stim's generated rotated_memory_z with all four knobs = p)
 * ------------------------------------------------------------------------
 *   DEPOLARIZE1(p) on every data qubit before each round
 *   DEPOLARIZE1(p) after every single-qubit Clifford
 *   DEPOLARIZE2(p) after every CNOT
 *   X_ERROR(p)     after every reset, and before every measurement
 *
 * DETECTORS
 * ---------
 *   round 0        : one per Z ancilla (deterministic 0 on |0...0>)
 *   rounds 1..R-1  : one per ancilla, comparing round t to round t-1
 *   final          : one per Z ancilla, comparing its last syndrome to the
 *                    parity of its data qubits' terminal Z measurements
 *   total          : (d^2 - 1) * R
 *
 * All detectors read 0 on a noiseless run; that is the primary correctness
 * gate, checked by --verify.
 */

#include "../src/backends/clifford/clifford.h"
#include "../src/backends/clifford/pauli_frame.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/*  Timing / memory helpers                                            */
/* ------------------------------------------------------------------ */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Resident-set high-water mark in bytes, or 0 if unavailable. */
static double peak_rss_bytes(void) {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return (double)info.resident_size_max;
#endif
    return 0.0;
}

/* ------------------------------------------------------------------ */
/*  Rotated surface-code layout                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    int  d;              /* code distance (odd)                       */
    int  n_data;         /* d^2                                       */
    int  n_anc;          /* d^2 - 1                                   */
    int  n_qubits;       /* 2d^2 - 1                                  */
    /* per-ancilla data */
    int* anc_is_x;       /* 1 if X-type stabilizer, 0 if Z-type       */
    int* anc_corner;     /* 4 entries per ancilla; data index or -1   */
    int* anc_weight;     /* 2 or 4                                    */
} sc_layout_t;

/* Qubit numbering: data 0..d^2-1 (row-major, index = r*d + c),
 * ancillas d^2 .. 2d^2-2 in discovery order. */
static int sc_data_index(const sc_layout_t* L, int c, int r) {
    if (c < 0 || c >= L->d || r < 0 || r >= L->d) return -1;
    return r * L->d + c;
}

static void sc_layout_free(sc_layout_t* L) {
    if (!L) return;
    free(L->anc_is_x);
    free(L->anc_corner);
    free(L->anc_weight);
    memset(L, 0, sizeof(*L));
}

static int sc_layout_build(sc_layout_t* L, int d) {
    memset(L, 0, sizeof(*L));
    if (d < 3 || (d % 2) == 0) return -1;
    L->d = d;
    L->n_data = d * d;
    L->n_anc = d * d - 1;
    L->n_qubits = 2 * d * d - 1;

    L->anc_is_x   = (int*)calloc((size_t)L->n_anc, sizeof(int));
    L->anc_corner = (int*)malloc((size_t)L->n_anc * 4 * sizeof(int));
    L->anc_weight = (int*)calloc((size_t)L->n_anc, sizeof(int));
    if (!L->anc_is_x || !L->anc_corner || !L->anc_weight) {
        sc_layout_free(L);
        return -1;
    }

    int a = 0;
    for (int ar = -1; ar <= d - 1; ar++) {
        for (int ac = -1; ac <= d - 1; ac++) {
            int is_x = (((ac + ar) % 2) + 2) % 2 == 0;
            int corner[4];
            corner[0] = sc_data_index(L, ac,     ar);       /* NW */
            corner[1] = sc_data_index(L, ac + 1, ar);       /* NE */
            corner[2] = sc_data_index(L, ac,     ar + 1);   /* SW */
            corner[3] = sc_data_index(L, ac + 1, ar + 1);   /* SE */
            int w = 0;
            for (int k = 0; k < 4; k++) if (corner[k] >= 0) w++;

            int keep = 0;
            if (w == 4) {
                keep = 1;                       /* interior plaquette   */
            } else if (w == 2) {
                int on_vertical_edge   = (ac == -1 || ac == d - 1);
                int on_horizontal_edge = (ar == -1 || ar == d - 1);
                /* X stabilizers cap the top/bottom edges, Z the sides. */
                if (on_horizontal_edge && !on_vertical_edge && is_x)  keep = 1;
                if (on_vertical_edge   && !on_horizontal_edge && !is_x) keep = 1;
            }
            if (!keep) continue;

            if (a >= L->n_anc) { sc_layout_free(L); return -1; }
            L->anc_is_x[a]  = is_x;
            L->anc_weight[a] = w;
            for (int k = 0; k < 4; k++) L->anc_corner[a * 4 + k] = corner[k];
            a++;
        }
    }
    if (a != L->n_anc) { sc_layout_free(L); return -1; }
    return 0;
}

static int sc_anc_qubit(const sc_layout_t* L, int a) { return L->n_data + a; }

/* ------------------------------------------------------------------ */
/*  Circuit construction                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    pf_circuit_op_t* ops;
    size_t n_ops;
    size_t cap;

    size_t* det_offsets;   /* n_det + 1 */
    uint32_t* det_indices;
    size_t n_det;
    size_t det_cap;
    size_t det_idx_len;
    size_t det_idx_cap;

    uint32_t* obs_indices;  /* logical observable measurement set */
    size_t n_obs_idx;

    size_t n_meas;
    size_t n_gate_ops;      /* H/CNOT/M/R only, for the "gates" headline */
} sc_circuit_t;

static void sc_circuit_free(sc_circuit_t* C) {
    if (!C) return;
    free(C->ops);
    free(C->det_offsets);
    free(C->det_indices);
    free(C->obs_indices);
    memset(C, 0, sizeof(*C));
}

static int sc_push(sc_circuit_t* C, int kind, uint32_t q0, uint32_t q1, double p) {
    if (C->n_ops == C->cap) {
        size_t nc = C->cap ? C->cap * 2 : 4096;
        pf_circuit_op_t* np = (pf_circuit_op_t*)realloc(C->ops, nc * sizeof(*np));
        if (!np) return -1;
        C->ops = np;
        C->cap = nc;
    }
    pf_circuit_op_t* o = &C->ops[C->n_ops++];
    o->kind = (uint8_t)kind;
    o->q0 = q0;
    o->q1 = q1;
    o->p = p;
    if (kind == PF_OP_H || kind == PF_OP_CNOT || kind == PF_OP_MEASURE ||
        kind == PF_OP_RESET)
        C->n_gate_ops++;
    if (kind == PF_OP_MEASURE || kind == PF_OP_MEASURE_NOISY) C->n_meas++;
    return 0;
}

static int sc_det_begin(sc_circuit_t* C) {
    if (C->n_det + 2 > C->det_cap) {
        size_t nc = C->det_cap ? C->det_cap * 2 : 4096;
        size_t* np = (size_t*)realloc(C->det_offsets, (nc + 1) * sizeof(*np));
        if (!np) return -1;
        C->det_offsets = np;
        C->det_cap = nc;
    }
    if (C->n_det == 0) C->det_offsets[0] = 0;
    return 0;
}

static int sc_det_add(sc_circuit_t* C, uint32_t meas_index) {
    if (C->det_idx_len == C->det_idx_cap) {
        size_t nc = C->det_idx_cap ? C->det_idx_cap * 2 : 8192;
        uint32_t* np = (uint32_t*)realloc(C->det_indices, nc * sizeof(*np));
        if (!np) return -1;
        C->det_indices = np;
        C->det_idx_cap = nc;
    }
    C->det_indices[C->det_idx_len++] = meas_index;
    return 0;
}

static void sc_det_end(sc_circuit_t* C) {
    C->n_det++;
    C->det_offsets[C->n_det] = C->det_idx_len;
}

/**
 * Build the distance-d, R-round rotated surface-code Z-memory circuit.
 * p == 0 emits a noiseless circuit (no noise ops at all).
 */
static int sc_build(sc_circuit_t* C, const sc_layout_t* L, int rounds, double p) {
    memset(C, 0, sizeof(*C));
    const int A = L->n_anc;
    const int D = L->n_data;
    const int noisy = (p > 0.0);

    /* Corner visit order per layer: X takes NW,NE,SW,SE; Z takes NW,SW,NE,SE. */
    static const int xorder[4] = {0, 1, 2, 3};
    static const int zorder[4] = {0, 2, 1, 3};

#define PUSH(k, a, b, pp) do { if (sc_push(C, (k), (uint32_t)(a), (uint32_t)(b), (pp))) goto fail; } while (0)

    /* ---- Initialisation: reset every qubit to |0> ---- */
    for (int q = 0; q < L->n_qubits; q++) PUSH(PF_OP_RESET, q, 0, 0.0);
    if (noisy)
        for (int q = 0; q < L->n_qubits; q++) PUSH(PF_OP_X_ERROR, q, 0, p);

    /* ---- Syndrome-extraction rounds ---- */
    for (int t = 0; t < rounds; t++) {
        /* before_round_data_depolarization */
        if (noisy)
            for (int k = 0; k < D; k++) PUSH(PF_OP_DEPOLARIZE1, k, 0, p);

        /* H on X ancillas */
        for (int a = 0; a < A; a++) {
            if (!L->anc_is_x[a]) continue;
            PUSH(PF_OP_H, sc_anc_qubit(L, a), 0, 0.0);
            if (noisy) PUSH(PF_OP_DEPOLARIZE1, sc_anc_qubit(L, a), 0, p);
        }

        /* Four collision-free CNOT layers */
        for (int layer = 0; layer < 4; layer++) {
            for (int a = 0; a < A; a++) {
                const int k = L->anc_is_x[a] ? xorder[layer] : zorder[layer];
                const int dq = L->anc_corner[a * 4 + k];
                if (dq < 0) continue;
                const int aq = sc_anc_qubit(L, a);
                if (L->anc_is_x[a]) PUSH(PF_OP_CNOT, aq, dq, 0.0);
                else                PUSH(PF_OP_CNOT, dq, aq, 0.0);
                if (noisy) {
                    if (L->anc_is_x[a]) PUSH(PF_OP_DEPOLARIZE2, aq, dq, p);
                    else                PUSH(PF_OP_DEPOLARIZE2, dq, aq, p);
                }
            }
        }

        /* H back on X ancillas */
        for (int a = 0; a < A; a++) {
            if (!L->anc_is_x[a]) continue;
            PUSH(PF_OP_H, sc_anc_qubit(L, a), 0, 0.0);
            if (noisy) PUSH(PF_OP_DEPOLARIZE1, sc_anc_qubit(L, a), 0, p);
        }

        /* Measure ancillas (measurement index t*A + a) */
        if (noisy)
            for (int a = 0; a < A; a++) PUSH(PF_OP_X_ERROR, sc_anc_qubit(L, a), 0, p);
        for (int a = 0; a < A; a++) PUSH(PF_OP_MEASURE, sc_anc_qubit(L, a), 0, 0.0);

        /* Reset ancillas for reuse */
        for (int a = 0; a < A; a++) PUSH(PF_OP_RESET, sc_anc_qubit(L, a), 0, 0.0);
        if (noisy)
            for (int a = 0; a < A; a++) PUSH(PF_OP_X_ERROR, sc_anc_qubit(L, a), 0, p);
    }

    /* ---- Terminal data measurement (index rounds*A + k) ---- */
    if (noisy)
        for (int k = 0; k < D; k++) PUSH(PF_OP_X_ERROR, k, 0, p);
    for (int k = 0; k < D; k++) PUSH(PF_OP_MEASURE, k, 0, 0.0);

#undef PUSH

    /* ---- Detectors ---- */
    /* Round 0: Z ancillas only (X ancillas are random on |0...0>). */
    for (int a = 0; a < A; a++) {
        if (L->anc_is_x[a]) continue;
        if (sc_det_begin(C)) goto fail;
        if (sc_det_add(C, (uint32_t)a)) goto fail;
        sc_det_end(C);
    }
    /* Rounds 1..R-1: every ancilla, against the previous round. */
    for (int t = 1; t < rounds; t++) {
        for (int a = 0; a < A; a++) {
            if (sc_det_begin(C)) goto fail;
            if (sc_det_add(C, (uint32_t)(t * A + a))) goto fail;
            if (sc_det_add(C, (uint32_t)((t - 1) * A + a))) goto fail;
            sc_det_end(C);
        }
    }
    /* Final: Z ancillas against the parity of their data qubits. */
    for (int a = 0; a < A; a++) {
        if (L->anc_is_x[a]) continue;
        if (sc_det_begin(C)) goto fail;
        if (sc_det_add(C, (uint32_t)((rounds - 1) * A + a))) goto fail;
        for (int k = 0; k < 4; k++) {
            const int dq = L->anc_corner[a * 4 + k];
            if (dq < 0) continue;
            if (sc_det_add(C, (uint32_t)(rounds * A + dq))) goto fail;
        }
        sc_det_end(C);
    }

    /* ---- Logical observable: Z on one ROW of data qubits ----
     *
     * The chain must commute with every X stabilizer.  X plaquettes cap the
     * top and bottom edges, so a *vertical* Z chain overlaps each weight-2
     * boundary X stabilizer in exactly one qubit and anticommutes with it.
     * A *horizontal* chain overlaps every X plaquette -- interior or
     * boundary -- in zero or two qubits, and terminates on the left/right
     * edges where the Z stabilizers live.  That is the logical Z. */
    C->obs_indices = (uint32_t*)malloc((size_t)L->d * sizeof(uint32_t));
    if (!C->obs_indices) goto fail;
    for (int c = 0; c < L->d; c++)
        C->obs_indices[c] = (uint32_t)(rounds * A + sc_data_index(L, c, 0));
    C->n_obs_idx = (size_t)L->d;

    return 0;
fail:
    sc_circuit_free(C);
    return -1;
}

/* ------------------------------------------------------------------ */
/*  Stim-format dump (so both engines run the identical circuit)       */
/* ------------------------------------------------------------------ */

static int sc_dump_stim(const sc_circuit_t* C, const sc_layout_t* L,
                        int rounds, const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    (void)L; (void)rounds;
    for (size_t i = 0; i < C->n_ops; i++) {
        const pf_circuit_op_t* o = &C->ops[i];
        switch (o->kind) {
            case PF_OP_H:      fprintf(f, "H %u\n", o->q0); break;
            case PF_OP_S:      fprintf(f, "S %u\n", o->q0); break;
            case PF_OP_S_DAG:  fprintf(f, "S_DAG %u\n", o->q0); break;
            case PF_OP_X:      fprintf(f, "X %u\n", o->q0); break;
            case PF_OP_Y:      fprintf(f, "Y %u\n", o->q0); break;
            case PF_OP_Z:      fprintf(f, "Z %u\n", o->q0); break;
            case PF_OP_CNOT:   fprintf(f, "CX %u %u\n", o->q0, o->q1); break;
            case PF_OP_CZ:     fprintf(f, "CZ %u %u\n", o->q0, o->q1); break;
            case PF_OP_SWAP:   fprintf(f, "SWAP %u %u\n", o->q0, o->q1); break;
            case PF_OP_RESET:  fprintf(f, "R %u\n", o->q0); break;
            case PF_OP_MEASURE:fprintf(f, "M %u\n", o->q0); break;
            case PF_OP_X_ERROR:     fprintf(f, "X_ERROR(%g) %u\n", o->p, o->q0); break;
            case PF_OP_Z_ERROR:     fprintf(f, "Z_ERROR(%g) %u\n", o->p, o->q0); break;
            case PF_OP_Y_ERROR:     fprintf(f, "Y_ERROR(%g) %u\n", o->p, o->q0); break;
            case PF_OP_DEPOLARIZE1: fprintf(f, "DEPOLARIZE1(%g) %u\n", o->p, o->q0); break;
            case PF_OP_DEPOLARIZE2: fprintf(f, "DEPOLARIZE2(%g) %u %u\n", o->p, o->q0, o->q1); break;
            case PF_OP_MEASURE_NOISY: fprintf(f, "M(%g) %u\n", o->p, o->q0); break;
            default: break;
        }
    }
    /* Detectors are emitted after all measurements, so rec[] offsets are
     * measured back from the total measurement count. */
    const long T = (long)C->n_meas;
    for (size_t dd = 0; dd < C->n_det; dd++) {
        fprintf(f, "DETECTOR");
        for (size_t k = C->det_offsets[dd]; k < C->det_offsets[dd + 1]; k++)
            fprintf(f, " rec[%ld]", (long)C->det_indices[k] - T);
        fprintf(f, "\n");
    }
    fprintf(f, "OBSERVABLE_INCLUDE(0)");
    for (size_t k = 0; k < C->n_obs_idx; k++)
        fprintf(f, " rec[%ld]", (long)C->obs_indices[k] - T);
    fprintf(f, "\n");
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Analysis pass (Moonlab's reference/deterministic tableau run)      */
/* ------------------------------------------------------------------ */

typedef struct {
    int      completed;        /* 1 if the whole circuit was analysed   */
    double   wall_s;           /* time actually spent                   */
    size_t   meas_done;        /* measurements resolved before stopping */
    size_t   meas_total;       /* measurements in the circuit           */
    size_t   det_meas_done;    /* of those, deterministic-branch ones   */
    double   projected_full_s; /* extrapolation to the full circuit     */
    double   peak_rss_bytes;
} sc_analysis_t;

/**
 * Replicates pf_compute_reference() through the public clifford_* API so
 * the analysis phase can be timed and, critically, *bounded*: at large d
 * the pass does not terminate in any reasonable time, so we run a prefix
 * and report the measured per-measurement rate plus the projection.
 *
 * budget_s <= 0 means run to completion.
 */
static int sc_analysis_run(const sc_circuit_t* C, size_t n_qubits,
                           double budget_s, uint64_t seed,
                           uint8_t* m_ref, uint8_t* m_kind,
                           sc_analysis_t* out) {
    memset(out, 0, sizeof(*out));
    out->meas_total = C->n_meas;

    clifford_tableau_t* t = clifford_tableau_create(n_qubits);
    if (!t) return -1;

    /* Mirrors the Z-eigenstate cache in pf_compute_reference(); see the
     * comment there for why it is sound.  Set MOONLAB_SC_NO_ZCACHE=1 to
     * disable it and reproduce the pre-fix timings. */
    signed char* zknown = (signed char*)malloc(n_qubits);
    if (!zknown) { clifford_tableau_free(t); return -1; }
    memset(zknown, -1, n_qubits);
    const int use_cache = (getenv("MOONLAB_SC_NO_ZCACHE") == NULL);

    uint64_t rng = seed ? seed : 0xA5A5A5A5DEADBEEFULL;
    size_t mi = 0, det_ct = 0;
    const double t0 = now_s();
    double elapsed = 0.0;
    int stopped = 0;

    for (size_t i = 0; i < C->n_ops && !stopped; i++) {
        const pf_circuit_op_t* o = &C->ops[i];
        const uint32_t q0 = o->q0, q1 = o->q1;
        int outcome = 0, kind = 0;
        switch (o->kind) {
            case PF_OP_H:     clifford_h(t, q0); zknown[q0] = -1; break;
            case PF_OP_S:     clifford_s(t, q0); zknown[q0] = -1; break;
            case PF_OP_S_DAG: clifford_s_dag(t, q0); zknown[q0] = -1; break;
            case PF_OP_X:     clifford_x(t, q0); zknown[q0] = -1; break;
            case PF_OP_Y:     clifford_y(t, q0); zknown[q0] = -1; break;
            case PF_OP_Z:     clifford_z(t, q0); zknown[q0] = -1; break;
            case PF_OP_CNOT:  clifford_cnot(t, q0, q1);
                              zknown[q0] = -1; zknown[q1] = -1; break;
            case PF_OP_CZ:    clifford_cz(t, q0, q1);
                              zknown[q0] = -1; zknown[q1] = -1; break;
            case PF_OP_SWAP:  clifford_swap(t, q0, q1);
                              zknown[q0] = -1; zknown[q1] = -1; break;
            case PF_OP_RESET:
                if (use_cache && zknown[q0] >= 0) outcome = zknown[q0];
                else clifford_measure(t, q0, &rng, &outcome, &kind);
                if (outcome) clifford_x(t, q0);
                zknown[q0] = 0;
                break;
            case PF_OP_MEASURE:
            case PF_OP_MEASURE_NOISY:
                if (use_cache && zknown[q0] >= 0) { outcome = zknown[q0]; kind = 0; }
                else clifford_measure(t, q0, &rng, &outcome, &kind);
                zknown[q0] = (signed char)(outcome & 1);
                if (m_ref)  m_ref[mi]  = (uint8_t)(outcome & 1);
                if (m_kind) m_kind[mi] = (uint8_t)(kind & 1);
                if (!kind) det_ct++;
                mi++;
                /* Check the deadline every 64 measurements. */
                if (budget_s > 0.0 && (mi & 63u) == 0u) {
                    elapsed = now_s() - t0;
                    if (elapsed > budget_s) stopped = 1;
                }
                break;
            default: break; /* noise channels contribute nothing here */
        }
    }

    elapsed = now_s() - t0;
    free(zknown);
    clifford_tableau_free(t);

    out->wall_s = elapsed;
    out->meas_done = mi;
    out->det_meas_done = det_ct;
    out->completed = !stopped;
    out->peak_rss_bytes = peak_rss_bytes();
    if (stopped && mi > 0) {
        /* The deterministic tableau measurement is O(n^2) in the qubit
         * count and independent of position in the circuit, so a linear
         * projection in the measurement count is the right first-order
         * extrapolation.  It is a LOWER bound: tableau density grows as
         * the circuit proceeds. */
        out->projected_full_s = elapsed * (double)C->n_meas / (double)mi;
    } else {
        out->projected_full_s = elapsed;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Sampling-only pass (frame phase without the analysis phase)        */
/* ------------------------------------------------------------------ */

/**
 * Faithful replica of pauli_frame.c's pf_run_block() built from the public
 * pauli_frame_batch_* API, so the SAMPLING phase can be timed at distances
 * where the analysis phase does not terminate.
 *
 * Two deliberate deviations, both cost-neutral to within a few percent and
 * both calibrated at every d where the real sampler also runs (the JSON
 * reports replica / real as sampling_replica_calibration):
 *
 *   - DEPOLARIZE2(p) has no public batch entry point.  It is emulated by
 *     two single-qubit depolarising calls at the correct single-qubit
 *     marginal 8p/15.  Noise injection walks geometric gaps, so its cost is
 *     proportional to the expected number of hit shots, and 2 * 8p/15 =
 *     1.07p reproduces the real channel's p to within 7%.
 *   - RESET additionally refreshes the Z frame with random bits inside the
 *     real sampler; the public reset only clears.  That is a W-word fill per
 *     reset against the ~3W word ops of a CNOT, and the calibration factor
 *     absorbs it.
 *
 * Returns 0 on success and writes the elapsed seconds to *out_wall.
 */
static int sc_sampling_only(const sc_circuit_t* C, const sc_layout_t* L,
                            int shots, int threads, uint64_t seed,
                            double* out_wall, double* out_det_fraction) {
    const size_t nmeas = C->n_meas;
    int nthreads = threads;
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = omp_get_max_threads();
#else
    if (nthreads <= 0) nthreads = 1;
#endif
    if (nthreads > shots) nthreads = shots;
    if (nthreads < 1) nthreads = 1;

    const size_t base = (size_t)shots / (size_t)nthreads;
    const size_t rem  = (size_t)shots % (size_t)nthreads;

    /* Detector-major output, exactly as the real sampler writes it. */
    uint8_t* out = (uint8_t*)malloc(C->n_det * (size_t)shots);
    if (!out) return -1;

    int err = 0;
    const double t0 = now_s();

#ifdef _OPENMP
#   pragma omp parallel for num_threads(nthreads) schedule(static, 1) reduction(|:err)
#endif
    for (int tid = 0; tid < nthreads; tid++) {
        const size_t bs = base + ((size_t)tid < rem ? 1u : 0u);
        const size_t off = (size_t)tid * base + ((size_t)tid < rem ? (size_t)tid : rem);
        if (bs == 0) continue;

        pauli_frame_batch_t* b = pauli_frame_batch_create((size_t)L->n_qubits, bs);
        uint8_t* mbuf = (uint8_t*)malloc(nmeas * bs);
        if (!b || !mbuf) { pauli_frame_batch_free(b); free(mbuf); err |= 1; continue; }

        uint64_t rng = seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(tid + 1);
        size_t mi = 0;
        for (size_t i = 0; i < C->n_ops; i++) {
            const pf_circuit_op_t* o = &C->ops[i];
            switch (o->kind) {
                case PF_OP_H:    pauli_frame_batch_h(b, o->q0); break;
                case PF_OP_S:
                case PF_OP_S_DAG: pauli_frame_batch_s(b, o->q0); break;
                case PF_OP_X: case PF_OP_Y: case PF_OP_Z: break;
                case PF_OP_CNOT: pauli_frame_batch_cnot(b, o->q0, o->q1); break;
                case PF_OP_CZ:   pauli_frame_batch_cz(b, o->q0, o->q1); break;
                case PF_OP_SWAP: pauli_frame_batch_swap(b, o->q0, o->q1); break;
                case PF_OP_RESET: pauli_frame_batch_reset_zero(b, o->q0); break;
                case PF_OP_MEASURE:
                case PF_OP_MEASURE_NOISY:
                    pauli_frame_batch_measure_z(b, o->q0, mbuf + mi * bs);
                    mi++;
                    break;
                case PF_OP_X_ERROR:
                    pauli_frame_batch_bit_flip(b, o->q0, o->p, &rng); break;
                case PF_OP_Z_ERROR: case PF_OP_Y_ERROR:
                case PF_OP_DEPOLARIZE1:
                    pauli_frame_batch_depolarising(b, o->q0, o->p, &rng); break;
                case PF_OP_DEPOLARIZE2:
                    pauli_frame_batch_depolarising(b, o->q0, o->p * 8.0 / 15.0, &rng);
                    pauli_frame_batch_depolarising(b, o->q1, o->p * 8.0 / 15.0, &rng);
                    break;
                default: break;
            }
        }

        /* Reduce the block's measurement record to detectors. */
        for (size_t dd = 0; dd < C->n_det; dd++) {
            uint8_t* dst = out + dd * (size_t)shots + off;
            const size_t k0 = C->det_offsets[dd], k1 = C->det_offsets[dd + 1];
            if (k0 == k1) { memset(dst, 0, bs); continue; }
            const uint8_t* src = mbuf + (size_t)C->det_indices[k0] * bs;
            for (size_t s = 0; s < bs; s++) dst[s] = src[s];
            for (size_t k = k0 + 1; k < k1; k++) {
                const uint8_t* m = mbuf + (size_t)C->det_indices[k] * bs;
                for (size_t s = 0; s < bs; s++) dst[s] ^= m[s];
            }
        }
        pauli_frame_batch_free(b);
        free(mbuf);
    }

    *out_wall = now_s() - t0;

    if (!err && out_det_fraction) {
        size_t fired = 0;
        const size_t tot = C->n_det * (size_t)shots;
        for (size_t k = 0; k < tot; k++) fired += out[k] ? 1u : 0u;
        *out_det_fraction = (double)fired / (double)tot;
    }
    free(out);
    return err ? -1 : 0;
}

/* ------------------------------------------------------------------ */
/*  Driver                                                             */
/* ------------------------------------------------------------------ */

static void usage(const char* p) {
    fprintf(stderr,
        "usage: %s --d D [--rounds R] [--p P] [--shots N] [--seed S]\n"
        "          [--analysis-budget SEC] [--dump-stim PATH] [--verify]\n"
        "          [--threads T] [--json]\n"
        "\n"
        "  --d D                code distance (odd, >= 3); rounds default to d\n"
        "  --p P                circuit-level depolarising strength (default 0.001)\n"
        "  --shots N            shots for the sampling phase (default 1024)\n"
        "  --analysis-budget S  cap the analysis pass at S seconds and project\n"
        "                       the remainder (default 60; <=0 means no cap)\n"
        "  --dump-stim PATH     write the identical circuit in stim text format\n"
        "  --det-rates PATH     write per-detector fire counts, one per line\n"
        "  --skip-analysis      skip the analysis pass and time sampling only\n"
        "  --slope              re-run sampling at 2N shots and take the slope, so\n"
        "                       the sampling figure is free of the analysis term\n"
        "  --verify             run the noiseless circuit and assert every\n"
        "                       detector is quiet in every shot\n",
        p);
}

int main(int argc, char** argv) {
    int d = 5, rounds = -1, shots = 1024, threads = 0, verify = 0, want_json = 0;
    int slope_cal = 0;
    int skip_analysis = 0;
    double p = 0.001, budget = 60.0;
    uint64_t seed = 12345;
    const char* dump_stim = NULL;
    const char* det_rates_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--d") && i + 1 < argc) d = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--p") && i + 1 < argc) p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--shots") && i + 1 < argc) shots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--analysis-budget") && i + 1 < argc) budget = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dump-stim") && i + 1 < argc) dump_stim = argv[++i];
        else if (!strcmp(argv[i], "--det-rates") && i + 1 < argc) det_rates_path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slope")) slope_cal = 1;
        else if (!strcmp(argv[i], "--skip-analysis")) skip_analysis = 1;
        else if (!strcmp(argv[i], "--verify")) verify = 1;
        else if (!strcmp(argv[i], "--json")) want_json = 1;
        else { usage(argv[0]); return 2; }
    }
    if (rounds < 0) rounds = d;

    sc_layout_t L;
    if (sc_layout_build(&L, d)) {
        fprintf(stderr, "error: distance %d is not a valid odd distance >= 3\n", d);
        return 2;
    }

    sc_circuit_t C;
    if (sc_build(&C, &L, rounds, verify ? 0.0 : p)) {
        fprintf(stderr, "error: circuit construction failed\n");
        sc_layout_free(&L);
        return 1;
    }

    if (dump_stim && sc_dump_stim(&C, &L, rounds, dump_stim)) {
        fprintf(stderr, "error: could not write %s\n", dump_stim);
        sc_circuit_free(&C); sc_layout_free(&L);
        return 1;
    }

    if (!want_json) {
        fprintf(stderr,
            "d=%d rounds=%d qubits=%d data=%d anc=%d ops=%zu gates=%zu "
            "meas=%zu det=%zu p=%g\n",
            d, rounds, L.n_qubits, L.n_data, L.n_anc,
            C.n_ops, C.n_gate_ops, C.n_meas, C.n_det, verify ? 0.0 : p);
    }

    /* ---- Phase 1: analysis ---- */
    uint8_t* m_ref  = (uint8_t*)calloc(C.n_meas ? C.n_meas : 1, 1);
    uint8_t* m_kind = (uint8_t*)calloc(C.n_meas ? C.n_meas : 1, 1);
    sc_analysis_t an;
    memset(&an, 0, sizeof(an));
    an.meas_total = C.n_meas;
    if (skip_analysis) {
        /* Sampling-phase-only run: the analysis pass is measured separately
         * (it costs minutes at large d) and its result is not needed by the
         * replica, which supplies its own all-zero reference. */
        if (!m_ref || !m_kind) {
            fprintf(stderr, "error: allocation failed\n");
            free(m_ref); free(m_kind); sc_circuit_free(&C); sc_layout_free(&L);
            return 1;
        }
    } else if (!m_ref || !m_kind ||
        sc_analysis_run(&C, (size_t)L.n_qubits, budget, seed, m_ref, m_kind, &an)) {
        fprintf(stderr, "error: analysis pass failed\n");
        free(m_ref); free(m_kind); sc_circuit_free(&C); sc_layout_free(&L);
        return 1;
    }

    /* ---- Phase 2: sampling (only meaningful if analysis completed) ---- */
    double sample_wall = 0.0, total_wall = 0.0, shots_per_s = 0.0;
    double det_fraction = -1.0;
    long   sample_rc = 0;
    int    sampled = 0;
    double sample_peak_rss = 0.0;
    int    verify_pass = -1;

    if (an.completed && shots > 0) {
        /* Detector output is one byte per (detector, shot). */
        const double bytes = (double)C.n_det * (double)shots;
        if (bytes < 32e9) {
            uint8_t* out = (uint8_t*)malloc((size_t)C.n_det * (size_t)shots);
            if (out) {
                const double t0 = now_s();
                sample_rc = pauli_frame_batch_sample_detectors(
                    (size_t)L.n_qubits, C.ops, C.n_ops,
                    C.det_offsets, C.det_indices, C.n_det,
                    (size_t)shots, seed, threads, out);
                total_wall = now_s() - t0;
                sample_peak_rss = peak_rss_bytes();
                if (sample_rc > 0) {
                    sampled = 1;
                    /* The published call bundles analysis + sampling.
                     * Subtracting the separately measured analysis is a
                     * difference of two large numbers and is far too noisy
                     * once analysis dominates, so --slope re-runs the call
                     * at 2N shots and takes the slope, which cancels the
                     * analysis term exactly. */
                    sample_wall = total_wall - an.wall_s;
                    if (slope_cal) {
                        uint8_t* out2 = (uint8_t*)malloc((size_t)C.n_det * (size_t)shots * 2u);
                        if (out2) {
                            const double t1 = now_s();
                            long rc2 = pauli_frame_batch_sample_detectors(
                                (size_t)L.n_qubits, C.ops, C.n_ops,
                                C.det_offsets, C.det_indices, C.n_det,
                                (size_t)shots * 2u, seed, threads, out2);
                            const double dt2 = now_s() - t1;
                            if (rc2 > 0 && dt2 > total_wall) sample_wall = dt2 - total_wall;
                            free(out2);
                        }
                    }
                    if (sample_wall <= 0.0) sample_wall = total_wall;
                    shots_per_s = (double)shots / sample_wall;

                    size_t fired = 0;
                    const size_t tot = (size_t)C.n_det * (size_t)shots;
                    for (size_t k = 0; k < tot; k++) fired += out[k] ? 1u : 0u;
                    det_fraction = (double)fired / (double)tot;
                    if (verify) verify_pass = (fired == 0);

                    /* Per-detector fire counts: the ground-truth gate against
                     * stim running this identical circuit. */
                    if (det_rates_path) {
                        FILE* rf = fopen(det_rates_path, "w");
                        if (rf) {
                            for (size_t dd = 0; dd < C.n_det; dd++) {
                                size_t cnt = 0;
                                const uint8_t* row = out + dd * (size_t)shots;
                                for (int s = 0; s < shots; s++) cnt += row[s] ? 1u : 0u;
                                fprintf(rf, "%zu\n", cnt);
                            }
                            fclose(rf);
                        }
                    }
                }
                free(out);
            }
        }
    }

    /* ---- Phase 2b: sampling-only replica ----
     * Runs regardless of whether the analysis pass completed, which is the
     * only way to get a sampling number at the distances where analysis
     * does not terminate.  Where phase 2 also ran, the two are compared so
     * the replica carries a stated calibration rather than an assumption. */
    double replica_wall = 0.0, replica_shots_per_s = 0.0, replica_det_fraction = -1.0;
    double replica_calibration = -1.0;
    int replica_ran = 0;
    if (shots > 0 && (double)C.n_det * (double)shots < 32e9) {
        double w = 0.0, dfrac = -1.0;
        if (sc_sampling_only(&C, &L, shots, threads, seed, &w, &dfrac) == 0) {
            replica_ran = 1;
            replica_wall = w;
            replica_shots_per_s = (double)shots / w;
            replica_det_fraction = dfrac;
            if (sampled && sample_wall > 0.0) replica_calibration = w / sample_wall;
        }
    }

    if (want_json) {
        printf("{\n");
        printf("  \"engine\": \"moonlab\",\n");
        printf("  \"d\": %d,\n", d);
        printf("  \"rounds\": %d,\n", rounds);
        printf("  \"p\": %.10g,\n", verify ? 0.0 : p);
        printf("  \"n_qubits\": %d,\n", L.n_qubits);
        printf("  \"n_data\": %d,\n", L.n_data);
        printf("  \"n_ancilla\": %d,\n", L.n_anc);
        printf("  \"n_ops\": %zu,\n", C.n_ops);
        printf("  \"n_gates\": %zu,\n", C.n_gate_ops);
        printf("  \"n_measurements\": %zu,\n", C.n_meas);
        printf("  \"n_detectors\": %zu,\n", C.n_det);
        printf("  \"simd_backend\": \"%s\",\n", pauli_frame_simd_backend());
        printf("  \"simd_lanes\": %d,\n", pauli_frame_simd_lanes());
        printf("  \"analysis\": {\n");
        printf("    \"completed\": %s,\n", an.completed ? "true" : "false");
        printf("    \"wall_s\": %.9g,\n", an.wall_s);
        printf("    \"measurements_resolved\": %zu,\n", an.meas_done);
        printf("    \"deterministic_measurements\": %zu,\n", an.det_meas_done);
        printf("    \"projected_full_s\": %.9g,\n", an.projected_full_s);
        printf("    \"peak_rss_bytes\": %.0f\n", an.peak_rss_bytes);
        printf("  },\n");
        printf("  \"sampling\": {\n");
        printf("    \"ran\": %s,\n", sampled ? "true" : "false");
        printf("    \"shots\": %d,\n", shots);
        printf("    \"threads\": %d,\n", threads);
        printf("    \"total_call_wall_s\": %.9g,\n", total_wall);
        printf("    \"sampling_only_wall_s\": %.9g,\n", sample_wall);
        printf("    \"shots_per_s\": %.9g,\n", shots_per_s);
        printf("    \"detector_fraction\": %.9g,\n", det_fraction);
        printf("    \"peak_rss_bytes\": %.0f\n", sample_peak_rss);
        printf("  },\n");
        printf("  \"sampling_replica\": {\n");
        printf("    \"ran\": %s,\n", replica_ran ? "true" : "false");
        printf("    \"wall_s\": %.9g,\n", replica_wall);
        printf("    \"shots_per_s\": %.9g,\n", replica_shots_per_s);
        printf("    \"detector_fraction\": %.9g,\n", replica_det_fraction);
        printf("    \"calibration_replica_over_real\": %.9g\n", replica_calibration);
        printf("  },\n");
        printf("  \"verify\": { \"requested\": %s, \"all_detectors_quiet\": %s }\n",
               verify ? "true" : "false",
               verify_pass < 0 ? "null" : (verify_pass ? "true" : "false"));
        printf("}\n");
    } else {
        fprintf(stderr, "analysis: %s %.3f s (%zu/%zu meas, %zu deterministic) "
                        "projected_full=%.3f s peak_rss=%.1f MB\n",
                an.completed ? "COMPLETE" : "TIMEOUT",
                an.wall_s, an.meas_done, an.meas_total, an.det_meas_done,
                an.projected_full_s, an.peak_rss_bytes / 1e6);
        if (sampled)
            fprintf(stderr, "sampling: %d shots in %.3f s -> %.1f shots/s "
                            "(detector fraction %.5f) peak_rss=%.1f MB\n",
                    shots, sample_wall, shots_per_s, det_fraction,
                    sample_peak_rss / 1e6);
        if (replica_ran)
            fprintf(stderr, "sampling(replica): %d shots in %.3f s -> %.1f shots/s "
                            "(detector fraction %.5f, calibration %.3f)\n",
                    shots, replica_wall, replica_shots_per_s,
                    replica_det_fraction, replica_calibration);
        if (verify)
            fprintf(stderr, "verify: %s\n",
                    verify_pass < 0 ? "NOT RUN" : (verify_pass ? "PASS" : "FAIL"));
    }

    free(m_ref); free(m_kind);
    sc_circuit_free(&C);
    sc_layout_free(&L);
    return (verify && verify_pass == 0) ? 1 : 0;
}
