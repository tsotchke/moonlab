/**
 * @file shor_ecdlp.c
 * @brief Logical and FTQC resource estimation for Shor-ECDLP.
 *
 * The logical-layer model composes the arithmetic cost of elliptic-
 * curve point addition inside a quantum phase estimation circuit.
 * The arithmetic block counts follow Roetteler et al. (2017)
 * refined with Gidney windowed-arithmetic (2019) and the GDB 2026
 * optimisations.  All formulas are reproduced here in a single
 * file so each numerical choice is auditable.
 *
 * STRUCTURAL COST MODEL (per QPE iteration)
 * -----------------------------------------
 * Shor-ECDLP uses two QPE-style registers: (i) a control register of
 * width m = 2n bits accumulating the phase, (ii) a point-coordinate
 * register holding (x, y) in affine or projective form.  Each control
 * bit drives one controlled point-addition.  A single controlled
 * point-addition on an n-bit curve costs:
 *
 *   Roetteler 2017 (standard, qubit-minimal):
 *     Toffolis:  C_add(n) = 224 n^2 log2 n
 *     Qubits:    Q(n)     = 9 n + 2 log2 n + 10
 *
 *   Gidney windowed arithmetic with window w:
 *     Toffolis:  C_add(n, w) ≈ 56 n^2 (2 + 1/w) log2 n + O(n^2)
 *     Qubits:    Q(n, w)     ≈ 9 n + (2^w / w) n / log2 n + 2 log2 n + 10
 *
 *   Gidney-Drake-Boneh 2026 (ECDLP-specialised, qubit-minimal):
 *     Toffolis:  C_add(n)   ≈ 1400 n                (at n=256 -> ~358k per add)
 *     Qubits:    Q(n)       ≈ 4.7 n                 (at n=256 -> ~1200)
 *     Aggregate over 2n controlled adds gives the headline ~90 M Toffolis,
 *     ~1200 qubits at secp256k1.
 *
 *   Gidney-Drake-Boneh 2026 (depth-minimal):
 *     Toffolis:  C_add(n)   ≈ 1100 n                (at n=256 -> ~280k per add)
 *     Qubits:    Q(n)       ≈ 5.7 n                 (at n=256 -> ~1450)
 *     Aggregate ~70 M Toffolis, ~1450 qubits.
 *
 * A full Shor-ECDLP QPE run does 2n controlled point-additions plus
 * classical post-processing (continued fractions or lattice reduction);
 * the Toffoli budget is therefore (2n) * C_add(n).
 *
 * The success probability model tracks two independent effects:
 * (i) the intrinsic Shor success probability on recovery (>= 4/pi^2
 *     per measurement in the period-finding variant; GDB report ~0.4
 *     for ECDLP at n=256 with standard post-processing), and
 * (ii) the decoherence / gate-error budget given the total gate count
 *     times the per-gate physical error, projected through the
 *     surface-code logical-error function in shor_ecdlp_ftqc_estimate.
 */

#include "shor_ecdlp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -------- helpers --------------------------------------------------- */

static double log2_d(double x) {
    return log(x) / log(2.0);
}

/* Toffoli count per controlled point-addition, parameterised by encoding.
 * Values calibrated so the aggregate (2n * C_add) matches published
 * paper totals at n = 256. */
static double toffolis_per_add(const shor_ecdlp_params_t* p) {
    double n = (double)p->curve_bits;
    switch (p->encoding) {
        case 0: /* Roetteler 2017: O(n^2 log n) per modular add */
            return 224.0 * n * n * log2_d(n);
        case 1: /* Gidney-Drake-Boneh 2026, windowed modular arithmetic.
                 * The full Shor-ECDLP primitive is O(n^3) in Toffolis
                 * (n point additions, each O(n^2) modular arithmetic
                 * via windowed adders).  We write that as 2n
                 * controlled-adds each of cost proportional to n^2,
                 * calibrated so that 2n * C_add(256) reproduces the
                 * paper's Table 1 values of 90e6 (qubit-minimal) or
                 * 70e6 (depth-minimal).  The previous formula used
                 * linear-in-n scaling inside the add, which matched
                 * n = 256 by calibration but missed O(n^3) at any
                 * other width (Chen/Drake-Boneh arXiv:2603.28846
                 * Sec. 5.2 gives the cubic scaling explicitly).  */
            {
                const double ratio = n / 256.0;
                const double calib = (p->time_space_tradeoff == 0) ? 90.0e6 : 70.0e6;
                /* total = 2n * C_add(n) = calib * (n/256)^3.
                 * Dividing by 2n gives C_add(n). */
                return (calib / (2.0 * 256.0)) * ratio * ratio;
            }
        default:
            return 224.0 * n * n * log2_d(n);
    }
}

/* Qubit count, parameterised by encoding. */
static double qubits_logical(const shor_ecdlp_params_t* p) {
    double n = (double)p->curve_bits;
    switch (p->encoding) {
        case 0: {
            /* Roetteler 2017: ~9n + 2 log2 n + 10 */
            double w = (p->window_bits > 0) ? (double)p->window_bits : 4.0;
            double window_qubits = (pow(2.0, w) / w) * n / log2_d(n);
            return 9.0 * n + window_qubits + 2.0 * log2_d(n) + 10.0;
        }
        case 1:
            /* Gidney-Drake-Boneh 2026 calibrated */
            return (p->time_space_tradeoff == 0)
                   ? (1200.0 / 256.0) * n      /* ~4.69 n */
                   : (1450.0 / 256.0) * n;     /* ~5.66 n */
        default:
            return 9.0 * n + 2.0 * log2_d(n) + 10.0;
    }
}

/* -------- public: logical estimate ---------------------------------- */

int shor_ecdlp_estimate(const shor_ecdlp_params_t* p,
                        shor_ecdlp_resources_t* out) {
    if (!p || !out) return -1;
    if (p->curve_bits < 4 || p->curve_bits > 4096) return -2;
    if (p->encoding != 0 && p->encoding != 1) return -3;

    double n = (double)p->curve_bits;
    double num_controlled_adds = 2.0 * n;

    double tof_per_add = toffolis_per_add(p);
    double tof_total   = num_controlled_adds * tof_per_add;

    out->logical_qubits           = (size_t)ceil(qubits_logical(p));
    out->toffoli_count            = (uint64_t)tof_total;
    /* Standard T-decomposition: 1 Toffoli = 7 T-gates (or 4 via magic
     * state distillation; we report the 7T form as the worst case). */
    out->t_count                  = 7ULL * out->toffoli_count;
    /* Each Toffoli decomposes to ~6 CNOTs in the standard network. */
    out->cnot_count               = 6ULL * out->toffoli_count;
    /* One measurement per control-register bit. */
    out->measurement_count        = (uint64_t)(2.0 * n);
    /* Serial circuit depth in Toffoli layers, assuming the GDB depth-
     * minimal encoding parallelises by a factor of ~n and the standard
     * one parallelises by a factor of ~log n. */
    double parallelism = (p->time_space_tradeoff == 1) ? n : log2_d(n);
    out->circuit_depth_toffolis   = tof_total / parallelism;
    /* Intrinsic Shor-ECDLP success probability on a single run, per
     * Roetteler 2017 / GDB 2026; classical lattice-reduction
     * post-processing recovers k from ~3-5 independent runs. */
    out->success_probability      = 0.40;

    return 0;
}

/* -------- public: FTQC overhead ------------------------------------- */

/* Fowler-Mariantoni-Martinis-Cleland logical-error-per-cycle for the
 * planar surface code at distance d and physical error rate p:
 *   P_L(d, p) = A * (p / p_th)^((d+1)/2)
 * with A ~ 0.03 and p_th ~ 0.01 under depolarising noise.  We use the
 * standard approximation; a real deployment would fit this curve to
 * measured hardware data. */
static double surface_code_per_cycle_error(double p_phys, size_t d) {
    const double A = 0.03;
    const double p_th = 0.01;
    if (p_phys >= p_th) return 1.0;
    return A * pow(p_phys / p_th, ((double)d + 1.0) / 2.0);
}

/* Minimum d so per-cycle error * total-cycles <= target_logical_error. */
static size_t pick_code_distance(double p_phys,
                                 double total_cycles,
                                 double target_err) {
    for (size_t d = 3; d <= 127; d += 2) {
        double per_cycle = surface_code_per_cycle_error(p_phys, d);
        if (per_cycle * total_cycles < target_err) return d;
    }
    return 127;
}

int shor_ecdlp_ftqc_estimate(const shor_ecdlp_resources_t* logical,
                             const shor_ecdlp_ftqc_params_t* ftqc,
                             shor_ecdlp_ftqc_resources_t* out) {
    if (!logical || !ftqc || !out) return -1;
    if (ftqc->physical_error_rate <= 0.0 ||
        ftqc->physical_error_rate >= 1.0) return -2;
    if (ftqc->code_cycle_time_s <= 0.0) return -3;

    /* Total logical cycles = circuit depth * surface-code timing;
     * treat one logical Toffoli layer as (d) surface-code cycles. */
    double tof_depth = logical->circuit_depth_toffolis;
    /* Auto-pick or use the provided distance. */
    size_t d = ftqc->code_distance;
    if (d == 0) {
        double approx_cycles = tof_depth * 15.0; /* d~15 initial guess */
        double tgt = (ftqc->target_logical_error > 0.0)
                     ? ftqc->target_logical_error : 1e-2;
        d = pick_code_distance(ftqc->physical_error_rate, approx_cycles, tgt);
    }
    if (d < 3) d = 3;
    if ((d & 1u) == 0) d += 1; /* surface code wants odd distance */

    double per_cycle_err = surface_code_per_cycle_error(
        ftqc->physical_error_rate, d);
    double total_cycles  = tof_depth * (double)d;
    double total_err     = per_cycle_err * total_cycles;

    /* Physical qubits: each logical qubit costs 2 d^2 for the patch,
     * plus ~1 magic-state factory per 100 Toffolis at 15 d^2 each.
     * These are loose defaults; GDB give tighter numbers for their
     * specific factory design. */
    double data_patches   = 2.0 * (double)d * (double)d;
    double data_phys      = data_patches * (double)logical->logical_qubits;
    double factory_phys   = 15.0 * (double)d * (double)d;
    /* Pipeline: factories are consumed in parallel with the data; the
     * peak physical-qubit count is what matters.  We assume ~10
     * factories operating concurrently at any time; the total Toffoli
     * count governs how many sequential factory-consumption rounds
     * happen but not the peak. */
    double concurrent_factories = 10.0;
    double factory_peak   = concurrent_factories * factory_phys;

    out->picked_code_distance    = d;
    out->physical_qubits         = (size_t)(data_phys + factory_peak);
    out->wall_clock_seconds      = total_cycles * ftqc->code_cycle_time_s;
    out->logical_error_per_gate  = per_cycle_err;
    out->total_logical_error     = total_err;

    return 0;
}

/* -------- named-curve helpers --------------------------------------- */

void shor_ecdlp_params_secp256k1(shor_ecdlp_params_t* p) {
    if (!p) return;
    p->curve_bits = 256;
    p->window_bits = 4;
    p->encoding = 1;             /* GDB 2026 */
    p->time_space_tradeoff = 0;  /* qubit-minimal: 1200 qubits, 90M Toffolis */
}

void shor_ecdlp_params_p256(shor_ecdlp_params_t* p) {
    if (!p) return;
    p->curve_bits = 256;
    p->window_bits = 4;
    p->encoding = 1;
    p->time_space_tradeoff = 0;
}

void shor_ecdlp_params_curve25519(shor_ecdlp_params_t* p) {
    if (!p) return;
    p->curve_bits = 255;
    p->window_bits = 4;
    p->encoding = 1;
    p->time_space_tradeoff = 0;
}
