/**
 * @file shor_ecdlp.h
 * @brief Shor's algorithm for the elliptic-curve discrete-logarithm problem:
 *        circuit synthesis and logical-layer resource estimation.
 *
 * OVERVIEW
 * --------
 * Given an elliptic-curve point @f$P@f$ of order @f$r@f$ and a target
 * point @f$Q = kP@f$ for an unknown scalar @f$k \in \mathbb Z_r@f$,
 * Shor's algorithm for ECDLP recovers @f$k@f$ with resources polynomial
 * in @f$\log_2 r@f$ by quantum-phase-estimating the eigenphase of the
 * point-addition operator.  The canonical construction is due to
 * Proos and Zalka (2003); the modern cost analysis used here is Roetteler
 * et al. (2017) with the windowed-arithmetic improvements of Gidney
 * (2019) and the ECDLP-specific optimisations of Gidney, Drake and Boneh
 * (2026).  The Gidney-Drake-Boneh paper's headline numbers for
 * @f$\mathrm{secp256k1}@f$ are ≤1200 logical qubits at ~90 M Toffoli
 * gates, or ≤1450 logical qubits at ~70 M Toffoli gates depending on
 * the space / time trade-off; these are the reference points the
 * implementation here reproduces.
 *
 * ROLE WITHIN MOONLAB
 * -------------------
 * Moonlab is a simulator; it cannot execute the full 1200-qubit Shor
 * circuit end-to-end on classical hardware.  What it *can* do, and
 * what this module delivers:
 *
 *  - @em Circuit synthesis: emit the Shor-ECDLP circuit at a given
 *    curve bit-width in a native IR, exportable to OpenQASM 3 or
 *    QIR.  Toy bit-widths (8, 16, 20) run on the 32-qubit state-vector
 *    simulator for correctness validation before any quantum execution.
 *  - @em Logical resource estimation: exact Toffoli / CNOT / T / qubit
 *    counts and circuit depth at any curve bit-width, with success
 *    probability as a function of decoherence budget.  Enables honest
 *    secp256k1 / Ed25519 / Curve25519 resource estimates.
 *  - @em FTQC overhead: map logical resources to physical-qubit counts
 *    and wall-clock at a given code distance and physical error rate
 *    using the Fowler-Martinis-Mariantoni-Cleland surface-code model.
 *
 * When deployed as the estimator tier of a distributed quantum compute
 * federation (orchestrated externally by QGTL), this module's outputs
 * feed the scheduler's node-assignment and code-distance decisions.
 *
 * WHAT THIS MODULE DELIBERATELY EXCLUDES
 * --------------------------------------
 * No attempt is made to target any specific third-party public key or
 * on-chain address.  Toy-curve demonstrations generate their own
 * keypairs synthetically; resource estimates for standard curves
 * (secp256k1, P-256, Curve25519) are purely structural.  A production
 * user responsible for their own wallet security who wishes to run this
 * against their own public key must supply the key explicitly through
 * the user-facing API and is responsible for the consent framing.
 *
 * REFERENCES
 * ----------
 *  - J. Proos and C. Zalka, "Shor's discrete logarithm quantum
 *    algorithm for elliptic curves", Quantum Inf. Comput. 3, 317
 *    (2003), arXiv:quant-ph/0301141.  Original Shor-ECDLP construction.
 *  - M. Roetteler, M. Naehrig, K. M. Svore and K. Lauter, "Quantum
 *    Resource Estimates for Computing Elliptic Curve Discrete
 *    Logarithms", ASIACRYPT 2017, arXiv:1706.06752.  Logical resource
 *    counts we reproduce at the arithmetic-block level.
 *  - C. Gidney, "Windowed quantum arithmetic",
 *    arXiv:1905.07682 (2019).  Windowed-arithmetic Toffoli reduction.
 *  - C. Gidney, N. Drake and D. Boneh, "Securing Elliptic Curve
 *    Cryptocurrencies against Quantum Vulnerabilities",
 *    arXiv:2603.28846 / IACR eprint 2026/625 (2026).  secp256k1-targeted
 *    ECDLP cost model; target of the reference numbers.
 *  - A. G. Fowler, M. Mariantoni, J. M. Martinis and A. N. Cleland,
 *    "Surface codes: Towards practical large-scale quantum
 *    computation", Phys. Rev. A 86, 032324 (2012), arXiv:1208.0928.
 *    FTQC overhead model for logical-to-physical mapping.
 *
 * @since  v0.2.0
 * @stability evolving
 */

#ifndef MOONLAB_SHOR_ECDLP_H
#define MOONLAB_SHOR_ECDLP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Logical-layer parameters ------------------------------------------- */

typedef struct {
    size_t curve_bits;           /**< n = log2(curve order); 256 for secp256k1 */
    size_t window_bits;          /**< windowed-arithmetic window; 4 is Gidney default */
    int    encoding;             /**< 0 = Roetteler 2017, 1 = Gidney-Drake-Boneh 2026 */
    int    time_space_tradeoff;  /**< 0 = qubit-minimal, 1 = depth-minimal (GDB variants) */
} shor_ecdlp_params_t;

typedef struct {
    size_t   logical_qubits;
    uint64_t toffoli_count;
    uint64_t cnot_count;
    uint64_t t_count;            /**< =7*toffoli_count under standard T-decomposition */
    uint64_t measurement_count;
    double   circuit_depth_toffolis;
    double   success_probability;
} shor_ecdlp_resources_t;

/**
 * @brief Logical-layer resource estimate for Shor-ECDLP at the given
 *        curve bit-width.
 *
 * The model reproduces the Gidney-Drake-Boneh 2026 secp256k1 numbers:
 *  - encoding = 1, time_space_tradeoff = 0: ~1200 qubits, ~90 M Toffolis.
 *  - encoding = 1, time_space_tradeoff = 1: ~1450 qubits, ~70 M Toffolis.
 *
 * @param p     parameters
 * @param out   result sink; must be non-NULL
 * @return 0 on success, non-zero on invalid arguments.
 */
int shor_ecdlp_estimate(const shor_ecdlp_params_t* p,
                        shor_ecdlp_resources_t* out);

/* --- FTQC overhead ------------------------------------------------------ */

typedef struct {
    double physical_error_rate;   /**< per-physical-gate error; 1e-3 is aspirational */
    double code_cycle_time_s;     /**< physical surface-code cycle; 1e-6 is typical */
    size_t code_distance;         /**< surface-code distance; 0 = auto-pick for target err */
    double target_logical_error;  /**< acceptable logical error for the whole computation */
} shor_ecdlp_ftqc_params_t;

typedef struct {
    size_t picked_code_distance;  /**< = code_distance if set, else auto */
    size_t physical_qubits;
    double wall_clock_seconds;
    double logical_error_per_gate;
    double total_logical_error;
} shor_ecdlp_ftqc_resources_t;

/**
 * @brief Fault-tolerant overhead model mapping logical resources to
 *        physical qubits and wall-clock.
 *
 * Uses the Fowler-Mariantoni-Martinis-Cleland planar surface-code
 * scaling: per-logical-qubit physical-qubit cost
 * @f$\approx 2\,d^{2}@f$ for a distance-@f$d@f$ code, plus a
 * magic-state-factory multiplier for the T-state consumption rate
 * (@f$\approx 15d^{2}@f$ physical qubits per factory, one factory per
 * 100 Toffolis is a loose default).
 */
int shor_ecdlp_ftqc_estimate(const shor_ecdlp_resources_t* logical,
                             const shor_ecdlp_ftqc_params_t* ftqc,
                             shor_ecdlp_ftqc_resources_t* out);

/* --- Named-curve convenience -------------------------------------------- */

/** Populate @p p with the parameters for secp256k1 and the
 *  GDB qubit-minimal encoding. */
void shor_ecdlp_params_secp256k1(shor_ecdlp_params_t* p);

/** Populate @p p with the parameters for NIST P-256. */
void shor_ecdlp_params_p256(shor_ecdlp_params_t* p);

/** Populate @p p with the parameters for Curve25519 / Ed25519. */
void shor_ecdlp_params_curve25519(shor_ecdlp_params_t* p);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_SHOR_ECDLP_H */
