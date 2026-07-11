/**
 * @file moonlab_export.h
 * @brief Stable C ABI surface for Moonlab (libquantumsim) consumers.
 *
 * This header defines the committed, versioned, downstream-facing ABI for
 * Moonlab. Downstream libraries (notably QGTL,
 * github.com:tsotchke/quantum_geometric_tensor) pin against the symbols
 * declared here and locate them at runtime via dlsym.
 *
 * ABI CONTRACT
 * ------------
 *  - Every symbol declared in this header is guaranteed to exist, with the
 *    same name and signature, across all 0.x releases. Semantic upgrades to
 *    a function's behavior will be delivered via a new function (with a
 *    version suffix such as `_v2`) rather than by breaking the existing one.
 *  - Removal of any symbol declared here requires a major-version bump
 *    (1.0 -> 2.0).
 *  - The header itself is allowed to grow: new symbols may be added, but
 *    existing declarations are locked.
 *  - Consumers should use `moonlab_abi_version()` at runtime to feature-gate
 *    newer capabilities.
 *
 * STABILITY
 * ---------
 *  - `moonlab_qrng_bytes` : stable since 0.1.2.
 *  - `moonlab_abi_version` : stable since 0.1.2.
 *
 * All future public APIs (PQC, quantum-geometry computation, Chern markers,
 * etc.) will be added here as they land in subsequent 0.x releases.
 *
 * @since 0.1.2
 * @copyright 2024-2026 tsotchke. Licensed under the Apache License, 2.0.
 */

#ifndef MOONLAB_EXPORT_H
#define MOONLAB_EXPORT_H

#include <stddef.h>
#include <stdint.h>
#include <complex.h>     /* for double _Complex in CA-MPS observable signatures */

/* MOONLAB_API visibility tag.  Lives in its own header so module
 * headers (ca_mps.h, dmrg.h, ...) can pick up the tag without
 * pulling in the full ABI surface declaration.  See moonlab_api.h
 * for the macro expansion contract. */
#include "moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version constants. The (major, minor, patch) triple is purely for
 * feature-discovery by downstream consumers; it is NOT linked to the
 * package version. Consumers should check (major, minor) and refuse to
 * bind if they require a newer minor than the installed library. */
#define MOONLAB_ABI_VERSION_MAJOR 0
#define MOONLAB_ABI_VERSION_MINOR 3
#define MOONLAB_ABI_VERSION_PATCH 0

/**
 * @brief Query the ABI version at runtime.
 *
 * Consumers dlsym this and check the major/minor values to decide whether
 * optional capabilities (added in later 0.x releases) are available.
 *
 * @param[out] major Major version component. May be NULL.
 * @param[out] minor Minor version component. May be NULL.
 * @param[out] patch Patch version component. May be NULL.
 *
 * @since 0.1.2
 */
MOONLAB_API void moonlab_abi_version(int* major, int* minor, int* patch);

/**
 * @brief Fill a buffer with cryptographically-strong quantum random bytes.
 *
 * Produces bytes from Moonlab's v3 QRNG engine, which combines a hardware
 * entropy pool (RDSEED / /dev/urandom / SecRandomCopyBytes) with a
 * Bell-verified quantum simulation layer. The function is thread-safe:
 * concurrent calls from multiple threads are serialised internally.
 *
 * The v3 context is lazily initialised on first call and released at
 * process exit via atexit; callers do not need to perform any setup.
 *
 * @param buf  Output buffer. Must be non-NULL whenever @p size > 0.
 * @param size Number of bytes to write. May be 0 (no-op, returns success).
 *
 * @return  0 on success.
 * @return -1 if @p buf is NULL and @p size > 0.
 * @return -2 if the v3 QRNG engine failed to initialise.
 * @return -3 if a subsequent byte-generation call failed (e.g. an
 *            internal Bell-verification epoch was rejected).
 *
 * @since 0.1.2
 */
MOONLAB_API int moonlab_qrng_bytes(uint8_t* buf, size_t size);

/* ---- Quantum geometric tensor (stable from 0.2.0) -------------------- */

/**
 * @brief Compute the integer Chern number of the lower band of the
 *        Qi-Wu-Zhang 2-band Chern insulator
 *
 *   H(k) = sin(kx) sigma_x + sin(ky) sigma_y
 *        + (m + cos(kx) + cos(ky)) sigma_z
 *
 * via the Fukui-Hatsugai-Suzuki link-variable method on a @p N x @p N
 * Brillouin-zone grid. The return value is the integer Chern number
 * (no gauge ambiguity) as rounded to nearest integer.
 *
 * This is the simplest primitive on the QGT ABI surface. Higher-level
 * functions (per-plaquette Berry curvature arrays, Fubini-Study
 * metric, custom models) are exposed on an opaque-handle interface via
 * the internal `src/algorithms/quantum_geometry/qgt.h` header; when
 * they stabilise they will be promoted here.
 *
 * Intended consumers: QGTL (github.com:tsotchke/quantum_geometric_tensor),
 * lilirrep, SbNN.
 *
 * @param m  QWZ mass parameter.
 * @param N  BZ grid side (N >= 4; 32 is plenty for a clean integer).
 * @param out_chern  Optional: writes the raw (pre-rounding) Chern
 *                   value; may be NULL.
 * @return integer Chern number rounded to nearest int, or INT_MIN on
 *         error (bad arguments / allocation failure).
 *
 * @since 0.2.0
 */
MOONLAB_API int moonlab_qwz_chern(double m, size_t N, double* out_chern);

/* ---- ML-KEM-512 (FIPS 203) PQC KEM (stable from 0.2.0) -------------- */

#define MOONLAB_MLKEM512_PUBLICKEYBYTES   800
#define MOONLAB_MLKEM512_SECRETKEYBYTES   1632
#define MOONLAB_MLKEM512_CIPHERTEXTBYTES  768
#define MOONLAB_MLKEM512_SHAREDSECRETBYTES 32

/**
 * @brief Generate an ML-KEM-512 key pair, with entropy sourced from
 *        Moonlab's Bell-verified quantum RNG.
 *
 * @param ek 800-byte output public key.
 * @param dk 1632-byte output secret key.
 * @return  0 on success, -1 on entropy failure.
 * @since 0.2.0
 */
MOONLAB_API int moonlab_mlkem512_keygen_qrng(uint8_t* ek, uint8_t* dk);

/**
 * @brief Encapsulate a shared secret against an ML-KEM-512 public key.
 *        Entropy for the inner message seed is drawn from
 *        @ref moonlab_qrng_bytes.
 *
 * @param c  768-byte output ciphertext.
 * @param K  32-byte output shared secret.
 * @param ek 800-byte public key.
 * @return  0 on success, -1 on entropy failure.
 * @since 0.2.0
 */
MOONLAB_API int moonlab_mlkem512_encaps_qrng(uint8_t* c, uint8_t* K, const uint8_t* ek);

/**
 * @brief Decapsulate an ML-KEM-512 ciphertext.  Constant-time;
 *        implicit-rejection on invalid ciphertexts.
 *
 * @param K  32-byte output shared secret (may be pseudorandom on
 *           tampered ciphertext).
 * @param c  768-byte ciphertext.
 * @param dk 1632-byte secret key.
 * @since 0.2.0
 */
MOONLAB_API void moonlab_mlkem512_decaps(uint8_t* K,
                              const uint8_t* c,
                              const uint8_t* dk);

/* ---- ML-KEM-768 (NIST-recommended default; stable from 0.2.0) ------- */

#define MOONLAB_MLKEM768_PUBLICKEYBYTES    1184
#define MOONLAB_MLKEM768_SECRETKEYBYTES    2400
#define MOONLAB_MLKEM768_CIPHERTEXTBYTES   1088
#define MOONLAB_MLKEM768_SHAREDSECRETBYTES 32

MOONLAB_API int  moonlab_mlkem768_keygen_qrng(uint8_t* ek, uint8_t* dk);
MOONLAB_API int  moonlab_mlkem768_encaps_qrng(uint8_t* c, uint8_t* K, const uint8_t* ek);
MOONLAB_API void moonlab_mlkem768_decaps(uint8_t* K, const uint8_t* c, const uint8_t* dk);

/* ---- ML-KEM-1024 (Category 5; stable from 0.2.0) --------------------- */

#define MOONLAB_MLKEM1024_PUBLICKEYBYTES    1568
#define MOONLAB_MLKEM1024_SECRETKEYBYTES    3168
#define MOONLAB_MLKEM1024_CIPHERTEXTBYTES   1568
#define MOONLAB_MLKEM1024_SHAREDSECRETBYTES 32

MOONLAB_API int  moonlab_mlkem1024_keygen_qrng(uint8_t* ek, uint8_t* dk);
MOONLAB_API int  moonlab_mlkem1024_encaps_qrng(uint8_t* c, uint8_t* K, const uint8_t* ek);
MOONLAB_API void moonlab_mlkem1024_decaps(uint8_t* K, const uint8_t* c, const uint8_t* dk);

/* ---- Clifford-Assisted MPS (stable from 0.2.1) ---------------------- */
/*
 * Hybrid state representation |psi> = C |phi> combining the
 * Aaronson-Gottesman tableau (Clifford prefactor C) with an MPS factor
 * |phi>.  Clifford gates are O(n) tableau updates with no MPS cost;
 * non-Clifford gates push a Pauli-string rotation into |phi>.  The
 * design is documented in `docs/research/ca_mps.md` and the bond-dim
 * advantage is benchmarked in `tests/performance/bench_ca_mps.c`
 * (64x at N=12 on a stabilizer state).
 *
 * Pauli-string arguments are arrays of `n` bytes from {0=I, 1=X, 2=Y,
 * 3=Z}.  Functions return 0 on success, non-zero on error.
 *
 * The handle type is forward-declared as opaque; consumers do not need
 * to include any internal Moonlab header.
 *
 * Intended consumers: QGTL, lilirrep, SbNN.
 */

typedef struct moonlab_ca_mps_t moonlab_ca_mps_t;

/** Allocate a CA-MPS state on @p num_qubits with bond cap @p max_bond_dim. */
MOONLAB_API moonlab_ca_mps_t* moonlab_ca_mps_create(uint32_t num_qubits, uint32_t max_bond_dim);
/** Release a CA-MPS state created by ::moonlab_ca_mps_create. */
MOONLAB_API void moonlab_ca_mps_free(moonlab_ca_mps_t* s);
/** Deep-clone (independent of the source). */
MOONLAB_API moonlab_ca_mps_t* moonlab_ca_mps_clone(const moonlab_ca_mps_t* s);

/** Number of qubits.  0 if @p s is NULL. */
MOONLAB_API uint32_t moonlab_ca_mps_num_qubits(const moonlab_ca_mps_t* s);
/** Configured bond-dimension cap. */
MOONLAB_API uint32_t moonlab_ca_mps_max_bond_dim(const moonlab_ca_mps_t* s);
/** Current peak bond dimension across the MPS factor. */
MOONLAB_API uint32_t moonlab_ca_mps_current_bond_dim(const moonlab_ca_mps_t* s);

/* Clifford gates: tableau-only (O(n) per gate, no MPS cost). */
MOONLAB_API int moonlab_ca_mps_h    (moonlab_ca_mps_t* s, uint32_t q);
MOONLAB_API int moonlab_ca_mps_s    (moonlab_ca_mps_t* s, uint32_t q);
MOONLAB_API int moonlab_ca_mps_sdag (moonlab_ca_mps_t* s, uint32_t q);
MOONLAB_API int moonlab_ca_mps_x    (moonlab_ca_mps_t* s, uint32_t q);
MOONLAB_API int moonlab_ca_mps_y    (moonlab_ca_mps_t* s, uint32_t q);
MOONLAB_API int moonlab_ca_mps_z    (moonlab_ca_mps_t* s, uint32_t q);
MOONLAB_API int moonlab_ca_mps_cnot (moonlab_ca_mps_t* s, uint32_t ctrl, uint32_t targ);
MOONLAB_API int moonlab_ca_mps_cz   (moonlab_ca_mps_t* s, uint32_t a,    uint32_t b);
MOONLAB_API int moonlab_ca_mps_swap (moonlab_ca_mps_t* s, uint32_t a,    uint32_t b);

/* Non-Clifford gates: push a Pauli-string rotation into the MPS factor. */
MOONLAB_API int moonlab_ca_mps_rx        (moonlab_ca_mps_t* s, uint32_t q, double theta);
MOONLAB_API int moonlab_ca_mps_ry        (moonlab_ca_mps_t* s, uint32_t q, double theta);
MOONLAB_API int moonlab_ca_mps_rz        (moonlab_ca_mps_t* s, uint32_t q, double theta);
MOONLAB_API int moonlab_ca_mps_t_gate    (moonlab_ca_mps_t* s, uint32_t q);
MOONLAB_API int moonlab_ca_mps_t_dagger  (moonlab_ca_mps_t* s, uint32_t q);
MOONLAB_API int moonlab_ca_mps_phase     (moonlab_ca_mps_t* s, uint32_t q, double theta);
MOONLAB_API int moonlab_ca_mps_crz       (moonlab_ca_mps_t* s, uint32_t c, uint32_t t, double theta);
MOONLAB_API int moonlab_ca_mps_crx       (moonlab_ca_mps_t* s, uint32_t c, uint32_t t, double theta);
MOONLAB_API int moonlab_ca_mps_cry       (moonlab_ca_mps_t* s, uint32_t c, uint32_t t, double theta);
MOONLAB_API int moonlab_ca_mps_u3        (moonlab_ca_mps_t* s, uint32_t q,
                                double theta, double phi, double lambda);
/** Toffoli (CCX): flips @p t when both @p c1 and @p c2 are |1>. */
MOONLAB_API int moonlab_ca_mps_toffoli   (moonlab_ca_mps_t* s,
                                uint32_t c1, uint32_t c2, uint32_t t);
/** Fredkin (CSWAP): swaps @p t1 and @p t2 when @p c is |1>. */
MOONLAB_API int moonlab_ca_mps_fredkin   (moonlab_ca_mps_t* s,
                                uint32_t c, uint32_t t1, uint32_t t2);

/** Apply exp(i theta P) for a Pauli string P (length = num_qubits). */
MOONLAB_API int moonlab_ca_mps_pauli_rotation(moonlab_ca_mps_t* s,
                                   const uint8_t* pauli_string, double theta);
/** Imaginary-time exp(-tau P) for a Pauli string P; non-unitary. */
MOONLAB_API int moonlab_ca_mps_imag_pauli_rotation(moonlab_ca_mps_t* s,
                                        const uint8_t* pauli_string, double tau);
/** Restore unit norm after non-unitary evolution. */
MOONLAB_API int moonlab_ca_mps_normalize(moonlab_ca_mps_t* s);
/** Current state norm. */
MOONLAB_API double moonlab_ca_mps_norm(const moonlab_ca_mps_t* s);

/** <psi|P|psi> for a Pauli string P. */
MOONLAB_API int moonlab_ca_mps_expect_pauli(const moonlab_ca_mps_t* s,
                                 const uint8_t* pauli_string,
                                 double _Complex* out_expval);
/** <psi|H|psi> for H = sum_k coeffs[k] * paulis[k]; paulis is laid out
 *  as `num_terms * num_qubits` bytes. */
MOONLAB_API int moonlab_ca_mps_expect_pauli_sum(const moonlab_ca_mps_t* s,
                                     const uint8_t* paulis,
                                     const double _Complex* coeffs,
                                     uint32_t num_terms,
                                     double _Complex* out_expval);
/** Marginal P(Z_qubit = +1), in [0, 1]. */
MOONLAB_API int moonlab_ca_mps_prob_z(const moonlab_ca_mps_t* s,
                           uint32_t qubit, double* out_prob);

/* ---- DMRG scalar entry points (stable from 0.2.1) ------------------ */
/*
 * Run DMRG to convergence on a built-in 1D model and return only the
 * resulting ground-state energy.  These are the simplest possible
 * stable-ABI surface for DMRG: scalar in, scalar out, no opaque
 * handle.  Suited to phase-diagram queries (sweep a parameter, read
 * one number per point) which is the dominant QGTL/SbNN use case.
 *
 * On error returns DBL_MAX (NaN cannot be used as an error sentinel
 * under -ffast-math).  Callers should compare against DBL_MAX.
 */

/**
 * @brief Transverse-field Ising model ground-state energy via DMRG.
 *
 * H = -J sum_i Z_i Z_{i+1} - h sum_i X_i, with @c J = 1 and @c g = h/J.
 *
 * @param num_sites    Chain length (>= 2).
 * @param g            h/J ratio.  g > 1 paramagnetic, g < 1 ordered.
 * @param max_bond_dim DMRG truncation cap; 32 is plenty up to N=20.
 * @param num_sweeps   Two-site sweeps (typically 5-10 to converge).
 * @return Ground-state energy, or DBL_MAX on error.
 *
 * @since 0.2.1
 */
MOONLAB_API double moonlab_dmrg_tfim_energy(uint32_t num_sites, double g,
                                 uint32_t max_bond_dim, uint32_t num_sweeps);

/**
 * @brief Heisenberg XXZ chain ground-state energy via DMRG.
 *
 * Pauli-operator convention (open boundary conditions):
 *   H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1})
 *       - h sum_i Z_i
 *
 * Eigenvalues of (X.X + Y.Y + Z.Z) on a single bond are -3 (singlet)
 * and +1 (triplet); the spin-operator form S.S = (1/4)(X.X + Y.Y + Z.Z)
 * gives the eigenvalues you see in textbooks divided by 4.  Reference
 * OBC ground states for J = Delta = 1, h = 0:
 *   N = 2  -> E_GS = -3
 *   N = 4  -> E_GS = -6.4641
 *   N = 8  -> E_GS = -13.4997
 *
 * @param num_sites    Chain length (>= 2).
 * @param J            Exchange coupling.
 * @param Delta        XXZ anisotropy (Delta = 1 for isotropic Heisenberg).
 * @param h            Longitudinal field.
 * @param max_bond_dim DMRG truncation cap (32-128 typical).
 * @param num_sweeps   Two-site sweeps to convergence.
 * @return Ground-state energy, or DBL_MAX on error.
 *
 * @since 0.2.1
 */
MOONLAB_API double moonlab_dmrg_heisenberg_energy(uint32_t num_sites,
                                       double J, double Delta, double h,
                                       uint32_t max_bond_dim,
                                       uint32_t num_sweeps);

/* ---- Variational-D ground-state search (stable from 0.2.1) --------- */

/**
 * @brief Variational-D alternating ground-state search on a CA-MPS.
 *
 * Drives @p state toward a low-energy CA-MPS approximation of the
 * ground state of @p paulis (a Pauli-sum Hamiltonian) by alternating
 * a greedy local-Clifford D-update with imag-time evolution on
 * |phi>.  See `docs/research/ca_mps.md` §5 and `MATH.md` §11.
 *
 * Entry-point wrapper around the internal
 * @c moonlab_ca_mps_optimize_var_d_alternating with sensible defaults
 * exposed via dedicated arguments.  The full-config form is available
 * to in-tree callers via `algorithms/tensor_network/ca_mps_var_d.h`.
 *
 * @param state                  CA-MPS handle (mutated in place).
 * @param paulis                 Flat (num_terms, num_qubits) uint8 Pauli
 *                               array (0=I, 1=X, 2=Y, 3=Z).
 * @param coeffs                 Real coefficients, length num_terms.
 * @param num_terms              Pauli-sum length.
 * @param max_outer_iters        Outer alternating-loop cap.
 * @param imag_time_dtau         Imag-time step size.
 * @param imag_time_steps_per_outer  Trotter cycles per outer iter.
 * @param clifford_passes_per_outer  Greedy D-update passes per outer iter.
 * @param composite_2gate        Pass 1 to enable 2-gate composite moves.
 * @param warmstart              0=IDENTITY, 1=H_ALL, 2=DUAL_TFIM,
 *                               3=FERRO_TFIM, 4=STABILIZER_SUBGROUP.
 * @param stab_paulis            For warmstart=4: (k, num_qubits) generators.
 *                               NULL for other warmstarts.
 * @param stab_num_gens          For warmstart=4: number of generators k.
 * @param[out] out_final_energy  Final variational energy (NULL if not wanted).
 *
 * @return 0 on success, negative ::ca_mps_error_t on failure.
 *
 * @since 0.2.1
 */
MOONLAB_API int moonlab_ca_mps_var_d_run(moonlab_ca_mps_t* state,
                              const uint8_t* paulis,
                              const double* coeffs,
                              uint32_t num_terms,
                              uint32_t max_outer_iters,
                              double imag_time_dtau,
                              uint32_t imag_time_steps_per_outer,
                              uint32_t clifford_passes_per_outer,
                              int composite_2gate,
                              int warmstart,
                              const uint8_t* stab_paulis,
                              uint32_t stab_num_gens,
                              double* out_final_energy);

/**
 * @brief Variational-D alternating loop with explicit convergence
 *        threshold control (v0.2.4 extension).
 *
 * Same semantics as ::moonlab_ca_mps_var_d_run but exposes
 * @p convergence_eps -- the early-exit threshold the alternating
 * outer loop tests after each iter.  The default-eps run wrapper
 * fixes this at 1e-7, which is unsuitable for SU(2)-symmetric
 * Heisenberg-class problems where the loop hits a non-GS fixed
 * point quickly: with eps=1e-7 the loop declares convergence
 * after a few iters at 11-19% residual; tightening eps below the
 * fixed-point's natural floor lets the search escape and resume
 * imag-time evolution.
 *
 * Pass @p convergence_eps <= 0 to use the default 1e-7.
 *
 * @since 0.2.4
 */
MOONLAB_API int moonlab_ca_mps_var_d_run_v2(moonlab_ca_mps_t* state,
                                 const uint8_t* paulis,
                                 const double* coeffs,
                                 uint32_t num_terms,
                                 uint32_t max_outer_iters,
                                 double imag_time_dtau,
                                 uint32_t imag_time_steps_per_outer,
                                 uint32_t clifford_passes_per_outer,
                                 int composite_2gate,
                                 int warmstart,
                                 const uint8_t* stab_paulis,
                                 uint32_t stab_num_gens,
                                 double convergence_eps,
                                 double* out_final_energy);

/**
 * @brief Apply the gauge-aware stabilizer-subgroup warmstart Clifford.
 *
 * Build and apply the symplectic-Gauss-Jordan Clifford that places
 * `state->D|0^n>` in the simultaneous +1 eigenspace of every supplied
 * Pauli generator.  Standalone use case (no var-D loop): ground-state
 * preparation for stabilizer-coded Hamiltonians, surface / toric /
 * repetition codes, and lattice gauge theories.
 *
 * Thin wrapper around
 * `moonlab_ca_mps_apply_stab_subgroup_warmstart` (the internal entry
 * point in `ca_mps_var_d_stab_warmstart.h`); the wrapper exists so
 * the stable ABI declaration uses an `int` return type independent
 * of the internal `ca_mps_error_t` enum.
 *
 * @param state       CA-MPS handle (mutated in place, gates absorbed
 *                    into the Clifford prefactor D).
 * @param paulis      Flat (num_gens, num_qubits) uint8 array of
 *                    pairwise-commuting generators.
 * @param num_gens    Number of generators (>= 1, <= num_qubits).
 *
 * @return 0 on success, ::CA_MPS_ERR_INVALID (-1) if generators
 *         don't pairwise commute or aren't independent.
 *
 * @since 0.2.1
 */
MOONLAB_API int moonlab_ca_mps_gauge_warmstart(moonlab_ca_mps_t* state,
                                     const uint8_t* paulis,
                                     uint32_t num_gens);

/* ---- 1+1D Z2 lattice gauge theory (stable from 0.2.1) -------------- */

/**
 * @brief Build the qubit Pauli sum for a 1+1D Z2 LGT Hamiltonian.
 *
 * Allocates `*out_paulis` and `*out_coeffs`; caller must free both.
 * Encoding: 0=I, 1=X, 2=Y, 3=Z; row-major (num_terms, num_qubits).
 * Hamiltonian written with exactly gauge-invariant kinetic terms;
 * see `docs/research/var_d_lattice_gauge_theory.md`.
 *
 * @param num_matter_sites N >= 2.
 * @param t_hop          Matter-hopping amplitude.
 * @param h_link         Electric-field strength on each link.
 * @param mass           Staggered fermion mass.
 * @param gauss_penalty  Lambda; redundant under exactly gauge-
 *                       invariant H, but kept for compatibility.
 * @param[out] out_paulis      Allocated Pauli-byte array.
 * @param[out] out_coeffs      Allocated coefficient array.
 * @param[out] out_num_terms   Pauli-sum length.
 * @param[out] out_num_qubits  Total qubit count = 2 * N - 1.
 *
 * @return 0 on success, negative on bad input or OOM.
 *
 * @since 0.2.1
 */
MOONLAB_API int moonlab_z2_lgt_1d_build(uint32_t num_matter_sites,
                              double t_hop, double h_link,
                              double mass, double gauss_penalty,
                              uint8_t** out_paulis,
                              double** out_coeffs,
                              uint32_t* out_num_terms,
                              uint32_t* out_num_qubits);

/**
 * @brief Write the Gauss-law operator at interior site x into out_pauli.
 *
 * `G_x = X_{2x-1} Z_{2x} X_{2x+1}` for x in [1, N-2].  out_pauli must
 * point to a buffer of length 2 * num_matter_sites - 1.
 *
 * @return 0 on success, negative on out-of-range x.
 *
 * @since 0.2.1
 */
MOONLAB_API int moonlab_z2_lgt_1d_gauss_law(uint32_t num_matter_sites,
                                  uint32_t site_x,
                                  uint8_t* out_pauli);

/* ---- Adaptive-bond two-site TDVP (stable from 0.4.1) --------------- */

/**
 * @brief Opaque handle for a v0.4 adaptive-bond two-site TDVP engine.
 *
 * Bundles the internal `tn_mps_state_t`, the Hamiltonian MPO, the
 * `tdvp_engine_t` state (including per-bond PID controller slots when
 * the adaptive controller is enabled), and a step-by-step
 * `tdvp_history_t`.  All members are private to libquantumsim;
 * downstream consumers manipulate the handle only through the
 * `moonlab_tdvp_*` accessors below.
 *
 * @since 0.4.1
 */
typedef struct moonlab_tdvp_engine_t moonlab_tdvp_engine_t;

/**
 * @brief Build a TDVP engine wrapping a Heisenberg-XXZ Hamiltonian.
 *
 * Constructs the Heisenberg MPO @c H = J * sum_i (X_i X_{i+1} + Y_i
 * Y_{i+1} + Delta * Z_i Z_{i+1}) - h * sum_i Z_i, seeds a random
 * initial MPS state of width @p initial_bond_dim (clamped to
 * @p max_bond_dim), and returns an engine configured for either
 * real-time or imag-time evolution.  When
 * @p adaptive_target_entropy is strictly positive the entropy-feedback
 * PID controller is enabled with reference-paper gains
 * (arXiv:2604.03960); pass 0 (or any non-positive value) to retain the
 * fixed-bond legacy path.
 *
 * @param num_sites               Chain length (>= 2).
 * @param J                       Exchange coupling.
 * @param Delta                   XXZ anisotropy (Delta = 1 for isotropic).
 * @param h                       Longitudinal field.
 * @param initial_bond_dim        Initial MPS bond dimension.
 * @param max_bond_dim            Legacy / adaptive ceiling cap.
 * @param dt                      Time-step size (positive).
 * @param imag_time               Non-zero for imag-time evolution
 *                                (exp(-H dt)); zero for real-time
 *                                (exp(-i H dt)).
 * @param adaptive_target_entropy Entropy-error budget for the PID
 *                                controller; <= 0 disables the
 *                                controller and runs at the fixed cap.
 * @return Engine handle on success, NULL on allocation failure.
 *
 * @since 0.4.1
 */
MOONLAB_API moonlab_tdvp_engine_t*
moonlab_tdvp_create_heisenberg(uint32_t num_sites,
                                double J,
                                double Delta,
                                double h,
                                uint32_t initial_bond_dim,
                                uint32_t max_bond_dim,
                                double dt,
                                int imag_time,
                                double adaptive_target_entropy);

/**
 * @brief Build a TDVP engine wrapping a transverse-field Ising
 *        Hamiltonian.
 *
 * Constructs @c H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i, seeds a
 * random initial MPS, and configures the engine identically to
 * ::moonlab_tdvp_create_heisenberg.
 *
 * @since 0.4.1
 */
MOONLAB_API moonlab_tdvp_engine_t*
moonlab_tdvp_create_tfim(uint32_t num_sites,
                          double J,
                          double h,
                          uint32_t initial_bond_dim,
                          uint32_t max_bond_dim,
                          double dt,
                          int imag_time,
                          double adaptive_target_entropy);

/**
 * @brief Advance the engine by one TDVP step.
 *
 * Performs a left-to-right + right-to-left two-site sweep with
 * step-size dt as configured at engine creation; appends the resulting
 * (time, energy, norm, bond-chi distribution) to the engine's internal
 * history.
 *
 * @return 0 on success, negative on integrator failure.
 * @since 0.4.1
 */
MOONLAB_API int moonlab_tdvp_step(moonlab_tdvp_engine_t* engine);

/**
 * @brief Advance the engine repeatedly until current time reaches
 *        @p target_time.
 *
 * Equivalent to calling ::moonlab_tdvp_step in a loop with dt
 * adjusted to land exactly on @p target_time.  Every step is recorded
 * in the engine's history.
 *
 * @return 0 on success, negative on integrator failure.
 * @since 0.4.1
 */
MOONLAB_API int moonlab_tdvp_evolve_to(moonlab_tdvp_engine_t* engine,
                                        double target_time);

/** Current evolution time accumulated by the engine. */
MOONLAB_API double moonlab_tdvp_current_time(const moonlab_tdvp_engine_t* engine);

/** Current variational energy @c <psi|H|psi> evaluated after the
 *  most recent step (or at engine creation if no steps have run). */
MOONLAB_API double moonlab_tdvp_current_energy(const moonlab_tdvp_engine_t* engine);

/** Current MPS norm @c <psi|psi>^{1/2}. */
MOONLAB_API double moonlab_tdvp_current_norm(const moonlab_tdvp_engine_t* engine);

/** Peak bond dimension across the MPS factor after the latest step. */
MOONLAB_API uint32_t moonlab_tdvp_current_max_bond_dim(const moonlab_tdvp_engine_t* engine);

/** Number of inter-site bonds (n_qubits - 1). */
MOONLAB_API uint32_t moonlab_tdvp_num_bonds(const moonlab_tdvp_engine_t* engine);

/**
 * @brief Current PID-selected chi for bond @p bond.
 *
 * Returns 0 when the adaptive controller is disabled or @p bond is
 * out of range; otherwise returns the dimension the controller chose
 * on the most recent two-site update of that bond.
 *
 * @since 0.4.1
 */
MOONLAB_API uint32_t moonlab_tdvp_bond_chi(const moonlab_tdvp_engine_t* engine,
                                            uint32_t bond);

/** Number of steps currently recorded in the engine's history. */
MOONLAB_API uint32_t moonlab_tdvp_history_num_steps(const moonlab_tdvp_engine_t* engine);

/**
 * @brief Read the scalar fields of step @p step from the engine's
 *        history.
 *
 * @param[in]  engine  Engine handle.
 * @param[in]  step    Step index; must be < ::moonlab_tdvp_history_num_steps.
 * @param[out] time    Out-time (may be NULL).
 * @param[out] energy  Out-energy (may be NULL).
 * @param[out] norm    Out-norm (may be NULL).
 * @return 0 on success, -1 if @p engine is NULL or @p step is out of
 *         range.
 * @since 0.4.1
 */
MOONLAB_API int moonlab_tdvp_history_get_step(const moonlab_tdvp_engine_t* engine,
                                               uint32_t step,
                                               double* time,
                                               double* energy,
                                               double* norm);

/**
 * @brief Read the recorded per-bond chi snapshot for step @p step into
 *        a caller-provided buffer.
 *
 * Writes ::moonlab_tdvp_num_bonds values into @p out_chi.  When the
 * adaptive controller was disabled at engine creation the buffer is
 * filled with zeroes (the legacy path does not record per-bond chi).
 *
 * @return 0 on success, -1 on out-of-range step or NULL buffer.
 * @since 0.4.1
 */
MOONLAB_API int moonlab_tdvp_history_get_bond_chi(const moonlab_tdvp_engine_t* engine,
                                                   uint32_t step,
                                                   uint32_t* out_chi,
                                                   uint32_t buf_capacity);

/** Release the engine and all internal allocations. */
MOONLAB_API void moonlab_tdvp_engine_free(moonlab_tdvp_engine_t* engine);

/* ---- Diagnostic stringifier (stable from 0.2.1) -------------------- */

/**
 * @brief Pretty-print a Moonlab status code for the named module.
 *
 * Returns a static string for canonical codes (SUCCESS, ERR_INVALID,
 * ERR_QUBIT, ERR_OOM, ERR_BACKEND); a module-specific name for known
 * extensions; a thread-local "<MODULE status N>" fallback otherwise.
 * Never returns NULL.
 *
 * @param module  One of the moonlab_status_module_t values:
 *                0=GENERIC, 1=CA_MPS, 2=CA_MPS_VAR_D,
 *                3=CA_MPS_STAB_WARMSTART, 4=CA_PEPS,
 *                5=TN_STATE, 6=TN_GATE, 7=TN_MEASURE, 8=TENSOR,
 *                9=CONTRACT, 10=SVD_COMPRESS, 11=CLIFFORD,
 *                12=PARTITION, 13=DIST_GATE, 14=MPI_BRIDGE.
 * @param status  Integer return code.
 *
 * @since 0.2.1
 */
MOONLAB_API const char* moonlab_status_string(int module, int status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MOONLAB_EXPORT_H */
