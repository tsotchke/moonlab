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
 *    same name and signature, throughout ABI major 0. Semantic upgrades to
 *    a function's behavior will be delivered via a new function (with a
 *    version suffix such as `_v2`) rather than by breaking the existing one.
 *  - Removal or an incompatible signature change requires an ABI-major bump.
 *  - The header itself is allowed to grow: new symbols may be added, but
 *    existing declarations are locked.
 *  - Consumers should use `moonlab_abi_version()` at runtime to feature-gate
 *    newer capabilities.
 *  - Linux shared-library builds are process-resident. `dlclose()` releases
 *    the caller's handle, but libquantumsim remains mapped until process exit;
 *    a later `dlopen()` can resolve and call the ABI again against the same
 *    process-global runtime state. Consumers must not reuse a closed handle or
 *    symbols obtained from it. Linux builds fail configuration when the
 *    linker cannot enforce this contract with ELF `DT_FLAGS_1 NODELETE`.
 *
 * STABILITY
 * ---------
 * Every declaration in this header is tagged MOONLAB_API and survives a
 * hidden-visibility build (`-DQSIM_HIDDEN_VISIBILITY=ON`).  The surface
 * spans the conditioned QRNG + status, ML-KEM-512/768/1024, the QGT
 * Chern / Z2 topology one-shots, the pointwise band geometry,
 * Clifford-Assisted MPS, DMRG scalar energies, variational-D, the 1+1D Z2
 * lattice-gauge builder, adaptive two-site TDVP, the exact VQE gradient
 * and its quantum geometry, and the status stringifier.
 * `tests/abi/test_moonlab_export_abi.c` dlopens the library and resolves
 * plus smokes this surface exactly as a downstream consumer does. On Linux it
 * also closes and reopens the library repeatedly, verifies process residency,
 * and calls representative APIs after every reopen.
 *
 * VERSION HISTORY
 * ---------------
 *  - 0.7.0  VQE quantum geometry (`moonlab_vqe_qgt`,
 *           `moonlab_vqe_berry_curvature`,
 *           `moonlab_vqe_natural_gradient`) and the pointwise band
 *           geometry one-shots (`moonlab_dsigma_metric_curvature`,
 *           `moonlab_qwz_curvature_at`, `moonlab_haldane_curvature_at`)
 *           declared, so QGTL / libirrep / SbNN reach the metric and
 *           Berry curvature over FFI without the opaque handle surface.
 *  - 0.6.0  `moonlab_ca_mps_conjugate_pauli` promoted to the stable
 *           surface (int return, independent of the internal
 *           `ca_mps_error_t` enum); the seven QGT topology one-shots
 *           (`moonlab_ssh_winding`, `moonlab_kitaev_chain_z2`,
 *           `moonlab_chern_qwz_proj`, `moonlab_chern_qwz_pt`,
 *           `moonlab_kane_mele_z2`, `moonlab_bhz_z2`,
 *           `moonlab_hofstadter_chern`) declared; portable
 *           `moonlab_complex_double` carrier introduced so the header
 *           parses under MSVC while keeping the C99 ABI identical.
 *  - 0.5.0  `moonlab_qrng_get_status` + the conditioned-QRNG status
 *           contract.
 *  - 0.4.1  adaptive-bond two-site TDVP engine surface.
 *  - 0.4.0  `moonlab_vqe_gradient` (exact adjoint / parameter-shift).
 *  - 0.2.x  QGT Chern, ML-KEM KEMs, CA-MPS, DMRG scalar energies,
 *           variational-D, Z2 LGT, status stringifier.
 *  - 0.1.2  `moonlab_qrng_bytes`, `moonlab_abi_version`.
 *
 * @since 0.1.2
 * @copyright 2024-2026 tsotchke. Licensed under the MIT License.
 */

#ifndef MOONLAB_EXPORT_H
#define MOONLAB_EXPORT_H

#include <stddef.h>
#include <stdint.h>
#if !defined(_MSC_VER) || defined(__clang__)
#include <complex.h>     /* native double _Complex carrier (see moonlab_complex_double) */
#endif

/* MOONLAB_API visibility tag.  Lives in its own header so module
 * headers (ca_mps.h, dmrg.h, ...) can pick up the tag without
 * pulling in the full ABI surface declaration.  See moonlab_api.h
 * for the macro expansion contract. */
#include "moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Portable complex carrier for the header boundary.
 *
 * MSVC's C mode has no C99 `double _Complex`, so a downstream consumer
 * compiling this header under MSVC cl.exe cannot parse the CA-MPS
 * observable signatures below.  We expose a layout-compatible carrier:
 *
 *   - GNU / Clang / any C99 compiler: the native `double _Complex`,
 *     so the ABI is byte-for-byte the C99 ABI with zero change.
 *   - MSVC (cl.exe): a `{ double re; double im; }` struct, which is
 *     exactly the storage of a C99 `double _Complex` (two contiguous
 *     doubles, real part first).
 *
 * Every stable-ABI use below passes this type only through a pointer, so
 * the by-value calling convention never enters the contract; only the
 * memory layout matters, and the static assert pins it. */
#if defined(_MSC_VER) && !defined(__clang__)
typedef struct { double re; double im; } moonlab_complex_double;
#else
typedef double _Complex moonlab_complex_double;
#endif

#if defined(__cplusplus)
static_assert(sizeof(moonlab_complex_double) == 2 * sizeof(double),
              "moonlab_complex_double must be two contiguous doubles");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(moonlab_complex_double) == 2 * sizeof(double),
               "moonlab_complex_double must be two contiguous doubles");
#endif

/* ABI version constants. The (major, minor, patch) triple is purely for
 * feature-discovery by downstream consumers; it is NOT linked to the
 * package version. Consumers should check (major, minor) and refuse to
 * bind if they require a newer minor than the installed library. */
#define MOONLAB_ABI_VERSION_MAJOR 0
#define MOONLAB_ABI_VERSION_MINOR 7
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
 * @stability stable
 */
MOONLAB_API void moonlab_abi_version(int* major, int* minor, int* patch);

/**
 * @brief Fill a buffer with cryptographically conditioned hybrid random bytes.
 *
 * The release path combines Moonlab's v3 quantum-simulation engine with fresh
 * hardware/OS entropy that is continuously checked by the SP 800-90B RCT and
 * APT health tests. Every request is domain-separated and conditioned through
 * SHAKE256. Simulated Bell epochs are checked before their bytes are released,
 * and any entropy, Bell, or conditioning failure is fail-closed with the
 * caller's buffer zeroed. The function is thread-safe: concurrent calls are
 * serialised internally.
 *
 * The Bell gate verifies Moonlab's quantum-simulation plumbing. It becomes a
 * device-independent certification only when paired with an independent
 * physical Bell-test provider and transcript verifier; the capability query
 * below deliberately distinguishes those two assurance levels. This module
 * is not claiming FIPS 140 validation.
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
 * @return -4 if fresh conditioner entropy could not be obtained.
 * @return -5 if the per-process request counter is exhausted.
 *
 * @since 0.1.2
 * @stability stable
 */
MOONLAB_API int moonlab_qrng_bytes(uint8_t* buf, size_t size);

/* Stable QRNG delivery capabilities.  A flag is set only when that property
 * is active in the process-wide release path. */
#define MOONLAB_QRNG_CAP_HARDWARE_OS_ENTROPY       (UINT64_C(1) << 0)
#define MOONLAB_QRNG_CAP_CONTINUOUS_HEALTH_TESTS   (UINT64_C(1) << 1)
#define MOONLAB_QRNG_CAP_SHAKE256_CONDITIONED      (UINT64_C(1) << 2)
#define MOONLAB_QRNG_CAP_BELL_SIMULATION_GATED     (UINT64_C(1) << 3)
#define MOONLAB_QRNG_CAP_THREAD_SAFE               (UINT64_C(1) << 4)
#define MOONLAB_QRNG_CAP_BELL_EPOCH_CERTIFIED      (UINT64_C(1) << 5)
#define MOONLAB_QRNG_CAP_DEVICE_INDEPENDENT_SOURCE (UINT64_C(1) << 6)
#define MOONLAB_QRNG_CAP_FIPS140_VALIDATED         (UINT64_C(1) << 7)

/** Runtime status for the process-wide conditioned QRNG delivery path. */
typedef struct {
    uint32_t struct_size;             /**< sizeof(moonlab_qrng_status_t) */
    uint32_t api_version;             /**< Status structure version (1) */
    uint64_t capabilities;            /**< MOONLAB_QRNG_CAP_* bitset */
    uint64_t conditioned_requests;    /**< Completed public API requests */
    uint64_t raw_bytes_generated;     /**< Bytes drawn through v3 */
    uint64_t bell_tests_performed;    /**< Simulated Bell plumbing gates */
    uint64_t bell_tests_passed;       /**< Gates accepted before delivery */
    double average_chsh;              /**< Mean simulated CHSH value */
    double minimum_chsh;              /**< Lowest simulated CHSH value */
} moonlab_qrng_status_t;

/**
 * @brief Query active QRNG protections and live health statistics.
 *
 * Lazily initializes the process-wide engine but does not draw output. The
 * first output request performs the first Bell epoch gate before delivery.
 *
 * @return 0 on success, -1 for NULL status, -2 on initialization failure.
 * @since ABI 0.5
 * @stability stable
 */
MOONLAB_API int moonlab_qrng_get_status(moonlab_qrng_status_t* status);

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
 * @stability stable
 */
MOONLAB_API int moonlab_qwz_chern(double m, size_t N, double* out_chern);

/* ---- Topology invariant one-shots (stable from 0.6.0) --------------- */
/*
 * Each function bundles model construction + invariant calculation +
 * teardown into a single scalar-in / scalar-out call, so a binding
 * consumer never marshals an opaque model handle across the FFI
 * boundary.  The heavier opaque-handle QGT surface lives in the
 * internal `src/algorithms/quantum_geometry/qgt.h`; these one-shots are
 * the frozen convenience layer.  All return the integer invariant, or
 * `INT_MIN` on allocation / argument error.
 *
 * Intended consumers: QGTL, libirrep, SbNN, and the JS / WASM TS layer.
 */

/** Winding number of the Su-Schrieffer-Heeger chain (t1 intra-cell,
 * @stability stable
 *  t2 inter-cell) on an @p N-point Brillouin-zone grid (N >= 4). */
MOONLAB_API int moonlab_ssh_winding(double t1, double t2, size_t N);

/** Z2 (0/1) invariant of the Kitaev p-wave chain in its BdG form for
 * @stability stable
 *  hopping @p t, chemical potential @p mu, pairing @p delta. */
MOONLAB_API int moonlab_kitaev_chain_z2(double t, double mu, double delta);

/** Chern number of the QWZ lower band via the projector (P dP dP)
 * @stability stable
 *  Berry-curvature grid on an @p N x @p N BZ mesh (N >= 4). */
MOONLAB_API int moonlab_chern_qwz_proj(double m, size_t N);

/** Chern number of the QWZ lower band via the parallel-transport
 * @stability stable
 *  (Wilson-loop) Berry-curvature grid on an @p N x @p N BZ mesh. */
MOONLAB_API int moonlab_chern_qwz_pt(double m, size_t N);

/** Z2 invariant of the Kane-Mele model (@p t hopping, @p lambda_so
 *  spin-orbit, @p lambda_r Rashba, @p lambda_v sublattice potential)
 * @stability stable
 *  on an @p N x @p N mesh (N >= 8, even). */
MOONLAB_API int moonlab_kane_mele_z2(double t, double lambda_so,
                                     double lambda_r, double lambda_v,
                                     size_t N);

/** Z2 invariant of the Bernevig-Hughes-Zhang model (@p A, @p B, @p M)
 * @stability stable
 *  on an @p N x @p N mesh (N >= 8, even). */
MOONLAB_API int moonlab_bhz_z2(double A, double B, double M, size_t N);

/** Chern number of the Hofstadter model at flux @p p / @p q with
 *  @p n_occupied filled sub-bands, on an @p N x @p N mesh (N >= 4,
 * @stability stable
 *  q >= 2, 1 <= n_occupied < q). */
MOONLAB_API int moonlab_hofstadter_chern(double t, size_t p, size_t q,
                                         size_t n_occupied, size_t N);

/* ---- ML-KEM-512 (FIPS 203) PQC KEM (stable from 0.2.0) -------------- */

#define MOONLAB_MLKEM512_PUBLICKEYBYTES   800
#define MOONLAB_MLKEM512_SECRETKEYBYTES   1632
#define MOONLAB_MLKEM512_CIPHERTEXTBYTES  768
#define MOONLAB_MLKEM512_SHAREDSECRETBYTES 32

/**
 * @brief Generate an ML-KEM-512 key pair using Moonlab's conditioned hybrid RNG.
 *
 * The entropy path is health-tested, SHAKE256-conditioned, Bell-epoch gated,
 * and fail-closed. Environments that require a validated cryptographic-module
 * boundary should use the explicit-seed API with that module's approved DRBG.
 *
 * @param ek 800-byte output public key.
 * @param dk 1632-byte output secret key.
 * @return  0 on success, -1 on entropy failure.
 * @since 0.2.0
 * @stability stable
 */
MOONLAB_API int moonlab_mlkem512_keygen_qrng(
    uint8_t ek[MOONLAB_MLKEM512_PUBLICKEYBYTES],
    uint8_t dk[MOONLAB_MLKEM512_SECRETKEYBYTES]);

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
 * @stability stable
 */
MOONLAB_API int moonlab_mlkem512_encaps_qrng(
    uint8_t c[MOONLAB_MLKEM512_CIPHERTEXTBYTES],
    uint8_t K[MOONLAB_MLKEM512_SHAREDSECRETBYTES],
    const uint8_t ek[MOONLAB_MLKEM512_PUBLICKEYBYTES]);

/**
 * @brief Decapsulate an ML-KEM-512 ciphertext with implicit rejection on
 *        invalid ciphertexts.  The source uses a branchless FO selector but
 *        has not been independently side-channel audited.
 *
 * @param K  32-byte output shared secret (may be pseudorandom on
 *           tampered ciphertext).
 * @param c  768-byte ciphertext.
 * @param dk 1632-byte secret key.
 * @since 0.2.0
 * @stability stable
 */
MOONLAB_API void moonlab_mlkem512_decaps(
    uint8_t K[MOONLAB_MLKEM512_SHAREDSECRETBYTES],
    const uint8_t c[MOONLAB_MLKEM512_CIPHERTEXTBYTES],
    const uint8_t dk[MOONLAB_MLKEM512_SECRETKEYBYTES]);

/* ---- ML-KEM-768 (NIST-recommended default; stable from 0.2.0) ------- */

#define MOONLAB_MLKEM768_PUBLICKEYBYTES    1184
#define MOONLAB_MLKEM768_SECRETKEYBYTES    2400
#define MOONLAB_MLKEM768_CIPHERTEXTBYTES   1088
#define MOONLAB_MLKEM768_SHAREDSECRETBYTES 32

/**
 * @brief Generate an ML-KEM-768 key pair using Moonlab's conditioned hybrid RNG.
 *
 * Draws 64 bytes from @ref moonlab_qrng_bytes and splits them into the
 * FIPS 203 seeds (d, z).  The entropy path is health-tested,
 * SHAKE256-conditioned, Bell-epoch gated, and fail-closed.
 * Environments that require a validated cryptographic-module boundary
 * should use the explicit-seed API with that module's approved DRBG.
 *
 * @param ek 1184-byte output public key.
 * @param dk 2400-byte output secret key.
 * @return  0 on success, -1 on entropy failure.
 * @since 0.2.0
 * @stability stable
 */
MOONLAB_API int moonlab_mlkem768_keygen_qrng(
    uint8_t ek[MOONLAB_MLKEM768_PUBLICKEYBYTES],
    uint8_t dk[MOONLAB_MLKEM768_SECRETKEYBYTES]);

/**
 * @brief Encapsulate a shared secret against an ML-KEM-768 public key.
 *        Entropy for the inner message seed is drawn from
 *        @ref moonlab_qrng_bytes.
 *
 * @p ek is validated against the FIPS 203 Section 7.2 modulus check
 * before any entropy is consumed.
 *
 * @param c  1088-byte output ciphertext.
 * @param K  32-byte output shared secret.
 * @param ek 1184-byte public key.
 * @return  0 on success, -1 on entropy failure, -2 if @p ek fails the
 *          Section 7.2 check.
 * @since 0.2.0
 * @stability stable
 */
MOONLAB_API int moonlab_mlkem768_encaps_qrng(
    uint8_t c[MOONLAB_MLKEM768_CIPHERTEXTBYTES],
    uint8_t K[MOONLAB_MLKEM768_SHAREDSECRETBYTES],
    const uint8_t ek[MOONLAB_MLKEM768_PUBLICKEYBYTES]);

/**
 * @brief Decapsulate an ML-KEM-768 ciphertext with implicit rejection on
 *        invalid ciphertexts.  The source uses a branchless FO selector but
 *        has not been independently side-channel audited.
 *
 * @param K  32-byte output shared secret (pseudorandom, derived from
 *           SHAKE256(z || c), on a tampered ciphertext).
 * @param c  1088-byte ciphertext.
 * @param dk 2400-byte secret key.
 * @since 0.2.0
 * @stability stable
 */
MOONLAB_API void moonlab_mlkem768_decaps(
    uint8_t K[MOONLAB_MLKEM768_SHAREDSECRETBYTES],
    const uint8_t c[MOONLAB_MLKEM768_CIPHERTEXTBYTES],
    const uint8_t dk[MOONLAB_MLKEM768_SECRETKEYBYTES]);

/* ---- ML-KEM-1024 (Category 5; stable from 0.2.0) --------------------- */

#define MOONLAB_MLKEM1024_PUBLICKEYBYTES    1568
#define MOONLAB_MLKEM1024_SECRETKEYBYTES    3168
#define MOONLAB_MLKEM1024_CIPHERTEXTBYTES   1568
#define MOONLAB_MLKEM1024_SHAREDSECRETBYTES 32

/**
 * @brief Generate an ML-KEM-1024 key pair using Moonlab's conditioned hybrid RNG.
 *
 * Draws 64 bytes from @ref moonlab_qrng_bytes and splits them into the
 * FIPS 203 seeds (d, z).  The entropy path is health-tested,
 * SHAKE256-conditioned, Bell-epoch gated, and fail-closed.
 * Environments that require a validated cryptographic-module boundary
 * should use the explicit-seed API with that module's approved DRBG.
 *
 * @param ek 1568-byte output public key.
 * @param dk 3168-byte output secret key.
 * @return  0 on success, -1 on entropy failure.
 * @since 0.2.0
 * @stability stable
 */
MOONLAB_API int moonlab_mlkem1024_keygen_qrng(
    uint8_t ek[MOONLAB_MLKEM1024_PUBLICKEYBYTES],
    uint8_t dk[MOONLAB_MLKEM1024_SECRETKEYBYTES]);

/**
 * @brief Encapsulate a shared secret against an ML-KEM-1024 public key.
 *        Entropy for the inner message seed is drawn from
 *        @ref moonlab_qrng_bytes.
 *
 * @p ek is validated against the FIPS 203 Section 7.2 modulus check
 * before any entropy is consumed.
 *
 * @param c  1568-byte output ciphertext.
 * @param K  32-byte output shared secret.
 * @param ek 1568-byte public key.
 * @return  0 on success, -1 on entropy failure, -2 if @p ek fails the
 *          Section 7.2 check.
 * @since 0.2.0
 * @stability stable
 */
MOONLAB_API int moonlab_mlkem1024_encaps_qrng(
    uint8_t c[MOONLAB_MLKEM1024_CIPHERTEXTBYTES],
    uint8_t K[MOONLAB_MLKEM1024_SHAREDSECRETBYTES],
    const uint8_t ek[MOONLAB_MLKEM1024_PUBLICKEYBYTES]);

/**
 * @brief Decapsulate an ML-KEM-1024 ciphertext with implicit rejection on
 *        invalid ciphertexts.  The source uses a branchless FO selector but
 *        has not been independently side-channel audited.
 *
 * @param K  32-byte output shared secret (pseudorandom, derived from
 *           SHAKE256(z || c), on a tampered ciphertext).
 * @param c  1568-byte ciphertext.
 * @param dk 3168-byte secret key.
 * @since 0.2.0
 * @stability stable
 */
MOONLAB_API void moonlab_mlkem1024_decaps(
    uint8_t K[MOONLAB_MLKEM1024_SHAREDSECRETBYTES],
    const uint8_t c[MOONLAB_MLKEM1024_CIPHERTEXTBYTES],
    const uint8_t dk[MOONLAB_MLKEM1024_SECRETKEYBYTES]);

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

/**
 * Allocate a CA-MPS state on @p num_qubits with bond cap @p max_bond_dim.
 * @stability stable
 */
MOONLAB_API moonlab_ca_mps_t* moonlab_ca_mps_create(uint32_t num_qubits, uint32_t max_bond_dim);
/**
 * Release a CA-MPS state created by ::moonlab_ca_mps_create.
 * @stability stable
 */
MOONLAB_API void moonlab_ca_mps_free(moonlab_ca_mps_t* s);
/**
 * Deep-clone (independent of the source).
 * @stability stable
 */
MOONLAB_API moonlab_ca_mps_t* moonlab_ca_mps_clone(const moonlab_ca_mps_t* s);

/**
 * Number of qubits.  0 if @p s is NULL.
 * @stability stable
 */
MOONLAB_API uint32_t moonlab_ca_mps_num_qubits(const moonlab_ca_mps_t* s);
/**
 * Configured bond-dimension cap.
 * @stability stable
 */
MOONLAB_API uint32_t moonlab_ca_mps_max_bond_dim(const moonlab_ca_mps_t* s);
/**
 * Current peak bond dimension across the MPS factor.
 * @stability stable
 */
MOONLAB_API uint32_t moonlab_ca_mps_current_bond_dim(const moonlab_ca_mps_t* s);

/* Clifford gates: tableau-only (O(n) per gate, no MPS cost). */

/**
 * Hadamard on @p q, composed onto the Clifford prefactor C.  The MPS
 * factor is untouched, so this costs O(n) tableau work and no bond
 * growth.
 * @param s CA-MPS state, mutated in place.
 * @param q Qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -4 if the tableau backend rejects the update.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_h    (moonlab_ca_mps_t* s, uint32_t q);
/**
 * Phase gate S = diag(1, i) on @p q, composed onto C.  Tableau-only.
 * @param s CA-MPS state, mutated in place.
 * @param q Qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -4 if the tableau backend rejects the update.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_s    (moonlab_ca_mps_t* s, uint32_t q);
/**
 * Inverse phase gate S-dagger on @p q, composed onto C as three S
 * applications.  Tableau-only.
 * @param s CA-MPS state, mutated in place.
 * @param q Qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -4 if the tableau backend rejects the update.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_sdag (moonlab_ca_mps_t* s, uint32_t q);
/**
 * Pauli X on @p q, composed onto C.  Tableau-only; a pure row-sign
 * update.
 * @param s CA-MPS state, mutated in place.
 * @param q Qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -4 if the tableau backend rejects the update.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_x    (moonlab_ca_mps_t* s, uint32_t q);
/**
 * Pauli Y on @p q, composed onto C.  Tableau-only; a pure row-sign
 * update.
 * @param s CA-MPS state, mutated in place.
 * @param q Qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -4 if the tableau backend rejects the update.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_y    (moonlab_ca_mps_t* s, uint32_t q);
/**
 * Pauli Z on @p q, composed onto C.  Tableau-only; a pure row-sign
 * update.
 * @param s CA-MPS state, mutated in place.
 * @param q Qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -4 if the tableau backend rejects the update.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_z    (moonlab_ca_mps_t* s, uint32_t q);
/**
 * CNOT composed onto C.  Tableau-only, so it stays O(n) regardless of
 * how far apart @p ctrl and @p targ are -- no SWAP chain and no bond
 * growth, unlike a plain MPS.
 * @param s    CA-MPS state, mutated in place.
 * @param ctrl Control qubit index in [0, num_qubits).
 * @param targ Target qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or an index is out of range;
 *         -4 if @p ctrl equals @p targ or the tableau backend rejects
 *         the update.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_cnot (moonlab_ca_mps_t* s, uint32_t ctrl, uint32_t targ);
/**
 * Controlled-Z composed onto C as H_b CNOT(a,b) H_b.  Tableau-only.
 * @param s CA-MPS state, mutated in place.
 * @param a First qubit index in [0, num_qubits).
 * @param b Second qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or an index is out of range;
 *         -4 if @p a equals @p b or the tableau backend rejects the
 *         update.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_cz   (moonlab_ca_mps_t* s, uint32_t a,    uint32_t b);
/**
 * SWAP composed onto C as three CNOTs.  Tableau-only, so it relabels
 * the Clifford prefactor rather than moving MPS tensors.
 * @param s CA-MPS state, mutated in place.
 * @param a First qubit index in [0, num_qubits).
 * @param b Second qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or an index is out of range;
 *         -4 if @p a equals @p b or the tableau backend rejects the
 *         update.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_swap (moonlab_ca_mps_t* s, uint32_t a,    uint32_t b);

/* Non-Clifford gates: push a Pauli-string rotation into the MPS factor. */

/**
 * RX(theta) = exp(-i theta X_q / 2) in the Qiskit/Cirq convention.
 * X_q is conjugated back through the stored Clifford C to a Pauli
 * string P' = C-dagger X_q C, and exp(-i theta P' / 2) is applied to
 * the MPS factor as a bond-dimension-2 MPO.  Bond growth is bounded by
 * the state's max_bond_dim.
 * @param s     CA-MPS state, mutated in place.
 * @param q     Qubit index in [0, num_qubits).
 * @param theta Rotation angle in radians.
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -3 on allocation failure; -4 if the MPO apply fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_rx        (moonlab_ca_mps_t* s, uint32_t q, double theta);
/**
 * RY(theta) = exp(-i theta Y_q / 2), applied by the same
 * conjugate-then-MPO path as ::moonlab_ca_mps_rx.
 * @param s     CA-MPS state, mutated in place.
 * @param q     Qubit index in [0, num_qubits).
 * @param theta Rotation angle in radians.
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -3 on allocation failure; -4 if the MPO apply fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_ry        (moonlab_ca_mps_t* s, uint32_t q, double theta);
/**
 * RZ(theta) = exp(-i theta Z_q / 2), applied by the same
 * conjugate-then-MPO path as ::moonlab_ca_mps_rx.
 * @param s     CA-MPS state, mutated in place.
 * @param q     Qubit index in [0, num_qubits).
 * @param theta Rotation angle in radians.
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -3 on allocation failure; -4 if the MPO apply fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_rz        (moonlab_ca_mps_t* s, uint32_t q, double theta);
/**
 * T = diag(1, e^{i pi/4}) on @p q.  Routed through
 * ::moonlab_ca_mps_rz with theta = pi/4, which differs from T only by
 * the global phase e^{i pi/8} and so leaves every observable
 * unchanged.
 * @param s CA-MPS state, mutated in place.
 * @param q Qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -3 on allocation failure; -4 if the MPO apply fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_t_gate    (moonlab_ca_mps_t* s, uint32_t q);
/**
 * T-dagger = diag(1, e^{-i pi/4}) on @p q, routed through
 * ::moonlab_ca_mps_rz with theta = -pi/4 (equal up to global phase).
 * @param s CA-MPS state, mutated in place.
 * @param q Qubit index in [0, num_qubits).
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -3 on allocation failure; -4 if the MPO apply fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_t_dagger  (moonlab_ca_mps_t* s, uint32_t q);
/**
 * Phase gate P(theta) = diag(1, e^{i theta}) on @p q, routed through
 * ::moonlab_ca_mps_rz with the same angle (equal up to the global
 * phase e^{i theta / 2}).
 * @param s     CA-MPS state, mutated in place.
 * @param q     Qubit index in [0, num_qubits).
 * @param theta Phase angle in radians.
 * @return 0 on success; -2 if @p s is NULL or @p q is out of range;
 *         -3 on allocation failure; -4 if the MPO apply fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_phase     (moonlab_ca_mps_t* s, uint32_t q, double theta);
/**
 * Controlled-RZ(theta), built as the standard phase-kickback sequence
 * RZ(t, theta/2), CNOT, RZ(t, -theta/2), CNOT over the primitives in
 * this header.
 * @param s     CA-MPS state, mutated in place.
 * @param c     Control qubit index in [0, num_qubits).
 * @param t     Target qubit index, distinct from @p c.
 * @param theta Rotation angle in radians.
 * @return 0 on success; -1 if @p s is NULL or @p c equals @p t; -2 if
 *         an index is out of range; -3 on allocation failure; -4 if a
 *         backend step fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_crz       (moonlab_ca_mps_t* s, uint32_t c, uint32_t t, double theta);
/**
 * Controlled-RX(theta), built as H_t CRZ(theta) H_t.
 * @param s     CA-MPS state, mutated in place.
 * @param c     Control qubit index in [0, num_qubits).
 * @param t     Target qubit index, distinct from @p c.
 * @param theta Rotation angle in radians.
 * @return 0 on success; -1 if @p s is NULL or @p c equals @p t; -2 if
 *         an index is out of range; -3 on allocation failure; -4 if a
 *         backend step fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_crx       (moonlab_ca_mps_t* s, uint32_t c, uint32_t t, double theta);
/**
 * Controlled-RY(theta), built as S_t CRX(theta) S-dagger_t.
 * @param s     CA-MPS state, mutated in place.
 * @param c     Control qubit index in [0, num_qubits).
 * @param t     Target qubit index, distinct from @p c.
 * @param theta Rotation angle in radians.
 * @return 0 on success; -1 if @p s is NULL or @p c equals @p t; -2 if
 *         an index is out of range; -3 on allocation failure; -4 if a
 *         backend step fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_cry       (moonlab_ca_mps_t* s, uint32_t c, uint32_t t, double theta);
/**
 * General single-qubit unitary U3, applied as the Euler sequence
 * RZ(lambda) then RY(theta) then RZ(phi) -- equal to the OpenQASM U3
 * up to the global phase e^{i(phi+lambda)/2}.  Costs three MPO
 * applications.
 * @param s      CA-MPS state, mutated in place.
 * @param q      Qubit index in [0, num_qubits).
 * @param theta  Polar angle in radians.
 * @param phi    Trailing Z-rotation angle in radians.
 * @param lambda Leading Z-rotation angle in radians.
 * @return 0 on success; -1 if @p s is NULL; -2 if @p q is out of range;
 *         -3 on allocation failure; -4 if an MPO apply fails.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_u3        (moonlab_ca_mps_t* s, uint32_t q,
                                double theta, double phi, double lambda);
/**
 * Toffoli (CCX): flips @p t when both @p c1 and @p c2 are |1>.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_toffoli   (moonlab_ca_mps_t* s,
                                uint32_t c1, uint32_t c2, uint32_t t);
/**
 * Fredkin (CSWAP): swaps @p t1 and @p t2 when @p c is |1>.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_fredkin   (moonlab_ca_mps_t* s,
                                uint32_t c, uint32_t t1, uint32_t t2);

/**
 * Apply exp(i theta P) for a Pauli string P (length = num_qubits).
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_pauli_rotation(moonlab_ca_mps_t* s,
                                   const uint8_t* pauli_string, double theta);
/**
 * Imaginary-time exp(-tau P) for a Pauli string P; non-unitary.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_imag_pauli_rotation(moonlab_ca_mps_t* s,
                                        const uint8_t* pauli_string, double tau);
/**
 * Restore unit norm after non-unitary evolution.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_normalize(moonlab_ca_mps_t* s);
/**
 * Current state norm.
 * @stability stable
 */
MOONLAB_API double moonlab_ca_mps_norm(const moonlab_ca_mps_t* s);

/**
 * <psi|P|psi> for a Pauli string P.
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_expect_pauli(const moonlab_ca_mps_t* s,
                                 const uint8_t* pauli_string,
                                 moonlab_complex_double* out_expval);
/** <psi|H|psi> for H = sum_k coeffs[k] * paulis[k]; paulis is laid out
 * @stability stable
 *  as `num_terms * num_qubits` bytes. */
MOONLAB_API int moonlab_ca_mps_expect_pauli_sum(const moonlab_ca_mps_t* s,
                                     const uint8_t* paulis,
                                     const moonlab_complex_double* coeffs,
                                     uint32_t num_terms,
                                     moonlab_complex_double* out_expval);
/**
 * Marginal P(Z_qubit = +1), in [0, 1].
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_prob_z(const moonlab_ca_mps_t* s,
                           uint32_t qubit, double* out_prob);

/**
 * @brief Q = C^dagger P C for the current Clifford C in @p s.
 *
 * Exposes the Clifford-conjugated Pauli string and accumulated phase so
 * consumers can inspect Q's support without recomputing the tableau
 * conjugation.  @p out_pauli must be at least
 * ::moonlab_ca_mps_num_qubits bytes; @p out_phase receives the
 * accumulated i^phase factor in {0,1,2,3}.
 *
 * Stable declaration with a plain `int` return.  The internal
 * `algorithms/tensor_network/ca_mps.h` declares the same symbol
 * returning `ca_mps_error_t`, which is a `typedef int`, so the two
 * declarations are compatible; re-declaring here with `int` keeps the
 * stable header free of the internal enum, exactly as the sibling
 * CA-MPS gate entry points above are already surfaced.
 *
 * @return 0 on success, negative on argument error.
 * @since 0.6.0
 * @stability stable
 */
MOONLAB_API int moonlab_ca_mps_conjugate_pauli(const moonlab_ca_mps_t* s,
                                    const uint8_t* in_pauli,
                                    uint8_t* out_pauli,
                                    int* out_phase);

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
 * @stability stable
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
 * @stability stable
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
 * @stability stable
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
 * @stability stable
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
 * @stability stable
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
 * @stability stable
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
 * @stability stable
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
 * @stability stable
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
 * @stability stable
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
 * @stability stable
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
 * @stability stable
 */
MOONLAB_API int moonlab_tdvp_evolve_to(moonlab_tdvp_engine_t* engine,
                                        double target_time);

/**
 * Current evolution time accumulated by the engine.
 * @stability stable
 */
MOONLAB_API double moonlab_tdvp_current_time(const moonlab_tdvp_engine_t* engine);

/** Current variational energy @c <psi|H|psi> evaluated after the
 * @stability stable
 *  most recent step (or at engine creation if no steps have run). */
MOONLAB_API double moonlab_tdvp_current_energy(const moonlab_tdvp_engine_t* engine);

/**
 * Current MPS norm @c <psi|psi>^{1/2}.
 * @stability stable
 */
MOONLAB_API double moonlab_tdvp_current_norm(const moonlab_tdvp_engine_t* engine);

/**
 * Peak bond dimension across the MPS factor after the latest step.
 * @stability stable
 */
MOONLAB_API uint32_t moonlab_tdvp_current_max_bond_dim(const moonlab_tdvp_engine_t* engine);

/**
 * Number of inter-site bonds (n_qubits - 1).
 * @stability stable
 */
MOONLAB_API uint32_t moonlab_tdvp_num_bonds(const moonlab_tdvp_engine_t* engine);

/**
 * @brief Current PID-selected chi for bond @p bond.
 *
 * Returns 0 when the adaptive controller is disabled or @p bond is
 * out of range; otherwise returns the dimension the controller chose
 * on the most recent two-site update of that bond.
 *
 * @since 0.4.1
 * @stability stable
 */
MOONLAB_API uint32_t moonlab_tdvp_bond_chi(const moonlab_tdvp_engine_t* engine,
                                            uint32_t bond);

/**
 * Number of steps currently recorded in the engine's history.
 * @stability stable
 */
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
 * @stability stable
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
 * @stability stable
 */
MOONLAB_API int moonlab_tdvp_history_get_bond_chi(const moonlab_tdvp_engine_t* engine,
                                                   uint32_t step,
                                                   uint32_t* out_chi,
                                                   uint32_t buf_capacity);

/**
 * Release the engine and all internal allocations.
 * @stability stable
 */
MOONLAB_API void moonlab_tdvp_engine_free(moonlab_tdvp_engine_t* engine);

/* ---- VQE exact gradient (stable from ABI 0.4.0) --------------------- */

/**
 * @brief Opaque handle to a VQE solver.
 *
 * The same object returned by @c vqe_solver_create (declared in
 * @c src/algorithms/vqe.h, installed as
 * @c quantumsim/algorithms/vqe.h).  Solver construction — Hamiltonian,
 * ansatz, optimizer, entropy context — lives on the exported
 * lower-level surface and is version-pinned rather than frozen; only
 * the gradient entry point below is part of the committed ABI.
 */
typedef struct vqe_solver moonlab_vqe_solver_t;

/**
 * @brief Compute the exact gradient dE/dtheta of a VQE energy.
 *
 * Thin stable-ABI wrapper over Moonlab's exact-gradient dispatch:
 * reverse-mode adjoint autograd for noise-free hardware-efficient
 * ansaetze (cost ~2 forward passes per Hamiltonian term, independent
 * of parameter count), analytic parameter-shift rule for every other
 * supported configuration.  Both paths are exact; this function never
 * computes a finite difference.  Downstream AD systems (e.g. the
 * Eshkol custom-VJP tape node) may therefore wrap it and multiply by
 * the incoming cotangent without violating an exact-AD contract.
 *
 * @param solver          Solver handle from @c vqe_solver_create.
 * @param parameters      Parameter vector theta; @p num_parameters
 *                        entries.
 * @param gradient_out    Output buffer for dE/dtheta; @p num_parameters
 *                        slots, fully overwritten on success.
 * @param num_parameters  Length of both arrays.  Must equal the
 *                        solver's ansatz parameter count; the
 *                        mismatch check is what makes this entry safe
 *                        to call through a language binding that
 *                        cannot see the ansatz struct.
 *
 * @return  0 on success.
 * @return -1 if any pointer is NULL.
 * @return -2 if @p num_parameters does not match the solver's ansatz.
 * @return -3 if the gradient computation failed internally.
 *
 * @since 1.1.0 (ABI 0.4.0)
 * @stability stable
 */
MOONLAB_API int moonlab_vqe_gradient(moonlab_vqe_solver_t* solver,
                                     const double* parameters,
                                     double* gradient_out,
                                     size_t num_parameters);

/* ---- VQE quantum geometry (stable from ABI 0.7.0) ------------------ */

/**
 * @brief Quantum geometric (Fubini-Study) metric of the ansatz state.
 *
 * @f$g_{ij} = \operatorname{Re}[\langle\partial_i\psi|\partial_j\psi\rangle -
 * \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle]@f$ for
 * the ideal (noise-free) trial state.  Exact analytic parameter derivatives
 * (generator insertion) for the built-in ansaetze; central differences for a
 * custom circuit, whose gate structure the library cannot see.
 *
 * @param solver          Solver handle from @c vqe_solver_create.
 * @param parameters      Parameter vector theta; @p num_parameters entries.
 * @param qgt_out         Output buffer, @p num_parameters *
 *                        @p num_parameters slots, written **row-major**:
 *                        element (i, j) is at index
 *                        `i * num_parameters + j`.  The matrix is real and
 *                        symmetric, so the layout is its own transpose, but
 *                        the index formula is the committed contract.
 * @param num_parameters  Length of @p parameters and the leading dimension
 *                        of @p qgt_out.  Must equal the solver's ansatz
 *                        parameter count.
 *
 * @return  0 on success.
 * @return -1 if any pointer is NULL.
 * @return -2 if @p num_parameters does not match the solver's ansatz.
 * @return -3 if the computation failed internally.
 *
 * @since 1.2.1 (ABI 0.7.0)
 * @stability stable
 */
MOONLAB_API int moonlab_vqe_qgt(moonlab_vqe_solver_t* solver,
                                const double* parameters,
                                double* qgt_out,
                                size_t num_parameters);

/**
 * @brief Berry curvature of the ansatz state: the imaginary half of the QGT.
 *
 * @f$F_{ij} = -2\operatorname{Im}[\langle\partial_i\psi|\partial_j\psi\rangle
 * - \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle]@f$,
 * from the same derivatives as ::moonlab_vqe_qgt.  Real and antisymmetric
 * (@f$F_{ii} = 0@f$, @f$F_{ji} = -F_{ij}@f$); the @f$-2\operatorname{Im}@f$
 * convention makes the flux integrate to the Berry phase around a closed
 * parameter loop.  A real ansatz gives @f$F = 0@f$ identically.
 *
 * @param solver          Solver handle from @c vqe_solver_create.
 * @param parameters      Parameter vector theta; @p num_parameters entries.
 * @param berry_out       Output buffer, @p num_parameters *
 *                        @p num_parameters slots, row-major as above.
 * @param num_parameters  Length of @p parameters.
 *
 * @return 0 on success; -1 NULL argument; -2 parameter-count mismatch;
 *         -3 internal failure.
 *
 * @since 1.2.1 (ABI 0.7.0)
 * @stability stable
 */
MOONLAB_API int moonlab_vqe_berry_curvature(moonlab_vqe_solver_t* solver,
                                            const double* parameters,
                                            double* berry_out,
                                            size_t num_parameters);

/**
 * @brief Natural-gradient direction @f$(g + \epsilon I)^{-1}\,\nabla E@f$.
 *
 * Preconditions @p gradient by the Tikhonov-regularised quantum geometric
 * tensor.  On failure (singular metric) callers fall back to @p gradient.
 *
 * @param solver          Solver handle from @c vqe_solver_create.
 * @param parameters      Parameter vector theta.
 * @param gradient        Energy gradient; @p num_parameters entries.
 * @param regularization  Shift added to the metric diagonal (e.g. 1e-3).
 * @param direction_out   Output direction; @p num_parameters slots.
 * @param num_parameters  Length of every array above.
 *
 * @return 0 on success; -1 NULL argument; -2 parameter-count mismatch;
 *         -3 internal failure.
 *
 * @since 1.2.1 (ABI 0.7.0)
 * @stability stable
 */
MOONLAB_API int moonlab_vqe_natural_gradient(moonlab_vqe_solver_t* solver,
                                             const double* parameters,
                                             const double* gradient,
                                             double regularization,
                                             double* direction_out,
                                             size_t num_parameters);

/* ---- Pointwise band geometry one-shots (stable from ABI 0.7.0) -----
 *
 * The momentum-space counterpart of the topology one-shots above: the
 * pointwise Fubini-Study metric and Berry curvature of a two-band Bloch
 * model's lower band, at one k.  Same convention as those one-shots --
 * scalar model parameters in, plain arrays out, no opaque handle across
 * the FFI boundary; the handle surface lives in the internal
 * `src/algorithms/quantum_geometry/qgt.h`.
 *
 * Intended consumers: QGTL, libirrep, SbNN, and the JS / WASM TS layer.
 */

/**
 * @brief Closed-form lower-band metric and Berry curvature of
 *        @f$H = \mathbf d\cdot\boldsymbol\sigma@f$ from an analytic
 *        @f$\mathbf d@f$ and its momentum gradients.
 *
 * @f$g_{\mu\nu} = \tfrac14\,\partial_\mu\hat{\mathbf d}\cdot
 * \partial_\nu\hat{\mathbf d}@f$ and @f$\Omega_{xy} = \tfrac12\,
 * \hat{\mathbf d}\cdot(\partial_x\hat{\mathbf d}\times
 * \partial_y\hat{\mathbf d})@f$.  Gauge-free and machine-precision: no
 * eigenvector, no finite difference.
 *
 * @param d         @f$\mathbf d@f$, 3 entries.
 * @param dx        @f$\partial_{k_x}\mathbf d@f$, 3 entries.
 * @param dy        @f$\partial_{k_y}\mathbf d@f$, 3 entries.
 * @param g_out     Row-major 2x2 metric, 4 slots.
 * @param omega_out Berry curvature @f$\Omega_{xy}@f$.  May be NULL.
 *
 * @return  0 on success.
 * @return -1 on a NULL argument.
 * @return -2 at a band touching, where the geometry is undefined.
 *
 * @since 1.2.1 (ABI 0.7.0)
 * @stability stable
 */
MOONLAB_API int moonlab_dsigma_metric_curvature(const double* d,
                                                const double* dx,
                                                const double* dy,
                                                double* g_out,
                                                double* omega_out);

/**
 * @brief Pointwise metric and Berry curvature of the QWZ lower band at @p k.
 *
 * Bundles model construction, the exact closed-form evaluation, and teardown
 * into one call.
 *
 * @param m         QWZ mass parameter.
 * @param k         Momentum, 2 entries.
 * @param g_out     Row-major 2x2 metric, 4 slots.
 * @param omega_out Berry curvature.  May be NULL.
 *
 * @return 0 on success; -1 NULL argument or allocation failure; -2 at a
 *         band touching.
 *
 * @since 1.2.1 (ABI 0.7.0)
 * @stability stable
 */
MOONLAB_API int moonlab_qwz_curvature_at(double m, const double* k,
                                         double* g_out, double* omega_out);

/**
 * @brief Pointwise metric and Berry curvature of the Haldane lower band
 *        at @p k.
 *
 * The next-nearest-neighbour identity term shifts both bands equally and does
 * not enter the geometry; the traceless part is what is evaluated.
 *
 * @param t1        Nearest-neighbour hopping.
 * @param t2        Next-nearest-neighbour hopping.
 * @param phi       Peierls phase.
 * @param m_stagger Sublattice staggering @f$M@f$.
 * @param k         Momentum, 2 entries.
 * @param g_out     Row-major 2x2 metric, 4 slots.
 * @param omega_out Berry curvature.  May be NULL.
 *
 * @return 0 on success; -1 NULL argument or allocation failure; -2 at a
 *         band touching.
 *
 * @since 1.2.1 (ABI 0.7.0)
 * @stability stable
 */
MOONLAB_API int moonlab_haldane_curvature_at(double t1, double t2, double phi,
                                             double m_stagger, const double* k,
                                             double* g_out, double* omega_out);

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
 * @stability stable
 */
MOONLAB_API const char* moonlab_status_string(int module, int status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MOONLAB_EXPORT_H */
