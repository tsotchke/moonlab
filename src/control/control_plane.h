/**
 * @file    control_plane.h
 * @brief   Minimal TCP control plane for remote circuit execution.
 *
 * Built on top of the v0.8.3 moonlab-circuit v1 wire format.  POSIX
 * sockets only -- no HTTP, no protobuf, no gRPC.  One request per
 * connection, length-prefixed binary, server is single-threaded.
 *
 * Wire protocol (server side, ASCII header + binary body):
 *
 *   Request (probability mode, v0.8.7+):
 *              `CIRCUIT <bytes>\n<bytes-of-moonlab-circuit-v1-text>`
 *
 *   Request (shots mode, v0.8.11+):
 *              `SHOTS <num_shots> <bytes>\n<bytes-of-...>`
 *
 *   Response on probability success:
 *              `OK <num_probabilities>\n<num_probabilities * 8 bytes of
 *               little-endian IEEE-754 doubles>`
 *
 *   Response on shots success:
 *              `SAMPLES <num_shots>\n<num_shots * 8 bytes of
 *               little-endian uint64 bitstring outcomes>`
 *
 *   Response on failure:
 *              `ERR <status_code> <message>\n`
 *
 * Client and server live in the same TU so future swap-in transports
 * (TLS, HTTP/2) only need to replace the two send/recv helpers.
 *
 * @since v0.8.7
 */

#ifndef MOONLAB_CONTROL_PLANE_H
#define MOONLAB_CONTROL_PLANE_H

#include "../applications/moonlab_api.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Status codes drawn from the
 *  `MOONLAB_STATUS_ERR_MODULE_BASE - 400..` band. */
#define MOONLAB_CONTROL_OK            ( 0)
#define MOONLAB_CONTROL_BAD_ARG       (-401)
#define MOONLAB_CONTROL_OOM           (-402)
#define MOONLAB_CONTROL_IO_ERROR      (-403) /**< socket I/O failure. */
#define MOONLAB_CONTROL_PROTOCOL      (-404) /**< malformed wire frame. */
#define MOONLAB_CONTROL_REJECTED      (-405) /**< server rejected the circuit. */
#define MOONLAB_CONTROL_TIMEOUT       (-406)

/**
 * @brief Run a single-shot, blocking control-plane server on
 *        `host`:`port`.  Returns after one client interaction, or
 *        after `max_iters` request/response cycles.
 *
 * @param[in]  host          Bind address, e.g. "127.0.0.1" or "0.0.0.0".
 * @param[in]  port          TCP port number.
 * @param[in]  max_iters     Number of request/response cycles to serve
 *                           before returning.  Use 1 for unit-test mode,
 *                           >=2 for soak tests, INT_MAX for daemon mode.
 * @param[out] out_port      Optional.  When `port == 0` the OS chooses a
 *                           port; on success this is filled with the
 *                           actual bound port.  Pass NULL to ignore.
 *
 * @return MOONLAB_CONTROL_OK if every cycle completed (or no client
 *         connected before SIGINT), or a negative code on bind/accept
 *         failure.
 */
MOONLAB_API int
moonlab_control_serve(const char *host,
                      uint16_t    port,
                      int         max_iters,
                      uint16_t   *out_port);

/**
 * @brief Submit a moonlab-circuit v1 text payload to a control-plane
 *        server and collect the probability vector.
 *
 * @param[in]  host          Server hostname / IP literal.
 * @param[in]  port          TCP port.
 * @param[in]  circuit_text  NUL-terminated serialized circuit.
 * @param[in]  text_len      Bytes in `circuit_text`; pass 0 to use
 *                           `strlen`.
 * @param[out] out_probs     On success, points to a malloc'd buffer of
 *                           `2^num_qubits` doubles.  Caller frees with
 *                           `free`.
 * @param[out] out_num       On success, the length of `*out_probs`.
 *
 * @return MOONLAB_CONTROL_OK on success or a negative code.
 *         On any failure, `*out_probs` is left NULL and `*out_num` 0.
 */
MOONLAB_API int
moonlab_control_submit_circuit(const char *host,
                               uint16_t    port,
                               const char *circuit_text,
                               size_t      text_len,
                               double    **out_probs,
                               size_t     *out_num);

/**
 * @brief Submit a circuit and request `num_shots` measurement
 *        samples rather than the full probability vector.  Returned
 *        outcomes are bitstring integers (least-significant bit =
 *        qubit 0); use bit tests against the result to score events.
 *
 * Since v0.8.11.
 *
 * @return MOONLAB_CONTROL_OK on success or a negative code.
 *         On any failure, `*out_outcomes` is left NULL.
 */
MOONLAB_API int
moonlab_control_submit_circuit_shots(const char *host,
                                     uint16_t    port,
                                     const char *circuit_text,
                                     size_t      text_len,
                                     int         num_shots,
                                     uint64_t  **out_outcomes,
                                     size_t     *out_num);

/* ------------------------------------------------------------------
 * Server lifecycle handle (since v0.8.13).
 *
 * Two-phase API for production deployment:
 *   1. `moonlab_control_server_open`  creates the listener (returns
 *      immediately; the OS-chosen port is filled into `out_port`).
 *   2. `moonlab_control_server_run`   blocks serving on the calling
 *      thread until shutdown is signalled or max_iters connections
 *      have been served.
 *   3. `moonlab_control_server_shutdown` may be called from any
 *      thread (or a signal handler) to wake the accept() in progress.
 *   4. `moonlab_control_server_close` releases the listener handle.
 *
 * The legacy `moonlab_control_serve` is still supported and is now
 * implemented atop these primitives.
 * ------------------------------------------------------------------ */

typedef struct moonlab_control_server moonlab_control_server_t;

/**
 * @brief Create + bind a control-plane listener without blocking.
 *
 * @param[in]  host       Bind address (e.g. "127.0.0.1" or "0.0.0.0").
 * @param[in]  port       TCP port; pass 0 to let the OS choose.
 * @param[out] out_server Owned handle.  Free via close().
 * @param[out] out_port   Optional.  Bound port (useful when port=0).
 *
 * @return MOONLAB_CONTROL_OK on success or a negative code.
 */
MOONLAB_API int
moonlab_control_server_open(const char                 *host,
                            uint16_t                    port,
                            moonlab_control_server_t  **out_server,
                            uint16_t                   *out_port);

/**
 * @brief Block serving on this thread until shutdown is signalled or
 *        `max_iters` connections have been served, whichever comes
 *        first.  Each accepted connection is dispatched to its own
 *        pthread; the function joins every worker before returning.
 *
 * @param[in]  server     Handle from `moonlab_control_server_open`.
 * @param[in]  max_iters  Max accepted connections.  Use `INT_MAX` for
 *                        daemon mode.
 *
 * @return MOONLAB_CONTROL_OK on clean shutdown, negative on I/O error.
 */
MOONLAB_API int
moonlab_control_server_run(moonlab_control_server_t *server,
                           int                       max_iters);

/**
 * @brief Signal a running server to stop after the current request.
 *        Thread-safe and safe to call from a signal handler (writes a
 *        single byte to a self-pipe; no malloc, no printf).
 */
MOONLAB_API void
moonlab_control_server_shutdown(moonlab_control_server_t *server);

/** @brief Release listener + self-pipe; idempotent on NULL. */
MOONLAB_API void
moonlab_control_server_close(moonlab_control_server_t *server);

/* ------------------------------------------------------------------
 * Authentication (since v0.8.15).
 *
 * Optional HMAC-SHA3-256 shared-secret token.  When a server is
 * configured with a non-empty secret, every request must arrive with
 * an `AUTH <hex-token>` line *before* the verb line:
 *
 *   AUTH 64-hex-char-HMAC\n
 *   CIRCUIT <bytes>\n<body>
 *
 * The token is `HMAC-SHA3-256(secret, verb-line)` -- i.e. the secret
 * is keyed across the second line, which itself carries the body
 * length.  An attacker who replays the wire then mutates the body
 * length will mismatch the HMAC and be rejected.  Tokens are
 * compared in constant time.
 *
 * The default (NULL or empty `secret`) is the legacy unauthenticated
 * mode for backward compatibility with the v0.8.7..v0.8.14 wire.
 * Set the env var `MOONLAB_CONTROL_SECRET` on the server to require
 * authentication; pass the same secret to the client via
 * `moonlab_control_submit_*_auth`.
 * ------------------------------------------------------------------ */

/**
 * @brief Set / clear the HMAC shared secret for this server.
 *        Pass NULL or empty `secret` to disable authentication.
 *        Safe to call after `open()` but before `run()`.
 *
 * @return MOONLAB_CONTROL_OK or MOONLAB_CONTROL_BAD_ARG.
 */
MOONLAB_API int
moonlab_control_server_set_secret(moonlab_control_server_t *server,
                                  const uint8_t            *secret,
                                  size_t                    secret_len);

/**
 * @brief Submit a circuit with an HMAC-SHA3-256 authentication token.
 *        Pass `secret = NULL` / `secret_len = 0` to fall back to the
 *        unauthenticated path (equivalent to
 *        `moonlab_control_submit_circuit`).
 */
MOONLAB_API int
moonlab_control_submit_circuit_auth(const char    *host,
                                    uint16_t       port,
                                    const uint8_t *secret,
                                    size_t         secret_len,
                                    const char    *circuit_text,
                                    size_t         text_len,
                                    double       **out_probs,
                                    size_t        *out_num);

/**
 * @brief Compute HMAC-SHA3-256(secret, msg).  Public so binding-language
 *        clients can construct the AUTH token themselves when they want
 *        to handle the socket directly rather than going through
 *        @ref moonlab_control_submit_circuit_auth.  Output is 32 bytes.
 */
MOONLAB_API void
moonlab_control_hmac_sha3_256(const uint8_t *secret, size_t secret_len,
                              const uint8_t *msg,    size_t msg_len,
                              uint8_t        out_digest[32]);

/* ------------------------------------------------------------------
 * TLS transport (since v0.8.17, gated by `MOONLAB_HAVE_TLS`).
 *
 * Wraps each connection in OpenSSL TLS 1.3.  HMAC auth still applies
 * inside the encrypted channel.  Server is configured with a PEM
 * certificate + private key; client either pins a CA bundle or runs
 * with `insecure = 1` for dev / self-signed certificates.
 *
 * `moonlab_control_*` calls return `MOONLAB_CONTROL_BAD_ARG` when the
 * shared library was built without `QSIM_ENABLE_TLS=ON`.
 * ------------------------------------------------------------------ */

/** @brief Status: TLS handshake / cert load / verification failure. */
#define MOONLAB_CONTROL_TLS_ERROR     (-407)

/**
 * @brief Configure the server to wrap every accepted connection in TLS.
 *        Loads the certificate + private key from PEM files.  Pass
 *        NULL for both to disable TLS on a previously configured server.
 *
 * @return MOONLAB_CONTROL_OK or a negative code.  Returns
 *         MOONLAB_CONTROL_BAD_ARG if the library was built without TLS.
 */
MOONLAB_API int
moonlab_control_server_use_tls(moonlab_control_server_t *server,
                               const char               *cert_path,
                               const char               *key_path);

/**
 * @brief Submit a circuit over a TLS-wrapped connection.
 *
 * @param[in]  ca_path   Optional CA bundle to pin against the server's
 *                       certificate.  Pass NULL when `insecure = 1`.
 * @param[in]  insecure  Set to 1 to skip peer verification (development /
 *                       self-signed certs).  Production deployments
 *                       should set 0 and supply `ca_path`.
 * @param[in]  secret / secret_len  Optional HMAC auth (matches the
 *                       v0.8.15 / v0.8.16 in-band AUTH prelude).
 */
MOONLAB_API int
moonlab_control_submit_circuit_tls(const char    *host,
                                   uint16_t       port,
                                   const char    *ca_path,
                                   int            insecure,
                                   const uint8_t *secret,
                                   size_t         secret_len,
                                   const char    *circuit_text,
                                   size_t         text_len,
                                   double       **out_probs,
                                   size_t        *out_num);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CONTROL_PLANE_H */
