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

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CONTROL_PLANE_H */
