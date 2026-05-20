/**
 * @file    control_plane.c
 * @brief   POSIX-sockets implementation of the v0.8.7 control plane.
 *
 * Single-threaded blocking server.  Future TLS / HTTP/2 transports can
 * replace `send_all` / `recv_until_newline` without touching the
 * dispatch logic.
 */

#include "control_plane.h"

#include "../applications/moonlab_qgtl_backend.h"
#include "../crypto/sha3/sha3.h"

#ifdef MOONLAB_HAVE_TLS
#  include <openssl/ssl.h>
#  include <openssl/err.h>
#endif

/* ------------------------------------------------------------------
 * Transport abstraction (since v0.8.17).
 *
 * A `moonlab_io_t` is a thin polymorphism between plain TCP and
 * TLS-wrapped fd.  When `ssl` is non-NULL the helpers go through
 * SSL_read / SSL_write; otherwise they call recv / send on the fd
 * directly.  All other code paths -- header parsing, HMAC verify,
 * body alloc, execute, response framing -- stay transport-blind.
 * ------------------------------------------------------------------ */
typedef struct {
    int   fd;
#ifdef MOONLAB_HAVE_TLS
    void *ssl; /* SSL* when TLS-wrapped, else NULL. */
#endif
} moonlab_io_t;

static int io_send(moonlab_io_t *io, const void *buf, size_t len);
static int io_recv(moonlab_io_t *io,       void *buf, size_t len);

#ifdef MOONLAB_HAVE_TLS
static int io_send_tls(SSL *ssl, const void *buf, size_t len);
static int io_recv_tls(SSL *ssl,       void *buf, size_t len);
#endif

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdatomic.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>

/* ------------------------------------------------------------------
 * Request observability (since v0.8.13).
 *
 * One line per request to stderr, gated by `MOONLAB_CONTROL_LOG`:
 *   `[moonlab.control] <verb> n_qubits=<n> body=<bytes> shots=<N>
 *    wall_ms=<ms> rc=<status>`
 * ------------------------------------------------------------------ */

static int log_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("MOONLAB_CONTROL_LOG");
        cached = (v && *v && *v != '0') ? 1 : 0;
    }
    return cached;
}

static double monotonic_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* Forward declarations for the v0.8.15 HMAC helpers (defined later
 * in the file alongside the server lifecycle handle). */
#define HMAC_BLOCK_SIZE  136
#define HMAC_DIGEST_SIZE 32
static void hmac_sha3_256(const uint8_t *key, size_t key_len,
                          const uint8_t *msg, size_t msg_len,
                          uint8_t out[HMAC_DIGEST_SIZE]);
static void hex_encode(const uint8_t *bin, size_t len, char *hex_out);
static int  ct_memcmp(const void *a, const void *b, size_t n);

/* Forward declarations for v0.8.23 metric counters (defined alongside
 * the server lifecycle struct). */
extern _Atomic uint64_t g_count_circuit;
extern _Atomic uint64_t g_count_shots;
extern _Atomic uint64_t g_count_health;
extern _Atomic uint64_t g_count_metrics;
extern _Atomic uint64_t g_count_rejected;
extern _Atomic uint64_t g_count_rate_limited;

/* ------------------------------------------------------------------
 * Wire helpers.
 * ------------------------------------------------------------------ */

static int send_all(moonlab_io_t *io, const void *buf, size_t len)
{
    const char *p = (const char *)buf;
    while (len > 0) {
        int sent = io_send(io, p, len);
        if (sent < 0) return MOONLAB_CONTROL_IO_ERROR;
        if (sent == 0) return MOONLAB_CONTROL_IO_ERROR;
        p   += (size_t)sent;
        len -= (size_t)sent;
    }
    return MOONLAB_CONTROL_OK;
}

static int recv_all(moonlab_io_t *io, void *buf, size_t len)
{
    char *p = (char *)buf;
    while (len > 0) {
        int got = io_recv(io, p, len);
        if (got < 0) return MOONLAB_CONTROL_IO_ERROR;
        if (got == 0) return MOONLAB_CONTROL_PROTOCOL; /* short read */
        p   += (size_t)got;
        len -= (size_t)got;
    }
    return MOONLAB_CONTROL_OK;
}

/* Read up to `cap-1` bytes, stopping after a '\n' (which is included).
 * NUL-terminates.  Returns OK on success or an error code. */
static int recv_until_newline(moonlab_io_t *io, char *buf, size_t cap, size_t *out_len)
{
    if (cap < 2) return MOONLAB_CONTROL_BAD_ARG;
    size_t pos = 0;
    while (pos + 1 < cap) {
        char c;
        int got = io_recv(io, &c, 1);
        if (got < 0) return MOONLAB_CONTROL_IO_ERROR;
        if (got == 0) return MOONLAB_CONTROL_PROTOCOL;
        buf[pos++] = c;
        if (c == '\n') break;
    }
    buf[pos] = '\0';
    if (out_len) *out_len = pos;
    return MOONLAB_CONTROL_OK;
}

static int send_err(moonlab_io_t *io, int status, const char *msg)
{
    char hdr[128];
    int n = snprintf(hdr, sizeof(hdr), "ERR %d %s\n",
                     status, msg ? msg : "error");
    if (n < 0 || (size_t)n >= sizeof(hdr)) return MOONLAB_CONTROL_IO_ERROR;
    return send_all(io, hdr, (size_t)n);
}

/* Transport dispatch.  Plain TCP routes through recv()/send(); TLS
 * routes through SSL_read()/SSL_write() when MOONLAB_HAVE_TLS. */
static int io_send(moonlab_io_t *io, const void *buf, size_t len)
{
#ifdef MOONLAB_HAVE_TLS
    if (io->ssl) return io_send_tls((SSL *)io->ssl, buf, len);
#endif
    while (1) {
        ssize_t n = send(io->fd, buf, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        return (int)n;
    }
}

static int io_recv(moonlab_io_t *io, void *buf, size_t len)
{
#ifdef MOONLAB_HAVE_TLS
    if (io->ssl) return io_recv_tls((SSL *)io->ssl, buf, len);
#endif
    while (1) {
        ssize_t n = recv(io->fd, buf, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        return (int)n;
    }
}

#ifdef MOONLAB_HAVE_TLS
static int io_send_tls(SSL *ssl, const void *buf, size_t len)
{
    int n = SSL_write(ssl, buf, (int)len);
    if (n <= 0) return -1;
    return n;
}
static int io_recv_tls(SSL *ssl, void *buf, size_t len)
{
    int n = SSL_read(ssl, buf, (int)len);
    if (n <= 0) {
        int err = SSL_get_error(ssl, n);
        if (err == SSL_ERROR_ZERO_RETURN) return 0; /* clean shutdown */
        return -1;
    }
    return n;
}
#endif

/* ------------------------------------------------------------------
 * Server: one request, one response.
 * ------------------------------------------------------------------ */

/* Maximum bytes the server accepts in a single CIRCUIT/SHOTS payload.
 * Tuned for the moonlab-circuit v1 format -- a 32-qubit dense gate list
 * (~30k gates) fits comfortably in 4 MB.  Larger payloads are likely
 * abusive; we ERR out before allocating. */
#define MOONLAB_CONTROL_MAX_BODY_BYTES (1L << 22)   /* 4 MB. */
#define MOONLAB_CONTROL_MAX_SHOTS      (1L << 20)   /* 1 M samples (8 MB). */

static int handle_one_request(moonlab_io_t  *io,
                              const uint8_t *secret,
                              size_t         secret_len)
{
    const double t0 = log_enabled() ? monotonic_ms() : 0.0;
    const char  *log_verb       = "?";
    int          log_n_qubits   = -1;
    long         log_body_bytes = -1;
    long         log_shots      = 0;
    int          log_rc         = MOONLAB_CONTROL_OK;
#define LOG_AND_RETURN(rc_value) do {                                 \
    log_rc = (rc_value);                                              \
    if (log_rc != MOONLAB_CONTROL_OK) {                               \
        atomic_fetch_add(&g_count_rejected, 1);                       \
    }                                                                 \
    if (log_enabled()) {                                              \
        const double dt = monotonic_ms() - t0;                        \
        fprintf(stderr,                                               \
            "[moonlab.control] verb=%s n_qubits=%d body=%ld shots=%ld" \
            " wall_ms=%.2f rc=%d\n",                                  \
            log_verb, log_n_qubits, log_body_bytes, log_shots,        \
            dt, log_rc);                                              \
        fflush(stderr);                                               \
    }                                                                 \
    return log_rc;                                                    \
} while (0)

    /* Optional auth prelude (v0.8.15):
     *   AUTH <64-hex-chars>\n
     * If the server has a secret configured, the AUTH line is
     * required and the token must be HMAC-SHA3-256(secret, verb_line).
     */
    char client_token_hex[128];
    int  saw_auth = 0;
    char first[96];
    size_t first_len = 0;
    int rc = recv_until_newline(io, first, sizeof(first), &first_len);
    if (rc != MOONLAB_CONTROL_OK) LOG_AND_RETURN(rc);

    /* Health probe (v0.8.21).  Short-circuit before AUTH /
     * CIRCUIT parsing -- load-balancer probes are unauthenticated. */
    if (strncmp(first, "HEALTH", 6) == 0 &&
        (first_len == 6 || first[6] == '\n')) {
        log_verb = "HEALTH";
        atomic_fetch_add(&g_count_health, 1);
        const char *resp = "OK alive\n";
        (void)send_all(io, resp, strlen(resp));
        LOG_AND_RETURN(MOONLAB_CONTROL_OK);
    }

    /* Metrics scrape (v0.8.23).  Same unauth justification as HEALTH:
     * monitoring scrapers shouldn't need credentials.  Emits a
     * Prometheus text-format exposition based on the process-wide
     * atomic counters. */
    if (strncmp(first, "METRICS", 7) == 0 &&
        (first_len == 7 || first[7] == '\n')) {
        log_verb = "METRICS";
        atomic_fetch_add(&g_count_metrics, 1);

        char body[1024];
        int n = snprintf(body, sizeof(body),
            "# HELP moonlab_control_requests_total Total control-plane requests by verb.\n"
            "# TYPE moonlab_control_requests_total counter\n"
            "moonlab_control_requests_total{verb=\"CIRCUIT\"} %llu\n"
            "moonlab_control_requests_total{verb=\"SHOTS\"} %llu\n"
            "moonlab_control_requests_total{verb=\"HEALTH\"} %llu\n"
            "moonlab_control_requests_total{verb=\"METRICS\"} %llu\n"
            "# HELP moonlab_control_rejected_total Requests rejected by the server (bad input, auth, execute failure).\n"
            "# TYPE moonlab_control_rejected_total counter\n"
            "moonlab_control_rejected_total %llu\n"
            "# HELP moonlab_control_rate_limited_total Requests refused at the accept-loop by the per-IP token bucket.\n"
            "# TYPE moonlab_control_rate_limited_total counter\n"
            "moonlab_control_rate_limited_total %llu\n",
            (unsigned long long)atomic_load(&g_count_circuit),
            (unsigned long long)atomic_load(&g_count_shots),
            (unsigned long long)atomic_load(&g_count_health),
            (unsigned long long)atomic_load(&g_count_metrics),
            (unsigned long long)atomic_load(&g_count_rejected),
            (unsigned long long)atomic_load(&g_count_rate_limited));
        if (n < 0 || (size_t)n >= sizeof(body)) {
            LOG_AND_RETURN(MOONLAB_CONTROL_IO_ERROR);
        }
        char hdr[64];
        int hn = snprintf(hdr, sizeof(hdr), "METRICS %d\n", n);
        if (hn < 0 || (size_t)hn >= sizeof(hdr)) {
            LOG_AND_RETURN(MOONLAB_CONTROL_IO_ERROR);
        }
        int rc2 = send_all(io, hdr, (size_t)hn);
        if (rc2 == MOONLAB_CONTROL_OK) {
            rc2 = send_all(io, body, (size_t)n);
        }
        LOG_AND_RETURN(rc2);
    }

    if (strncmp(first, "AUTH ", 5) == 0) {
        /* Pull the 64-hex token. */
        if (first_len < 5 + 64 + 1) {
            send_err(io, MOONLAB_CONTROL_PROTOCOL, "short AUTH");
            LOG_AND_RETURN(MOONLAB_CONTROL_PROTOCOL);
        }
        memcpy(client_token_hex, first + 5, 64);
        client_token_hex[64] = '\0';
        saw_auth = 1;
    }

    /* Verb line.  When AUTH preceded, this is the second physical line.
     * When no AUTH, it's the first line we already received. */
    char hdr[96];
    size_t hdr_len = first_len;
    if (saw_auth) {
        rc = recv_until_newline(io, hdr, sizeof(hdr), &hdr_len);
        if (rc != MOONLAB_CONTROL_OK) LOG_AND_RETURN(rc);
    } else {
        memcpy(hdr, first, first_len);
        hdr[first_len] = '\0';
    }

    /* If the server requires auth, validate the token against the
     * verb line.  HMAC keyed on `secret`, message = the verb line
     * (`hdr`, including its trailing newline). */
    if (secret_len > 0) {
        if (!saw_auth) {
            send_err(io, MOONLAB_CONTROL_REJECTED, "missing AUTH");
            LOG_AND_RETURN(MOONLAB_CONTROL_REJECTED);
        }
        uint8_t expected[HMAC_DIGEST_SIZE];
        hmac_sha3_256(secret, secret_len,
                      (const uint8_t *)hdr, hdr_len, expected);
        char expected_hex[2 * HMAC_DIGEST_SIZE + 1];
        hex_encode(expected, HMAC_DIGEST_SIZE, expected_hex);
        if (ct_memcmp(expected_hex, client_token_hex, 64) != 0) {
            send_err(io, MOONLAB_CONTROL_REJECTED, "bad token");
            LOG_AND_RETURN(MOONLAB_CONTROL_REJECTED);
        }
    } else if (saw_auth) {
        /* Server doesn't require auth but client sent one.  Accept
         * gracefully -- this lets a client probe authenticated paths
         * before the server is configured with a secret. */
    }


    int  mode_shots = 0;
    long num_shots  = 0;
    long body_bytes = -1;

    if (strncmp(hdr, "CIRCUIT ", 8) == 0) {
        log_verb = "CIRCUIT";
        atomic_fetch_add(&g_count_circuit, 1);
        if (sscanf(hdr + 8, "%ld", &body_bytes) != 1) {
            send_err(io, MOONLAB_CONTROL_PROTOCOL,
                     "expected CIRCUIT <N>");
            LOG_AND_RETURN(MOONLAB_CONTROL_PROTOCOL);
        }
    } else if (strncmp(hdr, "SHOTS ", 6) == 0) {
        log_verb = "SHOTS";
        atomic_fetch_add(&g_count_shots, 1);
        if (sscanf(hdr + 6, "%ld %ld", &num_shots, &body_bytes) != 2) {
            send_err(io, MOONLAB_CONTROL_PROTOCOL,
                     "expected SHOTS <shots> <bytes>");
            LOG_AND_RETURN(MOONLAB_CONTROL_PROTOCOL);
        }
        if (num_shots <= 0 || num_shots > MOONLAB_CONTROL_MAX_SHOTS) {
            send_err(io, MOONLAB_CONTROL_BAD_ARG,
                     "shots out of range");
            LOG_AND_RETURN(MOONLAB_CONTROL_BAD_ARG);
        }
        mode_shots = 1;
        log_shots  = num_shots;
    } else {
        send_err(io, MOONLAB_CONTROL_PROTOCOL,
                 "unknown verb");
        LOG_AND_RETURN(MOONLAB_CONTROL_PROTOCOL);
    }

    log_body_bytes = body_bytes;

    if (body_bytes <= 0 || body_bytes > MOONLAB_CONTROL_MAX_BODY_BYTES) {
        send_err(io, MOONLAB_CONTROL_BAD_ARG,
                 "body bytes out of range");
        LOG_AND_RETURN(MOONLAB_CONTROL_BAD_ARG);
    }

    char *body = (char *)malloc((size_t)body_bytes + 1);
    if (!body) {
        send_err(io, MOONLAB_CONTROL_OOM, "body alloc");
        LOG_AND_RETURN(MOONLAB_CONTROL_OOM);
    }
    rc = recv_all(io, body, (size_t)body_bytes);
    if (rc != MOONLAB_CONTROL_OK) {
        free(body);
        LOG_AND_RETURN(rc);
    }
    body[body_bytes] = '\0';

    int status = 0;
    moonlab_qgtl_circuit_t *c =
        moonlab_qgtl_circuit_deserialize(body, (size_t)body_bytes, &status);
    free(body);
    if (!c) {
        send_err(io, MOONLAB_CONTROL_REJECTED, "deserialize");
        LOG_AND_RETURN(MOONLAB_CONTROL_REJECTED);
    }
    log_n_qubits = moonlab_qgtl_circuit_num_qubits(c);

    moonlab_qgtl_exec_options_t opts;
    memset(&opts, 0, sizeof(opts));
    if (mode_shots) {
        opts.num_shots = (int)num_shots;
        opts.rng_seed  = 0; /* clock-based */
    } else {
        opts.return_probabilities = 1;
    }

    moonlab_qgtl_results_t res;
    memset(&res, 0, sizeof(res));

    int er = moonlab_qgtl_execute(c, &opts, &res);
    moonlab_qgtl_circuit_free(c);

    if (er != 0) {
        moonlab_qgtl_results_free(&res);
        send_err(io, MOONLAB_CONTROL_REJECTED, "execute");
        LOG_AND_RETURN(MOONLAB_CONTROL_REJECTED);
    }

    if (mode_shots) {
        if (!res.outcomes) {
            moonlab_qgtl_results_free(&res);
            send_err(io, MOONLAB_CONTROL_REJECTED, "no outcomes");
            LOG_AND_RETURN(MOONLAB_CONTROL_REJECTED);
        }
        char ok_hdr[64];
        int hn = snprintf(ok_hdr, sizeof(ok_hdr), "SAMPLES %d\n", res.num_shots);
        if (hn < 0 || (size_t)hn >= sizeof(ok_hdr)) {
            moonlab_qgtl_results_free(&res);
            LOG_AND_RETURN(MOONLAB_CONTROL_IO_ERROR);
        }
        rc = send_all(io, ok_hdr, (size_t)hn);
        if (rc == MOONLAB_CONTROL_OK) {
            rc = send_all(io, res.outcomes,
                          (size_t)res.num_shots * sizeof(uint64_t));
        }
        moonlab_qgtl_results_free(&res);
        LOG_AND_RETURN(rc);
    } else {
        if (!res.probabilities) {
            moonlab_qgtl_results_free(&res);
            send_err(io, MOONLAB_CONTROL_REJECTED, "no probabilities");
            LOG_AND_RETURN(MOONLAB_CONTROL_REJECTED);
        }
        const size_t dim = (size_t)1 << res.num_qubits;
        char ok_hdr[64];
        int hn = snprintf(ok_hdr, sizeof(ok_hdr), "OK %zu\n", dim);
        if (hn < 0 || (size_t)hn >= sizeof(ok_hdr)) {
            moonlab_qgtl_results_free(&res);
            LOG_AND_RETURN(MOONLAB_CONTROL_IO_ERROR);
        }
        rc = send_all(io, ok_hdr, (size_t)hn);
        if (rc == MOONLAB_CONTROL_OK) {
            rc = send_all(io, res.probabilities, dim * sizeof(double));
        }
        moonlab_qgtl_results_free(&res);
        LOG_AND_RETURN(rc);
    }
#undef LOG_AND_RETURN
}

/* Per-connection worker thread context: owns the client fd + optional
 * TLS handle. */
typedef struct {
    int            client_fd;
#ifdef MOONLAB_HAVE_TLS
    void          *ssl_ctx;  /* SSL_CTX*, owned by the server (shared). */
#endif
    uint8_t        secret[256];
    size_t         secret_len;
} worker_ctx_t;

static int handle_one_request(moonlab_io_t  *io,
                              const uint8_t *secret,
                              size_t         secret_len);
static int rl_take_token(moonlab_control_server_t *s, uint32_t ip);

#ifdef MOONLAB_HAVE_TLS
/* Defined just below.  Wraps an accepted client fd in TLS using the
 * server's SSL_CTX; returns a new SSL* on success or NULL on
 * handshake failure (caller closes the fd). */
static SSL *tls_accept_fd(SSL_CTX *ctx, int fd);
#endif

static void *worker_thread(void *arg)
{
    worker_ctx_t *ctx = (worker_ctx_t *)arg;
    int     fd   = ctx->client_fd;
    uint8_t sec[256];
    size_t  slen = ctx->secret_len;
    memcpy(sec, ctx->secret, slen);

    moonlab_io_t io;
    io.fd = fd;
#ifdef MOONLAB_HAVE_TLS
    io.ssl = NULL;
    if (ctx->ssl_ctx) {
        io.ssl = tls_accept_fd((SSL_CTX *)ctx->ssl_ctx, fd);
        if (!io.ssl) {
            memset(sec, 0, sizeof(sec));
            memset(ctx->secret, 0, sizeof(ctx->secret));
            free(ctx);
            close(fd);
            return NULL;
        }
    }
#endif
    memset(ctx->secret, 0, sizeof(ctx->secret));
    free(ctx);

    (void)handle_one_request(&io, slen > 0 ? sec : NULL, slen);

#ifdef MOONLAB_HAVE_TLS
    if (io.ssl) {
        SSL_shutdown((SSL *)io.ssl);
        SSL_free((SSL *)io.ssl);
    }
#endif
    memset(sec, 0, sizeof(sec));
    close(fd);
    return NULL;
}

/* ------------------------------------------------------------------
 * Server lifecycle handle (since v0.8.13).
 *
 * A self-pipe wakes any blocked accept() when shutdown is signalled.
 * The select() call multiplexes the listen socket and the read end of
 * the pipe; whichever fires first determines whether we accept or
 * exit the loop.  pthreads servicing in-flight requests are joined
 * before close() returns.
 * ------------------------------------------------------------------ */

/* HMAC-SHA3-256 (rate 136 bytes for SHA3-256). */
static void hmac_sha3_256(const uint8_t *key, size_t key_len,
                          const uint8_t *msg, size_t msg_len,
                          uint8_t out[HMAC_DIGEST_SIZE])
{
    uint8_t k_pad[HMAC_BLOCK_SIZE];
    if (key_len > HMAC_BLOCK_SIZE) {
        sha3_256(key, key_len, k_pad);
        memset(k_pad + HMAC_DIGEST_SIZE, 0, HMAC_BLOCK_SIZE - HMAC_DIGEST_SIZE);
    } else {
        memcpy(k_pad, key, key_len);
        memset(k_pad + key_len, 0, HMAC_BLOCK_SIZE - key_len);
    }

    uint8_t ipad[HMAC_BLOCK_SIZE], opad[HMAC_BLOCK_SIZE];
    for (size_t i = 0; i < HMAC_BLOCK_SIZE; i++) {
        ipad[i] = k_pad[i] ^ 0x36;
        opad[i] = k_pad[i] ^ 0x5C;
    }

    sha3_ctx_t inner;
    sha3_256_init(&inner);
    sha3_update(&inner, ipad, HMAC_BLOCK_SIZE);
    sha3_update(&inner, msg, msg_len);
    uint8_t inner_digest[HMAC_DIGEST_SIZE];
    sha3_final(&inner, inner_digest);

    sha3_ctx_t outer;
    sha3_256_init(&outer);
    sha3_update(&outer, opad, HMAC_BLOCK_SIZE);
    sha3_update(&outer, inner_digest, HMAC_DIGEST_SIZE);
    sha3_final(&outer, out);
}

static void hex_encode(const uint8_t *bin, size_t len, char *hex_out)
{
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        hex_out[2 * i]     = digits[(bin[i] >> 4) & 0xF];
        hex_out[2 * i + 1] = digits[ bin[i]       & 0xF];
    }
    hex_out[2 * len] = '\0';
}

/* Constant-time comparison.  Returns 0 if equal, nonzero otherwise. */
static int ct_memcmp(const void *a, const void *b, size_t n)
{
    const uint8_t *aa = (const uint8_t *)a;
    const uint8_t *bb = (const uint8_t *)b;
    uint8_t diff = 0;
    for (size_t i = 0; i < n; i++) diff |= aa[i] ^ bb[i];
    return diff;
}

/* Token bucket for per-IP rate limiting (since v0.8.21).  Indexed by
 * source IPv4 address (network byte order).  Small fixed-size hash
 * table; on collision we evict the oldest entry.  Protected by a
 * pthread_mutex when accessed by accept-loop + worker threads. */
#define RL_TABLE_SIZE 256
typedef struct {
    uint32_t       ip;            /* 0 = empty slot */
    double         tokens;
    double         last_refill_ms;
} rl_bucket_t;

/* Process-wide atomic metric counters (since v0.8.23).  Indexed by
 * verb; rejected / rate-limited are aggregate.  Atomic so worker
 * threads can increment without contending on the rate-limit mutex.
 * One shared instance: even with multiple `moonlab_control_server_t`
 * handles in-process, all requests roll up to the same counters --
 * matches Prometheus' "per-process" exposition model. */
_Atomic uint64_t g_count_circuit      = 0;
_Atomic uint64_t g_count_shots        = 0;
_Atomic uint64_t g_count_health       = 0;
_Atomic uint64_t g_count_metrics      = 0;
_Atomic uint64_t g_count_rejected     = 0;
_Atomic uint64_t g_count_rate_limited = 0;

struct moonlab_control_server {
    int     srv_fd;
    int     wake_pipe[2];   /* [0] = read end, [1] = write end. */
    uint8_t secret[256];
    size_t  secret_len;
#ifdef MOONLAB_HAVE_TLS
    SSL_CTX *ssl_ctx;
#endif
    /* Rate limit (since v0.8.21).  rate_rps == 0 disables. */
    int             rate_rps;
    int             burst;
    pthread_mutex_t rl_lock;
    rl_bucket_t     rl_table[RL_TABLE_SIZE];
};

#ifdef MOONLAB_HAVE_TLS
/* One-time OpenSSL library init.  In OpenSSL 3 this is mostly a no-op
 * (the library auto-inits on first use), but call OPENSSL_init_ssl
 * explicitly with default-config-disabled so the test binary doesn't
 * accidentally pick up a system openssl.cnf. */
static void tls_init_once(void)
{
    static int done = 0;
    if (done) return;
    OPENSSL_init_ssl(OPENSSL_INIT_NO_LOAD_CONFIG, NULL);
    done = 1;
}

static SSL *tls_accept_fd(SSL_CTX *ctx, int fd)
{
    SSL *ssl = SSL_new(ctx);
    if (!ssl) return NULL;
    if (SSL_set_fd(ssl, fd) != 1) { SSL_free(ssl); return NULL; }
    if (SSL_accept(ssl) <= 0) {
        SSL_free(ssl);
        return NULL;
    }
    return ssl;
}
#endif

int moonlab_control_server_open(const char                 *host,
                                uint16_t                    port,
                                moonlab_control_server_t  **out_server,
                                uint16_t                   *out_port)
{
    if (!host || !out_server) return MOONLAB_CONTROL_BAD_ARG;
    *out_server = NULL;

    moonlab_control_server_t *s = (moonlab_control_server_t *)calloc(1, sizeof(*s));
    if (!s) return MOONLAB_CONTROL_OOM;
    s->srv_fd      = -1;
    s->wake_pipe[0] = -1;
    s->wake_pipe[1] = -1;
    pthread_mutex_init(&s->rl_lock, NULL);

    if (pipe(s->wake_pipe) != 0) {
        free(s);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    /* Non-blocking read end so we never stall in select() processing. */
    int flags = fcntl(s->wake_pipe[0], F_GETFL, 0);
    fcntl(s->wake_pipe[0], F_SETFL, flags | O_NONBLOCK);

    s->srv_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (s->srv_fd < 0) {
        moonlab_control_server_close(s);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    int yes = 1;
    setsockopt(s->srv_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        moonlab_control_server_close(s);
        return MOONLAB_CONTROL_BAD_ARG;
    }

    if (bind(s->srv_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        moonlab_control_server_close(s);
        return MOONLAB_CONTROL_IO_ERROR;
    }

    if (out_port) {
        struct sockaddr_in actual;
        socklen_t alen = sizeof(actual);
        if (getsockname(s->srv_fd, (struct sockaddr *)&actual, &alen) == 0) {
            *out_port = ntohs(actual.sin_port);
        }
    }

    if (listen(s->srv_fd, 64) < 0) {
        moonlab_control_server_close(s);
        return MOONLAB_CONTROL_IO_ERROR;
    }

    *out_server = s;
    return MOONLAB_CONTROL_OK;
}

int moonlab_control_server_run(moonlab_control_server_t *s, int max_iters)
{
    if (!s || s->srv_fd < 0 || max_iters < 1) return MOONLAB_CONTROL_BAD_ARG;

    pthread_t *workers = (pthread_t *)calloc((size_t)max_iters, sizeof(pthread_t));
    if (!workers) return MOONLAB_CONTROL_OOM;

    int worker_count = 0;
    int served = 0;
    int rc = MOONLAB_CONTROL_OK;
    int shutting_down = 0;

    while (served < max_iters && !shutting_down) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(s->srv_fd, &readfds);
        FD_SET(s->wake_pipe[0], &readfds);
        int maxfd = s->srv_fd > s->wake_pipe[0] ? s->srv_fd : s->wake_pipe[0];

        int nr = select(maxfd + 1, &readfds, NULL, NULL, NULL);
        if (nr < 0) {
            if (errno == EINTR) continue;
            rc = MOONLAB_CONTROL_IO_ERROR;
            break;
        }

        if (FD_ISSET(s->wake_pipe[0], &readfds)) {
            /* Drain the pipe so a future open()->run() pair starts clean. */
            char drain[16];
            while (read(s->wake_pipe[0], drain, sizeof(drain)) > 0) { }
            shutting_down = 1;
            break;
        }

        if (!FD_ISSET(s->srv_fd, &readfds)) continue;

        struct sockaddr_in peer;
        socklen_t          peerlen = sizeof(peer);
        int client = accept(s->srv_fd, (struct sockaddr *)&peer, &peerlen);
        if (client < 0) {
            if (errno == EINTR) continue;
            rc = MOONLAB_CONTROL_IO_ERROR;
            break;
        }

        /* Rate-limit check (since v0.8.21).  IPv4 source address only;
         * IPv6 deferred.  Failed clients receive a single short
         * ERR line then we close the socket. */
        if (!rl_take_token(s, peer.sin_addr.s_addr)) {
            atomic_fetch_add(&g_count_rate_limited, 1);
            const char *resp = "ERR -408 rate limited\n";
            (void)send(client, resp, strlen(resp), 0);
            close(client);
            served++;
            continue;
        }

        worker_ctx_t *ctx = (worker_ctx_t *)malloc(sizeof(*ctx));
        if (!ctx) {
            moonlab_io_t inline_io = { 0 };
            inline_io.fd = client;
            (void)handle_one_request(&inline_io,
                                     s->secret_len > 0 ? s->secret : NULL,
                                     s->secret_len);
            close(client);
            served++;
            continue;
        }
        ctx->client_fd  = client;
#ifdef MOONLAB_HAVE_TLS
        ctx->ssl_ctx    = s->ssl_ctx;
#endif
        ctx->secret_len = s->secret_len;
        if (s->secret_len > 0) memcpy(ctx->secret, s->secret, s->secret_len);

        if (pthread_create(&workers[worker_count], NULL, worker_thread, ctx) != 0) {
            moonlab_io_t inline_io = { 0 };
            inline_io.fd = client;
            (void)handle_one_request(&inline_io,
                                     ctx->secret_len > 0 ? ctx->secret : NULL,
                                     ctx->secret_len);
            close(client);
            memset(ctx->secret, 0, sizeof(ctx->secret));
            free(ctx);
        } else {
            worker_count++;
        }
        served++;
    }

    for (int i = 0; i < worker_count; i++) {
        pthread_join(workers[i], NULL);
    }
    free(workers);
    return rc;
}

void moonlab_control_server_shutdown(moonlab_control_server_t *s)
{
    if (!s || s->wake_pipe[1] < 0) return;
    const char b = 1;
    /* `write` on a pipe with a single byte is async-signal-safe.
     * Ignore the result -- a closed pipe just means we already shut
     * down, and any partial-write retry isn't useful here. */
    ssize_t ignored = write(s->wake_pipe[1], &b, 1);
    (void)ignored;
}

void moonlab_control_server_close(moonlab_control_server_t *s)
{
    if (!s) return;
    if (s->srv_fd      >= 0) { close(s->srv_fd);      s->srv_fd      = -1; }
    if (s->wake_pipe[0] >= 0) { close(s->wake_pipe[0]); s->wake_pipe[0] = -1; }
    if (s->wake_pipe[1] >= 0) { close(s->wake_pipe[1]); s->wake_pipe[1] = -1; }
    pthread_mutex_destroy(&s->rl_lock);
#ifdef MOONLAB_HAVE_TLS
    if (s->ssl_ctx) { SSL_CTX_free(s->ssl_ctx); s->ssl_ctx = NULL; }
#endif
    memset(s->secret, 0, sizeof(s->secret));
    s->secret_len = 0;
    free(s);
}

int moonlab_control_server_require_client_cert(moonlab_control_server_t *s,
                                               const char               *client_ca_path)
{
#ifdef MOONLAB_HAVE_TLS
    if (!s) return MOONLAB_CONTROL_BAD_ARG;
    if (!s->ssl_ctx) return MOONLAB_CONTROL_BAD_ARG; /* must use_tls() first */

    if (!client_ca_path) {
        /* Clear mode: stop demanding client certs. */
        SSL_CTX_set_verify(s->ssl_ctx, SSL_VERIFY_NONE, NULL);
        return MOONLAB_CONTROL_OK;
    }

    if (SSL_CTX_load_verify_locations(s->ssl_ctx, client_ca_path, NULL) <= 0) {
        return MOONLAB_CONTROL_TLS_ERROR;
    }
    /* SSL_VERIFY_FAIL_IF_NO_PEER_CERT makes accept() reject any client
     * that doesn't present a cert at all.  SSL_VERIFY_PEER alone would
     * skip the check when no cert is offered. */
    SSL_CTX_set_verify(s->ssl_ctx,
                       SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT,
                       NULL);
    return MOONLAB_CONTROL_OK;
#else
    (void)s; (void)client_ca_path;
    return MOONLAB_CONTROL_BAD_ARG;
#endif
}

int moonlab_control_server_use_tls(moonlab_control_server_t *s,
                                   const char               *cert_path,
                                   const char               *key_path)
{
#ifdef MOONLAB_HAVE_TLS
    if (!s) return MOONLAB_CONTROL_BAD_ARG;
    tls_init_once();

    /* Clearing path: cert == NULL disables TLS for future accepts. */
    if (!cert_path && !key_path) {
        if (s->ssl_ctx) { SSL_CTX_free(s->ssl_ctx); s->ssl_ctx = NULL; }
        return MOONLAB_CONTROL_OK;
    }
    if (!cert_path || !key_path) return MOONLAB_CONTROL_BAD_ARG;

    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx) return MOONLAB_CONTROL_TLS_ERROR;

    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);

    if (SSL_CTX_use_certificate_file(ctx, cert_path, SSL_FILETYPE_PEM) <= 0) {
        SSL_CTX_free(ctx);
        return MOONLAB_CONTROL_TLS_ERROR;
    }
    if (SSL_CTX_use_PrivateKey_file(ctx, key_path, SSL_FILETYPE_PEM) <= 0) {
        SSL_CTX_free(ctx);
        return MOONLAB_CONTROL_TLS_ERROR;
    }
    if (SSL_CTX_check_private_key(ctx) <= 0) {
        SSL_CTX_free(ctx);
        return MOONLAB_CONTROL_TLS_ERROR;
    }

    if (s->ssl_ctx) SSL_CTX_free(s->ssl_ctx);
    s->ssl_ctx = ctx;
    return MOONLAB_CONTROL_OK;
#else
    (void)s; (void)cert_path; (void)key_path;
    return MOONLAB_CONTROL_BAD_ARG;
#endif
}

/* Public wrapper around the internal HMAC -- since v0.8.16, so
 * binding-language clients (Rust binding) can construct AUTH tokens
 * without re-implementing HMAC-SHA3-256.  Python uses `hashlib +
 * hmac` directly and doesn't need this entry point. */
void moonlab_control_hmac_sha3_256(const uint8_t *secret, size_t secret_len,
                                   const uint8_t *msg,    size_t msg_len,
                                   uint8_t        out_digest[32])
{
    hmac_sha3_256(secret, secret_len, msg, msg_len, out_digest);
}

int moonlab_control_server_set_rate_limit(moonlab_control_server_t *s,
                                          int rate_rps,
                                          int burst)
{
    if (!s || rate_rps < 0 || burst < 0) return MOONLAB_CONTROL_BAD_ARG;
    pthread_mutex_lock(&s->rl_lock);
    s->rate_rps = rate_rps;
    s->burst    = burst > 0 ? burst : (rate_rps > 0 ? rate_rps * 2 : 0);
    memset(s->rl_table, 0, sizeof(s->rl_table));
    pthread_mutex_unlock(&s->rl_lock);
    return MOONLAB_CONTROL_OK;
}

/* Returns 1 if the request from `ip` is permitted; 0 if the bucket is
 * empty (rate-limited).  Refills proportional to elapsed wall time. */
static int rl_take_token(moonlab_control_server_t *s, uint32_t ip)
{
    if (s->rate_rps <= 0) return 1;

    const double now_ms = monotonic_ms();
    const size_t slot   = (ip * 2654435761u) & (RL_TABLE_SIZE - 1);

    pthread_mutex_lock(&s->rl_lock);
    rl_bucket_t *b = &s->rl_table[slot];
    if (b->ip != ip) {
        /* Collision or empty slot.  Evict + reset. */
        b->ip             = ip;
        b->tokens         = (double)s->burst;
        b->last_refill_ms = now_ms;
    } else {
        const double dt_s = (now_ms - b->last_refill_ms) / 1000.0;
        b->tokens         = fmin((double)s->burst,
                                 b->tokens + dt_s * (double)s->rate_rps);
        b->last_refill_ms = now_ms;
    }
    int ok = 0;
    if (b->tokens >= 1.0) { b->tokens -= 1.0; ok = 1; }
    pthread_mutex_unlock(&s->rl_lock);
    return ok;
}

int moonlab_control_server_set_secret(moonlab_control_server_t *s,
                                      const uint8_t *secret,
                                      size_t         secret_len)
{
    if (!s) return MOONLAB_CONTROL_BAD_ARG;
    if (secret_len > sizeof(s->secret)) return MOONLAB_CONTROL_BAD_ARG;

    memset(s->secret, 0, sizeof(s->secret));
    if (secret && secret_len > 0) {
        memcpy(s->secret, secret, secret_len);
        s->secret_len = secret_len;
    } else {
        s->secret_len = 0;
    }
    return MOONLAB_CONTROL_OK;
}

int moonlab_control_serve(const char *host,
                          uint16_t    port,
                          int         max_iters,
                          uint16_t   *out_port)
{
    moonlab_control_server_t *s = NULL;
    int rc = moonlab_control_server_open(host, port, &s, out_port);
    if (rc != MOONLAB_CONTROL_OK) return rc;
    rc = moonlab_control_server_run(s, max_iters);
    moonlab_control_server_close(s);
    return rc;
}

/* ------------------------------------------------------------------
 * Client.
 * ------------------------------------------------------------------ */

int moonlab_control_submit_circuit(const char *host,
                                   uint16_t    port,
                                   const char *circuit_text,
                                   size_t      text_len,
                                   double    **out_probs,
                                   size_t     *out_num)
{
    if (!host || !circuit_text || !out_probs || !out_num) {
        return MOONLAB_CONTROL_BAD_ARG;
    }
    *out_probs = NULL;
    *out_num   = 0;

    if (text_len == 0) text_len = strlen(circuit_text);
    if (text_len == 0) return MOONLAB_CONTROL_BAD_ARG;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return MOONLAB_CONTROL_IO_ERROR;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd);
        return MOONLAB_CONTROL_BAD_ARG;
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    moonlab_io_t plain_io = { 0 };
    plain_io.fd = fd;

    char hdr[64];
    int hn = snprintf(hdr, sizeof(hdr), "CIRCUIT %zu\n", text_len);
    if (hn < 0 || (size_t)hn >= sizeof(hdr)) {
        close(fd);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    int rc = send_all(&plain_io, hdr, (size_t)hn);
    if (rc == MOONLAB_CONTROL_OK) {
        rc = send_all(&plain_io, circuit_text, text_len);
    }
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    char resp_hdr[128];
    size_t resp_len = 0;
    rc = recv_until_newline(&plain_io, resp_hdr, sizeof(resp_hdr), &resp_len);
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    if (strncmp(resp_hdr, "OK ", 3) == 0) {
        size_t num = 0;
        if (sscanf(resp_hdr + 3, "%zu", &num) != 1 || num == 0 || num > (1ULL << 30)) {
            close(fd);
            return MOONLAB_CONTROL_PROTOCOL;
        }
        double *probs = (double *)malloc(num * sizeof(double));
        if (!probs) { close(fd); return MOONLAB_CONTROL_OOM; }
        rc = recv_all(&plain_io, probs, num * sizeof(double));
        close(fd);
        if (rc != MOONLAB_CONTROL_OK) { free(probs); return rc; }
        *out_probs = probs;
        *out_num   = num;
        return MOONLAB_CONTROL_OK;
    } else {
        close(fd);
        return MOONLAB_CONTROL_REJECTED;
    }
}

int moonlab_control_submit_circuit_auth(const char    *host,
                                        uint16_t       port,
                                        const uint8_t *secret,
                                        size_t         secret_len,
                                        const char    *circuit_text,
                                        size_t         text_len,
                                        double       **out_probs,
                                        size_t        *out_num)
{
    if (!host || !circuit_text || !out_probs || !out_num) {
        return MOONLAB_CONTROL_BAD_ARG;
    }
    *out_probs = NULL;
    *out_num   = 0;

    if (text_len == 0) text_len = strlen(circuit_text);
    if (text_len == 0) return MOONLAB_CONTROL_BAD_ARG;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return MOONLAB_CONTROL_IO_ERROR;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd);
        return MOONLAB_CONTROL_BAD_ARG;
    }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    moonlab_io_t plain_io = { 0 };
    plain_io.fd = fd;

    char verb_line[64];
    int  vn = snprintf(verb_line, sizeof(verb_line),
                       "CIRCUIT %zu\n", text_len);
    if (vn < 0 || (size_t)vn >= sizeof(verb_line)) {
        close(fd);
        return MOONLAB_CONTROL_IO_ERROR;
    }

    int rc = MOONLAB_CONTROL_OK;
    if (secret && secret_len > 0) {
        uint8_t tok[HMAC_DIGEST_SIZE];
        hmac_sha3_256(secret, secret_len,
                      (const uint8_t *)verb_line, (size_t)vn, tok);
        char auth_line[5 + 2 * HMAC_DIGEST_SIZE + 2];
        memcpy(auth_line, "AUTH ", 5);
        hex_encode(tok, HMAC_DIGEST_SIZE, auth_line + 5);
        auth_line[5 + 64] = '\n';
        rc = send_all(&plain_io, auth_line, 5 + 64 + 1);
    }
    if (rc == MOONLAB_CONTROL_OK) {
        rc = send_all(&plain_io, verb_line, (size_t)vn);
    }
    if (rc == MOONLAB_CONTROL_OK) {
        rc = send_all(&plain_io, circuit_text, text_len);
    }
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    char resp_hdr[128];
    size_t resp_len = 0;
    rc = recv_until_newline(&plain_io, resp_hdr, sizeof(resp_hdr), &resp_len);
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    if (strncmp(resp_hdr, "OK ", 3) == 0) {
        size_t num = 0;
        if (sscanf(resp_hdr + 3, "%zu", &num) != 1 || num == 0 || num > (1ULL << 30)) {
            close(fd);
            return MOONLAB_CONTROL_PROTOCOL;
        }
        double *probs = (double *)malloc(num * sizeof(double));
        if (!probs) { close(fd); return MOONLAB_CONTROL_OOM; }
        rc = recv_all(&plain_io, probs, num * sizeof(double));
        close(fd);
        if (rc != MOONLAB_CONTROL_OK) { free(probs); return rc; }
        *out_probs = probs;
        *out_num   = num;
        return MOONLAB_CONTROL_OK;
    } else {
        close(fd);
        return MOONLAB_CONTROL_REJECTED;
    }
}

int moonlab_control_submit_circuit_shots(const char *host,
                                         uint16_t    port,
                                         const char *circuit_text,
                                         size_t      text_len,
                                         int         num_shots,
                                         uint64_t  **out_outcomes,
                                         size_t     *out_num)
{
    if (!host || !circuit_text || !out_outcomes || !out_num ||
        num_shots <= 0 || num_shots > MOONLAB_CONTROL_MAX_SHOTS) {
        return MOONLAB_CONTROL_BAD_ARG;
    }
    *out_outcomes = NULL;
    *out_num      = 0;

    if (text_len == 0) text_len = strlen(circuit_text);
    if (text_len == 0) return MOONLAB_CONTROL_BAD_ARG;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return MOONLAB_CONTROL_IO_ERROR;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd);
        return MOONLAB_CONTROL_BAD_ARG;
    }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    moonlab_io_t plain_io = { 0 };
    plain_io.fd = fd;

    char hdr[96];
    int hn = snprintf(hdr, sizeof(hdr), "SHOTS %d %zu\n",
                      num_shots, text_len);
    if (hn < 0 || (size_t)hn >= sizeof(hdr)) {
        close(fd);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    int rc = send_all(&plain_io, hdr, (size_t)hn);
    if (rc == MOONLAB_CONTROL_OK) {
        rc = send_all(&plain_io, circuit_text, text_len);
    }
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    char resp_hdr[128];
    size_t resp_len = 0;
    rc = recv_until_newline(&plain_io, resp_hdr, sizeof(resp_hdr), &resp_len);
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    if (strncmp(resp_hdr, "SAMPLES ", 8) == 0) {
        long shots_back = 0;
        if (sscanf(resp_hdr + 8, "%ld", &shots_back) != 1 ||
            shots_back <= 0 || shots_back > MOONLAB_CONTROL_MAX_SHOTS) {
            close(fd);
            return MOONLAB_CONTROL_PROTOCOL;
        }
        uint64_t *buf = (uint64_t *)malloc((size_t)shots_back * sizeof(uint64_t));
        if (!buf) { close(fd); return MOONLAB_CONTROL_OOM; }
        rc = recv_all(&plain_io, buf, (size_t)shots_back * sizeof(uint64_t));
        close(fd);
        if (rc != MOONLAB_CONTROL_OK) { free(buf); return rc; }
        *out_outcomes = buf;
        *out_num      = (size_t)shots_back;
        return MOONLAB_CONTROL_OK;
    } else {
        close(fd);
        return MOONLAB_CONTROL_REJECTED;
    }
}

#ifdef MOONLAB_HAVE_TLS
/* Shared TLS client core.  Both submit_circuit_tls (no client cert)
 * and submit_circuit_mtls (with client cert) delegate here. */
static int tls_client_submit_impl(const char *host, uint16_t port,
                                  const char *server_ca,
                                  const char *client_cert,
                                  const char *client_key,
                                  int insecure,
                                  const uint8_t *secret, size_t secret_len,
                                  const char *circuit_text, size_t text_len,
                                  double **out_probs, size_t *out_num)
{
    if (!host || !circuit_text || !out_probs || !out_num) {
        return MOONLAB_CONTROL_BAD_ARG;
    }
    *out_probs = NULL;
    *out_num   = 0;
    if (text_len == 0) text_len = strlen(circuit_text);
    if (text_len == 0) return MOONLAB_CONTROL_BAD_ARG;

    tls_init_once();
    SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) return MOONLAB_CONTROL_TLS_ERROR;
    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);

    if (insecure) {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, NULL);
    } else {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);
        if (server_ca) {
            if (SSL_CTX_load_verify_locations(ctx, server_ca, NULL) <= 0) {
                SSL_CTX_free(ctx); return MOONLAB_CONTROL_TLS_ERROR;
            }
        } else if (SSL_CTX_set_default_verify_paths(ctx) <= 0) {
            SSL_CTX_free(ctx); return MOONLAB_CONTROL_TLS_ERROR;
        }
    }

    if (client_cert) {
        if (SSL_CTX_use_certificate_file(ctx, client_cert, SSL_FILETYPE_PEM) <= 0) {
            SSL_CTX_free(ctx); return MOONLAB_CONTROL_TLS_ERROR;
        }
        if (!client_key) {
            SSL_CTX_free(ctx); return MOONLAB_CONTROL_BAD_ARG;
        }
        if (SSL_CTX_use_PrivateKey_file(ctx, client_key, SSL_FILETYPE_PEM) <= 0) {
            SSL_CTX_free(ctx); return MOONLAB_CONTROL_TLS_ERROR;
        }
        if (SSL_CTX_check_private_key(ctx) <= 0) {
            SSL_CTX_free(ctx); return MOONLAB_CONTROL_TLS_ERROR;
        }
    }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { SSL_CTX_free(ctx); return MOONLAB_CONTROL_IO_ERROR; }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd); SSL_CTX_free(ctx); return MOONLAB_CONTROL_BAD_ARG;
    }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd); SSL_CTX_free(ctx); return MOONLAB_CONTROL_IO_ERROR;
    }

    SSL *ssl = SSL_new(ctx);
    if (!ssl || SSL_set_fd(ssl, fd) != 1) {
        if (ssl) SSL_free(ssl);
        close(fd); SSL_CTX_free(ctx); return MOONLAB_CONTROL_TLS_ERROR;
    }
    if (!insecure) SSL_set_tlsext_host_name(ssl, host);
    if (SSL_connect(ssl) <= 0) {
        SSL_free(ssl); close(fd); SSL_CTX_free(ctx);
        return MOONLAB_CONTROL_TLS_ERROR;
    }

    moonlab_io_t tls_io = { 0 };
    tls_io.fd  = fd;
    tls_io.ssl = ssl;

    char verb_line[64];
    int vn = snprintf(verb_line, sizeof(verb_line), "CIRCUIT %zu\n", text_len);
    if (vn < 0 || (size_t)vn >= sizeof(verb_line)) {
        SSL_shutdown(ssl); SSL_free(ssl); close(fd); SSL_CTX_free(ctx);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    int rc = MOONLAB_CONTROL_OK;
    if (secret && secret_len > 0) {
        uint8_t tok[HMAC_DIGEST_SIZE];
        hmac_sha3_256(secret, secret_len,
                      (const uint8_t *)verb_line, (size_t)vn, tok);
        char auth_line[5 + 2 * HMAC_DIGEST_SIZE + 2];
        memcpy(auth_line, "AUTH ", 5);
        hex_encode(tok, HMAC_DIGEST_SIZE, auth_line + 5);
        auth_line[5 + 64] = '\n';
        rc = send_all(&tls_io, auth_line, 5 + 64 + 1);
    }
    if (rc == MOONLAB_CONTROL_OK) rc = send_all(&tls_io, verb_line, (size_t)vn);
    if (rc == MOONLAB_CONTROL_OK) rc = send_all(&tls_io, circuit_text, text_len);
    if (rc != MOONLAB_CONTROL_OK) {
        SSL_shutdown(ssl); SSL_free(ssl); close(fd); SSL_CTX_free(ctx);
        return rc;
    }

    char resp_hdr[128];
    size_t resp_len = 0;
    rc = recv_until_newline(&tls_io, resp_hdr, sizeof(resp_hdr), &resp_len);
    int final_rc = MOONLAB_CONTROL_PROTOCOL;
    if (rc == MOONLAB_CONTROL_OK && strncmp(resp_hdr, "OK ", 3) == 0) {
        size_t num = 0;
        if (sscanf(resp_hdr + 3, "%zu", &num) == 1 && num > 0 && num <= (1ULL << 30)) {
            double *probs = (double *)malloc(num * sizeof(double));
            if (probs) {
                if (recv_all(&tls_io, probs, num * sizeof(double)) == MOONLAB_CONTROL_OK) {
                    *out_probs = probs;
                    *out_num   = num;
                    final_rc   = MOONLAB_CONTROL_OK;
                } else {
                    free(probs);
                    final_rc = MOONLAB_CONTROL_IO_ERROR;
                }
            } else {
                final_rc = MOONLAB_CONTROL_OOM;
            }
        }
    } else if (rc == MOONLAB_CONTROL_OK && strncmp(resp_hdr, "ERR ", 4) == 0) {
        final_rc = MOONLAB_CONTROL_REJECTED;
    } else if (rc != MOONLAB_CONTROL_OK) {
        final_rc = rc;
    }

    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(fd);
    SSL_CTX_free(ctx);
    return final_rc;
}
#endif

int moonlab_control_submit_circuit_mtls(const char    *host,
                                        uint16_t       port,
                                        const char    *server_ca_path,
                                        const char    *client_cert_path,
                                        const char    *client_key_path,
                                        int            insecure,
                                        const uint8_t *secret,
                                        size_t         secret_len,
                                        const char    *circuit_text,
                                        size_t         text_len,
                                        double       **out_probs,
                                        size_t        *out_num)
{
#ifndef MOONLAB_HAVE_TLS
    (void)host; (void)port; (void)server_ca_path;
    (void)client_cert_path; (void)client_key_path; (void)insecure;
    (void)secret; (void)secret_len; (void)circuit_text; (void)text_len;
    (void)out_probs; (void)out_num;
    return MOONLAB_CONTROL_BAD_ARG;
#else
    return tls_client_submit_impl(host, port,
                                  server_ca_path,
                                  client_cert_path, client_key_path,
                                  insecure,
                                  secret, secret_len,
                                  circuit_text, text_len,
                                  out_probs, out_num);
#endif
}

/* ------------------------------------------------------------------
 * TLS client (since v0.8.17).
 * ------------------------------------------------------------------ */
int moonlab_control_submit_circuit_tls(const char    *host,
                                       uint16_t       port,
                                       const char    *ca_path,
                                       int            insecure,
                                       const uint8_t *secret,
                                       size_t         secret_len,
                                       const char    *circuit_text,
                                       size_t         text_len,
                                       double       **out_probs,
                                       size_t        *out_num)
{
#ifndef MOONLAB_HAVE_TLS
    (void)host; (void)port; (void)ca_path; (void)insecure;
    (void)secret; (void)secret_len; (void)circuit_text; (void)text_len;
    (void)out_probs; (void)out_num;
    return MOONLAB_CONTROL_BAD_ARG;
#else
    if (!host || !circuit_text || !out_probs || !out_num) {
        return MOONLAB_CONTROL_BAD_ARG;
    }
    *out_probs = NULL;
    *out_num   = 0;
    if (text_len == 0) text_len = strlen(circuit_text);
    if (text_len == 0) return MOONLAB_CONTROL_BAD_ARG;

    tls_init_once();
    SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) return MOONLAB_CONTROL_TLS_ERROR;
    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);

    if (insecure) {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, NULL);
    } else {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);
        if (ca_path) {
            if (SSL_CTX_load_verify_locations(ctx, ca_path, NULL) <= 0) {
                SSL_CTX_free(ctx); return MOONLAB_CONTROL_TLS_ERROR;
            }
        } else if (SSL_CTX_set_default_verify_paths(ctx) <= 0) {
            SSL_CTX_free(ctx); return MOONLAB_CONTROL_TLS_ERROR;
        }
    }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { SSL_CTX_free(ctx); return MOONLAB_CONTROL_IO_ERROR; }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd); SSL_CTX_free(ctx); return MOONLAB_CONTROL_BAD_ARG;
    }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd); SSL_CTX_free(ctx); return MOONLAB_CONTROL_IO_ERROR;
    }

    SSL *ssl = SSL_new(ctx);
    if (!ssl || SSL_set_fd(ssl, fd) != 1) {
        if (ssl) SSL_free(ssl);
        close(fd); SSL_CTX_free(ctx); return MOONLAB_CONTROL_TLS_ERROR;
    }
    if (!insecure) SSL_set_tlsext_host_name(ssl, host);
    if (SSL_connect(ssl) <= 0) {
        SSL_free(ssl); close(fd); SSL_CTX_free(ctx);
        return MOONLAB_CONTROL_TLS_ERROR;
    }

    moonlab_io_t tls_io = { 0 };
    tls_io.fd  = fd;
    tls_io.ssl = ssl;

    char verb_line[64];
    int vn = snprintf(verb_line, sizeof(verb_line), "CIRCUIT %zu\n", text_len);
    if (vn < 0 || (size_t)vn >= sizeof(verb_line)) {
        SSL_shutdown(ssl); SSL_free(ssl); close(fd); SSL_CTX_free(ctx);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    int rc = MOONLAB_CONTROL_OK;
    if (secret && secret_len > 0) {
        uint8_t tok[HMAC_DIGEST_SIZE];
        hmac_sha3_256(secret, secret_len,
                      (const uint8_t *)verb_line, (size_t)vn, tok);
        char auth_line[5 + 2 * HMAC_DIGEST_SIZE + 2];
        memcpy(auth_line, "AUTH ", 5);
        hex_encode(tok, HMAC_DIGEST_SIZE, auth_line + 5);
        auth_line[5 + 64] = '\n';
        rc = send_all(&tls_io, auth_line, 5 + 64 + 1);
    }
    if (rc == MOONLAB_CONTROL_OK) rc = send_all(&tls_io, verb_line, (size_t)vn);
    if (rc == MOONLAB_CONTROL_OK) rc = send_all(&tls_io, circuit_text, text_len);
    if (rc != MOONLAB_CONTROL_OK) {
        SSL_shutdown(ssl); SSL_free(ssl); close(fd); SSL_CTX_free(ctx);
        return rc;
    }

    char resp_hdr[128];
    size_t resp_len = 0;
    rc = recv_until_newline(&tls_io, resp_hdr, sizeof(resp_hdr), &resp_len);
    int final_rc = MOONLAB_CONTROL_PROTOCOL;
    if (rc == MOONLAB_CONTROL_OK && strncmp(resp_hdr, "OK ", 3) == 0) {
        size_t num = 0;
        if (sscanf(resp_hdr + 3, "%zu", &num) == 1 && num > 0 && num <= (1ULL << 30)) {
            double *probs = (double *)malloc(num * sizeof(double));
            if (probs) {
                if (recv_all(&tls_io, probs, num * sizeof(double)) == MOONLAB_CONTROL_OK) {
                    *out_probs = probs;
                    *out_num   = num;
                    final_rc   = MOONLAB_CONTROL_OK;
                } else {
                    free(probs);
                    final_rc = MOONLAB_CONTROL_IO_ERROR;
                }
            } else {
                final_rc = MOONLAB_CONTROL_OOM;
            }
        }
    } else if (rc == MOONLAB_CONTROL_OK && strncmp(resp_hdr, "ERR ", 4) == 0) {
        final_rc = MOONLAB_CONTROL_REJECTED;
    } else if (rc != MOONLAB_CONTROL_OK) {
        final_rc = rc;
    }

    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(fd);
    SSL_CTX_free(ctx);
    return final_rc;
#endif
}

/* ------------------------------------------------------------------
 * Health-check client (since v0.8.21).  Sends `HEALTH\n` over plain
 * TCP, expects `OK alive\n`.  No AUTH, no TLS -- intentionally simple
 * so a load-balancer probe can verify the listener without secrets.
 * ------------------------------------------------------------------ */
int moonlab_control_submit_metrics(const char *host, uint16_t port,
                                   char **out_text)
{
    if (!host || !out_text) return MOONLAB_CONTROL_BAD_ARG;
    *out_text = NULL;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return MOONLAB_CONTROL_IO_ERROR;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd); return MOONLAB_CONTROL_BAD_ARG;
    }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd); return MOONLAB_CONTROL_IO_ERROR;
    }

    moonlab_io_t plain_io = { 0 };
    plain_io.fd = fd;

    int rc = send_all(&plain_io, "METRICS\n", 8);
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    char resp_hdr[64];
    size_t resp_len = 0;
    rc = recv_until_newline(&plain_io, resp_hdr, sizeof(resp_hdr), &resp_len);
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }
    if (strncmp(resp_hdr, "METRICS ", 8) != 0) {
        close(fd); return MOONLAB_CONTROL_PROTOCOL;
    }

    long body_bytes = -1;
    if (sscanf(resp_hdr + 8, "%ld", &body_bytes) != 1 ||
        body_bytes <= 0 || body_bytes > (1L << 18)) {
        close(fd); return MOONLAB_CONTROL_PROTOCOL;
    }

    char *body = (char *)malloc((size_t)body_bytes + 1);
    if (!body) { close(fd); return MOONLAB_CONTROL_OOM; }
    rc = recv_all(&plain_io, body, (size_t)body_bytes);
    close(fd);
    if (rc != MOONLAB_CONTROL_OK) { free(body); return rc; }
    body[body_bytes] = '\0';
    *out_text = body;
    return MOONLAB_CONTROL_OK;
}

int moonlab_control_submit_health(const char *host, uint16_t port)
{
    if (!host) return MOONLAB_CONTROL_BAD_ARG;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return MOONLAB_CONTROL_IO_ERROR;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd); return MOONLAB_CONTROL_BAD_ARG;
    }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd); return MOONLAB_CONTROL_IO_ERROR;
    }

    moonlab_io_t plain_io = { 0 };
    plain_io.fd = fd;

    int rc = send_all(&plain_io, "HEALTH\n", 7);
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    char resp[64];
    size_t resp_len = 0;
    rc = recv_until_newline(&plain_io, resp, sizeof(resp), &resp_len);
    close(fd);
    if (rc != MOONLAB_CONTROL_OK) return rc;
    if (strncmp(resp, "OK alive", 8) == 0) return MOONLAB_CONTROL_OK;
    if (strncmp(resp, "ERR -408", 8) == 0) return MOONLAB_CONTROL_RATE_LIMITED;
    return MOONLAB_CONTROL_REJECTED;
}
