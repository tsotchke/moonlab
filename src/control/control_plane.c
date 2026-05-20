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

#include <errno.h>
#include <fcntl.h>
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

/* ------------------------------------------------------------------
 * Wire helpers.
 * ------------------------------------------------------------------ */

static int send_all(int fd, const void *buf, size_t len)
{
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return MOONLAB_CONTROL_IO_ERROR;
        }
        if (n == 0) return MOONLAB_CONTROL_IO_ERROR;
        p   += (size_t)n;
        len -= (size_t)n;
    }
    return MOONLAB_CONTROL_OK;
}

static int recv_all(int fd, void *buf, size_t len)
{
    char *p = (char *)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return MOONLAB_CONTROL_IO_ERROR;
        }
        if (n == 0) return MOONLAB_CONTROL_PROTOCOL; /* short read */
        p   += (size_t)n;
        len -= (size_t)n;
    }
    return MOONLAB_CONTROL_OK;
}

/* Read up to `cap-1` bytes, stopping after a '\n' (which is included).
 * NUL-terminates.  Returns OK on success or an error code. */
static int recv_until_newline(int fd, char *buf, size_t cap, size_t *out_len)
{
    if (cap < 2) return MOONLAB_CONTROL_BAD_ARG;
    size_t pos = 0;
    while (pos + 1 < cap) {
        char c;
        ssize_t n = recv(fd, &c, 1, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return MOONLAB_CONTROL_IO_ERROR;
        }
        if (n == 0) return MOONLAB_CONTROL_PROTOCOL;
        buf[pos++] = c;
        if (c == '\n') break;
    }
    buf[pos] = '\0';
    if (out_len) *out_len = pos;
    return MOONLAB_CONTROL_OK;
}

static int send_err(int fd, int status, const char *msg)
{
    char hdr[128];
    int n = snprintf(hdr, sizeof(hdr), "ERR %d %s\n",
                     status, msg ? msg : "error");
    if (n < 0 || (size_t)n >= sizeof(hdr)) return MOONLAB_CONTROL_IO_ERROR;
    return send_all(fd, hdr, (size_t)n);
}

/* ------------------------------------------------------------------
 * Server: one request, one response.
 * ------------------------------------------------------------------ */

/* Maximum bytes the server accepts in a single CIRCUIT/SHOTS payload.
 * Tuned for the moonlab-circuit v1 format -- a 32-qubit dense gate list
 * (~30k gates) fits comfortably in 4 MB.  Larger payloads are likely
 * abusive; we ERR out before allocating. */
#define MOONLAB_CONTROL_MAX_BODY_BYTES (1L << 22)   /* 4 MB. */
#define MOONLAB_CONTROL_MAX_SHOTS      (1L << 20)   /* 1 M samples (8 MB). */

static int handle_one_request(int client_fd)
{
    const double t0 = log_enabled() ? monotonic_ms() : 0.0;
    const char  *log_verb       = "?";
    int          log_n_qubits   = -1;
    long         log_body_bytes = -1;
    long         log_shots      = 0;
    int          log_rc         = MOONLAB_CONTROL_OK;
#define LOG_AND_RETURN(rc_value) do {                                 \
    log_rc = (rc_value);                                              \
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

    /* Header.  Two verbs:
     *   CIRCUIT <bytes>\n         -- probability mode (v0.8.7+)
     *   SHOTS   <shots> <bytes>\n -- shots       mode (v0.8.11+) */
    char hdr[96];
    size_t hdr_len = 0;
    int rc = recv_until_newline(client_fd, hdr, sizeof(hdr), &hdr_len);
    if (rc != MOONLAB_CONTROL_OK) LOG_AND_RETURN(rc);

    int  mode_shots = 0;
    long num_shots  = 0;
    long body_bytes = -1;

    if (strncmp(hdr, "CIRCUIT ", 8) == 0) {
        log_verb = "CIRCUIT";
        if (sscanf(hdr + 8, "%ld", &body_bytes) != 1) {
            send_err(client_fd, MOONLAB_CONTROL_PROTOCOL,
                     "expected CIRCUIT <N>");
            LOG_AND_RETURN(MOONLAB_CONTROL_PROTOCOL);
        }
    } else if (strncmp(hdr, "SHOTS ", 6) == 0) {
        log_verb = "SHOTS";
        if (sscanf(hdr + 6, "%ld %ld", &num_shots, &body_bytes) != 2) {
            send_err(client_fd, MOONLAB_CONTROL_PROTOCOL,
                     "expected SHOTS <shots> <bytes>");
            LOG_AND_RETURN(MOONLAB_CONTROL_PROTOCOL);
        }
        if (num_shots <= 0 || num_shots > MOONLAB_CONTROL_MAX_SHOTS) {
            send_err(client_fd, MOONLAB_CONTROL_BAD_ARG,
                     "shots out of range");
            LOG_AND_RETURN(MOONLAB_CONTROL_BAD_ARG);
        }
        mode_shots = 1;
        log_shots  = num_shots;
    } else {
        send_err(client_fd, MOONLAB_CONTROL_PROTOCOL,
                 "unknown verb");
        LOG_AND_RETURN(MOONLAB_CONTROL_PROTOCOL);
    }

    log_body_bytes = body_bytes;

    if (body_bytes <= 0 || body_bytes > MOONLAB_CONTROL_MAX_BODY_BYTES) {
        send_err(client_fd, MOONLAB_CONTROL_BAD_ARG,
                 "body bytes out of range");
        LOG_AND_RETURN(MOONLAB_CONTROL_BAD_ARG);
    }

    char *body = (char *)malloc((size_t)body_bytes + 1);
    if (!body) {
        send_err(client_fd, MOONLAB_CONTROL_OOM, "body alloc");
        LOG_AND_RETURN(MOONLAB_CONTROL_OOM);
    }
    rc = recv_all(client_fd, body, (size_t)body_bytes);
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
        send_err(client_fd, MOONLAB_CONTROL_REJECTED, "deserialize");
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
        send_err(client_fd, MOONLAB_CONTROL_REJECTED, "execute");
        LOG_AND_RETURN(MOONLAB_CONTROL_REJECTED);
    }

    if (mode_shots) {
        if (!res.outcomes) {
            moonlab_qgtl_results_free(&res);
            send_err(client_fd, MOONLAB_CONTROL_REJECTED, "no outcomes");
            LOG_AND_RETURN(MOONLAB_CONTROL_REJECTED);
        }
        char ok_hdr[64];
        int hn = snprintf(ok_hdr, sizeof(ok_hdr), "SAMPLES %d\n", res.num_shots);
        if (hn < 0 || (size_t)hn >= sizeof(ok_hdr)) {
            moonlab_qgtl_results_free(&res);
            LOG_AND_RETURN(MOONLAB_CONTROL_IO_ERROR);
        }
        rc = send_all(client_fd, ok_hdr, (size_t)hn);
        if (rc == MOONLAB_CONTROL_OK) {
            rc = send_all(client_fd, res.outcomes,
                          (size_t)res.num_shots * sizeof(uint64_t));
        }
        moonlab_qgtl_results_free(&res);
        LOG_AND_RETURN(rc);
    } else {
        if (!res.probabilities) {
            moonlab_qgtl_results_free(&res);
            send_err(client_fd, MOONLAB_CONTROL_REJECTED, "no probabilities");
            LOG_AND_RETURN(MOONLAB_CONTROL_REJECTED);
        }
        const size_t dim = (size_t)1 << res.num_qubits;
        char ok_hdr[64];
        int hn = snprintf(ok_hdr, sizeof(ok_hdr), "OK %zu\n", dim);
        if (hn < 0 || (size_t)hn >= sizeof(ok_hdr)) {
            moonlab_qgtl_results_free(&res);
            LOG_AND_RETURN(MOONLAB_CONTROL_IO_ERROR);
        }
        rc = send_all(client_fd, ok_hdr, (size_t)hn);
        if (rc == MOONLAB_CONTROL_OK) {
            rc = send_all(client_fd, res.probabilities, dim * sizeof(double));
        }
        moonlab_qgtl_results_free(&res);
        LOG_AND_RETURN(rc);
    }
#undef LOG_AND_RETURN
}

/* Per-connection worker thread context: owns the client fd. */
typedef struct {
    int client_fd;
} worker_ctx_t;

static void *worker_thread(void *arg)
{
    worker_ctx_t *ctx = (worker_ctx_t *)arg;
    int fd = ctx->client_fd;
    free(ctx);
    (void)handle_one_request(fd);
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

struct moonlab_control_server {
    int srv_fd;
    int wake_pipe[2];   /* [0] = read end, [1] = write end. */
};

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

        int client = accept(s->srv_fd, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            rc = MOONLAB_CONTROL_IO_ERROR;
            break;
        }

        worker_ctx_t *ctx = (worker_ctx_t *)malloc(sizeof(*ctx));
        if (!ctx) {
            (void)handle_one_request(client);
            close(client);
            served++;
            continue;
        }
        ctx->client_fd = client;

        if (pthread_create(&workers[worker_count], NULL, worker_thread, ctx) != 0) {
            (void)handle_one_request(client);
            close(client);
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
    free(s);
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

    char hdr[64];
    int hn = snprintf(hdr, sizeof(hdr), "CIRCUIT %zu\n", text_len);
    if (hn < 0 || (size_t)hn >= sizeof(hdr)) {
        close(fd);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    int rc = send_all(fd, hdr, (size_t)hn);
    if (rc == MOONLAB_CONTROL_OK) {
        rc = send_all(fd, circuit_text, text_len);
    }
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    char resp_hdr[128];
    size_t resp_len = 0;
    rc = recv_until_newline(fd, resp_hdr, sizeof(resp_hdr), &resp_len);
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    if (strncmp(resp_hdr, "OK ", 3) == 0) {
        size_t num = 0;
        if (sscanf(resp_hdr + 3, "%zu", &num) != 1 || num == 0 || num > (1ULL << 30)) {
            close(fd);
            return MOONLAB_CONTROL_PROTOCOL;
        }
        double *probs = (double *)malloc(num * sizeof(double));
        if (!probs) { close(fd); return MOONLAB_CONTROL_OOM; }
        rc = recv_all(fd, probs, num * sizeof(double));
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

    char hdr[96];
    int hn = snprintf(hdr, sizeof(hdr), "SHOTS %d %zu\n",
                      num_shots, text_len);
    if (hn < 0 || (size_t)hn >= sizeof(hdr)) {
        close(fd);
        return MOONLAB_CONTROL_IO_ERROR;
    }
    int rc = send_all(fd, hdr, (size_t)hn);
    if (rc == MOONLAB_CONTROL_OK) {
        rc = send_all(fd, circuit_text, text_len);
    }
    if (rc != MOONLAB_CONTROL_OK) { close(fd); return rc; }

    char resp_hdr[128];
    size_t resp_len = 0;
    rc = recv_until_newline(fd, resp_hdr, sizeof(resp_hdr), &resp_len);
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
        rc = recv_all(fd, buf, (size_t)shots_back * sizeof(uint64_t));
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
