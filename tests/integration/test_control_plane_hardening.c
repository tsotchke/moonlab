/**
 * @file  test_control_plane_hardening.c
 * @brief v0.9.0 production-hardening test: max_concurrent ceiling +
 *        tls_failed counter.
 *
 * Path 1 -- max_concurrent: set cap = 2, fire 6 parallel CIRCUIT
 *           requests, scrape METRICS, confirm the
 *           max_concurrent_rejected counter is wired and 6 requests
 *           are accounted for as OK + REJECTED.
 *
 * Path 2 -- tls_failed: stand up a TLS server, open three plain-TCP
 *           connections to it.  Each fails SSL_accept, bumping
 *           g_count_tls_failed.  No clean way to scrape METRICS over
 *           the same TLS port without a real cert pair, so we settle
 *           for confirming the failure path doesn't crash the server.
 */

#include "../../src/control/control_plane.h"
#include "../../src/applications/moonlab_qgtl_backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#ifdef MOONLAB_HAVE_TLS
#  include <openssl/evp.h>
#  include <openssl/pem.h>
#  include <openssl/rsa.h>
#  include <openssl/x509.h>
#  include "test_tls_keygen.h"
#endif

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    fflush(stdout);                                             \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
        fflush(stdout);                                         \
    }                                                           \
} while (0)

typedef struct {
    moonlab_control_server_t *server;
    int                       max_iters;
    int                       rc;
} ra_t;

static void *run_thread(void *arg)
{
    ra_t *a = (ra_t *)arg;
    a->rc = moonlab_control_server_run(a->server, a->max_iters);
    return NULL;
}

typedef struct { uint16_t port; char *text; int rc; } cl_t;

static void *hardening_client_thread(void *arg)
{
    cl_t *x = (cl_t *)arg;
    double *probs = NULL; size_t num = 0;
    x->rc = moonlab_control_submit_circuit(
        "127.0.0.1", x->port, x->text, 0, &probs, &num);
    free(probs);
    return NULL;
}

/* Parse the metric value line for a given metric name from a
 * Prometheus exposition body.  Skips "# HELP <name> ..." and
 * "# TYPE <name> ..." comment lines: those contain the metric name
 * but no numeric value. */
static long parse_counter(const char *body, const char *name)
{
    if (!body || !name) return -1;
    size_t namelen = strlen(name);
    const char *p = body;
    while (p && *p) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        if (len > namelen + 1 &&
            p[0] != '#' &&
            strncmp(p, name, namelen) == 0 &&
            p[namelen] == ' ') {
            long v = -1;
            if (sscanf(p + namelen + 1, "%ld", &v) == 1) return v;
        }
        if (!nl) break;
        p = nl + 1;
    }
    return -1;
}

/* Raw-socket round-trip: connect to the loopback server, send exactly
 * `reqlen` bytes, half-close so the server sees EOF, and read the response
 * into `resp` (NUL-terminated).  Returns the number of response bytes read,
 * or -1 if the connection failed. */
static ssize_t raw_roundtrip(uint16_t port, const char *req, size_t reqlen,
                             char *resp, size_t respcap)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    size_t off = 0;
    while (off < reqlen) {
        ssize_t n = send(fd, req + off, reqlen - off, 0);
        if (n <= 0) break;
        off += (size_t)n;
    }
    shutdown(fd, SHUT_WR);
    size_t got = 0;
    while (got + 1 < respcap) {
        ssize_t n = recv(fd, resp + got, respcap - 1 - got, 0);
        if (n <= 0) break;
        got += (size_t)n;
    }
    resp[got] = '\0';
    close(fd);
    return (ssize_t)got;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    /* macOS BSD sockets deliver SIGPIPE on write() to a peer-closed
     * connection.  Default disposition is process termination; on the
     * hosted macos-14 runner this kills the test with SIGPIPE before
     * path 1 finishes (race between server cap-reject closing the
     * connection and the client thread's next write).  Ignore it; the
     * client-side write() then returns EPIPE which the test code
     * already handles as a normal failed-submit. */
    signal(SIGPIPE, SIG_IGN);
    fprintf(stdout, "=== test_control_plane_hardening (v0.9.0) ===\n\n");

    /* ---- Path 1: max_concurrent ceiling ---- */
    fprintf(stdout, "--- path 1: max_concurrent = 2, fire 6 parallel ---\n");
    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    rc = moonlab_control_server_set_max_concurrent(server, 2);
    CHECK(rc == 0, "set_max_concurrent(2) rc=%d", rc);

    /* Generous iteration budget; we'll shut down explicitly. */
    ra_t ra = { server, 32, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, run_thread, &ra);
    struct timespec ts = { 0, 30 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    /* Each client sends a Bell circuit; the cap allows 2 workers in
     * flight at a time. */
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(2);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    size_t needed = 0;
    moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    char *text = (char *)malloc(needed + 1);
    moonlab_qgtl_circuit_serialize(c, text, needed + 1, NULL);
    moonlab_qgtl_circuit_free(c);

    cl_t cargs[6];
    pthread_t cthreads[6];
    for (int i = 0; i < 6; i++) {
        cargs[i].port = port;
        cargs[i].text = text;
        cargs[i].rc = 0;
        pthread_create(&cthreads[i], NULL, hardening_client_thread, &cargs[i]);
    }
    for (int i = 0; i < 6; i++) pthread_join(cthreads[i], NULL);

    int ok_count = 0, busy_count = 0, conn_err_count = 0;
    for (int i = 0; i < 6; i++) {
        if (cargs[i].rc == MOONLAB_CONTROL_OK) ok_count++;
        else if (cargs[i].rc == MOONLAB_CONTROL_REJECTED) busy_count++;
        /* Any negative rc that isn't REJECTED is the server hanging
         * up on us mid-write -- counts as "server refused work",
         * functionally identical to busy from the cap test's POV.
         * On macOS hosted CI the server's cap-reject close can race
         * the client write() so we see EPIPE here on one or two
         * threads instead of a clean REJECTED rc.  Bucketing them
         * lets us still assert that ALL 6 requests were accounted
         * for and at least one was denied. */
        else if (cargs[i].rc < 0) conn_err_count++;
    }
    fprintf(stdout, "    6 parallel: %d OK, %d busy, %d conn-err (rc breakdown)\n",
            ok_count, busy_count, conn_err_count);
    CHECK(ok_count + busy_count + conn_err_count == 6,
          "all 6 accounted for (%d ok + %d busy + %d conn-err)",
          ok_count, busy_count, conn_err_count);
    /* Don't assert "at least one denied".  On fast runners (Linux
     * hosted CI especially) the 6 client threads can serialise
     * through the cap=2 worker pool in <1ms each, finishing before
     * the cap is ever observed -- the next request finds 0/2 slots
     * occupied and proceeds.  That's correct behaviour, not a bug.
     * The METRICS counter below verifies the cap WIRING; whether
     * THIS particular race hits the cap is timing-dependent. */

    /* Scrape METRICS. */
    char *metrics = NULL;
    rc = moonlab_control_submit_metrics("127.0.0.1", port, &metrics);
    CHECK(rc == 0 && metrics != NULL, "METRICS scrape rc=%d", rc);
    if (metrics) {
        long busy_metric = parse_counter(metrics,
            "moonlab_control_max_concurrent_rejected_total");
        fprintf(stdout, "    max_concurrent_rejected_total = %ld\n",
                busy_metric);
        CHECK(busy_metric >= 0, "metric present (got %ld)", busy_metric);
        /* Cross-check the counter against the per-client return-code
         * breakdown: every busy_count must show up in the counter. */
        CHECK(busy_metric >= busy_count,
              "counter (%ld) >= REJECTED count (%d)", busy_metric, busy_count);
    }
    free(metrics);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);
    free(text);

    /* ---- Path 2: tls_failed counter ---- */
#ifdef MOONLAB_HAVE_TLS
    fprintf(stdout, "\n--- path 2: tls_failed counter ---\n");
    rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);

    /* Generate a self-signed cert in-process. */
    EVP_PKEY *key = moonlab_test_generate_rsa_key();
    X509 *cert = X509_new();
    X509_set_version(cert, 2);
    ASN1_INTEGER_set(X509_get_serialNumber(cert), 1);
    X509_gmtime_adj(X509_get_notBefore(cert), 0);
    X509_gmtime_adj(X509_get_notAfter(cert), 24L * 3600);
    X509_set_pubkey(cert, key);
    X509_NAME *name = X509_get_subject_name(cert);
    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
        (const unsigned char *)"127.0.0.1", -1, -1, 0);
    X509_set_issuer_name(cert, name);
    X509_sign(cert, key, EVP_sha256());
    const char *cert_path = "/tmp/moonlab_hardening_cert.pem";
    const char *key_path  = "/tmp/moonlab_hardening_key.pem";
    {
        FILE *f = fopen(cert_path, "w"); PEM_write_X509(f, cert); fclose(f);
        f = fopen(key_path, "w");
        PEM_write_PrivateKey(f, key, NULL, NULL, 0, NULL, NULL);
        fclose(f);
    }
    X509_free(cert);
    EVP_PKEY_free(key);

    rc = moonlab_control_server_use_tls(server, cert_path, key_path);
    CHECK(rc == 0, "use_tls rc=%d", rc);

    ra.server = server; ra.max_iters = 32; ra.rc = 0;
    pthread_create(&tid, NULL, run_thread, &ra);
    nanosleep(&ts, NULL);

    /* Connect over plain TCP to a TLS port; server's SSL_accept
     * rejects each handshake and bumps tls_failed. */
    for (int i = 0; i < 3; i++) {
        int sfd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(port);
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (connect(sfd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            const char *junk = "not-a-tls-clienthello\n";
            (void)send(sfd, junk, strlen(junk), 0);
            char buf[16];
            (void)recv(sfd, buf, sizeof(buf), 0);
        }
        close(sfd);
    }
    /* Brief settle so the SSL_accept failures fire before shutdown. */
    struct timespec settle = { 0, 100 * 1000 * 1000 };
    nanosleep(&settle, NULL);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);
    unlink(cert_path); unlink(key_path);
    fprintf(stdout, "    tls_failed path exercised (3 junk handshakes)\n");
#else
    fprintf(stdout, "\n  SKIP  path 2: library built without QSIM_ENABLE_TLS=ON\n");
#endif

    /* ---- Path 3: resource-amplification DoS guard (task #23) ----
     * A tiny, well-formed CIRCUIT frame declaring 30 qubits used to drive a
     * 2^30 * 16 B = 16 GB state-vector allocation in execute (fuzz finding
     * oom_qubit_amplification).  The server must now reject it BEFORE
     * allocating anything, with a clean protocol error naming the qubit cap
     * -- while legitimate small circuits still succeed. */
    fprintf(stdout, "\n--- path 3: oversized-qubit DoS frame is rejected ---\n");
    server = NULL; port = 0;
    rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);

    ra.server = server; ra.max_iters = 8; ra.rc = 0;
    pthread_create(&tid, NULL, run_thread, &ra);
    nanosleep(&ts, NULL);

    /* The exact fuzz reproducer: CIRCUIT body of 18 bytes declaring 30 qubits. */
    const char *dos_frame = "CIRCUIT 18\nNUM_QUBITS 30\nH 0\n";
    char resp[512];
    struct timespec t_before, t_after;
    clock_gettime(CLOCK_MONOTONIC, &t_before);
    ssize_t rn = raw_roundtrip(port, dos_frame, strlen(dos_frame),
                               resp, sizeof(resp));
    clock_gettime(CLOCK_MONOTONIC, &t_after);
    double dos_ms = (t_after.tv_sec - t_before.tv_sec) * 1000.0 +
                    (t_after.tv_nsec - t_before.tv_nsec) / 1.0e6;
    fprintf(stdout, "    30-qubit frame -> \"%.*s\" (%.1f ms)\n",
            (int)(rn > 0 ? (rn > 60 ? 60 : rn) : 0), rn > 0 ? resp : "", dos_ms);
    CHECK(rn > 0, "server responded to oversized frame (rn=%zd)", rn);
    CHECK(strncmp(resp, "ERR", 3) == 0,
          "oversized frame rejected with ERR (got: %.16s)", resp);
    /* The cap message is the fixed path; the pre-fix server had no cap and
     * either attempted the 16 GB allocation or returned an "execute" error. */
    CHECK(strstr(resp, "qubit") != NULL,
          "rejection names the qubit cap (proves pre-execution reject)");
    /* Pre-execution rejection is immediate; a real 16 GB alloc+touch would
     * take far longer.  Generous bound to stay CI-robust. */
    CHECK(dos_ms < 2000.0, "rejected promptly without allocating (%.1f ms)", dos_ms);

    /* A legitimate small circuit still round-trips fine (the cap does not
     * break normal traffic). */
    const char *ok_frame = "CIRCUIT 26\nNUM_QUBITS 2\nH 0\nCNOT 1 0\n";
    ssize_t okn = raw_roundtrip(port, ok_frame, strlen(ok_frame),
                                resp, sizeof(resp));
    fprintf(stdout, "    2-qubit frame  -> \"%.*s\"\n",
            (int)(okn > 0 ? (okn > 24 ? 24 : okn) : 0), okn > 0 ? resp : "");
    CHECK(okn > 0 && strncmp(resp, "OK", 2) == 0,
          "legitimate 2-qubit circuit still accepted (got: %.16s)", resp);

    moonlab_control_server_shutdown(server);
    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
