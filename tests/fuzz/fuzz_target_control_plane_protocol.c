/**
 * @file    fuzz_target_control_plane_protocol.c
 * @brief   Surface: the network-facing control-plane wire protocol.
 *
 * This is the highest-value target: `src/control/control_plane.c` parses
 * an untrusted, length-prefixed binary frame off a TCP socket -- the
 * `AUTH` / `CIRCUIT` / `SHOTS` / `HEALTH` / `METRICS` verbs, the tenant
 * form `AUTH <tenant>:<hmac>`, the `sscanf`-parsed length prefixes, and
 * the raw circuit body handed to the deserializer + simulator.
 *
 * The core dispatcher `handle_one_request` is `static`, so per the lane's
 * ownership rules we do NOT de-static it; instead we drive the exact
 * public entry point a remote client hits: `moonlab_control_server_open`
 * + `moonlab_control_server_run` bound to `127.0.0.1:0`, with the fuzzer
 * bytes streamed in as one connection's request.  This exercises the
 * real recv/parse/dispatch path end-to-end (framing, HMAC-absent path,
 * body allocation, deserialize, execute, response framing).
 *
 * Each input spins up a listener on an ephemeral port, serves exactly one
 * connection on a worker thread, and tears everything down -- no state
 * survives the call, so the persistent AFL loop and LeakSanitizer stay
 * clean.
 *
 * NOTE (handed off in FINDINGS.md): because there is no default admission
 * cap, a single <=4 MB CIRCUIT frame declaring ~30 qubits makes the
 * worker attempt a multi-GB state-vector allocation.  That is a resource-
 * amplification observation, not a memory-safety bug; the soak runner
 * bounds it with -malloc_limit_mb / -rss_limit_mb and the replay seeds
 * stay small.
 */

#include "fuzz_common.h"

#include "control/control_plane.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* One-time process setup: ignore SIGPIPE so a server-side early close
 * (bad verb, reject) mid-write never kills the fuzzer process. */
static void ignore_sigpipe_once(void)
{
    static int done = 0;
    if (!done) { signal(SIGPIPE, SIG_IGN); done = 1; }
}

/* Server-thread body: serve exactly one connection, then return. */
static void *serve_one(void *arg)
{
    moonlab_control_server_t *srv = (moonlab_control_server_t *)arg;
    (void)moonlab_control_server_run(srv, 1);
    return NULL;
}

static int connect_loopback(uint16_t port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

#ifdef SO_NOSIGPIPE
    int yes = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
    /* Linger 0: send an RST on close so we do not accumulate TIME_WAIT
     * sockets across millions of fuzz iterations. */
    struct linger lg = { 1, 0 };
    (void)setsockopt(fd, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void send_all_fd(int fd, const uint8_t *data, size_t size)
{
#ifdef MSG_NOSIGNAL
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    size_t off = 0;
    while (off < size) {
        ssize_t n = send(fd, data + off, size - off, flags);
        if (n <= 0) return; /* peer closed early -- fine, stop writing. */
        off += (size_t)n;
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    ignore_sigpipe_once();

    /* The server body cap is 4 MB; a little headroom covers the verb and
     * AUTH prelude.  Larger inputs carry no extra signal. */
    if (size > (5u * 1024u * 1024u)) size = 5u * 1024u * 1024u;

    moonlab_control_server_t *srv = NULL;
    uint16_t port = 0;
    if (moonlab_control_server_open("127.0.0.1", 0, &srv, &port)
            != MOONLAB_CONTROL_OK || srv == NULL) {
        return 0;
    }

    pthread_t th;
    if (pthread_create(&th, NULL, serve_one, srv) != 0) {
        moonlab_control_server_close(srv);
        return 0;
    }

    int fd = connect_loopback(port);
    if (fd >= 0) {
        send_all_fd(fd, data, size);
        shutdown(fd, SHUT_WR); /* signal EOF so the server stops recv'ing */

        /* Drain the response so the server's send() completes, bounded so
         * a hostile-looking response length can never spin forever. */
        char buf[1024];
        int guard = 0;
        while (guard++ < 200000) {
            ssize_t n = recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) break;
        }
        close(fd);
    } else {
        /* Never connected: wake the blocked accept() so the worker can
         * exit, otherwise the join below would hang. */
        moonlab_control_server_shutdown(srv);
    }

    pthread_join(th, NULL);
    moonlab_control_server_close(srv);
    return 0;
}
