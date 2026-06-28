/**
 * @file  test_control_plane_health_rate.c
 * @brief HEALTH probe + per-IP rate limit test for the v0.8.21 control
 *        plane.
 *
 * Path 1 -- HEALTH returns OK before any other interaction.
 * Path 2 -- per-IP rate limit: configure rate_rps=5 burst=5, fire 12
 *           HEALTH probes from the test's single source IP in a tight
 *           loop, expect at least one to come back rate-limited.
 */

#include "../../src/control/control_plane.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

typedef struct {
    moonlab_control_server_t *server;
    int                       max_iters;
    int                       rc;
} run_args_t;

static void *run_thread(void *arg)
{
    run_args_t *a = (run_args_t *)arg;
    a->rc = moonlab_control_server_run(a->server, a->max_iters);
    return NULL;
}

int main(void)
{
    fprintf(stdout, "=== test_control_plane_health_rate (v0.8.21) ===\n\n");

    /* ---- Path 1: HEALTH probe ---- */
    fprintf(stdout, "--- path 1: HEALTH probe ---\n");
    moonlab_control_server_t *server = NULL;
    uint16_t port = 0;
    int rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);

    run_args_t ra = { server, 1, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, run_thread, &ra);
    struct timespec ts = { 0, 30 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    rc = moonlab_control_submit_health("127.0.0.1", port);
    CHECK(rc == MOONLAB_CONTROL_OK, "submit_health rc=%d", rc);

    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    /* ---- Path 2: per-IP rate limit ---- */
    fprintf(stdout, "\n--- path 2: per-IP rate limit ---\n");
    server = NULL;
    rc = moonlab_control_server_open("127.0.0.1", 0, &server, &port);
    CHECK(rc == 0, "server_open rc=%d", rc);
    rc = moonlab_control_server_set_rate_limit(server, /* rps */ 5, /* burst */ 5);
    CHECK(rc == 0, "set_rate_limit(5, 5) rc=%d", rc);

    const int N = 12;
    ra.server = server; ra.max_iters = N; ra.rc = 0;
    pthread_create(&tid, NULL, run_thread, &ra);
    nanosleep(&ts, NULL);

    int ok_count = 0, limited_count = 0;
    for (int i = 0; i < N; i++) {
        rc = moonlab_control_submit_health("127.0.0.1", port);
        if (rc == MOONLAB_CONTROL_OK)             ok_count++;
        else if (rc == MOONLAB_CONTROL_RATE_LIMITED) limited_count++;
    }
    fprintf(stdout, "    fired %d probes: %d OK, %d rate-limited\n",
            N, ok_count, limited_count);
    CHECK(ok_count >= 5,
          "at least burst=5 succeed (got %d)", ok_count);
    CHECK(limited_count > 0,
          "rate limiter kicks in (got %d limited)", limited_count);
    CHECK(ok_count + limited_count == N,
          "every probe accounted for (%d + %d == %d)",
          ok_count, limited_count, N);

    pthread_join(tid, NULL);
    moonlab_control_server_close(server);

    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
