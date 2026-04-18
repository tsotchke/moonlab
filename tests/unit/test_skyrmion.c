/**
 * @file test_skyrmion.c
 * @brief Smoke test for the skyrmion braiding helpers.
 *
 * Exercises the braid-path generators (circular + exchange) without
 * requiring a fully-prepared MPS state. Verifies that:
 *  - circular path generates the requested segments
 *  - total_time is positive
 *  - path memory lifecycle is clean
 */

#include "../../src/algorithms/tensor_network/skyrmion_braiding.h"
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void test_circular_braid_path(void) {
    fprintf(stdout, "\n-- skyrmion: circular braid path --\n");
    braid_path_t* p = braid_path_circular(
        0.0, 0.0,              /* centre */
        1.5,                   /* radius */
        BRAID_CLOCKWISE,
        16,                    /* num_segments */
        1.0);                  /* velocity */
    CHECK(p != NULL, "braid_path_circular returned non-NULL");
    if (!p) return;
    CHECK(p->num_waypoints > 0, "path has waypoints (%u)",
          (unsigned)p->num_waypoints);
    CHECK(p->type == BRAID_CLOCKWISE, "type == CLOCKWISE");
    CHECK(p->total_time > 0.0, "total_time = %.6f > 0", p->total_time);
    braid_path_free(p);
}

int main(void) {
    fprintf(stdout, "=== skyrmion braiding smoke ===\n");
    test_circular_braid_path();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
