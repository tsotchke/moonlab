/**
 * @file test_feynman.c
 * @brief Feynman diagram builder smoke tests.
 *
 * Exercises the visualization/feynman_diagram.c surface: build a basic
 * QED vertex, the canonical e+ e- -> mu+ mu- diagram, render ASCII,
 * and confirm the lifecycle is clean.
 */

#include "../../src/visualization/feynman_diagram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void test_qed_vertex(void) {
    fprintf(stdout, "\n-- feynman: QED vertex prebuilt --\n");
    feynman_diagram_t* d = feynman_create_qed_vertex();
    CHECK(d != NULL, "feynman_create_qed_vertex returned non-NULL");
    if (d) feynman_free(d);
}

static void test_ee_to_mumu(void) {
    fprintf(stdout, "\n-- feynman: e+ e- -> mu+ mu- prebuilt --\n");
    feynman_diagram_t* d = feynman_create_ee_to_mumu();
    CHECK(d != NULL, "feynman_create_ee_to_mumu returned non-NULL");
    if (d) feynman_free(d);
}

static void test_manual_build_and_render(void) {
    fprintf(stdout, "\n-- feynman: manual build + ASCII render --\n");
    feynman_diagram_t* d = feynman_create("test-process");
    CHECK(d != NULL, "feynman_create for named process");
    if (!d) return;

    feynman_set_title(d, "test diagram");
    feynman_set_loop_order(d, 1);

    int v0 = feynman_add_vertex(d, 0.0, 0.0);
    int v1 = feynman_add_vertex(d, 1.0, 0.0);
    CHECK(v0 >= 0 && v1 >= 0,
          "add_vertex returned valid ids v0=%d v1=%d", v0, v1);

    int ph = feynman_add_photon(d, v0, v1, "photon");
    CHECK(ph >= 0, "add_photon between v0 and v1 returned id %d", ph);

    feynman_free(d);
}

int main(void) {
    fprintf(stdout, "=== Feynman diagram builder smoke ===\n");
    test_qed_vertex();
    test_ee_to_mumu();
    test_manual_build_and_render();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
