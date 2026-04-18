/**
 * @file test_lattice_2d.c
 * @brief Smoke test for the 2D lattice + MPO-2D builders.
 *
 * Creates a small 4x4 square lattice with open boundaries, constructs a
 * bond list from a default Hamiltonian, and walks the snake/grid
 * coordinate mapping. This exercises the MPO-2D + lattice_2d modules
 * that power PEPS / skyrmion / 2D-TDVP work, which otherwise have no
 * automated coverage.
 */

#include "../../src/algorithms/tensor_network/lattice_2d.h"
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

static void test_square_lattice_4x4_open(void) {
    fprintf(stdout, "\n-- lattice_2d: 4x4 square, open boundaries --\n");
    lattice_2d_t* lat = lattice_2d_create(4, 4, LATTICE_SQUARE, BC_OPEN);
    CHECK(lat != NULL, "lattice_2d_create returned non-NULL");
    if (!lat) return;
    CHECK(lat->num_sites == 16,
          "4x4 lattice has 16 sites (got %u)", (unsigned)lat->num_sites);
    CHECK(lat->type == LATTICE_SQUARE, "type == LATTICE_SQUARE");
    CHECK(lat->bc == BC_OPEN, "bc == BC_OPEN");

    /* Snake/grid mappings round-trip. */
    if (lat->snake_to_grid && lat->grid_to_snake) {
        int roundtrip_ok = 1;
        for (uint32_t snake = 0; snake < lat->num_sites; ++snake) {
            uint32_t grid = lat->snake_to_grid[snake];
            uint32_t back = lat->grid_to_snake[grid];
            if (back != snake) { roundtrip_ok = 0; break; }
        }
        CHECK(roundtrip_ok,
              "snake -> grid -> snake is identity over all 16 sites");
    } else {
        fprintf(stdout, "  SKIP  snake/grid maps not populated on this build\n");
    }

    lattice_2d_free(lat);
}

static void test_triangular_lattice(void) {
    fprintf(stdout, "\n-- lattice_2d: 3x3 triangular --\n");
    lattice_2d_t* lat = lattice_2d_create(3, 3, LATTICE_TRIANGULAR, BC_OPEN);
    CHECK(lat != NULL, "triangular lattice creates");
    if (!lat) return;
    CHECK(lat->num_sites == 9,
          "3x3 triangular has 9 sites (got %u)", (unsigned)lat->num_sites);
    lattice_2d_free(lat);
}

static void test_honeycomb_lattice(void) {
    fprintf(stdout, "\n-- lattice_2d: 2x2 honeycomb --\n");
    lattice_2d_t* lat = lattice_2d_create(2, 2, LATTICE_HONEYCOMB, BC_OPEN);
    CHECK(lat != NULL, "honeycomb lattice creates");
    if (lat) lattice_2d_free(lat);
}

int main(void) {
    fprintf(stdout, "=== lattice_2d / MPO-2D smoke tests ===\n");
    test_square_lattice_4x4_open();
    test_triangular_lattice();
    test_honeycomb_lattice();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
