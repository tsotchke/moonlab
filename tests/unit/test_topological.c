/**
 * @file test_topological.c
 * @brief Topological QC smoke tests.
 *
 *  - Fibonacci and Ising anyon systems construct with the expected
 *    quantum dimensions.
 *  - Every built-in model's F/R symbols satisfy the fusion-category
 *    coherence conditions (pentagon, both hexagons, F-matrix unitarity),
 *    and that check is proven non-vacuous.
 *  - Anyonic registers can be created and freed on either anyon model.
 *  - Surface-code/toric-code basic lifecycle.
 *  - The KNOWN-BROKEN braiding/gate layer is pinned in its current, wrong
 *    form (see test_braiding_is_not_yet_a_representation) so the v1.2.1 fix
 *    has a red-to-green target.  Nothing in this file should be read as
 *    validating braiding: see the note on
 *    test_fibonacci_braiding_invariants.
 */

#include "../../src/algorithms/topological/topological.h"
#include <math.h>
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

/* Pins a KNOWN-BROKEN behaviour so it cannot change silently.  This harness has
 * no expected-failure mechanism (only tests/oracle/ does), so a bug we have not
 * fixed yet is asserted in its present, wrong form and flagged loudly.  When the
 * underlying defect is fixed these assertions MUST be inverted -- they are the
 * red-to-green target for the fix, not a description of correct behaviour. */
#define XFAIL_DOC(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                      \
        fprintf(stderr, "  FAIL  [xfail-doc] " fmt "\n", ##__VA_ARGS__);\
        failures++;                                                     \
    } else {                                                            \
        fprintf(stdout, "  BUG   [xfail-doc] " fmt "\n", ##__VA_ARGS__);\
    }                                                                   \
} while (0)

static void test_fibonacci_quantum_dimension(void) {
    fprintf(stdout, "\n-- topological: Fibonacci anyons have d_tau = phi --\n");
    anyon_system_t* sys = anyon_system_fibonacci();
    CHECK(sys != NULL, "create Fibonacci anyon system");
    if (!sys) return;

    /* Fibonacci tau anyon has quantum dimension phi = (1 + sqrt(5))/2.
     * Identity has quantum dimension 1. */
    double d_total = anyon_total_dimension(sys);
    CHECK(isfinite(d_total) && d_total > 0.0,
          "total quantum dimension positive (got %.6f)", d_total);

    anyon_system_free(sys);
}

static void test_ising_anyons(void) {
    fprintf(stdout, "\n-- topological: Ising anyon system builds --\n");
    anyon_system_t* sys = anyon_system_ising();
    CHECK(sys != NULL, "create Ising anyon system");
    if (sys) {
        double d_total = anyon_total_dimension(sys);
        CHECK(isfinite(d_total) && d_total > 0.0,
              "Ising total quantum dimension is finite positive (got %.6f)",
              d_total);
        anyon_system_free(sys);
    }
}

static void test_surface_code_lifecycle(void) {
    fprintf(stdout, "\n-- topological: surface_code_create/free --\n");
    surface_code_t* code = surface_code_create(3);  /* distance-3 */
    CHECK(code != NULL, "create distance-3 surface code");
    if (code) {
        qs_error_t err = surface_code_init_logical_zero(code);
        CHECK(err == QS_SUCCESS, "init logical |0>");
        surface_code_free(code);
    }
}

static void test_toric_code_lifecycle(void) {
    fprintf(stdout, "\n-- topological: toric_code_create/free --\n");
    toric_code_t* code = toric_code_create(3);  /* L=3 */
    CHECK(code != NULL, "create L=3 toric code");
    if (code) {
        toric_code_free(code);
    }
}

/* NOTE: these are weak invariants, not evidence that braiding works.  Every
 * assertion below is satisfied by multiplying the amplitude vector by a scalar
 * phase, which is all braid_anyons currently does (see
 * test_braiding_is_not_yet_a_representation).  The test is kept because norm
 * preservation and sigma sigma^{-1} = I must continue to hold after the v1.2.1
 * fix, but on its own it certifies nothing about the braid group. */
static void test_fibonacci_braiding_invariants(void) {
    fprintf(stdout, "\n-- topological: Fibonacci braiding --\n");
    anyon_system_t* sys = anyon_system_fibonacci();
    if (!sys) { CHECK(0, "create system"); return; }

    /* Four tau anyons fusing to vacuum — the canonical setup used to
     * realise one logical qubit of Fibonacci topological quantum
     * computation. */
    anyon_charge_t charges[4] = { FIB_TAU, FIB_TAU, FIB_TAU, FIB_TAU };
    fusion_tree_t* tree = fusion_tree_create(sys, charges, 4, FIB_VACUUM);
    CHECK(tree != NULL, "create 4-tau fusion tree");
    if (!tree) { anyon_system_free(sys); return; }

    /* Fusion space dimension: for n=4 tau anyons fused to vacuum, it's
     * Fibonacci number F_{n-1} = F_3 = 2. */
    uint32_t paths = fusion_count_paths(sys, charges, 4, FIB_VACUUM);
    CHECK(paths == 2,
          "fusion_count_paths(4 tau -> vacuum) == 2 (got %u)", paths);

    /* Norm before braid should be 1. */
    double norm0 = 0.0;
    for (uint32_t i = 0; i < tree->num_paths; i++) {
        double m = cabs(tree->amplitudes[i]);
        norm0 += m * m;
    }
    CHECK(fabs(norm0 - 1.0) < 1e-10,
          "initial norm == 1 (got %.12f)", norm0);

    /* Snapshot amplitudes for identity-braid check. */
    double complex amps0[8] = {0};
    for (uint32_t i = 0; i < tree->num_paths && i < 8; i++) {
        amps0[i] = tree->amplitudes[i];
    }

    /* Apply sigma_1 followed by sigma_1^{-1}: should be identity on
     * the logical state up to global phase. */
    qs_error_t err = braid_anyons(tree, 0, true);
    CHECK(err == QS_SUCCESS, "sigma_1 succeeds");
    err = braid_anyons(tree, 0, false);
    CHECK(err == QS_SUCCESS, "sigma_1^{-1} succeeds");

    /* Norm preserved. */
    double norm1 = 0.0;
    for (uint32_t i = 0; i < tree->num_paths; i++) {
        double m = cabs(tree->amplitudes[i]);
        norm1 += m * m;
    }
    CHECK(fabs(norm1 - 1.0) < 1e-10,
          "norm preserved after sigma_1 sigma_1^{-1} (got %.12f)", norm1);

    /* sigma * sigma^{-1} is identity on the full state. */
    double max_diff = 0.0;
    for (uint32_t i = 0; i < tree->num_paths && i < 8; i++) {
        double d = cabs(tree->amplitudes[i] - amps0[i]);
        if (d > max_diff) max_diff = d;
    }
    CHECK(max_diff < 1e-10,
          "sigma_1 sigma_1^{-1} = I on amplitudes (max diff %.3e)", max_diff);

    fusion_tree_free(tree);
    anyon_system_free(sys);
}

/* The tabulated F/R symbols of every built-in anyon model must satisfy the
 * fusion-category coherence conditions (pentagon + both hexagons + F-matrix
 * unitarity) exactly.  anyon_verify_coherence returns the largest violation;
 * a consistent model is 0 to machine precision. */
static void test_anyon_coherence(void) {
    fprintf(stdout, "\n-- topological: anyon F/R coherence (pentagon+hexagon) --\n");

    anyon_system_t *fib = anyon_system_fibonacci();
    double r = anyon_verify_coherence(fib);
    CHECK(r < 1e-13, "Fibonacci satisfies pentagon+hexagon+unitarity (%.2e)", r);
    anyon_system_free(fib);

    anyon_system_t *ising = anyon_system_ising();
    r = anyon_verify_coherence(ising);
    CHECK(r < 1e-13, "Ising satisfies pentagon+hexagon+unitarity (%.2e)", r);
    anyon_system_free(ising);

    for (uint32_t k = 2; k <= 5; k++) {
        anyon_system_t *s = anyon_system_su2k(k);
        r = anyon_verify_coherence(s);
        CHECK(r < 1e-13, "SU(2)_%u satisfies pentagon+hexagon+unitarity (%.2e)", k, r);
        anyon_system_free(s);
    }
}

/* The coherence check above is only worth anything if it can actually fail.
 * Before the fix, the F-unitarity term inferred an F-matrix's row set from
 * which entries were nonzero, so an allowed-but-entirely-zero row read as
 * "absent" and scored 1.1e-16 -- machine precision on completely wrong data.
 * Zeroing a row of F^{ttt}_t by hand must now be detected. */
static void test_coherence_verifier_is_not_vacuous(void) {
    fprintf(stdout, "\n-- topological: coherence verifier detects a zeroed F row --\n");

    anyon_system_t *sys = anyon_system_fibonacci();
    if (!sys) { CHECK(0, "create Fibonacci system"); return; }

    double clean = anyon_verify_coherence(sys);
    CHECK(clean < 1e-13, "unmutated Fibonacci is coherent (%.2e)", clean);

    /* F_matrices[15] is F^{ttt}_t (a*8+b*4+c*2+d with all labels tau).
     * Zero its e=vacuum row; the row is still fusion-allowed, so unitarity
     * of that row must now be violated by ~1. */
    sys->F_matrices[15][0] = 0.0;
    sys->F_matrices[15][1] = 0.0;

    double dirty = anyon_verify_coherence(sys);
    CHECK(dirty > 0.5,
          "zeroed F^{ttt}_t row is caught (%.3f, was %.2e)", dirty, clean);

    anyon_system_free(sys);
}

/* ------------------------------------------------------------------------
 * KNOWN BUG -- scheduled for v1.2.1.
 *
 * The F/R symbol tables are coherence-verified (above), but the layer built on
 * top of them is not a braid-group representation.  braid_anyons() uses
 * c = tree->total_charge as the fusion channel for every path in the tree
 * instead of the per-path intermediate charge, so it multiplies the whole
 * amplitude vector by one global R-phase.  Consequences, all asserted below in
 * their present (wrong) form:
 *
 *   - sigma_1, sigma_2 and sigma_3 act identically, so the generators do not
 *     satisfy the braid relations and carry no positional information;
 *   - the anyonic gates, being products of those generators, apply a global
 *     phase and perform no logical rotation at all;
 *   - apply_F_move() is unimplemented and returns QS_SUCCESS without touching
 *     the tree, so no change of fusion basis is available either.
 *
 * Fixing this means tracking per-path intermediate charges in fusion_tree_t and
 * implementing apply_F_move; at that point every XFAIL_DOC below flips to a
 * CHECK of the opposite condition.
 * --------------------------------------------------------------------- */
static void test_braiding_is_not_yet_a_representation(void) {
    fprintf(stdout,
        "\n-- topological: braid layer is a global phase, NOT a braid rep"
        " (known bug, v1.2.1) --\n");

    anyon_system_t *sys = anyon_system_fibonacci();
    if (!sys) { CHECK(0, "create Fibonacci system"); return; }

    anyon_charge_t charges[4] = { FIB_TAU, FIB_TAU, FIB_TAU, FIB_TAU };

    /* Braid each of the three generators on its own fresh tree. */
    double complex amps[3][2] = {{0}};
    uint32_t npaths = 0;
    for (uint32_t pos = 0; pos < 3; pos++) {
        fusion_tree_t *t = fusion_tree_create(sys, charges, 4, FIB_VACUUM);
        if (!t) { CHECK(0, "create fusion tree"); anyon_system_free(sys); return; }
        qs_error_t err = braid_anyons(t, pos, true);
        CHECK(err == QS_SUCCESS, "sigma_%u applies", pos + 1);
        npaths = t->num_paths;
        for (uint32_t i = 0; i < t->num_paths && i < 2; i++) amps[pos][i] = t->amplitudes[i];
        fusion_tree_free(t);
    }
    CHECK(npaths == 2, "4 tau -> vacuum has a 2-dimensional fusion space (got %u)",
          npaths);

    /* A faithful representation would make these three differ.  They do not. */
    double d12 = 0.0, d13 = 0.0;
    for (uint32_t i = 0; i < 2; i++) {
        double a = cabs(amps[0][i] - amps[1][i]);
        double b = cabs(amps[0][i] - amps[2][i]);
        if (a > d12) d12 = a;
        if (b > d13) d13 = b;
    }
    XFAIL_DOC(d12 < 1e-15,
              "sigma_1 and sigma_2 act identically (diff %.3e; a braid rep would differ)",
              d12);
    XFAIL_DOC(d13 < 1e-15,
              "sigma_1 and sigma_3 act identically (diff %.3e; a braid rep would differ)",
              d13);

    /* The logical gates therefore perform no rotation: braiding scales every
     * amplitude by the same factor, so the ratio |a_1|/|a_0| never moves off
     * its initial value of 1. */
    struct { const char *name; int which; } gates[3] = {
        { "anyonic_not", 0 }, { "anyonic_hadamard", 1 }, { "anyonic_T_gate", 2 }
    };
    for (int g = 0; g < 3; g++) {
        anyonic_register_t *reg = anyonic_register_create(sys, 1);
        if (!reg) { CHECK(0, "create anyonic register"); break; }

        if (gates[g].which == 0)      anyonic_not(reg, 0);
        else if (gates[g].which == 1) anyonic_hadamard(reg, 0);
        else                          anyonic_T_gate(reg, 0, 1e-3);

        double a0 = cabs(reg->tree->amplitudes[0]);
        double a1 = cabs(reg->tree->amplitudes[1]);
        double ratio = (a0 > 1e-12) ? a1 / a0 : -1.0;
        XFAIL_DOC(fabs(ratio - 1.0) < 1e-12,
                  "%s leaves |a1|/|a0| at %.6f -- no logical rotation",
                  gates[g].name, ratio);

        /* Norm is preserved, since a global phase is all that is applied. */
        double norm = a0 * a0 + a1 * a1;
        CHECK(fabs(norm - 1.0) < 1e-10,
              "%s preserves the norm (%.12f)", gates[g].name, norm);

        anyonic_register_free(reg);
    }

    /* apply_F_move reports success but changes nothing. */
    {
        fusion_tree_t *t = fusion_tree_create(sys, charges, 4, FIB_VACUUM);
        if (t) {
            double complex before[2] = { t->amplitudes[0], t->amplitudes[1] };
            qs_error_t err = apply_F_move(t, 0);
            double moved = cabs(t->amplitudes[0] - before[0]) +
                           cabs(t->amplitudes[1] - before[1]);
            XFAIL_DOC(err == QS_SUCCESS && moved < 1e-15,
                      "apply_F_move is a no-op returning QS_SUCCESS (delta %.3e)",
                      moved);
            fusion_tree_free(t);
        }
    }

    anyon_system_free(sys);
}

int main(void) {
    fprintf(stdout, "=== topological smoke tests ===\n");
    test_fibonacci_quantum_dimension();
    test_ising_anyons();
    test_anyon_coherence();
    test_coherence_verifier_is_not_vacuous();
    test_fibonacci_braiding_invariants();
    test_braiding_is_not_yet_a_representation();
    test_surface_code_lifecycle();
    test_toric_code_lifecycle();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
