/**
 * @file  test_qgtl_circuit_io.c
 * @brief Roundtrip tests for moonlab_qgtl_circuit serialization
 *        (v0.8.3 portable circuit format).
 *
 * Build a circuit covering every gate type, serialize, deserialize,
 * confirm gate-by-gate match.  Then exercise file save / load.
 */

#include "../../src/applications/moonlab_qgtl_backend.h"

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

static moonlab_qgtl_circuit_t *build_full_coverage_circuit(void)
{
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(4);
    if (!c) return NULL;

    /* One of every gate type. */
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_I,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_X,    1, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_Y,    2, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_Z,    3, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_S,    1, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_T,    2, 0, NULL);

    const double pi_3 = 3.14159265358979323846 / 3.0;
    const double pi_4 = 3.14159265358979323846 / 4.0;
    const double pi_7 = 3.14159265358979323846 / 7.0;

    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_RX,   0, 0, &pi_3);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_RY,   1, 0, &pi_4);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_RZ,   2, 0, &pi_7);

    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CY,   2, 1, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CZ,   3, 2, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_SWAP, 0, 3, NULL);

    return c;
}

static void test_size_query_then_serialize(void)
{
    fprintf(stdout, "\n--- size-query + serialize ---\n");
    moonlab_qgtl_circuit_t *c = build_full_coverage_circuit();
    CHECK(c != NULL, "build full-coverage circuit");
    if (!c) return;
    CHECK(moonlab_qgtl_circuit_num_gates(c) == 14, "expected 14 gates");

    size_t needed = 0;
    int rc = moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    CHECK(rc == MOONLAB_QGTL_OK, "size-query rc OK");
    CHECK(needed > 0, "size-query reports nonzero size (got %zu)", needed);

    char *buf = (char *)malloc(needed + 1);
    CHECK(buf != NULL, "malloc(%zu)", needed + 1);
    if (!buf) { moonlab_qgtl_circuit_free(c); return; }

    rc = moonlab_qgtl_circuit_serialize(c, buf, needed + 1, NULL);
    CHECK(rc == MOONLAB_QGTL_OK, "serialize into buf rc OK");
    CHECK(buf[needed] == '\0', "buf NUL-terminated");
    CHECK(strstr(buf, "# moonlab-circuit v1") != NULL, "header present");
    CHECK(strstr(buf, "NUM_QUBITS 4") != NULL, "num-qubits present");
    CHECK(strstr(buf, "H 0\n") != NULL, "H gate emitted");
    CHECK(strstr(buf, "CNOT 1 0\n") != NULL, "CNOT gate emitted");
    CHECK(strstr(buf, "RX 0 ") != NULL, "RX gate emitted with theta");

    free(buf);
    moonlab_qgtl_circuit_free(c);
}

static void test_roundtrip_matches(void)
{
    fprintf(stdout, "\n--- serialize -> deserialize roundtrip ---\n");
    moonlab_qgtl_circuit_t *c1 = build_full_coverage_circuit();
    CHECK(c1 != NULL, "build c1");
    if (!c1) return;

    size_t needed = 0;
    moonlab_qgtl_circuit_serialize(c1, NULL, 0, &needed);
    char *buf = (char *)malloc(needed + 1);
    moonlab_qgtl_circuit_serialize(c1, buf, needed + 1, NULL);

    int status = -1;
    moonlab_qgtl_circuit_t *c2 =
        moonlab_qgtl_circuit_deserialize(buf, needed, &status);
    CHECK(status == MOONLAB_QGTL_OK, "deserialize status OK");
    CHECK(c2 != NULL, "deserialize produced circuit");
    if (!c2) { free(buf); moonlab_qgtl_circuit_free(c1); return; }

    CHECK(moonlab_qgtl_circuit_num_qubits(c2) ==
          moonlab_qgtl_circuit_num_qubits(c1), "num_qubits match");
    CHECK(moonlab_qgtl_circuit_num_gates(c2) ==
          moonlab_qgtl_circuit_num_gates(c1), "num_gates match (got %d / %d)",
          moonlab_qgtl_circuit_num_gates(c2),
          moonlab_qgtl_circuit_num_gates(c1));

    /* Roundtrip-of-roundtrip: serializing c2 should match c1's text. */
    size_t needed2 = 0;
    moonlab_qgtl_circuit_serialize(c2, NULL, 0, &needed2);
    char *buf2 = (char *)malloc(needed2 + 1);
    moonlab_qgtl_circuit_serialize(c2, buf2, needed2 + 1, NULL);
    CHECK(needed == needed2, "serialized size identical (%zu == %zu)",
          needed, needed2);
    CHECK(memcmp(buf, buf2, needed) == 0,
          "byte-exact round-trip serialization");

    free(buf);
    free(buf2);
    moonlab_qgtl_circuit_free(c1);
    moonlab_qgtl_circuit_free(c2);
}

static void test_oom_path(void)
{
    fprintf(stdout, "\n--- buf-too-small OOM path ---\n");
    moonlab_qgtl_circuit_t *c = build_full_coverage_circuit();
    if (!c) return;

    char tiny[16];
    size_t needed = 0;
    int rc = moonlab_qgtl_circuit_serialize(c, tiny, sizeof(tiny), &needed);
    CHECK(rc == MOONLAB_QGTL_OOM, "tiny buf -> OOM");
    CHECK(needed > sizeof(tiny), "OOM path reports required size");

    moonlab_qgtl_circuit_free(c);
}

static void test_file_save_load(void)
{
    fprintf(stdout, "\n--- file save / load ---\n");
    moonlab_qgtl_circuit_t *c1 = build_full_coverage_circuit();
    if (!c1) return;

    const char *path = "test_qgtl_circuit_io_roundtrip.qcir";

    int rc = moonlab_qgtl_circuit_save(c1, path);
    CHECK(rc == MOONLAB_QGTL_OK, "save rc OK");

    int status = -1;
    moonlab_qgtl_circuit_t *c2 = moonlab_qgtl_circuit_load(path, &status);
    CHECK(status == MOONLAB_QGTL_OK, "load status OK");
    CHECK(c2 != NULL, "load produced circuit");
    if (!c2) { remove(path); moonlab_qgtl_circuit_free(c1); return; }

    CHECK(moonlab_qgtl_circuit_num_gates(c2) ==
          moonlab_qgtl_circuit_num_gates(c1),
          "load preserved gate count");

    remove(path);
    moonlab_qgtl_circuit_free(c1);
    moonlab_qgtl_circuit_free(c2);
}

static void test_bad_input(void)
{
    fprintf(stdout, "\n--- bad-input rejection ---\n");

    int status = 0;
    moonlab_qgtl_circuit_t *c =
        moonlab_qgtl_circuit_deserialize("garbage\n", 8, &status);
    CHECK(c == NULL, "garbage rejected (no NUM_QUBITS)");
    CHECK(status == MOONLAB_QGTL_BAD_ARG, "status is BAD_ARG");

    c = moonlab_qgtl_circuit_deserialize(
            "NUM_QUBITS 3\nWAT 0\n", 19, &status);
    CHECK(c == NULL, "unknown gate WAT rejected");

    c = moonlab_qgtl_circuit_deserialize(
            "NUM_QUBITS 99\nH 0\n", 18, &status);
    CHECK(c == NULL, "num_qubits=99 rejected (out of range)");

    /* Explicit lengths are authoritative.  A NUL inside that extent must be
     * rejected instead of terminating a prefix or acting as a line break. */
    static const char with_embedded_nul[] =
        "NUM_QUBITS 2\n"
        "H 0\0"
        "X 1\n";
    status = 12345;
    c = moonlab_qgtl_circuit_deserialize(
            with_embedded_nul, sizeof(with_embedded_nul) - 1, &status);
    CHECK(c == NULL, "explicit-length embedded NUL rejected");
    CHECK(status == MOONLAB_QGTL_BAD_ARG,
          "embedded NUL status is BAD_ARG");
    if (c) moonlab_qgtl_circuit_free(c);

    /* Regression: next_line used to report both EOF and an overlong line as
     * zero.  After a valid header that silently accepted a truncated prefix. */
    static const char header[] = "NUM_QUBITS 2\n";
    char overlong[sizeof(header) - 1 + 256];
    memcpy(overlong, header, sizeof(header) - 1);
    memset(overlong + sizeof(header) - 1, 'A', 256);
    status = 12345;
    c = moonlab_qgtl_circuit_deserialize(overlong, sizeof(overlong), &status);
    CHECK(c == NULL, "overlong final line rejected instead of treated as EOF");
    CHECK(status == MOONLAB_QGTL_BAD_ARG,
          "overlong final line status is BAD_ARG");
    if (c) moonlab_qgtl_circuit_free(c);

    /* Comments + blank lines tolerated. */
    const char *with_comments =
        "# header\n"
        "\n"
        "NUM_QUBITS 2\n"
        "# inline comment\n"
        "H 0\n"
        "CNOT 1 0\n";
    c = moonlab_qgtl_circuit_deserialize(with_comments, strlen(with_comments), &status);
    CHECK(c != NULL && status == MOONLAB_QGTL_OK,
          "comments + blank lines tolerated");
    if (c) {
        CHECK(moonlab_qgtl_circuit_num_gates(c) == 2,
              "comment-laden parse: 2 gates");
        moonlab_qgtl_circuit_free(c);
    }
}

int main(void)
{
    fprintf(stdout, "=== test_qgtl_circuit_io (v0.8.3) ===\n");
    test_size_query_then_serialize();
    test_roundtrip_matches();
    test_oom_path();
    test_file_save_load();
    test_bad_input();
    fprintf(stdout, "\n=== %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
