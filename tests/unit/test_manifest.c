/**
 * @file test_manifest.c
 * @brief Smoke-check the reproducibility manifest.
 *
 * Tight checks on the structural content (non-empty git SHA of the
 * right length, valid hostname, non-NULL label, correct version vs
 * VERSION.txt), plus a round-trip that parses the compact JSON emitted
 * by @c moonlab_manifest_write_json and verifies it contains the
 * expected keys with reasonable values.  We do not bundle a JSON
 * library, so the parser is a hand-rolled tolerance check on the
 * text stream; misses would show up as regressions if the emitter
 * changed its key set or format.
 */

#include "../../src/utils/manifest.h"
#include "moonlab_build_info.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#define strdup _strdup
#endif

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                               \
    if (!(cond)) {                                               \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);     \
        failures++;                                              \
    } else {                                                     \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);     \
    }                                                            \
} while (0)

/* Return non-zero iff @p haystack contains @p needle. */
static int contains(const char* haystack, const char* needle) {
    return strstr(haystack, needle) != NULL;
}

typedef void (*manifest_writer_t)(const moonlab_manifest_t* m, FILE* out);

static char* capture_manifest_json(const moonlab_manifest_t* m,
                                   manifest_writer_t writer,
                                   size_t* out_len) {
    FILE* mem = tmpfile();
    if (!mem) return NULL;

    writer(m, mem);
    fflush(mem);

    if (fseek(mem, 0, SEEK_END) != 0) {
        fclose(mem);
        return NULL;
    }

    long len = ftell(mem);
    if (len < 0 || fseek(mem, 0, SEEK_SET) != 0) {
        fclose(mem);
        return NULL;
    }

    char* buf = malloc((size_t)len + 1);
    if (!buf) {
        fclose(mem);
        return NULL;
    }

    size_t nread = fread(buf, 1, (size_t)len, mem);
    fclose(mem);
    if (nread != (size_t)len) {
        free(buf);
        return NULL;
    }

    buf[nread] = '\0';
    if (out_len) *out_len = nread;
    return buf;
}

static void test_build_info_macros(void) {
    fprintf(stdout, "\n-- build_info macros --\n");
    CHECK(strlen(MOONLAB_VERSION_STRING) > 0, "VERSION string non-empty: %s",
          MOONLAB_VERSION_STRING);
    CHECK(MOONLAB_VERSION_MAJOR >= 0, "VERSION major = %d", MOONLAB_VERSION_MAJOR);
    CHECK(strlen(MOONLAB_GIT_SHA) == 40 ||
          strcmp(MOONLAB_GIT_SHA, "unknown") == 0,
          "git SHA is 40 hex or 'unknown': %s", MOONLAB_GIT_SHA);
    CHECK(strlen(MOONLAB_GIT_SHA_SHORT) >= 7 ||
          strcmp(MOONLAB_GIT_SHA_SHORT, "unknown") == 0,
          "git short SHA >=7 chars: %s", MOONLAB_GIT_SHA_SHORT);
    CHECK(MOONLAB_GIT_DIRTY == 0 || MOONLAB_GIT_DIRTY == 1,
          "git_dirty is 0 or 1: %d", MOONLAB_GIT_DIRTY);
    CHECK(strlen(MOONLAB_BUILD_TIMESTAMP) == 20,
          "build_timestamp ISO-8601 20 chars: %s", MOONLAB_BUILD_TIMESTAMP);
    CHECK(strlen(MOONLAB_COMPILER_ID) > 0,
          "compiler_id: %s", MOONLAB_COMPILER_ID);
    CHECK(strlen(MOONLAB_PLATFORM_NAME) > 0,
          "platform: %s", MOONLAB_PLATFORM_NAME);
}

static void test_capture_release(void) {
    fprintf(stdout, "\n-- capture / release --\n");
    moonlab_manifest_t m;
    int rc = moonlab_manifest_capture(&m, "test_manifest", 42);
    CHECK(rc == 0, "capture rc=0");
    CHECK(m.run_label && strcmp(m.run_label, "test_manifest") == 0, "run_label");
    CHECK(m.seed == 42, "seed = 42");
    CHECK(m.hostname && *m.hostname, "hostname non-empty: %s", m.hostname);
    CHECK(m.cpu_count >= 1, "cpu_count >= 1: %d", m.cpu_count);
    CHECK(m.mem_total_bytes > 0, "mem_total_bytes > 0");
    CHECK(m.run_start_iso && strlen(m.run_start_iso) == 20,
          "run_start_iso is 20 chars: %s", m.run_start_iso);
    CHECK(m.version && strcmp(m.version, MOONLAB_VERSION_STRING) == 0,
          "version matches build-info: %s", m.version);

    /* Release twice -- second call must be a no-op. */
    moonlab_manifest_release(&m);
    moonlab_manifest_release(&m);
    CHECK(m.run_label == NULL && m.hostname == NULL,
          "release zeros the owned pointers");
}

static void test_default_label_when_null(void) {
    fprintf(stdout, "\n-- null-label handling --\n");
    moonlab_manifest_t m;
    CHECK(moonlab_manifest_capture(&m, NULL, 0) == 0, "capture(NULL) rc=0");
    CHECK(m.run_label && strcmp(m.run_label, "(unnamed)") == 0,
          "null label becomes (unnamed)");
    moonlab_manifest_release(&m);
}

static void test_json_emission(void) {
    fprintf(stdout, "\n-- compact + pretty JSON --\n");
    moonlab_manifest_t m;
    moonlab_manifest_capture(&m, "my_bench", 0xCAFEBABE);
    m.metrics_json = "{\"throughput_gflops\":612.3,\"notes\":\"hot\\ncache\"}";

    size_t buflen = 0;
    char* buf = capture_manifest_json(&m, moonlab_manifest_write_json, &buflen);

    CHECK(buf && buflen > 0, "JSON emitted %zu bytes", buflen);
    CHECK(buf[0] == '{' && buf[buflen - 1] == '}', "JSON wrapped in braces");
    CHECK(contains(buf, "\"run_label\": \"my_bench\""), "has run_label");
    CHECK(contains(buf, "\"seed\":"),  "has seed key");
    CHECK(contains(buf, "3405691582"),
          "seed 0xCAFEBABE = 3405691582 embedded as decimal");
    CHECK(contains(buf, "\"version\": \""),    "has version");
    CHECK(contains(buf, "\"git_sha\": \""),    "has git_sha");
    CHECK(contains(buf, "\"hostname\": \""),   "has hostname");
    CHECK(contains(buf, "\"metrics\": {"),     "has metrics object");
    CHECK(contains(buf, "\"throughput_gflops\":612.3"),
          "metrics embedded verbatim");
    /* The raw newline in the caller's JSON fragment is preserved as-is
     * because metrics_json is spliced, not re-escaped. */
    CHECK(contains(buf, "hot\\ncache"),
          "caller's pre-escaped newline passes through unchanged");
    /* No double quote before run_label (catches a common off-by-one
     * where the emitter forgot to open the brace). */
    CHECK(strncmp(buf, "{\"run_label\"", 12) == 0,
          "JSON opens with {\"run_label\"");

    free(buf);
    moonlab_manifest_release(&m);
}

static void test_pretty_roundtrip(void) {
    fprintf(stdout, "\n-- pretty-print roundtrip --\n");
    moonlab_manifest_t m;
    moonlab_manifest_capture(&m, "pretty", 7);
    size_t buflen = 0;
    char* buf = capture_manifest_json(&m, moonlab_manifest_write_json_pretty, &buflen);
    CHECK(buf && buflen > 0, "pretty JSON emitted %zu bytes", buflen);
    CHECK(strncmp(buf, "{\n  \"run_label\"", 15) == 0,
          "pretty JSON starts with {\\n<2-space indent>");
    CHECK(buf[buflen - 1] == '\n', "pretty JSON ends with trailing newline");
    free(buf);
    moonlab_manifest_release(&m);
}

static void test_stamp_finish(void) {
    fprintf(stdout, "\n-- stamp_finish --\n");
    moonlab_manifest_t m;
    moonlab_manifest_capture(&m, "timed", 0);
    CHECK(m.run_finish_iso && *m.run_finish_iso == '\0',
          "run_finish_iso empty before stamp");
    moonlab_manifest_stamp_finish(&m);
    CHECK(m.run_finish_iso && strlen(m.run_finish_iso) == 20,
          "stamp_finish writes 20-char ISO: %s", m.run_finish_iso);
    CHECK(m.run_elapsed_seconds >= 0.0,
          "elapsed seconds non-negative: %.2f", m.run_elapsed_seconds);
    /* Idempotent. */
    char* first_copy = strdup(m.run_finish_iso);
    moonlab_manifest_stamp_finish(&m);
    CHECK(strcmp(m.run_finish_iso, first_copy) == 0,
          "stamp_finish is idempotent");
    free(first_copy);
    moonlab_manifest_release(&m);
}

int main(void) {
    fprintf(stdout, "=== moonlab_manifest unit tests ===\n");
    test_build_info_macros();
    test_capture_release();
    test_default_label_when_null();
    test_json_emission();
    test_pretty_roundtrip();
    test_stamp_finish();
    fprintf(stdout, "\n%d failure(s)\n", failures);
    return (failures == 0) ? 0 : 1;
}
