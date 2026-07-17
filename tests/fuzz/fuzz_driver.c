/**
 * @file    fuzz_driver.c
 * @brief   Shared AFL++ persistent-mode shim + plain replay driver.
 *
 * This TU provides `main` for the non-libFuzzer builds of every target.
 * It is compiled and linked with each `<target>_replay` executable (and
 * with the AFL++ build when `afl-clang-fast` is used).  The libFuzzer
 * build does NOT link this file -- the fuzzer runtime owns `main` there.
 *
 * Two modes, chosen at compile time:
 *
 *   - AFL++ (`__AFL_COMPILER` defined by afl-clang-fast/-lto): drive the
 *     target through the persistent-mode loop, reusing one process for
 *     the whole campaign with a shared-memory testcase buffer.
 *
 *   - plain (any other compiler): a one-shot / batch replay driver.  With
 *     no arguments it reads a single testcase from stdin; with file
 *     arguments it replays each file in turn.  Directories are expanded
 *     to their regular-file entries so `run_fuzz.sh` can hand a whole
 *     seed-corpus directory to the binary.  This is what CI and ctest use
 *     to exercise the corpus under ASan/UBSan without a fuzzing runtime.
 *
 * Exit status is nonzero only when a sanitizer aborts the process; the
 * driver itself always returns 0 on clean completion so a green corpus
 * replay is unambiguous.
 */

#include <dirent.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* Optional one-time init hook, mirroring libFuzzer's LLVMFuzzerInitialize.
 * Provided here as a weak *definition* (with a default no-op body) rather
 * than a weak declaration: a weak undefined reference does not resolve
 * cleanly with the macOS static linker, whereas a weak definition links
 * everywhere and is silently overridden if a target supplies its own
 * strong LLVMFuzzerInitialize. */
__attribute__((weak)) int LLVMFuzzerInitialize(int *argc, char ***argv)
{
    (void)argc; (void)argv;
    return 0;
}

#define FUZZ_MAX_INPUT (16u * 1024u * 1024u)

#ifdef __AFL_COMPILER

/* AFL++ persistent mode.  __AFL_FUZZ_INIT / __AFL_LOOP /
 * __AFL_FUZZ_TESTCASE_BUF / __AFL_FUZZ_TESTCASE_LEN are provided by the
 * AFL instrumenting compiler. */
__AFL_FUZZ_INIT();

int main(int argc, char **argv)
{
    if (LLVMFuzzerInitialize) LLVMFuzzerInitialize(&argc, &argv);

#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif
    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(100000)) {
        size_t len = __AFL_FUZZ_TESTCASE_LEN;
        LLVMFuzzerTestOneInput(buf, len);
    }
    return 0;
}

#else /* plain replay driver */

static void replay_buf(const uint8_t *data, size_t size)
{
    LLVMFuzzerTestOneInput(data, size);
}

/* Read an entire stream into a heap buffer and replay it once. */
static int replay_stream(FILE *f)
{
    size_t cap = 4096, len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return -1;
    for (;;) {
        if (len == cap) {
            if (cap >= FUZZ_MAX_INPUT) break;
            cap *= 2;
            uint8_t *nb = (uint8_t *)realloc(buf, cap);
            if (!nb) { free(buf); return -1; }
            buf = nb;
        }
        size_t got = fread(buf + len, 1, cap - len, f);
        len += got;
        if (got == 0) break;
    }
    replay_buf(buf, len);
    free(buf);
    return 0;
}

static int replay_path(const char *path);

static int replay_dir(const char *path)
{
    DIR *d = opendir(path);
    if (!d) return -1;
    struct dirent *e;
    int rc = 0;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;         /* skip . .. hidden */
        char child[4096];
        int n = snprintf(child, sizeof(child), "%s/%s", path, e->d_name);
        if (n < 0 || (size_t)n >= sizeof(child)) continue;
        rc |= replay_path(child);
    }
    closedir(d);
    return rc;
}

static int replay_path(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "[fuzz-replay] cannot stat %s\n", path);
        return -1;
    }
    if (S_ISDIR(st.st_mode)) return replay_dir(path);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[fuzz-replay] cannot open %s\n", path);
        return -1;
    }
    int rc = replay_stream(f);
    fclose(f);
    return rc;
}

int main(int argc, char **argv)
{
    if (LLVMFuzzerInitialize) LLVMFuzzerInitialize(&argc, &argv);

    if (argc < 2) {
        return replay_stream(stdin) == 0 ? 0 : 1;
    }
    int rc = 0;
    for (int i = 1; i < argc; i++) rc |= replay_path(argv[i]);
    return rc == 0 ? 0 : 1;
}

#endif /* __AFL_COMPILER */
