/**
 * @file test_moonlab_export_abi.c
 * @brief Smoke test for the committed stable ABI surface.
 *
 * This test dlopens libquantumsim at runtime and resolves every symbol
 * declared in `src/applications/moonlab_export.h` via dlsym — exactly as a
 * downstream consumer (e.g. QGTL) would. A regression here means a
 * downstream integration just broke.
 *
 * Exits 0 on success, non-zero with a printed reason on any failure.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef int (*moonlab_qrng_bytes_fn)(uint8_t* buf, size_t size);
typedef void (*moonlab_abi_version_fn)(int* major, int* minor, int* patch);

static const char* const LIB_CANDIDATES[] = {
    "libquantumsim.dylib",
    "libquantumsim.0.dylib",
    "libquantumsim.so",
    "libquantumsim.so.0",
    NULL,
};

static void* open_library(void) {
    for (const char* const* name = LIB_CANDIDATES; *name; ++name) {
        void* h = dlopen(*name, RTLD_NOW | RTLD_LOCAL);
        if (h) {
            fprintf(stdout, "opened %s\n", *name);
            return h;
        }
    }
    return NULL;
}

static int test_version(void* h) {
    dlerror();
    moonlab_abi_version_fn fn =
        (moonlab_abi_version_fn)dlsym(h, "moonlab_abi_version");
    const char* err = dlerror();
    if (!fn || err) {
        fprintf(stderr, "dlsym(moonlab_abi_version) failed: %s\n",
                err ? err : "null");
        return 1;
    }

    int major = -1, minor = -1, patch = -1;
    fn(&major, &minor, &patch);
    if (major < 0 || minor < 0 || patch < 0) {
        fprintf(stderr, "moonlab_abi_version returned negative components: "
                        "%d.%d.%d\n", major, minor, patch);
        return 1;
    }
    fprintf(stdout, "ABI version: %d.%d.%d\n", major, minor, patch);

    /* Accept a NULL-tolerant call shape. */
    fn(NULL, NULL, NULL);
    return 0;
}

static int test_qrng(void* h) {
    dlerror();
    moonlab_qrng_bytes_fn fn =
        (moonlab_qrng_bytes_fn)dlsym(h, "moonlab_qrng_bytes");
    const char* err = dlerror();
    if (!fn || err) {
        fprintf(stderr, "dlsym(moonlab_qrng_bytes) failed: %s\n",
                err ? err : "null");
        return 1;
    }

    /* Zero-size call must succeed and not touch the buffer. */
    if (fn(NULL, 0) != 0) {
        fprintf(stderr, "moonlab_qrng_bytes(NULL, 0) did not succeed\n");
        return 1;
    }

    /* NULL buffer with nonzero size must fail gracefully. */
    if (fn(NULL, 16) == 0) {
        fprintf(stderr, "moonlab_qrng_bytes(NULL, 16) must reject NULL buf\n");
        return 1;
    }

    /* Draw some bytes and verify they are not all zero (entropy sanity). */
    uint8_t buf[64];
    memset(buf, 0, sizeof buf);
    int rc = fn(buf, sizeof buf);
    if (rc != 0) {
        fprintf(stderr, "moonlab_qrng_bytes(buf, 64) returned %d\n", rc);
        return 1;
    }

    int nonzero = 0;
    for (size_t i = 0; i < sizeof buf; ++i) {
        if (buf[i] != 0) { nonzero = 1; break; }
    }
    if (!nonzero) {
        fprintf(stderr, "moonlab_qrng_bytes produced an all-zero 64B block\n");
        return 1;
    }

    /* A second draw must differ from the first (probabilistically — a
     * 64-byte collision has vanishing probability for a functioning
     * source). */
    uint8_t buf2[64];
    memset(buf2, 0, sizeof buf2);
    if (fn(buf2, sizeof buf2) != 0) {
        fprintf(stderr, "second moonlab_qrng_bytes draw failed\n");
        return 1;
    }
    if (memcmp(buf, buf2, sizeof buf) == 0) {
        fprintf(stderr,
                "two consecutive 64B draws produced identical output — "
                "the QRNG appears broken\n");
        return 1;
    }

    fprintf(stdout, "QRNG draws look healthy (64B + 64B, distinct, non-zero)\n");
    return 0;
}

int main(void) {
    void* h = open_library();
    if (!h) {
        fprintf(stderr,
                "could not dlopen libquantumsim; tried: ");
        for (const char* const* name = LIB_CANDIDATES; *name; ++name) {
            fprintf(stderr, "%s%s", *name, *(name + 1) ? ", " : "\n");
        }
        fprintf(stderr, "dlerror: %s\n", dlerror());
        return 1;
    }

    int failures = 0;
    failures += test_version(h);
    failures += test_qrng(h);

    dlclose(h);

    if (failures) {
        fprintf(stderr, "ABI smoke test FAILED (%d failures)\n", failures);
        return 1;
    }
    fprintf(stdout, "ABI smoke test OK\n");
    return 0;
}
