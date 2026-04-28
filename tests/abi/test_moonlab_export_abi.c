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
#include <complex.h>

typedef int (*moonlab_qrng_bytes_fn)(uint8_t* buf, size_t size);
typedef void (*moonlab_abi_version_fn)(int* major, int* minor, int* patch);
typedef int (*moonlab_qwz_chern_fn)(double m, size_t N, double* out_chern);

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

    /* moonlab_qwz_chern: topological / trivial phases of QWZ. */
    dlerror();
    moonlab_qwz_chern_fn qwz =
        (moonlab_qwz_chern_fn)dlsym(h, "moonlab_qwz_chern");
    if (!qwz || dlerror()) {
        fprintf(stderr, "dlsym(moonlab_qwz_chern) failed\n");
        failures++;
    } else {
        double c;
        int ct = qwz(+1.0, 32, &c);
        if (ct != -1) {
            fprintf(stderr, "moonlab_qwz_chern(m=+1) = %d (raw %.3f), "
                    "expected -1\n", ct, c);
            failures++;
        }
        int cz = qwz(+3.0, 32, NULL);
        if (cz != 0) {
            fprintf(stderr, "moonlab_qwz_chern(m=+3) = %d, expected 0\n", cz);
            failures++;
        }
        if (!failures) fprintf(stdout, "moonlab_qwz_chern OK\n");
    }

    /* CA-MPS ABI surface: probe the create / Clifford / non-Clifford /
     * observable entry points.  Don't drive a full simulation -- the
     * goal is verifying the symbols are exported with the documented
     * signatures.  Two-qubit Bell-state round-trip pins the most-used
     * shape (H + CNOT, then <Z_0 Z_1> = +1 on the +Phi+ Bell state). */
    typedef void* (*camps_create_fn)(uint32_t, uint32_t);
    typedef void  (*camps_free_fn)(void*);
    typedef int   (*camps_q_fn)(void*, uint32_t);
    typedef int   (*camps_cc_fn)(void*, uint32_t, uint32_t);
    typedef int   (*camps_expect_fn)(const void*, const uint8_t*, double _Complex*);

    dlerror();
    camps_create_fn  camps_create  = (camps_create_fn)  dlsym(h, "moonlab_ca_mps_create");
    camps_free_fn    camps_free    = (camps_free_fn)    dlsym(h, "moonlab_ca_mps_free");
    camps_q_fn       camps_h       = (camps_q_fn)       dlsym(h, "moonlab_ca_mps_h");
    camps_cc_fn      camps_cnot    = (camps_cc_fn)      dlsym(h, "moonlab_ca_mps_cnot");
    camps_expect_fn  camps_expect  = (camps_expect_fn)  dlsym(h, "moonlab_ca_mps_expect_pauli");

    if (!camps_create || !camps_free || !camps_h || !camps_cnot || !camps_expect) {
        fprintf(stderr, "dlsym for CA-MPS ABI symbols failed\n");
        failures++;
    } else {
        void* s = camps_create(2, 4);
        if (!s) {
            fprintf(stderr, "moonlab_ca_mps_create(2, 4) returned NULL\n");
            failures++;
        } else {
            (void)camps_h(s, 0);          /* prepare Bell state |Phi+> */
            (void)camps_cnot(s, 0, 1);
            uint8_t zz[2] = {3, 3};       /* Z_0 Z_1 */
            double _Complex e = 0.0;
            int rc = camps_expect(s, zz, &e);
            if (rc != 0) {
                fprintf(stderr, "expect_pauli rc = %d\n", rc); failures++;
            } else {
                double diff = creal(e) - 1.0;
                if (diff < 0) diff = -diff;
                if (diff > 1e-12) {
                    fprintf(stderr, "Bell <Z_0 Z_1> = %.6f, expected +1\n", creal(e));
                    failures++;
                } else {
                    fprintf(stdout, "moonlab_ca_mps_* (Bell <ZZ> = %.6f) OK\n", creal(e));
                }
            }
            camps_free(s);
        }
    }

    /* DMRG scalar entry point: TFIM ground-state energy at g = 1
     * (the quantum critical point), N = 8 chain, chi = 16, 5 sweeps.
     * Per-site energy at the critical point is ~ -4/pi for large N;
     * for N = 8 with periodic BCs we accept anything in [-1.3, -1.0]
     * per site, which is generous and only catches gross regressions
     * (a returned 0, +inf, or DBL_MAX). */
    typedef double (*dmrg_tfim_fn)(uint32_t, double, uint32_t, uint32_t);
    dlerror();
    dmrg_tfim_fn dmrg_tfim = (dmrg_tfim_fn)dlsym(h, "moonlab_dmrg_tfim_energy");
    if (!dmrg_tfim) {
        fprintf(stderr, "dlsym(moonlab_dmrg_tfim_energy) failed\n");
        failures++;
    } else {
        double E = dmrg_tfim(/*N=*/8, /*g=*/1.0, /*chi=*/16, /*sweeps=*/5);
        if (E >= 1e30) {
            fprintf(stderr, "moonlab_dmrg_tfim_energy returned sentinel %.3e\n", E);
            failures++;
        } else {
            double e_per_site = E / 8.0;
            if (e_per_site < -1.3 || e_per_site > -1.0) {
                fprintf(stderr, "moonlab_dmrg_tfim_energy/N = %.4f outside "
                        "expected [-1.3, -1.0]\n", e_per_site);
                failures++;
            } else {
                fprintf(stdout, "moonlab_dmrg_tfim_energy(N=8, g=1) = %.4f "
                        "(%.4f / site) OK\n", E, e_per_site);
            }
        }
    }

    /* DMRG Heisenberg: isotropic OBC chain N = 8, J = 1, Delta = 1,
     * h = 0.  The MPO uses Pauli convention (X.X + Y.Y + Z.Z per bond),
     * so the OBC ground-state energy from direct ED is -13.4997.  DMRG
     * with chi = 32, 8 sweeps converges to that within 1e-3.  Accept
     * [-13.6, -13.4] as the smoke band. */
    typedef double (*dmrg_heis_fn)(uint32_t, double, double, double,
                                    uint32_t, uint32_t);
    dlerror();
    dmrg_heis_fn dmrg_heis =
        (dmrg_heis_fn)dlsym(h, "moonlab_dmrg_heisenberg_energy");
    if (!dmrg_heis) {
        fprintf(stderr, "dlsym(moonlab_dmrg_heisenberg_energy) failed\n");
        failures++;
    } else {
        double E = dmrg_heis(/*N=*/8, /*J=*/1.0, /*Delta=*/1.0, /*h=*/0.0,
                             /*chi=*/32, /*sweeps=*/8);
        if (E >= 1e30) {
            fprintf(stderr, "moonlab_dmrg_heisenberg_energy returned sentinel %.3e\n", E);
            failures++;
        } else if (E > -13.4 || E < -13.6) {
            fprintf(stderr, "moonlab_dmrg_heisenberg_energy = %.4f outside "
                    "[-13.6, -13.4] (OBC ED = -13.4997)\n", E);
            failures++;
        } else {
            fprintf(stdout, "moonlab_dmrg_heisenberg_energy(N=8) = %.4f OK\n", E);
        }
    }

    dlclose(h);

    if (failures) {
        fprintf(stderr, "ABI smoke test FAILED (%d failures)\n", failures);
        return 1;
    }
    fprintf(stdout, "ABI smoke test OK\n");
    return 0;
}
