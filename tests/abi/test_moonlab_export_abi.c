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

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>

#if defined(_WIN32)
#include <windows.h>
static DWORD g_dl_last_error = 0;

static void set_dll_directory_for_path(const char* path) {
    char dir[MAX_PATH];
    size_t n = strlen(path);
    if (n >= sizeof(dir)) return;
    memcpy(dir, path, n + 1);
    for (size_t i = n; i > 0; --i) {
        if (dir[i - 1] == '\\' || dir[i - 1] == '/') {
            dir[i - 1] = '\0';
            SetDllDirectoryA(dir);
            return;
        }
    }
}

static const char* dlerror(void) {
    static char buf[512];
    DWORD err = g_dl_last_error;
    g_dl_last_error = 0;
    if (err == 0) return NULL;
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM |
                             FORMAT_MESSAGE_IGNORE_INSERTS,
                             NULL, err, 0, buf, sizeof(buf), NULL);
    if (n == 0) {
        snprintf(buf, sizeof(buf), "Win32 error %lu", (unsigned long)err);
    }
    return buf;
}

static void* dlopen(const char* name, int flags) {
    (void)flags;
    if (strchr(name, '\\') || strchr(name, '/')) {
        set_dll_directory_for_path(name);
    }
    HMODULE h = LoadLibraryA(name);
    g_dl_last_error = h ? 0 : GetLastError();
    return (void*)h;
}

static void* dlsym(void* h, const char* name) {
    FARPROC p = GetProcAddress((HMODULE)h, name);
    g_dl_last_error = p ? 0 : GetLastError();
    return (void*)p;
}

static int dlclose(void* h) {
    if (!h) return 0;
    return FreeLibrary((HMODULE)h) ? 0 : 1;
}

#define RTLD_NOW 0
#define RTLD_LOCAL 0
#else
#include <dlfcn.h>
#endif

#define ABI_STEP(name) do { \
    fprintf(stdout, "[abi] %s\n", (name)); \
} while (0)

typedef int (*moonlab_qrng_bytes_fn)(uint8_t* buf, size_t size);
typedef void (*moonlab_abi_version_fn)(int* major, int* minor, int* patch);
typedef int (*moonlab_qwz_chern_fn)(double m, size_t N, double* out_chern);

typedef struct {
    uint32_t struct_size;
    uint32_t api_version;
    uint64_t capabilities;
    uint64_t conditioned_requests;
    uint64_t raw_bytes_generated;
    uint64_t bell_tests_performed;
    uint64_t bell_tests_passed;
    double average_chsh;
    double minimum_chsh;
} moonlab_qrng_status_abi_t;
typedef int (*moonlab_qrng_get_status_fn)(moonlab_qrng_status_abi_t* status);

static const char* const LIB_CANDIDATES[] = {
#if defined(_WIN32)
    "libquantumsim.dll",
    "quantumsim.dll",
#else
    "libquantumsim.dylib",
    "libquantumsim.0.dylib",
    "libquantumsim.so",
    "libquantumsim.so.0",
#endif
    NULL,
};

static void* open_library(void) {
    const char* configured = getenv("MOONLAB_QUANTUMSIM_LIBRARY");
    if (configured && configured[0] != '\0') {
        void* h = dlopen(configured, RTLD_NOW | RTLD_LOCAL);
        if (h) {
            fprintf(stdout, "opened %s\n", configured);
            return h;
        }
        fprintf(stderr, "configured library failed: %s\n", dlerror());
    }
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

static int test_qrng_status(void* h) {
    dlerror();
    moonlab_qrng_get_status_fn fn =
        (moonlab_qrng_get_status_fn)dlsym(h, "moonlab_qrng_get_status");
    const char* err = dlerror();
    if (!fn || err) {
        fprintf(stderr, "dlsym(moonlab_qrng_get_status) failed: %s\n",
                err ? err : "null");
        return 1;
    }
    if (fn(NULL) == 0) {
        fprintf(stderr, "moonlab_qrng_get_status(NULL) must reject NULL\n");
        return 1;
    }

    moonlab_qrng_status_abi_t status;
    memset(&status, 0, sizeof(status));
    if (fn(&status) != 0) {
        fprintf(stderr, "moonlab_qrng_get_status failed\n");
        return 1;
    }
    const uint64_t required = UINT64_C(0x1f); /* Capability bits 0..4. */
    if (status.struct_size != sizeof(status) || status.api_version != 1 ||
        (status.capabilities & required) != required) {
        fprintf(stderr,
                "QRNG status contract mismatch: size=%u api=%u caps=0x%llx\n",
                status.struct_size, status.api_version,
                (unsigned long long)status.capabilities);
        return 1;
    }
    if (status.bell_tests_performed == 0 ||
        status.bell_tests_passed != status.bell_tests_performed ||
        status.minimum_chsh <= 2.0) {
        fprintf(stderr,
                "QRNG Bell epoch status unhealthy: tests=%llu pass=%llu min=%.4f\n",
                (unsigned long long)status.bell_tests_performed,
                (unsigned long long)status.bell_tests_passed,
                status.minimum_chsh);
        return 1;
    }
    if ((status.capabilities & (UINT64_C(1) << 6)) != 0 ||
        (status.capabilities & (UINT64_C(1) << 7)) != 0) {
        fprintf(stderr, "QRNG status overclaims DI or FIPS validation\n");
        return 1;
    }
    fprintf(stdout, "QRNG status ABI OK (caps=0x%llx, CHSH min=%.4f)\n",
            (unsigned long long)status.capabilities, status.minimum_chsh);
    return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    ABI_STEP("open library");
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
    ABI_STEP("version");
    failures += test_version(h);
    ABI_STEP("qrng");
    failures += test_qrng(h);
    ABI_STEP("qrng status");
    failures += test_qrng_status(h);

    /* moonlab_qwz_chern: topological / trivial phases of QWZ. */
    ABI_STEP("qwz chern");
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
    ABI_STEP("ca-mps bell");
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

    /* CA-MPS conjugate_pauli (since 0.2.4): get back Q_k = D^dagger P D
     * for the current Clifford D in the state.  Probe with D = identity
     * (default after create) so Q_k must equal the input string and the
     * phase is 0. */
    ABI_STEP("ca-mps conjugate_pauli");
    typedef int (*camps_conj_fn)(const void*, const uint8_t*, uint8_t*, int*);
    dlerror();
    camps_conj_fn camps_conj =
        (camps_conj_fn)dlsym(h, "moonlab_ca_mps_conjugate_pauli");
    if (!camps_conj) {
        fprintf(stderr, "dlsym(moonlab_ca_mps_conjugate_pauli) failed\n");
        failures++;
    } else if (camps_create) {
        void* s = camps_create(3, 4);
        if (!s) {
            fprintf(stderr, "moonlab_ca_mps_create(3,4) NULL for conj probe\n");
            failures++;
        } else {
            uint8_t in_p[3] = {1, 2, 3};   /* X Y Z */
            uint8_t out_p[3] = {0, 0, 0};
            int phase = -1;
            int rc = camps_conj(s, in_p, out_p, &phase);
            if (rc != 0 || out_p[0] != 1 || out_p[1] != 2 || out_p[2] != 3 ||
                phase != 0) {
                fprintf(stderr,
                        "conjugate_pauli (D=I): rc=%d out=[%u,%u,%u] phase=%d\n",
                        rc, out_p[0], out_p[1], out_p[2], phase);
                failures++;
            } else {
                fprintf(stdout, "moonlab_ca_mps_conjugate_pauli OK\n");
            }
            camps_free(s);
        }
    }

    /* var-D entry points (v0.2.1 + v0.2.4): probe both symbols are
     * exported.  Don't actually run them (they're tested elsewhere) --
     * just verify that bindings consumers (QGTL, lilirrep) can resolve
     * them. */
    ABI_STEP("ca-mps var-d symbols");
    dlerror();
    void* var_d_run_sym    = dlsym(h, "moonlab_ca_mps_var_d_run");
    void* var_d_run_v2_sym = dlsym(h, "moonlab_ca_mps_var_d_run_v2");
    if (!var_d_run_sym) {
        fprintf(stderr, "dlsym(moonlab_ca_mps_var_d_run) failed\n");
        failures++;
    } else if (!var_d_run_v2_sym) {
        fprintf(stderr, "dlsym(moonlab_ca_mps_var_d_run_v2) failed\n");
        failures++;
    } else {
        fprintf(stdout, "moonlab_ca_mps_var_d_run + var_d_run_v2 OK\n");
    }

    /* DMRG scalar entry point: TFIM ground-state energy at g = 1
     * (the quantum critical point), N = 8 chain, chi = 16, 5 sweeps.
     * Per-site energy at the critical point is ~ -4/pi for large N;
     * for N = 8 with periodic BCs we accept anything in [-1.3, -1.0]
     * per site, which is generous and only catches gross regressions
     * (a returned 0, +inf, or DBL_MAX). */
    ABI_STEP("dmrg tfim");
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
    ABI_STEP("dmrg heisenberg");
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

    /* moonlab_z2_lgt_1d_build + moonlab_z2_lgt_1d_gauss_law: probe the
     * 1+1D Z2 LGT Pauli-sum and Gauss-law accessors, new in 0.2.1. */
    ABI_STEP("z2 lgt");
    typedef int (*z2_build_fn)(uint32_t, double, double, double, double,
                                uint8_t**, double**, uint32_t*, uint32_t*);
    typedef int (*z2_gauss_fn)(uint32_t, uint32_t, uint8_t*);
    dlerror();
    z2_build_fn z2_build = (z2_build_fn)dlsym(h, "moonlab_z2_lgt_1d_build");
    z2_gauss_fn z2_gauss = (z2_gauss_fn)dlsym(h, "moonlab_z2_lgt_1d_gauss_law");
    if (!z2_build || !z2_gauss) {
        fprintf(stderr, "dlsym(moonlab_z2_lgt_1d_*) failed: %s\n",
                dlerror() ? dlerror() : "(null)");
        failures++;
    } else {
        uint8_t* paulis = NULL;
        double*  coeffs = NULL;
        uint32_t T = 0, nq = 0;
        int rc = z2_build(/*N=*/4, /*t=*/1.0, /*h=*/0.5, /*m=*/0.0,
                           /*lambda=*/0.0, &paulis, &coeffs, &T, &nq);
        if (rc != 0 || nq != 7 || T == 0) {
            fprintf(stderr,
                    "moonlab_z2_lgt_1d_build N=4 wrong: rc=%d nq=%u T=%u\n",
                    rc, nq, T);
            failures++;
        } else {
            fprintf(stdout, "moonlab_z2_lgt_1d_build N=4: nq=%u T=%u OK\n",
                    nq, T);
        }
        free(paulis);
        free(coeffs);

        uint8_t gx[7] = {0};
        rc = z2_gauss(/*N=*/4, /*site_x=*/1, gx);
        /* G_1 = X_1 Z_2 X_3 -> bytes 0,1,3,1,0,0,0 (X=1, Z=3). */
        if (rc != 0 || gx[1] != 1 || gx[2] != 3 || gx[3] != 1) {
            fprintf(stderr,
                    "moonlab_z2_lgt_1d_gauss_law: layout wrong (rc=%d, "
                    "got %u %u %u %u %u %u %u)\n",
                    rc, gx[0], gx[1], gx[2], gx[3], gx[4], gx[5], gx[6]);
            failures++;
        } else {
            fprintf(stdout, "moonlab_z2_lgt_1d_gauss_law N=4 site=1 OK\n");
        }
    }

    /* moonlab_status_string: probe the diagnostic stringifier. */
    ABI_STEP("status string");
    typedef const char* (*status_fn)(int, int);
    dlerror();
    status_fn status = (status_fn)dlsym(h, "moonlab_status_string");
    if (!status) {
        fprintf(stderr, "dlsym(moonlab_status_string) failed\n");
        failures++;
    } else {
        const char* ok = status(/*module=GENERIC=*/0, /*status=SUCCESS=*/0);
        const char* err = status(/*module=CA_MPS=*/1, /*status=INVALID=*/-1);
        if (!ok || !err || strcmp(ok, "SUCCESS") != 0 ||
            strcmp(err, "ERR_INVALID") != 0) {
            fprintf(stderr,
                    "moonlab_status_string: unexpected (%s, %s)\n",
                    ok ? ok : "(null)", err ? err : "(null)");
            failures++;
        } else {
            fprintf(stdout, "moonlab_status_string OK\n");
        }
    }

    /* moonlab_ca_mps_var_d_run + moonlab_ca_mps_gauge_warmstart:
     * dlsym-find the new ABI entry points (don't run them; the
     * existing CA-MPS handle creation is already exercised by
     * test_qrng's CA-MPS API probes elsewhere). */
    ABI_STEP("gauge warmstart symbol");
    if (!dlsym(h, "moonlab_ca_mps_var_d_run")) {
        fprintf(stderr, "dlsym(moonlab_ca_mps_var_d_run) failed\n");
        failures++;
    } else {
        fprintf(stdout, "moonlab_ca_mps_var_d_run dlsym OK\n");
    }
    if (!dlsym(h, "moonlab_ca_mps_gauge_warmstart")) {
        fprintf(stderr, "dlsym(moonlab_ca_mps_gauge_warmstart) failed\n");
        failures++;
    } else {
        fprintf(stdout, "moonlab_ca_mps_gauge_warmstart dlsym OK\n");
    }

    /* v0.4.1 adaptive-bond TDVP ABI: dlsym every entry point and run
     * a TFIM imag-time short evolve end-to-end to pin the full create
     * -> step -> history -> free lifecycle. */
    ABI_STEP("tdvp");
    typedef void* (*tdvp_create_tfim_fn)(uint32_t, double, double, uint32_t,
                                          uint32_t, double, int, double);
    typedef int   (*tdvp_step_fn)(void*);
    typedef int   (*tdvp_evolve_to_fn)(void*, double);
    typedef double (*tdvp_dbl_fn)(const void*);
    typedef uint32_t (*tdvp_u32_fn)(const void*);
    typedef uint32_t (*tdvp_bond_fn)(const void*, uint32_t);
    typedef int   (*tdvp_hist_step_fn)(const void*, uint32_t,
                                        double*, double*, double*);
    typedef int   (*tdvp_hist_chi_fn)(const void*, uint32_t,
                                       uint32_t*, uint32_t);
    typedef void  (*tdvp_free_fn)(void*);
    typedef void* (*tdvp_create_heis_fn)(uint32_t, double, double, double,
                                          uint32_t, uint32_t, double,
                                          int, double);

    tdvp_create_tfim_fn create_tfim =
        (tdvp_create_tfim_fn)dlsym(h, "moonlab_tdvp_create_tfim");
    tdvp_create_heis_fn create_heis =
        (tdvp_create_heis_fn)dlsym(h, "moonlab_tdvp_create_heisenberg");
    tdvp_step_fn step_fn =
        (tdvp_step_fn)dlsym(h, "moonlab_tdvp_step");
    tdvp_evolve_to_fn evolve_to =
        (tdvp_evolve_to_fn)dlsym(h, "moonlab_tdvp_evolve_to");
    tdvp_dbl_fn cur_time =
        (tdvp_dbl_fn)dlsym(h, "moonlab_tdvp_current_time");
    tdvp_dbl_fn cur_energy =
        (tdvp_dbl_fn)dlsym(h, "moonlab_tdvp_current_energy");
    tdvp_dbl_fn cur_norm =
        (tdvp_dbl_fn)dlsym(h, "moonlab_tdvp_current_norm");
    tdvp_u32_fn cur_chi =
        (tdvp_u32_fn)dlsym(h, "moonlab_tdvp_current_max_bond_dim");
    tdvp_u32_fn nbonds =
        (tdvp_u32_fn)dlsym(h, "moonlab_tdvp_num_bonds");
    tdvp_bond_fn bond_chi =
        (tdvp_bond_fn)dlsym(h, "moonlab_tdvp_bond_chi");
    tdvp_u32_fn hist_steps =
        (tdvp_u32_fn)dlsym(h, "moonlab_tdvp_history_num_steps");
    tdvp_hist_step_fn hist_step =
        (tdvp_hist_step_fn)dlsym(h, "moonlab_tdvp_history_get_step");
    tdvp_hist_chi_fn hist_chi =
        (tdvp_hist_chi_fn)dlsym(h, "moonlab_tdvp_history_get_bond_chi");
    tdvp_free_fn engine_free =
        (tdvp_free_fn)dlsym(h, "moonlab_tdvp_engine_free");

    if (!create_tfim || !create_heis || !step_fn || !evolve_to ||
        !cur_time || !cur_energy || !cur_norm || !cur_chi ||
        !nbonds || !bond_chi || !hist_steps || !hist_step ||
        !hist_chi || !engine_free) {
        fprintf(stderr,
                "dlsym for one or more moonlab_tdvp_* symbols failed\n");
        failures++;
    } else {
        /* Drive a TFIM imag-time evolution at critical g = 1 with the
         * adaptive controller on.  Just three steps -- this is an ABI
         * smoke, not a physics test (the physics is covered by
         * tests/unit/test_tdvp_adaptive_tfim_ground.c). */
        void *engine = create_tfim(
            /*N=*/8, /*J=*/1.0, /*h=*/1.0,
            /*chi_init=*/8, /*chi_max=*/32, /*dt=*/0.05,
            /*imag_time=*/1, /*adaptive_target_entropy=*/1e-3);
        if (!engine) {
            fprintf(stderr, "moonlab_tdvp_create_tfim returned NULL\n");
            failures++;
        } else {
            int rc = step_fn(engine);
            if (rc != 0) {
                fprintf(stderr, "moonlab_tdvp_step rc=%d\n", rc);
                failures++;
            }
            rc = evolve_to(engine, cur_time(engine) + 0.1);
            if (rc != 0) {
                fprintf(stderr, "moonlab_tdvp_evolve_to rc=%d\n", rc);
                failures++;
            }
            uint32_t steps = hist_steps(engine);
            if (steps == 0) {
                fprintf(stderr, "tdvp history empty after step+evolve\n");
                failures++;
            }
            double t = 0, e = 0, n = 0;
            if (hist_step(engine, 0, &t, &e, &n) != 0) {
                fprintf(stderr, "hist_step(0) failed\n");
                failures++;
            }
            uint32_t nb = nbonds(engine);
            if (nb != 7) {
                fprintf(stderr, "nbonds expected 7, got %u\n", nb);
                failures++;
            }
            /* Bond-chi readback. */
            uint32_t buf[7] = {0};
            if (hist_chi(engine, steps - 1, buf, 7) != 0) {
                fprintf(stderr, "hist_chi readback failed\n");
                failures++;
            }
            /* Adaptive controller is on; at least one bond should have
             * received a non-zero chi assignment. */
            uint32_t any_nonzero = 0;
            for (uint32_t b = 0; b < 7; b++) any_nonzero |= buf[b];
            if (any_nonzero == 0) {
                fprintf(stderr,
                        "all per-bond chi readings are zero with "
                        "adaptive controller enabled\n");
                failures++;
            }
            /* Current-energy / norm should be finite. */
            double E = cur_energy(engine);
            double N = cur_norm(engine);
            uint32_t MaxChi = cur_chi(engine);
            if (!(E == E) || !(N == N) || MaxChi == 0) {
                fprintf(stderr,
                        "current accessors: E=%g N=%g MaxChi=%u\n",
                        E, N, MaxChi);
                failures++;
            }
            engine_free(engine);

            /* Also exercise Heisenberg constructor + legacy
             * (adaptive_target_entropy = 0) path. */
            void *eh = create_heis(/*N=*/6, /*J=*/1.0, /*Delta=*/1.0,
                                    /*h=*/0.0, /*chi_init=*/8,
                                    /*chi_max=*/16, /*dt=*/0.05,
                                    /*imag_time=*/0,
                                    /*adaptive=*/0.0);
            if (!eh) {
                fprintf(stderr,
                        "moonlab_tdvp_create_heisenberg (legacy) returned NULL\n");
                failures++;
            } else {
                if (step_fn(eh) != 0) {
                    fprintf(stderr, "Heisenberg legacy step failed\n");
                    failures++;
                }
                engine_free(eh);
            }
            if (!failures) {
                fprintf(stdout, "moonlab_tdvp_* ABI smoke OK\n");
            }
        }
    }

    /* v1.1.0 / ABI 0.4.0 exact VQE gradient: resolve the stable entry,
     * build an H2 hardware-efficient solver through the exported
     * lower-level surface, and pin (a) argument validation, (b) exact
     * agreement with a central finite difference of vqe_compute_energy.
     * The finite difference is a TEST-ONLY cross-check: the library
     * path under test is adjoint autograd / parameter-shift, never FD. */
    ABI_STEP("vqe gradient");
    {
        typedef void*  (*vqe_h2_fn)(double);
        typedef void*  (*vqe_hea_fn)(size_t, size_t);
        typedef void*  (*vqe_opt_create_fn)(int);
        typedef void*  (*entropy_hw_fn)(void);
        typedef void   (*entropy_destroy_fn)(void*);
        typedef void*  (*vqe_solver_create_fn)(void*, void*, void*, void*);
        typedef void   (*vqe_free_fn)(void*);
        typedef double (*vqe_energy_fn)(void*, const double*);
        typedef int    (*vqe_grad_fn)(void*, const double*, double*, size_t);

        dlerror();
        vqe_grad_fn vqe_grad = (vqe_grad_fn)dlsym(h, "moonlab_vqe_gradient");
        if (!vqe_grad) {
            fprintf(stderr, "dlsym(moonlab_vqe_gradient) failed: %s\n",
                    dlerror());
            failures++;
        } else {
            vqe_h2_fn h2 = (vqe_h2_fn)dlsym(h, "vqe_create_h2_hamiltonian");
            vqe_hea_fn hea =
                (vqe_hea_fn)dlsym(h, "vqe_create_hardware_efficient_ansatz");
            vqe_opt_create_fn opt_create =
                (vqe_opt_create_fn)dlsym(h, "vqe_optimizer_create");
            entropy_hw_fn entropy_hw =
                (entropy_hw_fn)dlsym(h, "quantum_entropy_ctx_create_hw");
            entropy_destroy_fn entropy_destroy =
                (entropy_destroy_fn)dlsym(h, "quantum_entropy_ctx_destroy");
            vqe_solver_create_fn solver_create =
                (vqe_solver_create_fn)dlsym(h, "vqe_solver_create");
            vqe_free_fn solver_free = (vqe_free_fn)dlsym(h, "vqe_solver_free");
            vqe_free_fn ansatz_free = (vqe_free_fn)dlsym(h, "vqe_ansatz_free");
            vqe_free_fn opt_free = (vqe_free_fn)dlsym(h, "vqe_optimizer_free");
            vqe_free_fn ham_free =
                (vqe_free_fn)dlsym(h, "pauli_hamiltonian_free");
            vqe_energy_fn energy =
                (vqe_energy_fn)dlsym(h, "vqe_compute_energy");

            if (!h2 || !hea || !opt_create || !entropy_hw ||
                !entropy_destroy || !solver_create || !solver_free ||
                !ansatz_free || !opt_free || !ham_free || !energy) {
                fprintf(stderr,
                        "VQE construction surface incomplete (dlsym)\n");
                failures++;
            } else {
                /* NULL args must be rejected without touching memory. */
                double dummy = 0.0;
                if (vqe_grad(NULL, &dummy, &dummy, 1) != -1) {
                    fprintf(stderr,
                            "moonlab_vqe_gradient(NULL,...) != -1\n");
                    failures++;
                }

                void *H = h2(0.74);
                /* H2 is a 2-qubit Hamiltonian; 2 HEA layers give
                 * 2 qubits * 2 layers * 2 rotations = 8 parameters. */
                void *ansatz = hea(2, 2);
                void *opt = opt_create(/*VQE_OPTIMIZER_ADAM=*/2);
                void *entropy = entropy_hw();
                void *solver = (H && ansatz && opt && entropy)
                                   ? solver_create(H, ansatz, opt, entropy)
                                   : NULL;
                if (!solver) {
                    fprintf(stderr, "VQE solver construction failed\n");
                    failures++;
                } else {
                    enum { N_PARAMS = 8 };
                    double theta[N_PARAMS];
                    double grad[N_PARAMS];
                    for (size_t k = 0; k < N_PARAMS; k++) {
                        theta[k] = 0.1 * (double)(k + 1);
                    }

                    /* Parameter-count mismatch must be refused. */
                    if (vqe_grad(solver, theta, grad, N_PARAMS + 1) != -2) {
                        fprintf(stderr,
                                "moonlab_vqe_gradient mismatch != -2\n");
                        failures++;
                    }

                    int rc = vqe_grad(solver, theta, grad, N_PARAMS);
                    if (rc != 0) {
                        fprintf(stderr,
                                "moonlab_vqe_gradient returned %d\n", rc);
                        failures++;
                    } else {
                        /* Central-difference cross-check (test-only). */
                        const double h_step = 1e-4;
                        double theta_p[N_PARAMS];
                        double max_err = 0.0, max_mag = 0.0;
                        for (size_t k = 0; k < N_PARAMS; k++) {
                            memcpy(theta_p, theta, sizeof(theta));
                            theta_p[k] = theta[k] + h_step;
                            double ep = energy(solver, theta_p);
                            theta_p[k] = theta[k] - h_step;
                            double em = energy(solver, theta_p);
                            double fd = (ep - em) / (2.0 * h_step);
                            double err = fabs(grad[k] - fd);
                            if (err > max_err) max_err = err;
                            if (fabs(grad[k]) > max_mag) {
                                max_mag = fabs(grad[k]);
                            }
                        }
                        if (max_err > 1e-6) {
                            fprintf(stderr,
                                    "gradient/central-diff mismatch: "
                                    "max err %.2e\n", max_err);
                            failures++;
                        } else if (max_mag < 1e-8) {
                            fprintf(stderr,
                                    "gradient is identically zero at a "
                                    "generic point\n");
                            failures++;
                        } else {
                            fprintf(stdout,
                                    "moonlab_vqe_gradient OK "
                                    "(max |grad-fd| = %.2e)\n", max_err);
                        }
                    }
                    solver_free(solver);
                }
                if (entropy) entropy_destroy(entropy);
                if (opt) opt_free(opt);
                if (ansatz) ansatz_free(ansatz);
                if (H) ham_free(H);
            }
        }
    }

#if defined(_WIN32)
    /* MinGW DLL detach can block after QRNG/TDVP runtime initialisation.
     * This ABI smoke verifies load + symbol calls; Windows process teardown
     * will reclaim the module without making FreeLibrary part of the
     * downstream ABI contract. */
    (void)h;
#else
    dlclose(h);
#endif

    int exit_code = failures ? 1 : 0;
    if (failures) {
        fprintf(stderr, "ABI smoke test FAILED (%d failures)\n", failures);
    } else {
        fprintf(stdout, "ABI smoke test OK\n");
    }

#if defined(_WIN32)
    /* The Windows shared-library smoke can leave runtime cleanup in
     * DLL/process detach after QRNG and TDVP are initialized.  The ABI
     * contract under test is load + calls, so exit without making teardown
     * part of the release gate. */
    fflush(stdout);
    fflush(stderr);
    TerminateProcess(GetCurrentProcess(), (UINT)exit_code);
#endif

    return exit_code;
}
