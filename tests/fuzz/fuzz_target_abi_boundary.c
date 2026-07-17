/**
 * @file    fuzz_target_abi_boundary.c
 * @brief   Surface: opaque-handle stable-ABI boundary argument fuzzing.
 *
 * The stable ABI is what language bindings (Rust, Python, Eshkol) and
 * sibling libraries (QGTL, libirrep, SbNN) call across an FFI boundary
 * where the caller cannot see the underlying structs.  A binding passing
 * a mismatched size or an out-of-range index must receive the documented
 * error code, not a crash.  This target derives those arguments from
 * fuzz bytes and asserts the contract for three families:
 *
 *   - `moonlab_qrng_bytes`      -- length + NULL-buffer handling; the
 *                                  documented returns are 0 / -1 / -2 /
 *                                  -3 / -4 / -5.
 *   - `moonlab_ca_mps_*`        -- structural fuzzing: create with a
 *                                  fuzzed (num_qubits, max_bond_dim), then
 *                                  apply gates with fuzzed qubit indices
 *                                  (including out-of-range), evaluate
 *                                  Pauli expectations and Z-marginals with
 *                                  fuzzed operator strings, clone, free.
 *   - `moonlab_vqe_gradient`    -- num_parameters vs. the solver's real
 *                                  ansatz count; documented returns are
 *                                  0 / -1 / -2 / -3.
 *
 * The asserts below are the actual test oracle: a return value outside
 * the documented set aborts the process and surfaces as a finding.
 */

#include "fuzz_common.h"

#include "applications/moonlab_export.h"
#include "algorithms/vqe.h"
#include "utils/quantum_entropy.h"

#include <assert.h>
#include <complex.h>
#include <stdlib.h>

/* ---- moonlab_qrng_bytes ------------------------------------------------ */

static void fuzz_qrng(const uint8_t **p, const uint8_t *end)
{
    uint32_t n = fuzz_u32(p, end) % 4097u; /* 0..4096 bytes */

    /* NULL buffer with size > 0 must be rejected with -1, never written. */
    int rc_null = moonlab_qrng_bytes(NULL, n);
    if (n > 0) {
        assert(rc_null == -1);
    }

    if (n == 0) {
        /* size 0 is a documented no-op success. */
        assert(moonlab_qrng_bytes(NULL, 0) == 0);
        return;
    }

    uint8_t *buf = (uint8_t *)malloc(n);
    if (!buf) return;
    int rc = moonlab_qrng_bytes(buf, n);
    /* Documented return set: 0 success, -1..-5 failure modes. */
    assert(rc == 0 || (rc <= -1 && rc >= -5));
    free(buf);
}

/* ---- moonlab_ca_mps_* -------------------------------------------------- */

static void fuzz_ca_mps(const uint8_t **p, const uint8_t *end)
{
    /* Keep the state small so throughput stays high; the point is the
     * index/size validation, not large-system simulation. */
    uint32_t nq  = 1u + (fuzz_u8(p, end) % 12u);  /* 1..12 qubits */
    uint32_t chi = 1u + (fuzz_u8(p, end) % 16u);  /* bond cap 1..16 */

    moonlab_ca_mps_t *s = moonlab_ca_mps_create(nq, chi);
    if (!s) return;

    /* num_qubits accessor must agree with what we asked for. */
    assert(moonlab_ca_mps_num_qubits(s) == nq);

    /* Apply a bounded sequence of gates with fuzzed -- possibly
     * out-of-range -- qubit indices.  Every call must return a status
     * (0 or negative); none may crash on a bad index. */
    int ops = 0;
    while (*p < end && ops < 256) {
        uint8_t op = fuzz_u8(p, end);
        uint32_t a = fuzz_u8(p, end);           /* deliberately unbounded */
        uint32_t b = fuzz_u8(p, end);
        double   t = (double)(int8_t)fuzz_u8(p, end) * 0.1;
        switch (op % 12u) {
        case 0:  (void)moonlab_ca_mps_h(s, a); break;
        case 1:  (void)moonlab_ca_mps_x(s, a); break;
        case 2:  (void)moonlab_ca_mps_z(s, a); break;
        case 3:  (void)moonlab_ca_mps_s(s, a); break;
        case 4:  (void)moonlab_ca_mps_cnot(s, a, b); break;
        case 5:  (void)moonlab_ca_mps_cz(s, a, b); break;
        case 6:  (void)moonlab_ca_mps_swap(s, a, b); break;
        case 7:  (void)moonlab_ca_mps_rx(s, a, t); break;
        case 8:  (void)moonlab_ca_mps_rz(s, a, t); break;
        case 9:  (void)moonlab_ca_mps_crz(s, a, b, t); break;
        case 10: (void)moonlab_ca_mps_toffoli(s, a, b, fuzz_u8(p, end)); break;
        default: (void)moonlab_ca_mps_phase(s, a, t); break;
        }
        ops++;
    }

    /* Pauli-string observables: length must be num_qubits.  Build a
     * string from fuzz bytes (values outside {I,X,Y,Z} exercise the
     * operator validator). */
    uint8_t pauli[12];
    for (uint32_t i = 0; i < nq; i++) pauli[i] = fuzz_u8(p, end);
    double _Complex ev = 0;
    (void)moonlab_ca_mps_expect_pauli(s, pauli, &ev);

    /* Pauli sum with a small fuzzed term count. */
    uint32_t nterms = 1u + (fuzz_u8(p, end) % 3u);
    uint8_t paulis[3 * 12];
    double _Complex coeffs[3];
    for (uint32_t k = 0; k < nterms; k++) {
        for (uint32_t i = 0; i < nq; i++)
            paulis[k * nq + i] = fuzz_u8(p, end);
        coeffs[k] = (double)(int8_t)fuzz_u8(p, end);
    }
    (void)moonlab_ca_mps_expect_pauli_sum(s, paulis, coeffs, nterms, &ev);

    /* Z-marginal with a possibly out-of-range qubit index. */
    double prob = 0.0;
    (void)moonlab_ca_mps_prob_z(s, fuzz_u8(p, end), &prob);

    /* Clone then free both -- exercises the deep-copy path. */
    moonlab_ca_mps_t *clone = moonlab_ca_mps_clone(s);
    if (clone) moonlab_ca_mps_free(clone);
    moonlab_ca_mps_free(s);
}

/* ---- moonlab_vqe_gradient --------------------------------------------- */

/* One process-wide solver, built lazily and kept reachable for the whole
 * campaign (so LeakSanitizer treats it as live, not leaked).  We hold
 * every component pointer so the graph stays rooted. */
static vqe_solver_t        *g_solver;
static vqe_ansatz_t        *g_ansatz;
static pauli_hamiltonian_t *g_ham;
static vqe_optimizer_t     *g_opt;
static quantum_entropy_ctx_t *g_ent;
static int                  g_solver_tried;

static void ensure_solver(void)
{
    if (g_solver_tried) return;
    g_solver_tried = 1;

    g_ham = vqe_create_h2_hamiltonian(0.74);
    g_ansatz = vqe_create_hardware_efficient_ansatz(2, 1);
    g_opt = vqe_optimizer_create(VQE_OPTIMIZER_GRADIENT_DESCENT);
    g_ent = quantum_entropy_ctx_create_hw();
    if (g_ham && g_ansatz && g_opt && g_ent) {
        g_solver = vqe_solver_create(g_ham, g_ansatz, g_opt, g_ent);
    }
}

static void fuzz_vqe(const uint8_t **p, const uint8_t *end)
{
    /* NULL guard is always reachable, solver or not. */
    double one = 0.0;
    assert(moonlab_vqe_gradient(NULL, &one, &one, 1) == -1);

    ensure_solver();
    if (!g_solver) return;

    /* Choose num_parameters: sometimes the true ansatz count (success /
     * -3 path), sometimes a fuzzed value (the -2 mismatch path). */
    size_t real_n = g_ansatz->num_parameters;
    size_t n = (fuzz_u8(p, end) & 1u)
                   ? real_n
                   : (size_t)(fuzz_u32(p, end) % 64u);

    double *params = (double *)calloc(n ? n : 1, sizeof(double));
    double *grad   = (double *)calloc(n ? n : 1, sizeof(double));
    if (params && grad) {
        for (size_t i = 0; i < n; i++)
            params[i] = (double)(int8_t)fuzz_u8(p, end) * 0.05;
        int rc = moonlab_vqe_gradient(g_solver, params, grad, n);
        /* Documented set: 0 ok, -1 NULL, -2 size mismatch, -3 internal. */
        assert(rc == 0 || rc == -1 || rc == -2 || rc == -3);
        if (n != real_n) assert(rc == -2);
    }
    free(params);
    free(grad);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    const uint8_t *p   = data;
    const uint8_t *end = data + size;

    switch (fuzz_u8(&p, end) % 3u) {
    case 0: fuzz_qrng(&p, end);   break;
    case 1: fuzz_ca_mps(&p, end); break;
    default: fuzz_vqe(&p, end);   break;
    }
    return 0;
}
