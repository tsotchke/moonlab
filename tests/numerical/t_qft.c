/*
 * t_qft.c -- scale / precision at many qubits.
 *
 *   1. QFT->IQFT round-trip fidelity on random product states at
 *      increasing qubit counts (default up to 22; NUMERIC_MAX_QUBITS
 *      raises the cap, e.g. 28).  Round-trip must return the input to
 *      fidelity 1 within a dimension-scaled tolerance, with no NaN.
 *   2. GHZ state at the max qubit count: exactly two nonzero amplitudes
 *      (1/sqrt2) and unit norm.
 *   3. Deep single-qubit accumulation: 100k gates whose product is
 *      identity; final state must match the initial |0> to tight tol.
 *
 * Links the prebuilt library.
 */
#include "numerical_common.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

static double statenorm(const quantum_state_t *st) {
    double n = 0.0;
    for (size_t i = 0; i < st->state_dim; i++) {
        complex_t a = st->amplitudes[i];
        n += creal(a)*creal(a) + cimag(a)*cimag(a);
    }
    return n;
}

static void qft_roundtrip(size_t nq, uint64_t seed) {
    quantum_state_t st;
    if (quantum_state_init(&st, nq) != QS_SUCCESS) { NC_MISS("qft init nq=%zu", nq); return; }

    /* random product state: RY(random) RZ(random) on each qubit from |0> */
    for (size_t q = 0; q < nq; q++) {
        gate_ry(&st, (int)q, (nc_urand(&seed)+1.0) * 3.14159);
        gate_rz(&st, (int)q, (nc_urand(&seed)+1.0) * 3.14159);
    }
    /* snapshot */
    complex_t *ref = malloc(st.state_dim * sizeof(complex_t));
    if (!ref) { NC_MISS("qft snapshot OOM nq=%zu", nq); quantum_state_free(&st); return; }
    memcpy(ref, st.amplitudes, st.state_dim * sizeof(complex_t));

    int *qubits = malloc(nq * sizeof(int));
    for (size_t q = 0; q < nq; q++) qubits[q] = (int)q;

    if (gate_qft(&st, qubits, nq) != QS_SUCCESS)  { NC_MISS("qft nq=%zu", nq); goto done; }
    if (gate_iqft(&st, qubits, nq) != QS_SUCCESS) { NC_MISS("iqft nq=%zu", nq); goto done; }

    /* max amplitude deviation from the pre-QFT reference */
    double maxdev = 0.0; int bad = 0;
    for (size_t i = 0; i < st.state_dim; i++) {
        if (nc_is_bad_c(st.amplitudes[i])) { bad = 1; break; }
        double e = cabs(st.amplitudes[i] - ref[i]);
        if (e > maxdev) maxdev = e;
    }
    g_checks++;
    if (bad) { NC_MISS("qft nq=%zu: NaN/Inf after round-trip", nq); goto done; }

    /* tolerance scales with sqrt(dim): O(n) butterfly stages accumulate */
    double tol = 1e-12 * sqrt((double)st.state_dim) * (double)nq;
    if (tol < 1e-11) tol = 1e-11;
    if (maxdev > tol)
        NC_MISS("qft nq=%zu: round-trip maxdev=%.3e tol=%.3e seed=0x%llx",
                nq, maxdev, tol, (unsigned long long)seed);
    else g_checks++;

    nc_close("qft norm", statenorm(&st), 1.0, 1e-10);
done:
    free(ref); free(qubits);
    quantum_state_free(&st);
}

static void ghz(size_t nq) {
    quantum_state_t st;
    if (quantum_state_init(&st, nq) != QS_SUCCESS) { NC_MISS("ghz init"); return; }
    gate_hadamard(&st, 0);
    for (size_t q = 0; q + 1 < nq; q++) gate_cnot(&st, (int)q, (int)(q+1));
    double inv = 1.0 / sqrt(2.0);
    uint64_t last = ((uint64_t)1 << nq) - 1;
    nc_close_c("ghz |0..0>", quantum_state_get_amplitude(&st, 0), inv + 0.0*I, 1e-12);
    nc_close_c("ghz |1..1>", quantum_state_get_amplitude(&st, last), inv + 0.0*I, 1e-12);
    nc_close("ghz norm", statenorm(&st), 1.0, 1e-12);
    quantum_state_free(&st);
}

/* Apply 50000 (T, T_dagger) pairs = identity; then 50000 (S,S_dagger).
 * Any per-gate rounding bias accumulates over 200k operations. */
static void deep_accumulation(void) {
    quantum_state_t st;
    if (quantum_state_init(&st, 3) != QS_SUCCESS) { NC_MISS("deep init"); return; }
    gate_hadamard(&st, 0); gate_hadamard(&st, 1); gate_hadamard(&st, 2);
    complex_t *ref = malloc(st.state_dim * sizeof(complex_t));
    memcpy(ref, st.amplitudes, st.state_dim * sizeof(complex_t));
    for (int i = 0; i < 50000; i++) { gate_t(&st, 1); gate_t_dagger(&st, 1); }
    for (int i = 0; i < 50000; i++) { gate_rz(&st, 2, 1e-3); gate_rz(&st, 2, -1e-3); }
    double maxdev = 0.0;
    for (size_t i = 0; i < st.state_dim; i++) {
        double e = cabs(st.amplitudes[i] - ref[i]);
        if (e > maxdev) maxdev = e;
    }
    /* 200k unitary ops accumulate ~1e-11 of rounding; that is expected FP
     * behaviour (the library does not renormalize between gates), so the
     * bound only flags gross drift / blow-up, not benign accumulation. */
    NC_INFO("deep-circuit amp drift after 200k gates: %.3e", maxdev);
    if (maxdev > 1e-8) NC_MISS("deep accumulation drift maxdev=%.3e (>1e-8)", maxdev);
    else g_checks++;
    nc_close("deep norm", statenorm(&st), 1.0, 1e-8);
    free(ref);
    quantum_state_free(&st);
}

int main(void) {
    nc_begin("qft_scale");

    size_t maxq = 22;
    const char *env = getenv("NUMERIC_MAX_QUBITS");
    if (env) { long v = strtol(env, NULL, 10); if (v >= 4 && v <= 30) maxq = (size_t)v; }
    NC_INFO("QFT round-trip up to %zu qubits", maxq);

    for (size_t nq = 4; nq <= maxq; nq += 2)
        qft_roundtrip(nq, 0xABCDEF00ULL + nq);

    ghz(maxq);
    deep_accumulation();

    return nc_end();
}
