/**
 * @file    moonlab_qgtl_backend.c
 * @brief   Implementation of the QGTL-shaped circuit-ingestion surface.
 *
 * The circuit-record is just a growable array of `qgtl_gate_record_t`
 * tuples.  `execute` runs through them once, dispatches each to the
 * matching `gate_*` call against a freshly-initialised
 * `quantum_state_t`, then optionally extracts the probability
 * distribution + samples measurement outcomes.
 *
 * No optimisation, transpilation, or gate fusion -- QGTL handles that
 * upstream.  This TU is intentionally the thinnest possible bridge.
 */

#include "moonlab_qgtl_backend.h"

#include "../quantum/state.h"
#include "../quantum/gates.h"
#include "../quantum/measurement.h"

#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    moonlab_qgtl_gate_t type;
    int     target;
    int     control;
    double  theta;     /**< Captured at `add_gate` time so params[] can vanish. */
    int     has_param; /**< 1 if `theta` is meaningful. */
} qgtl_gate_record_t;

struct moonlab_qgtl_circuit {
    int                 num_qubits;
    int                 num_gates;
    int                 capacity;
    qgtl_gate_record_t *gates;
};

moonlab_qgtl_circuit_t *
moonlab_qgtl_circuit_create(int num_qubits)
{
    if (num_qubits < 1 || num_qubits > 32) return NULL;
    moonlab_qgtl_circuit_t *c = (moonlab_qgtl_circuit_t *)
        calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->num_qubits = num_qubits;
    return c;
}

void moonlab_qgtl_circuit_free(moonlab_qgtl_circuit_t *c)
{
    if (!c) return;
    free(c->gates);
    free(c);
}

int moonlab_qgtl_circuit_num_qubits(const moonlab_qgtl_circuit_t *c)
{
    return c ? c->num_qubits : MOONLAB_QGTL_BAD_ARG;
}

int moonlab_qgtl_circuit_num_gates(const moonlab_qgtl_circuit_t *c)
{
    return c ? c->num_gates : MOONLAB_QGTL_BAD_ARG;
}

static int gate_takes_param(moonlab_qgtl_gate_t t)
{
    return t == MOONLAB_QGTL_GATE_RX ||
           t == MOONLAB_QGTL_GATE_RY ||
           t == MOONLAB_QGTL_GATE_RZ;
}

static int gate_is_two_qubit(moonlab_qgtl_gate_t t)
{
    return t == MOONLAB_QGTL_GATE_CNOT ||
           t == MOONLAB_QGTL_GATE_CY   ||
           t == MOONLAB_QGTL_GATE_CZ   ||
           t == MOONLAB_QGTL_GATE_SWAP;
}

int moonlab_qgtl_add_gate(moonlab_qgtl_circuit_t *c,
                          moonlab_qgtl_gate_t    type,
                          int                    target,
                          int                    control,
                          const double          *params)
{
    if (!c) return MOONLAB_QGTL_BAD_ARG;
    if (target < 0 || target >= c->num_qubits) return MOONLAB_QGTL_BAD_ARG;
    if (gate_is_two_qubit(type)) {
        if (control < 0 || control >= c->num_qubits) return MOONLAB_QGTL_BAD_ARG;
        if (control == target) return MOONLAB_QGTL_BAD_ARG;
    }
    if (gate_takes_param(type) && params == NULL) return MOONLAB_QGTL_BAD_ARG;

    if (c->num_gates >= c->capacity) {
        const int new_cap = c->capacity ? c->capacity * 2 : 16;
        qgtl_gate_record_t *n = (qgtl_gate_record_t *)
            realloc(c->gates, (size_t)new_cap * sizeof(*n));
        if (!n) return MOONLAB_QGTL_OOM;
        c->gates    = n;
        c->capacity = new_cap;
    }

    qgtl_gate_record_t *g = &c->gates[c->num_gates++];
    g->type    = type;
    g->target  = target;
    g->control = control;
    g->theta     = gate_takes_param(type) ? params[0] : 0.0;
    g->has_param = gate_takes_param(type) ? 1 : 0;
    return MOONLAB_QGTL_OK;
}

static int apply_one_gate(quantum_state_t *s, const qgtl_gate_record_t *g)
{
    qs_error_t rc;
    switch (g->type) {
    case MOONLAB_QGTL_GATE_I:
        rc = 0; /* identity is a no-op */
        break;
    case MOONLAB_QGTL_GATE_X:    rc = gate_pauli_x(s, g->target); break;
    case MOONLAB_QGTL_GATE_Y:    rc = gate_pauli_y(s, g->target); break;
    case MOONLAB_QGTL_GATE_Z:    rc = gate_pauli_z(s, g->target); break;
    case MOONLAB_QGTL_GATE_H:    rc = gate_hadamard(s, g->target); break;
    case MOONLAB_QGTL_GATE_S:    rc = gate_s(s, g->target); break;
    case MOONLAB_QGTL_GATE_T:    rc = gate_t(s, g->target); break;
    case MOONLAB_QGTL_GATE_RX:   rc = gate_rx(s, g->target, g->theta); break;
    case MOONLAB_QGTL_GATE_RY:   rc = gate_ry(s, g->target, g->theta); break;
    case MOONLAB_QGTL_GATE_RZ:   rc = gate_rz(s, g->target, g->theta); break;
    case MOONLAB_QGTL_GATE_CNOT: rc = gate_cnot(s, g->control, g->target); break;
    case MOONLAB_QGTL_GATE_CZ:   rc = gate_cz(s, g->control, g->target); break;
    case MOONLAB_QGTL_GATE_SWAP: rc = gate_swap(s, g->control, g->target); break;
    case MOONLAB_QGTL_GATE_CY:
        /* CY = (I tensor S^dagger) * CNOT * (I tensor S).  Build by composition
         * since moonlab doesn't ship a direct gate_cy. */
        rc = gate_s(s, g->target);
        if (rc == 0) rc = gate_cnot(s, g->control, g->target);
        if (rc == 0) rc = gate_s_dagger(s, g->target);
        break;
    default:
        return MOONLAB_QGTL_UNSUPPORTED;
    }
    return (rc == 0) ? MOONLAB_QGTL_OK : MOONLAB_QGTL_INTERNAL;
}

static uint64_t default_seed(void)
{
    /* Deterministic-from-clock seed; QGTL passes a real seed when it
     * needs reproducibility. */
    return (uint64_t)time(NULL) ^ 0x9e3779b97f4a7c15ULL;
}

/* xorshift64 -- one-line PRNG good enough for shot sampling. */
static inline uint64_t xorshift64_step(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

int moonlab_qgtl_execute(moonlab_qgtl_circuit_t           *c,
                         const moonlab_qgtl_exec_options_t *opts,
                         moonlab_qgtl_results_t           *out)
{
    if (!c || !opts || !out) return MOONLAB_QGTL_BAD_ARG;

    /* Initialise an N-qubit |0...0> state. */
    quantum_state_t state = {0};
    if (quantum_state_init(&state, (size_t)c->num_qubits) != 0) {
        return MOONLAB_QGTL_OOM;
    }

    /* Walk the gate list. */
    int rc = MOONLAB_QGTL_OK;
    for (int i = 0; i < c->num_gates; i++) {
        rc = apply_one_gate(&state, &c->gates[i]);
        if (rc != MOONLAB_QGTL_OK) {
            quantum_state_free(&state);
            return rc;
        }
    }

    /* Initialise output record. */
    memset(out, 0, sizeof(*out));
    out->num_qubits = c->num_qubits;
    out->num_shots  = opts->num_shots;

    const size_t dim = (size_t)1 << c->num_qubits;

    /* Optional: dump the probability vector. */
    if (opts->return_probabilities) {
        out->probabilities = (double *)malloc(dim * sizeof(double));
        if (!out->probabilities) {
            quantum_state_free(&state);
            return MOONLAB_QGTL_OOM;
        }
        for (size_t b = 0; b < dim; b++) {
            const double re = creal(state.amplitudes[b]);
            const double im = cimag(state.amplitudes[b]);
            out->probabilities[b] = re * re + im * im;
        }
    }

    /* Optional: shot sampling.  We need to preserve the post-circuit
     * state across shots, so clone-and-measure each shot.  At
     * num_shots > ~1000 the per-shot clone overhead dominates --
     * QGTL is expected to either keep shots small (it's a
     * cross-validation step, not a production run) or pre-flag
     * stim-style sampling, which can come in a later release. */
    if (opts->num_shots > 0) {
        out->outcomes = (uint64_t *)malloc((size_t)opts->num_shots * sizeof(uint64_t));
        if (!out->outcomes) {
            free(out->probabilities);
            out->probabilities = NULL;
            quantum_state_free(&state);
            return MOONLAB_QGTL_OOM;
        }
        uint64_t seed = opts->rng_seed != 0 ? opts->rng_seed : default_seed();
        for (int s = 0; s < opts->num_shots; s++) {
            quantum_state_t shot = {0};
            if (quantum_state_clone(&shot, &state) != 0) {
                free(out->outcomes);
                free(out->probabilities);
                out->outcomes = NULL;
                out->probabilities = NULL;
                quantum_state_free(&state);
                return MOONLAB_QGTL_OOM;
            }
            /* Draw a uniform random in [0, 1). */
            const uint64_t r = xorshift64_step(&seed);
            const double u = (double)(r >> 11) / (double)(1ULL << 53);
            out->outcomes[s] = measurement_all_qubits(&shot, u);
            quantum_state_free(&shot);
        }
    }

    quantum_state_free(&state);
    return MOONLAB_QGTL_OK;
}

void moonlab_qgtl_results_free(moonlab_qgtl_results_t *r)
{
    if (!r) return;
    free(r->outcomes);
    free(r->probabilities);
    r->outcomes      = NULL;
    r->probabilities = NULL;
}

/* ------------------------------------------------------------------
 * Serialization (since v0.8.3).
 * ------------------------------------------------------------------ */

static const char *gate_name(moonlab_qgtl_gate_t t)
{
    switch (t) {
    case MOONLAB_QGTL_GATE_I:    return "I";
    case MOONLAB_QGTL_GATE_X:    return "X";
    case MOONLAB_QGTL_GATE_Y:    return "Y";
    case MOONLAB_QGTL_GATE_Z:    return "Z";
    case MOONLAB_QGTL_GATE_H:    return "H";
    case MOONLAB_QGTL_GATE_S:    return "S";
    case MOONLAB_QGTL_GATE_T:    return "T";
    case MOONLAB_QGTL_GATE_RX:   return "RX";
    case MOONLAB_QGTL_GATE_RY:   return "RY";
    case MOONLAB_QGTL_GATE_RZ:   return "RZ";
    case MOONLAB_QGTL_GATE_CNOT: return "CNOT";
    case MOONLAB_QGTL_GATE_CY:   return "CY";
    case MOONLAB_QGTL_GATE_CZ:   return "CZ";
    case MOONLAB_QGTL_GATE_SWAP: return "SWAP";
    }
    return NULL;
}

static int gate_from_name(const char *s, moonlab_qgtl_gate_t *out)
{
    if      (strcmp(s, "I")    == 0) *out = MOONLAB_QGTL_GATE_I;
    else if (strcmp(s, "X")    == 0) *out = MOONLAB_QGTL_GATE_X;
    else if (strcmp(s, "Y")    == 0) *out = MOONLAB_QGTL_GATE_Y;
    else if (strcmp(s, "Z")    == 0) *out = MOONLAB_QGTL_GATE_Z;
    else if (strcmp(s, "H")    == 0) *out = MOONLAB_QGTL_GATE_H;
    else if (strcmp(s, "S")    == 0) *out = MOONLAB_QGTL_GATE_S;
    else if (strcmp(s, "T")    == 0) *out = MOONLAB_QGTL_GATE_T;
    else if (strcmp(s, "RX")   == 0) *out = MOONLAB_QGTL_GATE_RX;
    else if (strcmp(s, "RY")   == 0) *out = MOONLAB_QGTL_GATE_RY;
    else if (strcmp(s, "RZ")   == 0) *out = MOONLAB_QGTL_GATE_RZ;
    else if (strcmp(s, "CNOT") == 0) *out = MOONLAB_QGTL_GATE_CNOT;
    else if (strcmp(s, "CY")   == 0) *out = MOONLAB_QGTL_GATE_CY;
    else if (strcmp(s, "CZ")   == 0) *out = MOONLAB_QGTL_GATE_CZ;
    else if (strcmp(s, "SWAP") == 0) *out = MOONLAB_QGTL_GATE_SWAP;
    else return MOONLAB_QGTL_BAD_ARG;
    return MOONLAB_QGTL_OK;
}

static int emit_chunk(char  *buf,
                      size_t buf_size,
                      size_t *pos,
                      const char *chunk,
                      int    n)
{
    if (buf_size > 0) {
        if (*pos + (size_t)n + 1 > buf_size) {
            return MOONLAB_QGTL_OOM;
        }
        memcpy(buf + *pos, chunk, (size_t)n);
    }
    *pos += (size_t)n;
    return MOONLAB_QGTL_OK;
}

int moonlab_qgtl_circuit_serialize(const moonlab_qgtl_circuit_t *c,
                                   char  *buf,
                                   size_t buf_size,
                                   size_t *out_written)
{
    if (!c) return MOONLAB_QGTL_BAD_ARG;
    if (buf_size > 0 && !buf) return MOONLAB_QGTL_BAD_ARG;

    size_t pos = 0;
    char tmp[96];
    int n;

    n = snprintf(tmp, sizeof(tmp), "# moonlab-circuit v1\n");
    if (n < 0) return MOONLAB_QGTL_INTERNAL;
    if (emit_chunk(buf, buf_size, &pos, tmp, n) != MOONLAB_QGTL_OK) {
        if (out_written) *out_written = pos + (size_t)n + 1;
        return MOONLAB_QGTL_OOM;
    }

    n = snprintf(tmp, sizeof(tmp), "NUM_QUBITS %d\n", c->num_qubits);
    if (n < 0) return MOONLAB_QGTL_INTERNAL;
    if (emit_chunk(buf, buf_size, &pos, tmp, n) != MOONLAB_QGTL_OK) {
        if (out_written) *out_written = pos + (size_t)n + 1;
        return MOONLAB_QGTL_OOM;
    }

    for (int i = 0; i < c->num_gates; i++) {
        const qgtl_gate_record_t *g = &c->gates[i];
        const char *name = gate_name(g->type);
        if (!name) return MOONLAB_QGTL_INTERNAL;

        if (gate_is_two_qubit(g->type)) {
            n = snprintf(tmp, sizeof(tmp), "%s %d %d\n",
                         name, g->target, g->control);
        } else if (g->has_param) {
            n = snprintf(tmp, sizeof(tmp), "%s %d %.17g\n",
                         name, g->target, g->theta);
        } else {
            n = snprintf(tmp, sizeof(tmp), "%s %d\n", name, g->target);
        }
        if (n < 0) return MOONLAB_QGTL_INTERNAL;
        if (emit_chunk(buf, buf_size, &pos, tmp, n) != MOONLAB_QGTL_OK) {
            if (out_written) *out_written = pos + (size_t)n + 1;
            return MOONLAB_QGTL_OOM;
        }
    }

    if (buf_size > 0) {
        if (pos + 1 > buf_size) {
            if (out_written) *out_written = pos + 1;
            return MOONLAB_QGTL_OOM;
        }
        buf[pos] = '\0';
    }
    if (out_written) *out_written = pos;
    return MOONLAB_QGTL_OK;
}

/* Read the next non-blank, non-comment line into `line` (terminated).
 * Returns 1 on success, 0 on EOF.  Advances `*p` past the consumed
 * bytes (and the trailing newline). */
static int next_line(const char **p, const char *end, char *line, size_t line_cap)
{
    while (*p < end) {
        const char *q = *p;
        while (q < end && *q != '\n' && *q != '\0') q++;
        const size_t len = (size_t)(q - *p);
        const char *start = *p;
        /* Advance past the terminator we stopped on. q < end means we halted
         * on a '\n' OR an interior '\0'; both are line boundaries and must be
         * consumed, otherwise a payload with an embedded NUL parks *p on that
         * NUL forever (blank-line continue never advances) -- an untrusted-input
         * hang first caught by circuit_deserialize_fuzz. */
        *p = (q < end) ? q + 1 : q;

        /* Trim leading whitespace. */
        const char *ls = start;
        while (ls < start + len && isspace((unsigned char)*ls)) ls++;
        if (ls >= start + len) continue;             /* blank line */
        if (*ls == '#') continue;                    /* comment */

        const size_t copy = (size_t)((start + len) - ls);
        if (copy + 1 > line_cap) return 0; /* overlong line -> bail */
        memcpy(line, ls, copy);
        line[copy] = '\0';
        return 1;
    }
    return 0;
}

moonlab_qgtl_circuit_t *
moonlab_qgtl_circuit_deserialize(const char *buf,
                                 size_t      buf_size,
                                 int        *out_status)
{
    int status = MOONLAB_QGTL_BAD_ARG;
    moonlab_qgtl_circuit_t *c = NULL;

    if (!buf) goto done;
    if (buf_size == 0) buf_size = strlen(buf);
    const char *p   = buf;
    const char *end = buf + buf_size;

    char line[256];
    int got_header = 0;
    int num_qubits = -1;

    while (next_line(&p, end, line, sizeof(line))) {
        if (!got_header) {
            int n = -1;
            if (sscanf(line, "NUM_QUBITS %d", &n) != 1 || n < 1 || n > 32) {
                status = MOONLAB_QGTL_BAD_ARG;
                goto done;
            }
            num_qubits = n;
            c = moonlab_qgtl_circuit_create(num_qubits);
            if (!c) { status = MOONLAB_QGTL_OOM; goto done; }
            got_header = 1;
            continue;
        }

        char name[8] = {0};
        int target = -1;
        int control = -1;
        double theta = 0.0;

        /* Pull the gate name first. */
        int consumed = 0;
        if (sscanf(line, "%7s %n", name, &consumed) != 1) {
            status = MOONLAB_QGTL_BAD_ARG; goto done;
        }
        moonlab_qgtl_gate_t type;
        if (gate_from_name(name, &type) != MOONLAB_QGTL_OK) {
            status = MOONLAB_QGTL_BAD_ARG; goto done;
        }
        const char *rest = line + consumed;

        if (gate_is_two_qubit(type)) {
            if (sscanf(rest, "%d %d", &target, &control) != 2) {
                status = MOONLAB_QGTL_BAD_ARG; goto done;
            }
            int rc = moonlab_qgtl_add_gate(c, type, target, control, NULL);
            if (rc != MOONLAB_QGTL_OK) { status = rc; goto done; }
        } else if (gate_takes_param(type)) {
            if (sscanf(rest, "%d %lf", &target, &theta) != 2) {
                status = MOONLAB_QGTL_BAD_ARG; goto done;
            }
            int rc = moonlab_qgtl_add_gate(c, type, target, 0, &theta);
            if (rc != MOONLAB_QGTL_OK) { status = rc; goto done; }
        } else {
            if (sscanf(rest, "%d", &target) != 1) {
                status = MOONLAB_QGTL_BAD_ARG; goto done;
            }
            int rc = moonlab_qgtl_add_gate(c, type, target, 0, NULL);
            if (rc != MOONLAB_QGTL_OK) { status = rc; goto done; }
        }
    }

    if (!got_header) { status = MOONLAB_QGTL_BAD_ARG; goto done; }

    status = MOONLAB_QGTL_OK;
    if (out_status) *out_status = status;
    return c;

done:
    if (c) moonlab_qgtl_circuit_free(c);
    if (out_status) *out_status = status;
    return NULL;
}

int moonlab_qgtl_circuit_save(const moonlab_qgtl_circuit_t *c,
                              const char *path)
{
    if (!c || !path) return MOONLAB_QGTL_BAD_ARG;

    size_t needed = 0;
    int rc = moonlab_qgtl_circuit_serialize(c, NULL, 0, &needed);
    if (rc != MOONLAB_QGTL_OK) return rc;

    char *buf = (char *)malloc(needed + 1);
    if (!buf) return MOONLAB_QGTL_OOM;

    rc = moonlab_qgtl_circuit_serialize(c, buf, needed + 1, NULL);
    if (rc != MOONLAB_QGTL_OK) { free(buf); return rc; }

    FILE *fp = fopen(path, "w");
    if (!fp) { free(buf); return MOONLAB_QGTL_INTERNAL; }

    size_t wrote = fwrite(buf, 1, needed, fp);
    int io_ok = (wrote == needed && fclose(fp) == 0);
    free(buf);
    return io_ok ? MOONLAB_QGTL_OK : MOONLAB_QGTL_INTERNAL;
}

moonlab_qgtl_circuit_t *
moonlab_qgtl_circuit_load(const char *path, int *out_status)
{
    if (!path) {
        if (out_status) *out_status = MOONLAB_QGTL_BAD_ARG;
        return NULL;
    }

    FILE *fp = fopen(path, "r");
    if (!fp) {
        if (out_status) *out_status = MOONLAB_QGTL_INTERNAL;
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        if (out_status) *out_status = MOONLAB_QGTL_INTERNAL;
        return NULL;
    }
    long sz = ftell(fp);
    if (sz < 0) {
        fclose(fp);
        if (out_status) *out_status = MOONLAB_QGTL_INTERNAL;
        return NULL;
    }
    rewind(fp);

    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(fp);
        if (out_status) *out_status = MOONLAB_QGTL_OOM;
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)sz, fp);
    buf[got] = '\0';
    fclose(fp);

    moonlab_qgtl_circuit_t *c =
        moonlab_qgtl_circuit_deserialize(buf, got, out_status);
    free(buf);
    return c;
}
