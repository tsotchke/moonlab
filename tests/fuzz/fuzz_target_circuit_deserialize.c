/**
 * @file    fuzz_target_circuit_deserialize.c
 * @brief   Surface: moonlab-circuit v1 text deserialization.
 *
 * Drives `moonlab_qgtl_circuit_deserialize` -- the public parser the
 * control-plane backend and any QGTL/libirrep/SbNN consumer runs over
 * an untrusted circuit payload.  It parses a NUM_QUBITS header plus a
 * line-oriented gate list (I X Y Z H S T RX RY RZ CNOT CY CZ SWAP) with
 * `sscanf`, and allocates a growable gate array.
 *
 * The parser is exercised directly (no networking) which makes this the
 * fast, high-throughput sibling of the control-plane target: both reach
 * `moonlab_qgtl_circuit_deserialize`, but this one skips the socket and
 * the simulator execution entirely.
 *
 * Contract under test: for any byte string the function returns either a
 * valid owned handle (freed here) or NULL with a negative status, and
 * never reads out of bounds, over-runs a fixed line buffer, or leaks.
 */

#include "fuzz_common.h"

#include "applications/moonlab_qgtl_backend.h"

#include <assert.h>
#include <ctype.h>
#include <stdlib.h>

/* Inputs with a bounded NUL or a non-comment logical line larger than the
 * parser's 255-byte grammar limit are required to fail closed. */
static int requires_parse_rejection(const uint8_t *data, size_t size)
{
    if (size > 0 && memchr(data, '\0', size) != NULL) return 1;

    size_t line_start = 0;
    while (line_start < size) {
        size_t line_end = line_start;
        while (line_end < size && data[line_end] != '\n') line_end++;

        size_t content = line_start;
        while (content < line_end && isspace((unsigned char)data[content])) {
            content++;
        }
        if (content < line_end && data[content] != '#' &&
            line_end - content > 255) {
            return 1;
        }

        line_start = (line_end < size) ? line_end + 1 : line_end;
    }
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    /* Mirror every real caller of this function: the control-plane
     * backend and moonlab_qgtl_circuit_load both hand it a
     * NUL-terminated buffer.  The documented contract is that
     * buf_size == 0 means "strlen the buffer", so a caller that passes
     * size 0 MUST supply a terminator -- feeding a non-terminated buffer
     * with size 0 would run strlen off the end (a caller-contract
     * violation, noted as a hardening item in FINDINGS.md, not a library
     * defect).  We copy into a NUL-terminated buffer so both the
     * length-bounded (size > 0) and the strlen (size == 0) code paths are
     * fuzzed exactly as production reaches them. */
    char *buf = (char *)malloc(size + 1);
    if (!buf) return 0;
    if (size) memcpy(buf, data, size);
    buf[size] = '\0';

    int status = 12345; /* sentinel: must be overwritten */
    moonlab_qgtl_circuit_t *c =
        moonlab_qgtl_circuit_deserialize(buf, size, &status);

    if (requires_parse_rejection(data, size)) {
        assert(c == NULL);
        assert(status == MOONLAB_QGTL_BAD_ARG);
    }

    if (c) {
        /* Valid parse: exercise the introspection accessors too, then
         * release.  A non-NULL handle must carry a sane qubit count. */
        assert(status == MOONLAB_QGTL_OK);
        (void)moonlab_qgtl_circuit_num_qubits(c);
        (void)moonlab_qgtl_circuit_num_gates(c);
        moonlab_qgtl_circuit_free(c);
    } else {
        assert(status < 0);
    }

    free(buf);
    return 0;
}
