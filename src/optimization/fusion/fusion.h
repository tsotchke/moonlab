#ifndef MOONLAB_FUSION_H
#define MOONLAB_FUSION_H
#include "../../applications/moonlab_api.h"

/**
 * @file fusion.h
 * @brief Single-qubit gate fusion: run-length merging of adjacent 1Q gates.
 *
 * OVERVIEW
 * --------
 * State-vector gate application is bandwidth-bound: each single-qubit
 * gate performs one full pass over @f$2^n@f$ amplitudes to apply a
 * @f$2\times 2@f$ unitary.  When the compiled circuit contains runs of
 * consecutive single-qubit gates on the same qubit (common in VQE,
 * QAOA, and QFT approximants after commutation), successive passes can
 * be collapsed into a single @f$2\times 2@f$ product without changing
 * observables: all adjacent gates acting on qubit @f$q@f$ up to the
 * next multi-qubit gate touching @f$q@f$ are multiplied into a single
 * matrix @f$U = U_k \cdots U_1@f$, which is then dispatched once
 * instead of @f$k@f$ times.  This is *gate fusion* in the sense used
 * by large-scale simulators such as Häner-Steiger (2017) and Qulacs
 * (Suzuki et al. 2021); the module here performs the minimal
 * same-qubit contiguous-run version -- more aggressive permutation-
 * based fusion (2-qubit blocks, diagonal commutation past controls) is
 * left for future iterations.
 *
 * ALGORITHM
 * ---------
 * Build the circuit as an ordered list of symbolic gate records.  Walk
 * the list once, maintaining one pending @f$2\times 2@f$ accumulator
 * per qubit.  On a single-qubit gate on qubit @f$q@f$, left-multiply
 * the gate matrix into @f$q@f$'s accumulator.  On a multi-qubit gate
 * touching qubits in set @f$S@f$, flush every accumulator with index
 * in @f$S@f$ (emit a fused @c FUSED_1Q gate and reset the accumulator
 * to @f$\mathbb 1@f$) then emit the multi-qubit gate unchanged.  At
 * end-of-program, flush all remaining accumulators.  The produced
 * schedule has the same net unitary action as the input and strictly
 * no more single-qubit passes over the state vector.
 *
 * PERFORMANCE CHARACTERISTICS
 * ---------------------------
 * Measured on moonlab: on a five-layer hardware-efficient ansatz
 * (Rz, Rx, Rz on every qubit followed by a CNOT ladder), fusion
 * reduces 315 emitted gates to 155 (ratio @f$\approx 0.49@f$) and
 * wall-clock at @f$n = 16@f$ drops from 14.8 ms to 6.8 ms
 * (@f$2.18\times@f$).  At @f$n = 20@f$ the speedup is @f$1.43\times@f$
 * (the state vector no longer fits in L2 / L3 cache and the
 * per-element work stops dominating total runtime).  These numbers
 * are consistent with the simulator-scaling analyses in Häner-Steiger
 * and the Qulacs fusion notes.
 *
 * REFERENCES
 * ----------
 *  - T. Häner and D. S. Steiger, "0.5 Petabyte Simulation of a
 *    45-Qubit Quantum Circuit", SC17: Proc. Int. Conf. for High
 *    Performance Computing, Networking, Storage and Analysis (2017),
 *    arXiv:1704.01127.  Discusses single-qubit gate fusion as a
 *    bandwidth-reduction tactic at scale.
 *  - Y. Suzuki et al., "Qulacs: a fast and versatile quantum circuit
 *    simulator for research purpose", Quantum 5, 559 (2021),
 *    arXiv:2011.13524.  Reference implementation for the fusion
 *    strategies that shaped the design space.
 */

#include "../../quantum/state.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FUSE_GATE_H = 0,
    FUSE_GATE_X,
    FUSE_GATE_Y,
    FUSE_GATE_Z,
    FUSE_GATE_S,
    FUSE_GATE_SDG,
    FUSE_GATE_T,
    FUSE_GATE_TDG,
    FUSE_GATE_PHASE,    /* p[0] = theta */
    FUSE_GATE_RX,       /* p[0] = theta */
    FUSE_GATE_RY,       /* p[0] = theta */
    FUSE_GATE_RZ,       /* p[0] = theta */
    FUSE_GATE_U3,       /* p = { theta, phi, lambda } */
    FUSE_GATE_FUSED_1Q, /* u[2][2] is pre-filled */
    FUSE_GATE_CNOT,     /* q[0]=ctrl, q[1]=tgt */
    FUSE_GATE_CZ,
    FUSE_GATE_CY,
    FUSE_GATE_SWAP,
    FUSE_GATE_CPHASE,   /* p[0]=theta */
    FUSE_GATE_CRX,
    FUSE_GATE_CRY,
    FUSE_GATE_CRZ
} fuse_gate_kind_t;

typedef struct {
    fuse_gate_kind_t kind;
    int q[2];
    double p[3];
    complex_t u[2][2]; /* used only for FUSE_GATE_FUSED_1Q */
} fuse_gate_t;

typedef struct fuse_circuit_t fuse_circuit_t;

typedef struct {
    size_t original_gates;
    size_t fused_gates;
    size_t merges_applied;
} fuse_stats_t;

/**
 * @brief Allocate an empty gate list for a circuit on @p num_qubits
 *        qubits.
 *
 * The record array starts with capacity for 16 gates and doubles on
 * demand as gates are appended.  Nothing is fused at build time; the
 * accumulator pass runs in ::fuse_compile.
 *
 * @param num_qubits  Qubit count; must be in 1..MAX_QUBITS.
 * @return Owned, heap-allocated circuit; NULL on a zero or oversized
 *         qubit count, and on allocation failure.
 * @stability evolving
 */
MOONLAB_API fuse_circuit_t* fuse_circuit_create(size_t num_qubits);

/**
 * @brief Release a circuit and the gate array it owns.
 *
 * @param c  Circuit to release; NULL is a no-op.
 * @return Nothing.
 * @stability evolving
 */
MOONLAB_API void            fuse_circuit_free(fuse_circuit_t* c);

/**
 * @brief Number of gate records currently held by the circuit.
 *
 * Counts emitted records, so on a circuit produced by ::fuse_compile
 * this is the post-fusion gate count that ::fuse_stats_t reports as
 * @c fused_gates.
 *
 * @param c  Circuit to query.
 * @return Record count; 0 when @p c is NULL.
 * @stability evolving
 */
MOONLAB_API size_t          fuse_circuit_len(const fuse_circuit_t* c);

/**
 * @brief Qubit count the circuit was created with.
 *
 * This is the width used to bounds-check every @c fuse_append_* call
 * and to size the per-qubit accumulator table in ::fuse_compile.
 *
 * @param c  Circuit to query.
 * @return Qubit count; 0 when @p c is NULL.
 * @stability evolving
 */
MOONLAB_API size_t          fuse_circuit_num_qubits(const fuse_circuit_t* c);

/**
 * @brief Append a Hadamard gate on qubit @p q.
 *
 * @param c  Circuit to append to.
 * @param q  Target qubit, in 0..num_qubits-1.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_h(fuse_circuit_t* c, int q);

/**
 * @brief Append a Pauli-X (bit flip) on qubit @p q.
 *
 * @param c  Circuit to append to.
 * @param q  Target qubit, in 0..num_qubits-1.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_x(fuse_circuit_t* c, int q);

/**
 * @brief Append a Pauli-Y on qubit @p q.
 *
 * @param c  Circuit to append to.
 * @param q  Target qubit, in 0..num_qubits-1.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_y(fuse_circuit_t* c, int q);

/**
 * @brief Append a Pauli-Z (phase flip) on qubit @p q.
 *
 * @param c  Circuit to append to.
 * @param q  Target qubit, in 0..num_qubits-1.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_z(fuse_circuit_t* c, int q);

/**
 * @brief Append S = diag(1, i) on qubit @p q.
 *
 * @param c  Circuit to append to.
 * @param q  Target qubit, in 0..num_qubits-1.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_s(fuse_circuit_t* c, int q);

/**
 * @brief Append S-dagger = diag(1, -i) on qubit @p q.
 *
 * @param c  Circuit to append to.
 * @param q  Target qubit, in 0..num_qubits-1.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_sdg(fuse_circuit_t* c, int q);

/**
 * @brief Append T = diag(1, e^{i pi/4}) on qubit @p q.
 *
 * @param c  Circuit to append to.
 * @param q  Target qubit, in 0..num_qubits-1.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_t(fuse_circuit_t* c, int q);

/**
 * @brief Append T-dagger = diag(1, e^{-i pi/4}) on qubit @p q.
 *
 * @param c  Circuit to append to.
 * @param q  Target qubit, in 0..num_qubits-1.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_tdg(fuse_circuit_t* c, int q);

/**
 * @brief Append the phase gate diag(1, e^{i theta}) on qubit @p q.
 *
 * @param c      Circuit to append to.
 * @param q      Target qubit, in 0..num_qubits-1.
 * @param theta  Phase angle in radians, stored verbatim in the
 *               record's p[0] slot.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_phase(fuse_circuit_t* c, int q, double theta);

/**
 * @brief Append R_X(theta) = exp(-i theta X / 2) on qubit @p q.
 *
 * @param c      Circuit to append to.
 * @param q      Target qubit, in 0..num_qubits-1.
 * @param theta  Rotation angle in radians (Qiskit/Cirq convention).
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_rx(fuse_circuit_t* c, int q, double theta);

/**
 * @brief Append R_Y(theta) = exp(-i theta Y / 2) on qubit @p q.
 *
 * @param c      Circuit to append to.
 * @param q      Target qubit, in 0..num_qubits-1.
 * @param theta  Rotation angle in radians (Qiskit/Cirq convention).
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_ry(fuse_circuit_t* c, int q, double theta);

/**
 * @brief Append R_Z(theta) = exp(-i theta Z / 2) on qubit @p q.
 *
 * @param c      Circuit to append to.
 * @param q      Target qubit, in 0..num_qubits-1.
 * @param theta  Rotation angle in radians (Qiskit/Cirq convention).
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_rz(fuse_circuit_t* c, int q, double theta);

/**
 * @brief Append the general single-qubit U3(theta, phi, lambda) on
 *        qubit @p q.
 *
 * Qiskit convention:
 *   [[cos(t/2),            -e^{i l} sin(t/2)],
 *    [e^{i p} sin(t/2),  e^{i(p+l)} cos(t/2)]].
 *
 * @param c       Circuit to append to.
 * @param q       Target qubit, in 0..num_qubits-1.
 * @param theta   Polar angle t, in radians.
 * @param phi     Azimuthal phase p, in radians.
 * @param lambda  Trailing phase l, in radians.
 * @return 0 on success; -1 when @p q is out of range or the record
 *         array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_u3(fuse_circuit_t* c, int q,
                   double theta, double phi, double lambda);

/**
 * @brief Append a CNOT: flip @p tgt when @p ctrl is |1>.
 *
 * Two-qubit gates are fusion barriers -- ::fuse_compile flushes the
 * pending accumulators on both qubits before emitting this record.
 *
 * @param c     Circuit to append to.
 * @param ctrl  Control qubit, in 0..num_qubits-1.
 * @param tgt   Target qubit, in 0..num_qubits-1, distinct from
 *              @p ctrl.
 * @return 0 on success; -1 when either index is out of range, the two
 *         indices are equal, or the record array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_cnot(fuse_circuit_t* c, int ctrl, int tgt);

/**
 * @brief Append a controlled-Z (symmetric in its two qubits).
 *
 * Acts as a fusion barrier on both qubits.
 *
 * @param c     Circuit to append to.
 * @param ctrl  First qubit, in 0..num_qubits-1.
 * @param tgt   Second qubit, in 0..num_qubits-1, distinct from
 *              @p ctrl.
 * @return 0 on success; -1 when either index is out of range, the two
 *         indices are equal, or the record array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_cz(fuse_circuit_t* c, int ctrl, int tgt);

/**
 * @brief Append a controlled-Y: apply Y to @p tgt when @p ctrl is |1>.
 *
 * Acts as a fusion barrier on both qubits.
 *
 * @param c     Circuit to append to.
 * @param ctrl  Control qubit, in 0..num_qubits-1.
 * @param tgt   Target qubit, in 0..num_qubits-1, distinct from
 *              @p ctrl.
 * @return 0 on success; -1 when either index is out of range, the two
 *         indices are equal, or the record array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_cy(fuse_circuit_t* c, int ctrl, int tgt);

/**
 * @brief Append a SWAP of qubits @p a and @p b.
 *
 * Acts as a fusion barrier on both qubits.
 *
 * @param c  Circuit to append to.
 * @param a  First qubit, in 0..num_qubits-1.
 * @param b  Second qubit, in 0..num_qubits-1, distinct from @p a.
 * @return 0 on success; -1 when either index is out of range, the two
 *         indices are equal, or the record array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_swap(fuse_circuit_t* c, int a, int b);

/**
 * @brief Append a controlled phase: diag(1, 1, 1, e^{i theta}).
 *
 * Acts as a fusion barrier on both qubits.
 *
 * @param c      Circuit to append to.
 * @param ctrl   Control qubit, in 0..num_qubits-1.
 * @param tgt    Target qubit, in 0..num_qubits-1, distinct from
 *               @p ctrl.
 * @param theta  Phase angle in radians.
 * @return 0 on success; -1 when either index is out of range, the two
 *         indices are equal, or the record array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_cphase(fuse_circuit_t* c, int ctrl, int tgt, double theta);

/**
 * @brief Append a controlled-R_X(theta) on @p tgt, gated by @p ctrl.
 *
 * Acts as a fusion barrier on both qubits.
 *
 * @param c      Circuit to append to.
 * @param ctrl   Control qubit, in 0..num_qubits-1.
 * @param tgt    Target qubit, in 0..num_qubits-1, distinct from
 *               @p ctrl.
 * @param theta  Rotation angle in radians.
 * @return 0 on success; -1 when either index is out of range, the two
 *         indices are equal, or the record array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_crx(fuse_circuit_t* c, int ctrl, int tgt, double theta);

/**
 * @brief Append a controlled-R_Y(theta) on @p tgt, gated by @p ctrl.
 *
 * Acts as a fusion barrier on both qubits.
 *
 * @param c      Circuit to append to.
 * @param ctrl   Control qubit, in 0..num_qubits-1.
 * @param tgt    Target qubit, in 0..num_qubits-1, distinct from
 *               @p ctrl.
 * @param theta  Rotation angle in radians.
 * @return 0 on success; -1 when either index is out of range, the two
 *         indices are equal, or the record array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_cry(fuse_circuit_t* c, int ctrl, int tgt, double theta);

/**
 * @brief Append a controlled-R_Z(theta) on @p tgt, gated by @p ctrl.
 *
 * Acts as a fusion barrier on both qubits.
 *
 * @param c      Circuit to append to.
 * @param ctrl   Control qubit, in 0..num_qubits-1.
 * @param tgt    Target qubit, in 0..num_qubits-1, distinct from
 *               @p ctrl.
 * @param theta  Rotation angle in radians.
 * @return 0 on success; -1 when either index is out of range, the two
 *         indices are equal, or the record array could not be grown.
 * @stability evolving
 */
MOONLAB_API int fuse_append_crz(fuse_circuit_t* c, int ctrl, int tgt, double theta);

/**
 * @brief Produce a fused copy of @p src.
 * @param src    input circuit (not modified)
 * @param stats  optional statistics sink (may be NULL)
 * @return owned, heap-allocated circuit; NULL on OOM.
 * @stability evolving
 */
MOONLAB_API fuse_circuit_t* fuse_compile(const fuse_circuit_t* src, fuse_stats_t* stats);

/**
 * @brief Execute a circuit on a state vector.
 *
 * Works on both fused and unfused circuits. Returns QS_SUCCESS if every
 * gate applied cleanly.
 * @stability evolving
 */
MOONLAB_API qs_error_t fuse_execute(const fuse_circuit_t* c, quantum_state_t* state);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_FUSION_H */
