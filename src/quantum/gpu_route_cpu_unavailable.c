/*
 * CPU-only definitions of the optional GPU-backend entry points.
 *
 * These symbols are provided by the CUDA translation units
 * (cuda_statevec.cu / state_gpu.cu) when CUDA is compiled in. When it is not,
 * exactly one definition must still exist so the library links: the gate
 * dispatcher and the distributed layer reference them and only ever call them
 * when state->gpu_state is non-NULL, which never happens on a CPU-only build.
 *
 * This file is compiled ONLY when the CUDA sources are not (the else-branch of
 * if(QSIM_HAS_CUDA) in CMakeLists.txt), so there is never more than one
 * definition. It deliberately uses plain, strong definitions rather than
 * __attribute__((weak)): clang-cl on Windows/COFF does not honour the weak
 * attribute on function definitions and emits duplicate symbols, which broke
 * the Windows link. One strong definition per configuration is portable
 * everywhere.
 */

#include "state.h"

#include <stdint.h>

void moonlab_cuda_state_free(void *p) { (void)p; }

int qsim_gpu_route_hadamard(quantum_state_t *s, int q) { (void)s; (void)q; return -1; }
int qsim_gpu_route_pauli_x(quantum_state_t *s, int q) { (void)s; (void)q; return -1; }
int qsim_gpu_route_cnot(quantum_state_t *s, int c, int t) {
    (void)s; (void)c; (void)t; return -1;
}
int qsim_gpu_route_apply_1q_matrix(quantum_state_t *s, int q, const double m[8]) {
    (void)s; (void)q; (void)m; return -1;
}
int qsim_gpu_route_apply_2q_matrix(quantum_state_t *s, int q0, int q1, const double m[32]) {
    (void)s; (void)q0; (void)q1; (void)m; return -1;
}
int qsim_gpu_route_mcx(quantum_state_t *s, uint64_t mask, int t) {
    (void)s; (void)mask; (void)t; return -1;
}
int qsim_gpu_route_mcz(quantum_state_t *s, uint64_t mask) {
    (void)s; (void)mask; return -1;
}
int qsim_gpu_route_fredkin(quantum_state_t *s, int c, int t1, int t2) {
    (void)s; (void)c; (void)t1; (void)t2; return -1;
}

int moonlab_cuda_runtime_probe_discrete(void) { return 0; }

int moonlab_cuda_apply_1q(void *state, uint32_t target, const double m[8]) {
    (void)state; (void)target; (void)m; return -1;
}
int moonlab_cuda_apply_cnot(void *state, uint32_t control, uint32_t target) {
    (void)state; (void)control; (void)target; return -1;
}
int moonlab_cuda_state_create(uint32_t n_qubits, void **out_state) {
    (void)n_qubits;
    if (out_state) *out_state = NULL;
    return -1;
}
int moonlab_cuda_state_copy_to_host(const void *state, double *out) {
    (void)state; (void)out; return -1;
}
int moonlab_cuda_state_copy_from_host(void *state, const double *in) {
    (void)state; (void)in; return -1;
}
int moonlab_cuda_select_device_for_rank(int local_rank,
                                        int *device_id,
                                        int *device_count) {
    (void)local_rank;
    if (device_id) *device_id = -1;
    if (device_count) *device_count = 0;
    return -1;
}
