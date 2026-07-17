/**
 * @file moonlab_vqe_export.c
 * @brief Stable-ABI wrapper for the exact VQE gradient.
 *
 * Definition for the `moonlab_vqe_gradient` entry point declared in
 * `moonlab_export.h`.  The wrapper validates the caller's view of the
 * parameter count against the solver's ansatz before delegating to
 * `vqe_compute_gradient`.  On a noise-free solver the gradient is exact:
 * reverse-mode adjoint autograd for hardware-efficient ansaetze, analytic
 * parameter-shift otherwise — never a finite difference, so downstream AD
 * systems can wrap it as an exact VJP.  With a noise model attached,
 * `vqe_compute_gradient` returns `VQE_GRADIENT_ERR_NOT_EXACT` (which this
 * wrapper surfaces as -3) unless the caller has explicitly opted into a
 * stochastic estimate via `vqe_solver_set_allow_stochastic_gradient`; the
 * exact-AD contract is never silently violated.
 *
 * Returns 0 on success, -1 on NULL argument, -2 on parameter-count
 * mismatch, -3 on gradient failure (including the not-exact-under-noise case).
 *
 * @since 1.1.0 (ABI 0.4.0)
 */

#include "moonlab_export.h"

#include "../algorithms/vqe.h"

int moonlab_vqe_gradient(moonlab_vqe_solver_t* solver,
                         const double* parameters,
                         double* gradient_out,
                         size_t num_parameters)
{
    if (!solver || !parameters || !gradient_out) {
        return -1;
    }
    if (!solver->ansatz || solver->ansatz->num_parameters != num_parameters) {
        return -2;
    }
    if (vqe_compute_gradient(solver, parameters, gradient_out) != 0) {
        return -3;
    }
    return 0;
}
