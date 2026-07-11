/**
 * @file moonlab_vqe_export.c
 * @brief Stable-ABI wrapper for the exact VQE gradient.
 *
 * Definition for the `moonlab_vqe_gradient` entry point declared in
 * `moonlab_export.h`.  The wrapper validates the caller's view of the
 * parameter count against the solver's ansatz before delegating to
 * `vqe_compute_gradient`, whose dispatch is exact on every path:
 * reverse-mode adjoint autograd for noise-free hardware-efficient
 * ansaetze, analytic parameter-shift otherwise.  No finite
 * differences — downstream AD systems wrap this as an exact VJP.
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
