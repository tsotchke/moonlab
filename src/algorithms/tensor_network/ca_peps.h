/**
 * @file ca_peps.h
 * @brief Clifford-Assisted PEPS -- 2D generalisation of CA-MPS.
 *
 * The 2D extension promised in section 7 of docs/research/ca_mps.md.
 * The state lives on an Lx by Ly square lattice; Clifford gates update
 * the Aaronson-Gottesman tableau D only (free), and non-Clifford gates
 * push through D as Pauli-string rotations applied to the physical
 * factor |phi>.
 *
 * v0.2.1 implementation: |phi> is stored as a row-major MPS over the
 * Lx * Ly qubits via the existing CA-MPS engine (q = x + Lx * y).  This
 * is *not* a true PEPS factor -- a real 2D tensor with four bond indices
 * per site, plus split-CTMRG / boundary-MPS contraction, lands in v0.3.
 * The row-major embedding nonetheless gives correct results for any
 * circuit at the current ABI surface (single + 2-qubit Cliffords + RX/
 * RY/RZ + Pauli expectation), since:
 *   - Clifford updates are tableau-only and don't touch |phi>.
 *   - Single-qubit rotations conjugate to a Pauli string and apply as a
 *     bond-dim-2 MPO regardless of the underlying tensor topology.
 *   - Pauli expectation values reduce to <phi | conj_pauli | phi> and
 *     contract through the existing CA-MPS path.
 *
 * The trade-off is bond-growth scaling: dense non-Clifford 2D circuits
 * push the row-major MPS bond dimension up faster than a true PEPS
 * would.  For Clifford-heavy circuits (the regime CA-* targets) the
 * difference is irrelevant.  Two-qubit non-Clifford gates are not in
 * the public API yet; they need real PEPS to be added without giving
 * up the scaling argument.
 *
 * The API mirrors @c moonlab_ca_mps_* one-to-one so consumer code can
 * switch between 1D and 2D by changing the type and the @c create call.
 */

#ifndef MOONLAB_CA_PEPS_H
#define MOONLAB_CA_PEPS_H

#include <stddef.h>
#include <stdint.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Error codes returned by every public CA-PEPS entry point.  All
 * non-zero values indicate failure.  @c CA_PEPS_ERR_NOT_IMPLEMENTED
 * is retained for ABI compatibility and is reserved for entry points
 * that still depend on real PEPS infrastructure (none in v0.2.1). */
typedef enum {
    CA_PEPS_SUCCESS              =  0,
    CA_PEPS_ERR_INVALID          = -1,
    CA_PEPS_ERR_QUBIT            = -2,
    CA_PEPS_ERR_OOM              = -3,
    CA_PEPS_ERR_BACKEND          = -4,
    CA_PEPS_ERR_NOT_IMPLEMENTED  = -100
} ca_peps_error_t;

typedef struct moonlab_ca_peps_t moonlab_ca_peps_t;

/* ================================================================== */
/*  Lifecycle                                                          */
/* ================================================================== */

/**
 * @brief Construct a CA-PEPS on an Lx by Ly square lattice with
 *        per-bond bond dimension cap @p chi_bond.
 *
 * The physical state is initialised to |0>^(Lx*Ly) with the Clifford
 * prefactor D = I.
 */
moonlab_ca_peps_t* moonlab_ca_peps_create(uint32_t Lx, uint32_t Ly,
                                            uint32_t chi_bond);

void moonlab_ca_peps_free(moonlab_ca_peps_t* s);
moonlab_ca_peps_t* moonlab_ca_peps_clone(const moonlab_ca_peps_t* s);

uint32_t moonlab_ca_peps_lx(const moonlab_ca_peps_t* s);
uint32_t moonlab_ca_peps_ly(const moonlab_ca_peps_t* s);
uint32_t moonlab_ca_peps_num_qubits(const moonlab_ca_peps_t* s);
uint32_t moonlab_ca_peps_max_bond_dim(const moonlab_ca_peps_t* s);

/* ================================================================== */
/*  Clifford gates -- tableau-only updates (O(n) bit ops).            */
/* ================================================================== */

/* Single-qubit indexed by linear (x + Lx*y).  Two-qubit must be on
 * adjacent sites in the square lattice. */

ca_peps_error_t moonlab_ca_peps_h(moonlab_ca_peps_t* s, uint32_t q);
ca_peps_error_t moonlab_ca_peps_s(moonlab_ca_peps_t* s, uint32_t q);
ca_peps_error_t moonlab_ca_peps_sdag(moonlab_ca_peps_t* s, uint32_t q);
ca_peps_error_t moonlab_ca_peps_x(moonlab_ca_peps_t* s, uint32_t q);
ca_peps_error_t moonlab_ca_peps_y(moonlab_ca_peps_t* s, uint32_t q);
ca_peps_error_t moonlab_ca_peps_z(moonlab_ca_peps_t* s, uint32_t q);
ca_peps_error_t moonlab_ca_peps_cnot(moonlab_ca_peps_t* s, uint32_t c, uint32_t t);
ca_peps_error_t moonlab_ca_peps_cz(moonlab_ca_peps_t* s, uint32_t a, uint32_t b);

/* ================================================================== */
/*  Non-Clifford gates -- push into PEPS via Pauli-rotation MPO.       */
/* ================================================================== */

ca_peps_error_t moonlab_ca_peps_rx(moonlab_ca_peps_t* s, uint32_t q, double theta);
ca_peps_error_t moonlab_ca_peps_ry(moonlab_ca_peps_t* s, uint32_t q, double theta);
ca_peps_error_t moonlab_ca_peps_rz(moonlab_ca_peps_t* s, uint32_t q, double theta);

/* ================================================================== */
/*  Measurement -- requires split-CTMRG environment for a 2D PEPS      */
/*                 contraction.                                        */
/* ================================================================== */

ca_peps_error_t moonlab_ca_peps_expect_pauli(const moonlab_ca_peps_t* s,
                                              const uint8_t* pauli,
                                              double _Complex* out_expval);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CA_PEPS_H */
