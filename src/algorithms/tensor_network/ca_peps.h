/**
 * @file ca_peps.h
 * @brief Clifford-Assisted PEPS -- 2D generalisation of CA-MPS.
 *
 * The 2D extension promised in §7 of docs/research/ca_mps.md: replace
 * the MPS factor |phi> with a Projected Entangled Pair State (PEPS)
 * while keeping the Clifford prefactor D as the same Aaronson-Gottesman
 * tableau used by ca_mps.h.  Clifford gates update only the tableau
 * (free); non-Clifford gates apply Pauli-string rotations to the PEPS
 * via split-CTMRG environment-tensor contraction.
 *
 * **Status: scaffold only as of 2026-04-29.**  This header defines the
 * public API surface for a downstream consumer (e.g. QGTL) to plan
 * against.  The implementation in @c ca_peps.c stubs every entry point
 * with @c CA_PEPS_ERR_NOT_IMPLEMENTED.  The reason: PEPS itself doesn't
 * exist yet in Moonlab -- no @c tn_peps_state_t type, no split-CTMRG
 * environment routine, no PEPS-specific SVD compression.  Building
 * those is the actual work; CA-PEPS sits on top.  Estimated 2 weeks
 * for the full implementation per the design doc.
 *
 * The scaffolded API is designed to mirror @c moonlab_ca_mps_* one-to-one
 * so consumer code can switch between 1D and 2D by changing the type
 * and the @c create call.
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
 * non-zero values indicate failure; @c CA_PEPS_ERR_NOT_IMPLEMENTED
 * is the placeholder used by stubs in the v0.1 scaffolding. */
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
