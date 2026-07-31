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
#include "../../applications/moonlab_api.h"

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
 * @stability evolving
 */
MOONLAB_API moonlab_ca_peps_t* moonlab_ca_peps_create(uint32_t Lx, uint32_t Ly,
                                            uint32_t chi_bond);

/**
 * Release the handle and the underlying CA-MPS factor.  No-op on NULL.
 * @param s  CA-PEPS handle to free.
 * @return Nothing.
 * @stability evolving
 */
MOONLAB_API void moonlab_ca_peps_free(moonlab_ca_peps_t* s);

/**
 * @brief Deep-copy a CA-PEPS.
 *
 * Copies the lattice extents and bond cap, then clones the backing
 * CA-MPS (tableau D plus the MPS factor |phi>).
 *
 * @param s  Source handle.
 * @return A newly owned handle, or NULL when @p s is NULL or the
 *         clone allocation fails.
 * @stability evolving
 */
MOONLAB_API moonlab_ca_peps_t* moonlab_ca_peps_clone(const moonlab_ca_peps_t* s);

/**
 * Lattice extent along x, as passed to @c moonlab_ca_peps_create.
 * @param s  CA-PEPS handle.
 * @return Lx; 0 when @p s is NULL.
 * @stability evolving
 */
MOONLAB_API uint32_t moonlab_ca_peps_lx(const moonlab_ca_peps_t* s);

/**
 * Lattice extent along y, as passed to @c moonlab_ca_peps_create.
 * @param s  CA-PEPS handle.
 * @return Ly; 0 when @p s is NULL.
 * @stability evolving
 */
MOONLAB_API uint32_t moonlab_ca_peps_ly(const moonlab_ca_peps_t* s);

/**
 * Number of lattice sites, Lx * Ly.  This is the length every Pauli
 * string passed to the measurement entry points must have.
 * @param s  CA-PEPS handle.
 * @return Lx * Ly; 0 when @p s is NULL.
 * @stability evolving
 */
MOONLAB_API uint32_t moonlab_ca_peps_num_qubits(const moonlab_ca_peps_t* s);

/**
 * Bond-dimension cap @c chi_bond the state was created with.
 * @param s  CA-PEPS handle.
 * @return The cap; 0 when @p s is NULL.
 * @stability evolving
 */
MOONLAB_API uint32_t moonlab_ca_peps_max_bond_dim(const moonlab_ca_peps_t* s);

/**
 * @brief Largest bond dimension currently carried by the row-major MPS
 *        factor |phi>.
 *
 * Reflects working storage after the most recent truncation, not the
 * intrinsic entanglement -- use
 * ::moonlab_ca_peps_max_half_cut_entropy for the latter.
 *
 * @param s  CA-PEPS handle.
 * @return Current max bond dimension (>= 1); 0 when @p s is NULL.
 * @stability evolving
 */
MOONLAB_API uint32_t moonlab_ca_peps_current_bond_dim(const moonlab_ca_peps_t* s);

/**
 * @brief Maximum half-cut von Neumann entanglement entropy of @c |phi>
 *        across all bipartitions, in nats.
 *
 * Same yardstick as the CA-MPS analogue: representation-independent
 * compactness measure.
 * @stability evolving
 */
MOONLAB_API double moonlab_ca_peps_max_half_cut_entropy(const moonlab_ca_peps_t* s);

/* ================================================================== */
/*  Clifford gates -- tableau-only updates (O(n) bit ops).            */
/* ================================================================== */

/* Single-qubit indexed by linear (x + Lx*y).  Two-qubit must be on
 * adjacent sites in the square lattice. */

/**
 * Hadamard on site @p q.  Tableau-only, so |phi> is untouched.
 * @param s  CA-PEPS handle.
 * @param q  Linear site index x + Lx*y, in 0..Lx*Ly-1.
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when @p s is NULL or
 *         @p q is off-lattice, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_h(moonlab_ca_peps_t* s, uint32_t q);

/**
 * S = sqrt(Z) = diag(1, i) on site @p q.  Tableau-only.
 * @param s  CA-PEPS handle.
 * @param q  Linear site index x + Lx*y, in 0..Lx*Ly-1.
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when @p s is NULL or
 *         @p q is off-lattice, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_s(moonlab_ca_peps_t* s, uint32_t q);

/**
 * S-dagger = diag(1, -i) on site @p q.  Tableau-only.
 * @param s  CA-PEPS handle.
 * @param q  Linear site index x + Lx*y, in 0..Lx*Ly-1.
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when @p s is NULL or
 *         @p q is off-lattice, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_sdag(moonlab_ca_peps_t* s, uint32_t q);

/**
 * Pauli X (bit flip) on site @p q.  Tableau-only.
 * @param s  CA-PEPS handle.
 * @param q  Linear site index x + Lx*y, in 0..Lx*Ly-1.
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when @p s is NULL or
 *         @p q is off-lattice, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_x(moonlab_ca_peps_t* s, uint32_t q);

/**
 * Pauli Y on site @p q.  Tableau-only.
 * @param s  CA-PEPS handle.
 * @param q  Linear site index x + Lx*y, in 0..Lx*Ly-1.
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when @p s is NULL or
 *         @p q is off-lattice, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_y(moonlab_ca_peps_t* s, uint32_t q);

/**
 * Pauli Z (phase flip) on site @p q.  Tableau-only.
 * @param s  CA-PEPS handle.
 * @param q  Linear site index x + Lx*y, in 0..Lx*Ly-1.
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when @p s is NULL or
 *         @p q is off-lattice, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_z(moonlab_ca_peps_t* s, uint32_t q);

/**
 * @brief Controlled-NOT between two lattice neighbours.
 *
 * The tableau update itself is geometry-free, but the call still
 * enforces square-lattice adjacency (no periodic boundary) so callers
 * cannot silently treat distant sites as neighbours.
 *
 * @param s  CA-PEPS handle.
 * @param c  Control site, linear index x + Lx*y.
 * @param t  Target site; must be a horizontal or vertical neighbour
 *           of @p c.
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when either index is
 *         off-lattice or the two sites are not adjacent, or a mapped
 *         CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_cnot(moonlab_ca_peps_t* s, uint32_t c, uint32_t t);

/**
 * @brief Controlled-Z between two lattice neighbours (symmetric in
 *        @p a / @p b).
 *
 * Adjacency is enforced exactly as in ::moonlab_ca_peps_cnot.
 *
 * @param s  CA-PEPS handle.
 * @param a  First site, linear index x + Lx*y.
 * @param b  Second site; must be a horizontal or vertical neighbour
 *           of @p a.
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when either index is
 *         off-lattice or the two sites are not adjacent, or a mapped
 *         CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_cz(moonlab_ca_peps_t* s, uint32_t a, uint32_t b);

/* ================================================================== */
/*  Non-Clifford gates -- push into PEPS via Pauli-rotation MPO.       */
/* ================================================================== */

/**
 * @brief R_X(theta) = exp(-i theta X / 2) on site @p q.
 *
 * Non-Clifford: the generator is conjugated through D into a Pauli
 * string and applied to |phi> as a bond-dim-2 MPO, so the cost is
 * independent of where @p q sits on the lattice.
 *
 * @param s      CA-PEPS handle.
 * @param q      Linear site index x + Lx*y, in 0..Lx*Ly-1.
 * @param theta  Rotation angle in radians (Qiskit/Cirq convention).
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when @p s is NULL or
 *         @p q is off-lattice, ::CA_PEPS_ERR_OOM on allocation
 *         failure, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_rx(moonlab_ca_peps_t* s, uint32_t q, double theta);

/**
 * @brief R_Y(theta) = exp(-i theta Y / 2) on site @p q.
 *
 * Same Pauli-rotation push-through as ::moonlab_ca_peps_rx.
 *
 * @param s      CA-PEPS handle.
 * @param q      Linear site index x + Lx*y, in 0..Lx*Ly-1.
 * @param theta  Rotation angle in radians (Qiskit/Cirq convention).
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when @p s is NULL or
 *         @p q is off-lattice, ::CA_PEPS_ERR_OOM on allocation
 *         failure, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_ry(moonlab_ca_peps_t* s, uint32_t q, double theta);

/**
 * @brief R_Z(theta) = exp(-i theta Z / 2) on site @p q.
 *
 * Same Pauli-rotation push-through as ::moonlab_ca_peps_rx.
 *
 * @param s      CA-PEPS handle.
 * @param q      Linear site index x + Lx*y, in 0..Lx*Ly-1.
 * @param theta  Rotation angle in radians (Qiskit/Cirq convention).
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_QUBIT when @p s is NULL or
 *         @p q is off-lattice, ::CA_PEPS_ERR_OOM on allocation
 *         failure, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_rz(moonlab_ca_peps_t* s, uint32_t q, double theta);

/** T = R_Z(pi/4) up to a global phase. */
ca_peps_error_t moonlab_ca_peps_t_gate(moonlab_ca_peps_t* s, uint32_t q);

/** T-dagger = R_Z(-pi/4) up to a global phase. */
ca_peps_error_t moonlab_ca_peps_t_dagger(moonlab_ca_peps_t* s, uint32_t q);

/** P(theta) = diag(1, e^{i theta}); equals R_Z(theta) up to a global phase. */
ca_peps_error_t moonlab_ca_peps_phase(moonlab_ca_peps_t* s,
                                       uint32_t q, double theta);

/**
 * @brief Apply exp(i theta P) for an n-qubit Pauli string P.
 *
 * @param pauli  Length-(Lx*Ly) byte array, 0=I, 1=X, 2=Y, 3=Z.
 * @param theta  Rotation angle in radians.
 */
ca_peps_error_t moonlab_ca_peps_pauli_rotation(moonlab_ca_peps_t* s,
                                                const uint8_t* pauli,
                                                double theta);

/**
 * @brief Apply exp(-tau P) for an n-qubit Pauli string P.
 *
 * Imaginary-time-step primitive (non-unitary).  The caller is
 * responsible for renormalisation via @c moonlab_ca_peps_normalize.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_imag_pauli_rotation(moonlab_ca_peps_t* s,
                                                     const uint8_t* pauli,
                                                     double tau);

/**
 * Rescale the internal MPS factor to unit norm.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_normalize(moonlab_ca_peps_t* s);

/**
 * Return the norm of the underlying state.
 * @stability evolving
 */
MOONLAB_API double moonlab_ca_peps_norm(const moonlab_ca_peps_t* s);

/* ================================================================== */
/*  Measurement                                                         */
/* ================================================================== */

/**
 * @brief Compute <psi | P | psi> for a Pauli string P on the lattice.
 *
 * Delegates to the CA-MPS contraction, which absorbs the Clifford
 * prefactor via <psi|P|psi> = <phi| D^dagger P D |phi>.  The result is
 * geometry-independent given the row-major embedding.
 *
 * @param s           CA-PEPS handle.
 * @param pauli       Length-(Lx*Ly) byte array, 0=I, 1=X, 2=Y, 3=Z,
 *                    indexed by x + Lx*y.
 * @param out_expval  Receives the expectation value.
 * @return ::CA_PEPS_SUCCESS, ::CA_PEPS_ERR_INVALID when any argument
 *         is NULL, or a mapped CA-MPS backend error.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_expect_pauli(const moonlab_ca_peps_t* s,
                                              const uint8_t* pauli,
                                              double _Complex* out_expval);

/**
 * @brief Compute <psi|H|psi> for a Hamiltonian H = sum_k c_k P_k.
 *
 * @param paulis     Flat (num_terms, num_qubits) uint8 Pauli array.
 * @param coeffs     Length-num_terms complex coefficients.
 * @param num_terms  Pauli-sum length.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_expect_pauli_sum(const moonlab_ca_peps_t* s,
                                                  const uint8_t* paulis,
                                                  const double _Complex* coeffs,
                                                  uint32_t num_terms,
                                                  double _Complex* out_expval);

/**
 * Marginal P(Z_q = +1).  See ca_mps analogue for the fine print.
 * @stability evolving
 */
MOONLAB_API ca_peps_error_t moonlab_ca_peps_prob_z(const moonlab_ca_peps_t* s,
                                        uint32_t q, double* out_prob);

/* ================================================================== */
/*  Variational-D ground-state search (delegated to CA-MPS engine)    */
/* ================================================================== */

/**
 * @brief Run the variational-D alternating loop for a Pauli-sum
 *        Hamiltonian on the row-major-MPS-backed CA-PEPS.
 *
 * Thin wrapper around @c moonlab_ca_mps_var_d_run.  Greedy local
 * Clifford search on D is alternated with imaginary-time evolution on
 * the underlying |phi>.  Local-Clifford moves operate on linear-index
 * adjacent qubits, which corresponds to horizontal lattice neighbours
 * in the row-major embedding -- vertical correlations are still
 * captured by D through composite multi-qubit Cliffords (CNOT chains
 * along the linear ordering), but a real PEPS factor would expose
 * them more efficiently.  v0.3 milestone.
 *
 * @param state                  CA-PEPS handle (mutated in place).
 * @param paulis                 Flat (num_terms, Lx*Ly) uint8 array.
 * @param coeffs                 Real coefficients, length num_terms.
 * @param num_terms              Pauli-sum length.
 * @param max_outer_iters        Outer alternating-loop cap.
 * @param imag_time_dtau         Imag-time step size.
 * @param imag_time_steps_per_outer  Trotter cycles per outer iter.
 * @param clifford_passes_per_outer  Greedy D-update passes per outer iter.
 * @param composite_2gate        Pass 1 to enable 2-gate composite moves.
 * @param warmstart              0=I, 1=H_ALL, 2=DUAL_TFIM, 3=FERRO_TFIM,
 *                               4=STABILIZER_SUBGROUP.
 * @param stab_paulis            For warmstart=4: (k, num_qubits) generators.
 * @param stab_num_gens          For warmstart=4: number of generators k.
 * @param[out] out_final_energy  Final variational energy (NULL ok).
 *
 * @return 0 on success, negative ::ca_peps_error_t on failure.
 *
 * @since v0.2.1
 * @stability evolving
 */
MOONLAB_API int moonlab_ca_peps_var_d_run(moonlab_ca_peps_t* state,
                               const uint8_t* paulis,
                               const double* coeffs,
                               uint32_t num_terms,
                               uint32_t max_outer_iters,
                               double imag_time_dtau,
                               uint32_t imag_time_steps_per_outer,
                               uint32_t clifford_passes_per_outer,
                               int composite_2gate,
                               int warmstart,
                               const uint8_t* stab_paulis,
                               uint32_t stab_num_gens,
                               double* out_final_energy);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CA_PEPS_H */
