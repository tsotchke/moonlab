/**
 * @file ca_mps_var_d_stab_warmstart.h
 * @brief Stabilizer-subgroup warmstart for var-D CA-MPS.
 *
 * Given a list of commuting Pauli strings g_0, ..., g_{k-1} on n qubits
 * (the generators of a stabilizer subgroup S of the n-qubit Pauli group),
 * builds a Clifford circuit C such that C|0^n> is in the simultaneous +1
 * eigenspace of every g_i, and applies that circuit to a CA-MPS state.
 *
 * The construction is symplectic Gauss-Jordan elimination on the Pauli
 * tableau (Aaronson-Gottesman 2004, "Improved Simulation of Stabilizer
 * Circuits", arXiv:quant-ph/0406196).  Conjugation rules for H, S, CNOT
 * applied as column operations bring the tableau to canonical form
 * { Z_{q_0}, ..., Z_{q_{k-1}} }; the inverse-and-reversed circuit
 * applied to |0^n> (with X flips on rows whose final phase is -1) is
 * then a state in the +1 eigenspace of the original generators.
 *
 * Use case: 1+1D Z2 lattice gauge theory.  The Gauss-law operators
 * G_x = X_{2x-1} X_{2x+1} Z_{2x} commute and generate a stabilizer
 * subgroup whose +1 eigenspace is the gauge-invariant (physical) sector.
 * Standard var-D warmstarts (TFIM-dual, FERRO) violate the Gauss law
 * and the greedy Clifford search has no single-gate descent toward a
 * gauge-projecting Clifford.  This warmstart starts var-D inside the
 * physical sector; the alternating loop then only has to recover the
 * remaining matter + electric-field dynamics on top.
 *
 * Generalises directly to any stabilizer-subgroup Hamiltonian:
 * surface code, toric code, repetition code, color code, ZN gauge
 * theories (with composite single-qubit Cliffords), etc.  For Z2 (qubit
 * stabilizers) the algorithm is exact and the Clifford circuit has at
 * most O(k * n) gates.
 */

#ifndef MOONLAB_CA_MPS_VAR_D_STAB_WARMSTART_H
#define MOONLAB_CA_MPS_VAR_D_STAB_WARMSTART_H

#include "ca_mps.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apply a stabilizer-subgroup warmstart Clifford to a CA-MPS state.
 *
 * Builds and applies (in place, into @p state->D since every emitted
 * gate is Clifford) a Clifford circuit C such that the resulting
 * state |psi> = C|phi_0> is in the +1 eigenspace of each generator
 * g_i, where |phi_0> is the current @p state contents (typically
 * |0^n> for a freshly created CA-MPS).
 *
 * @param[in,out] state       CA-MPS handle.  Mutated in place.
 * @param[in]     paulis      Generators as a flat row-major
 *                            (num_gens, num_qubits) uint8_t array.
 *                            Encoding: 0=I, 1=X, 2=Y, 3=Z.  Same
 *                            convention as ::moonlab_ca_mps_expect_pauli_sum
 *                            and the LGT Pauli-sum builder.
 * @param[in]     num_gens    Number of generators (k).  Must be >= 1
 *                            and <= num_qubits.
 *
 * Preconditions: all generators must pairwise commute, and they must
 * be linearly independent over the symplectic F_2 vector space.  If
 * either fails, the function returns CA_MPS_ERR_INVALID without
 * modifying @p state.
 *
 * Phase: each input Pauli string is treated as a +1-coefficient
 * Hermitian Pauli (i.e. real product of Pauli matrices, no leading
 * factor of i).  This matches the Z2 LGT Gauss-law operators and
 * stabilizer codes typically used in Moonlab.  The internal
 * elimination still tracks phase bits correctly so that future
 * extensions to negative-phase generators are local to one helper.
 *
 * @return CA_MPS_SUCCESS on success.
 *         CA_MPS_ERR_INVALID on bad arguments or violated
 *         commutation/independence preconditions.
 *         Other ca_mps_error_t codes propagated from the underlying
 *         Clifford gate applications.
 */
ca_mps_error_t moonlab_ca_mps_apply_stab_subgroup_warmstart(
    moonlab_ca_mps_t* state,
    const uint8_t* paulis,
    uint32_t num_gens);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CA_MPS_VAR_D_STAB_WARMSTART_H */
