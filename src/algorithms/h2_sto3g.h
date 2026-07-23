/**
 * @file h2_sto3g.h
 * @brief First-principles STO-3G H2 Pauli coefficients as smooth functions of
 *        bond length -- a differentiable potential energy surface.
 *
 * The two-qubit H2 Hamiltonian H = g0 II + g1 IZ + g2 ZI + g3 ZZ + g4 XX has
 * coefficients that are computed here from genuine STO-3G Gaussian integrals
 * (own Boys/erf), Slater-Condon reduction to the seniority-zero space, and the
 * standard Jordan-Wigner map.  Unlike a Morse/exponential interpolation, these
 * are smooth and C-infinity in r (in particular they have NO kink at
 * equilibrium), so a finite-difference or automatic derivative of the VQE energy
 * with respect to geometry yields a correct interatomic force.
 *
 * Validated against O'Malley et al. PRX 6, 031007 (2016): the map reconstructs
 * the exact FCI ground-state energy at every bond length to machine precision;
 * absolute coefficients agree with O'Malley to the inter-implementation STO-3G
 * integral floor (~1e-3), which cancels in the r-DEPENDENCE used for forces.
 */
#ifndef MOONLAB_H2_STO3G_H
#define MOONLAB_H2_STO3G_H

#include "../applications/moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute the five STO-3G H2 Pauli coefficients at a bond length.
 * @param r_angstrom  Internuclear distance in Angstroms.
 * @param g           Output array [g0=II, g1=IZ, g2=ZI, g3=ZZ, g4=XX] (Hartree),
 *                    the electronic coefficients (nuclear repulsion is separate).
 */
MOONLAB_API void h2_sto3g_pauli_coeffs(double r_angstrom, double g[5]);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_H2_STO3G_H */
