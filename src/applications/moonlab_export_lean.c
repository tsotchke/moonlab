/**
 * @file moonlab_export_lean.c
 * @brief WASM-safe half of the v0.2.x stable C export surface.
 *
 * Implements the @c moonlab_* ABI functions declared in
 * @c moonlab_export.h that do NOT depend on the hardware-entropy /
 * quantum-RNG pipeline.  The qrng-backed @c moonlab_qrng_bytes
 * lives in the sibling file @c moonlab_qrng_export.c so that the
 * lean half can be included in builds (notably the emscripten
 * WASM build) that exclude qrng + hardware_entropy for size
 * reasons.
 *
 * Split is internal-only: the public ABI surface in
 * @c moonlab_export.h is unchanged.  Native builds link both
 * files and downstream consumers (QGTL et al.) see exactly the
 * symbols they always saw.
 *
 * @since v0.5.0 (file split out of moonlab_qrng_export.c).
 */

#include "moonlab_export.h"
#include "../algorithms/quantum_geometry/qgt.h"
#include "../algorithms/tensor_network/dmrg.h"
#include "../algorithms/tensor_network/tn_state.h"
#include "../algorithms/tensor_network/ca_mps.h"
#include "../algorithms/tensor_network/ca_mps_var_d.h"
#include "../algorithms/tensor_network/ca_mps_var_d_stab_warmstart.h"
#include "hep/lattice_z2_1d.h"
#include "../utils/moonlab_status.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

void moonlab_abi_version(int* major, int* minor, int* patch) {
    if (major) *major = MOONLAB_ABI_VERSION_MAJOR;
    if (minor) *minor = MOONLAB_ABI_VERSION_MINOR;
    if (patch) *patch = MOONLAB_ABI_VERSION_PATCH;
}

int moonlab_qwz_chern(double m, size_t N, double* out_chern) {
    if (N < 4) return INT_MIN;
    qgt_system_t* sys = qgt_model_qwz(m);
    if (!sys) return INT_MIN;
    qgt_berry_grid_t g;
    int rc = qgt_berry_grid(sys, N, &g);
    if (rc != 0) { qgt_free(sys); return INT_MIN; }
    double c = g.chern;
    if (out_chern) *out_chern = c;
    qgt_berry_grid_free(&g);
    qgt_free(sys);
    return (int)lround(c);
}

double moonlab_dmrg_tfim_energy(uint32_t num_sites, double g,
                                 uint32_t max_bond_dim, uint32_t num_sweeps) {
    if (num_sites < 2 || max_bond_dim < 1 || num_sweeps == 0) return DBL_MAX;

    dmrg_config_t cfg = dmrg_config_default();
    cfg.max_bond_dim = max_bond_dim;
    cfg.max_sweeps   = (int)num_sweeps;

    dmrg_result_t* dmrg_res = NULL;
    tn_mps_state_t* mps = dmrg_tfim_ground_state(num_sites, g, &cfg, &dmrg_res);
    if (!mps) return DBL_MAX;

    double energy = DBL_MAX;
    if (dmrg_res) {
        energy = dmrg_res->ground_energy;
        dmrg_result_free(dmrg_res);
    } else {
        mpo_t* mpo = mpo_tfim_create(num_sites, /*J=*/1.0, g);
        if (mpo) {
            energy = dmrg_compute_energy(mps, mpo);
            mpo_free(mpo);
        }
    }
    tn_mps_free(mps);
    return energy;
}

double moonlab_dmrg_heisenberg_energy(uint32_t num_sites,
                                       double J, double Delta, double h,
                                       uint32_t max_bond_dim,
                                       uint32_t num_sweeps) {
    if (num_sites < 2 || max_bond_dim < 1 || num_sweeps == 0) return DBL_MAX;

    mpo_t* mpo = mpo_heisenberg_create(num_sites, J, Delta, h);
    if (!mpo) return DBL_MAX;

    tn_state_config_t mps_cfg = tn_state_config_default();
    mps_cfg.max_bond_dim = max_bond_dim;
    uint32_t chi_init = (max_bond_dim > 8) ? 8 : max_bond_dim;
    tn_mps_state_t* mps = dmrg_init_random_mps(num_sites, chi_init, &mps_cfg);
    if (!mps) {
        mpo_free(mpo);
        return DBL_MAX;
    }

    dmrg_config_t cfg = dmrg_config_default();
    cfg.max_bond_dim = max_bond_dim;
    cfg.max_sweeps   = (int)num_sweeps;

    double energy = DBL_MAX;
    dmrg_result_t* res = dmrg_ground_state(mps, mpo, &cfg);
    if (res) {
        energy = res->ground_energy;
        dmrg_result_free(res);
    } else {
        energy = dmrg_compute_energy(mps, mpo);
    }

    tn_mps_free(mps);
    mpo_free(mpo);
    return energy;
}

/* ================================================================== */
/*  Variational-D ABI wrappers (since 0.2.1).                          */
/* ================================================================== */

int moonlab_ca_mps_var_d_run_v2(moonlab_ca_mps_t* state,
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
                                 double convergence_eps,
                                 double* out_final_energy) {
    if (!state || !paulis || !coeffs || num_terms == 0) {
        return CA_MPS_ERR_INVALID;
    }
    ca_mps_warmstart_t ws = CA_MPS_WARMSTART_IDENTITY;
    switch (warmstart) {
        case 1: ws = CA_MPS_WARMSTART_H_ALL; break;
        case 2: ws = CA_MPS_WARMSTART_DUAL_TFIM; break;
        case 3: ws = CA_MPS_WARMSTART_FERRO_TFIM; break;
        case 4: ws = CA_MPS_WARMSTART_STABILIZER_SUBGROUP; break;
        default: ws = CA_MPS_WARMSTART_IDENTITY; break;
    }
    ca_mps_var_d_alt_config_t cfg = ca_mps_var_d_alt_config_default();
    cfg.max_outer_iters             = (int)max_outer_iters;
    cfg.imag_time_dtau              = imag_time_dtau;
    cfg.imag_time_steps_per_outer   = (int)imag_time_steps_per_outer;
    cfg.clifford_passes_per_outer   = (int)clifford_passes_per_outer;
    cfg.composite_2gate             = composite_2gate;
    cfg.warmstart                   = ws;
    cfg.warmstart_stab_paulis       = stab_paulis;
    cfg.warmstart_stab_num_gens     = stab_num_gens;
    cfg.verbose                     = 0;
    if (convergence_eps > 0.0) {
        cfg.convergence_eps = convergence_eps;
    }

    ca_mps_var_d_alt_result_t res = {0};
    ca_mps_error_t e = moonlab_ca_mps_optimize_var_d_alternating(
        state, paulis, coeffs, num_terms, &cfg, &res);
    if (out_final_energy) *out_final_energy = res.final_energy;
    return (int)e;
}

int moonlab_ca_mps_var_d_run(moonlab_ca_mps_t* state,
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
                              double* out_final_energy) {
    return moonlab_ca_mps_var_d_run_v2(state, paulis, coeffs, num_terms,
                                        max_outer_iters, imag_time_dtau,
                                        imag_time_steps_per_outer,
                                        clifford_passes_per_outer,
                                        composite_2gate, warmstart,
                                        stab_paulis, stab_num_gens,
                                        /*convergence_eps*/ 0.0,
                                        out_final_energy);
}

int moonlab_ca_mps_gauge_warmstart(moonlab_ca_mps_t* state,
                                     const uint8_t* paulis,
                                     uint32_t num_gens) {
    return (int)moonlab_ca_mps_apply_stab_subgroup_warmstart(state,
                                                               paulis,
                                                               num_gens);
}

/* ================================================================== */
/*  1+1D Z2 LGT ABI wrappers (since 0.2.1).                            */
/* ================================================================== */

int moonlab_z2_lgt_1d_build(uint32_t num_matter_sites,
                              double t_hop, double h_link,
                              double mass, double gauss_penalty,
                              uint8_t** out_paulis,
                              double** out_coeffs,
                              uint32_t* out_num_terms,
                              uint32_t* out_num_qubits) {
    z2_lgt_config_t cfg = {
        .num_matter_sites = num_matter_sites,
        .t_hop            = t_hop,
        .h_link           = h_link,
        .mass             = mass,
        .gauss_penalty    = gauss_penalty
    };
    return z2_lgt_1d_build_pauli_sum(&cfg, out_paulis, out_coeffs,
                                       out_num_terms, out_num_qubits);
}

int moonlab_z2_lgt_1d_gauss_law(uint32_t num_matter_sites,
                                  uint32_t site_x,
                                  uint8_t* out_pauli) {
    z2_lgt_config_t cfg = { .num_matter_sites = num_matter_sites };
    return z2_lgt_1d_gauss_law_pauli(&cfg, site_x, out_pauli);
}

/* ================================================================== */
/*  Diagnostic stringifier (since 0.2.1).                              */
/* ================================================================== */

const char* moonlab_status_string(int module, int status) {
    return moonlab_status_to_string((moonlab_status_module_t)module,
                                      (moonlab_status_t)status);
}
