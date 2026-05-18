/**
 * @file moonlab_tdvp_export.c
 * @brief Stable-ABI wrappers for the v0.4 adaptive-bond two-site TDVP
 *        engine.
 *
 * Definitions for the `moonlab_tdvp_*` entry points declared in
 * `moonlab_export.h`.  The opaque `moonlab_tdvp_engine_t` bundles the
 * MPS state, the Hamiltonian MPO, the internal `tdvp_engine_t`
 * (including per-bond PID slots when the adaptive controller is
 * enabled), and a step-by-step `tdvp_history_t`.  All Moonlab
 * internals stay private to libquantumsim; downstream consumers
 * (QGTL, language bindings) bind through this surface only.
 *
 * @since 0.4.1
 */

#include "moonlab_export.h"

#include "../algorithms/tensor_network/tn_state.h"
#include "../algorithms/tensor_network/dmrg.h"
#include "../algorithms/tensor_network/tdvp.h"

#include <stdlib.h>
#include <string.h>

struct moonlab_tdvp_engine_t {
    tn_mps_state_t  *mps;
    mpo_t           *mpo;
    tdvp_engine_t   *engine;
    tdvp_history_t  *history;
    double           current_energy;
    double           current_norm;
    uint32_t         current_max_bond_dim;
};

/* Internal: construct an engine from a caller-supplied MPO and the
 * common configuration knobs.  Takes ownership of @p mpo on success
 * (frees it on failure).  Returns NULL on any allocation failure. */
static moonlab_tdvp_engine_t* moonlab_tdvp_engine_from_mpo(
        uint32_t num_sites,
        mpo_t *mpo,
        uint32_t initial_bond_dim,
        uint32_t max_bond_dim,
        double dt,
        int imag_time,
        double adaptive_target_entropy)
{
    if (!mpo || num_sites < 2 || max_bond_dim == 0 || dt <= 0.0) {
        if (mpo) mpo_free(mpo);
        return NULL;
    }

    moonlab_tdvp_engine_t *h =
        (moonlab_tdvp_engine_t *)calloc(1, sizeof(*h));
    if (!h) {
        mpo_free(mpo);
        return NULL;
    }
    h->mpo = mpo;

    /* Random initial MPS: bond width clamped by tn_state_config. */
    tn_state_config_t mps_cfg = tn_state_config_create(max_bond_dim, 1e-10);
    uint32_t chi_init = initial_bond_dim;
    if (chi_init == 0) chi_init = 4;
    if (chi_init > max_bond_dim) chi_init = max_bond_dim;

    h->mps = dmrg_init_random_mps(num_sites, chi_init, &mps_cfg);
    if (!h->mps) {
        moonlab_tdvp_engine_free(h);
        return NULL;
    }

    tdvp_config_t cfg = (adaptive_target_entropy > 0.0)
        ? tdvp_config_adaptive(adaptive_target_entropy)
        : tdvp_config_default();
    cfg.dt = dt;
    cfg.max_bond_dim = max_bond_dim;
    cfg.evolution_type = imag_time ? TDVP_IMAGINARY_TIME : TDVP_REAL_TIME;

    h->engine = tdvp_engine_create(h->mps, h->mpo, &cfg);
    if (!h->engine) {
        moonlab_tdvp_engine_free(h);
        return NULL;
    }

    h->history = tdvp_history_create(64);
    if (!h->history) {
        moonlab_tdvp_engine_free(h);
        return NULL;
    }

    /* Seed cached scalars with the initial state's energy and norm so
     * accessors return meaningful values before the first step. */
    h->current_energy = dmrg_compute_energy(h->mps, h->mpo);
    h->current_norm = tn_mps_norm(h->mps);
    h->current_max_bond_dim = 0;
    for (uint32_t s = 0; s + 1 < num_sites; s++) {
        if (h->mps->bond_dims[s] > h->current_max_bond_dim) {
            h->current_max_bond_dim = h->mps->bond_dims[s];
        }
    }

    return h;
}

moonlab_tdvp_engine_t*
moonlab_tdvp_create_heisenberg(uint32_t num_sites,
                                double J,
                                double Delta,
                                double h_field,
                                uint32_t initial_bond_dim,
                                uint32_t max_bond_dim,
                                double dt,
                                int imag_time,
                                double adaptive_target_entropy)
{
    mpo_t *mpo = mpo_heisenberg_create(num_sites, J, Delta, h_field);
    return moonlab_tdvp_engine_from_mpo(num_sites, mpo,
                                         initial_bond_dim, max_bond_dim,
                                         dt, imag_time,
                                         adaptive_target_entropy);
}

moonlab_tdvp_engine_t*
moonlab_tdvp_create_tfim(uint32_t num_sites,
                          double J,
                          double h_field,
                          uint32_t initial_bond_dim,
                          uint32_t max_bond_dim,
                          double dt,
                          int imag_time,
                          double adaptive_target_entropy)
{
    mpo_t *mpo = mpo_tfim_create(num_sites, J, h_field);
    return moonlab_tdvp_engine_from_mpo(num_sites, mpo,
                                         initial_bond_dim, max_bond_dim,
                                         dt, imag_time,
                                         adaptive_target_entropy);
}

int moonlab_tdvp_step(moonlab_tdvp_engine_t* engine)
{
    if (!engine || !engine->engine) return -1;

    tdvp_result_t result = {0};
    int rc = tdvp_step(engine->engine, &result);
    if (rc == 0) {
        tdvp_history_add(engine->history, &result);
        engine->current_energy = result.energy;
        engine->current_norm = result.norm;
        engine->current_max_bond_dim = result.max_bond_dim;
    }
    tdvp_result_clear(&result);
    return rc;
}

int moonlab_tdvp_evolve_to(moonlab_tdvp_engine_t* engine,
                            double target_time)
{
    if (!engine || !engine->engine) return -1;

    int rc = tdvp_evolve_to(engine->engine, target_time, engine->history);
    if (rc == 0 && engine->history->num_steps > 0) {
        uint32_t last = engine->history->num_steps - 1;
        engine->current_energy = engine->history->energies[last];
        engine->current_norm = engine->history->norms[last];
        /* tdvp_evolve_to does not stage max_bond_dim into history;
         * recompute from the live MPS. */
        engine->current_max_bond_dim = 0;
        for (uint32_t s = 0; s + 1 < engine->mps->num_qubits; s++) {
            if (engine->mps->bond_dims[s] > engine->current_max_bond_dim) {
                engine->current_max_bond_dim = engine->mps->bond_dims[s];
            }
        }
    }
    return rc;
}

double moonlab_tdvp_current_time(const moonlab_tdvp_engine_t* engine)
{
    return (engine && engine->engine) ? tdvp_get_time(engine->engine) : 0.0;
}

double moonlab_tdvp_current_energy(const moonlab_tdvp_engine_t* engine)
{
    return engine ? engine->current_energy : 0.0;
}

double moonlab_tdvp_current_norm(const moonlab_tdvp_engine_t* engine)
{
    return engine ? engine->current_norm : 0.0;
}

uint32_t moonlab_tdvp_current_max_bond_dim(const moonlab_tdvp_engine_t* engine)
{
    return engine ? engine->current_max_bond_dim : 0u;
}

uint32_t moonlab_tdvp_num_bonds(const moonlab_tdvp_engine_t* engine)
{
    if (!engine || !engine->mps || engine->mps->num_qubits == 0) return 0u;
    return engine->mps->num_qubits - 1u;
}

uint32_t moonlab_tdvp_bond_chi(const moonlab_tdvp_engine_t* engine,
                                uint32_t bond)
{
    if (!engine || !engine->engine) return 0u;
    return tdvp_bond_chi(engine->engine, bond);
}

uint32_t moonlab_tdvp_history_num_steps(const moonlab_tdvp_engine_t* engine)
{
    return (engine && engine->history) ? engine->history->num_steps : 0u;
}

int moonlab_tdvp_history_get_step(const moonlab_tdvp_engine_t* engine,
                                    uint32_t step,
                                    double* out_time,
                                    double* out_energy,
                                    double* out_norm)
{
    if (!engine || !engine->history) return -1;
    if (step >= engine->history->num_steps) return -1;
    if (out_time)   *out_time   = engine->history->times[step];
    if (out_energy) *out_energy = engine->history->energies[step];
    if (out_norm)   *out_norm   = engine->history->norms[step];
    return 0;
}

int moonlab_tdvp_history_get_bond_chi(const moonlab_tdvp_engine_t* engine,
                                        uint32_t step,
                                        uint32_t* out_chi,
                                        uint32_t buf_capacity)
{
    if (!engine || !engine->history || !out_chi) return -1;
    if (step >= engine->history->num_steps) return -1;

    uint32_t n = moonlab_tdvp_num_bonds(engine);
    if (buf_capacity < n) return -1;

    if (!engine->history->bond_chi_history || engine->history->n_bonds == 0) {
        /* Legacy fixed-bond path: no per-bond chi recorded. */
        for (uint32_t b = 0; b < n; b++) out_chi[b] = 0u;
        return 0;
    }

    const uint32_t stride = engine->history->n_bonds;
    const uint32_t copy_count = (stride < n) ? stride : n;
    const uint32_t *row = engine->history->bond_chi_history
                        + (size_t)step * stride;
    for (uint32_t b = 0; b < copy_count; b++) out_chi[b] = row[b];
    for (uint32_t b = copy_count; b < n; b++) out_chi[b] = 0u;
    return 0;
}

void moonlab_tdvp_engine_free(moonlab_tdvp_engine_t* engine)
{
    if (!engine) return;
    if (engine->history) tdvp_history_free(engine->history);
    if (engine->engine) tdvp_engine_free(engine->engine);
    if (engine->mps) tn_mps_free(engine->mps);
    if (engine->mpo) mpo_free(engine->mpo);
    free(engine);
}
