/*
 * t_dmrg.c -- DMRG / TDVP ill-conditioning and drift.
 *
 *   1. DMRG ground energy of the open-chain TFIM
 *        H = -sum_i Z_i Z_{i+1} - g sum_i X_i
 *      vs a dense exact diagonalization (via the library's own
 *      hermitian_eigen_decomposition) for N in {6,8,10} across
 *      g in {0.5, 1.0 (critical), 1.5, 2.0}.  Near-critical g=1 is the
 *      small-gap / ill-conditioned regime.
 *   2. TDVP real-time evolution: energy must be (approximately) conserved
 *      over many steps and never NaN/blow up.
 *   3. TDVP imaginary-time evolution: energy must stay finite and decrease
 *      toward the ground state over a long run (log_norm_factor accumulation
 *      / underflow probe).
 *
 * Links the prebuilt library.
 */
#include "numerical_common.h"
#include "../../src/utils/matrix_math.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tdvp.h"

/* dense open-chain TFIM ground energy via library eigensolver */
static int tfim_exact_ground(int N, double g, double *out) {
    size_t n = (size_t)1 << N;
    complex_t *H = calloc(n * n, sizeof(complex_t));
    double *ev = calloc(n, sizeof(double));
    complex_t *V = calloc(n * n, sizeof(complex_t));
    if (!H || !ev || !V) { free(H); free(ev); free(V); return -1; }

    /* diagonal: -sum Z_i Z_{i+1} */
    for (size_t b = 0; b < n; b++) {
        double diag = 0.0;
        for (int i = 0; i + 1 < N; i++) {
            int zi = ((b >> i) & 1) ? -1 : 1;
            int zj = ((b >> (i+1)) & 1) ? -1 : 1;
            diag += -(double)(zi * zj);
        }
        H[b*n + b] += diag;
    }
    /* off-diagonal: -g sum X_i (flips bit i) */
    for (size_t b = 0; b < n; b++)
        for (int i = 0; i < N; i++) {
            size_t bf = b ^ ((size_t)1 << i);
            H[b*n + bf] += -g;
        }

    int rc = hermitian_eigen_decomposition(H, n, ev, V, 0, 1e-12);
    if (rc == 0) *out = ev[n - 1]; /* ascending-reversed => last is smallest */
    free(H); free(ev); free(V);
    return rc;
}

static void dmrg_vs_exact(int N, double g) {
    double exact = 0.0;
    if (tfim_exact_ground(N, g, &exact) != 0) { NC_MISS("exact diag failed N=%d g=%.2f", N, g); return; }

    dmrg_config_t cfg = dmrg_config_default();
    cfg.max_bond_dim = 64;
    cfg.max_sweeps = 30;
    cfg.energy_tol = 1e-10;
    cfg.seed = 12345;

    dmrg_result_t *res = NULL;
    tn_mps_state_t *mps = dmrg_tfim_ground_state((uint32_t)N, g, &cfg, &res);
    if (!mps || !res) { NC_MISS("DMRG returned NULL N=%d g=%.2f", N, g); if (res) dmrg_result_free(res); if (mps) tn_mps_free(mps); return; }

    double e = res->ground_energy;
    g_checks++;
    if (nc_is_bad(e)) { NC_MISS("DMRG energy NaN/Inf N=%d g=%.2f", N, g); }
    else {
        double tol = 1e-3 * (fabs(exact) + 1.0);
        if (fabs(e - exact) > tol)
            NC_MISS("DMRG N=%d g=%.2f: E=%.10f exact=%.10f err=%.3e tol=%.3e seed=%llu",
                    N, g, e, exact, fabs(e - exact), tol, (unsigned long long)cfg.seed);
        else
            NC_INFO("DMRG N=%d g=%.2f: E=%.8f exact=%.8f (ok)", N, g, e, exact);
    }
    /* sweep-energy trace must be finite */
    if (res->sweep_energies)
        for (uint32_t s = 0; s < res->num_sweeps; s++)
            nc_finite("DMRG sweep energy", res->sweep_energies[s]);
    nc_finite("DMRG variance", res->energy_variance);
    nc_finite("DMRG truncation_error", res->truncation_error);

    dmrg_result_free(res);
    tn_mps_free(mps);
}

static void tdvp_realtime_drift(int N, double g) {
    mpo_t *mpo = mpo_tfim_create((uint32_t)N, 1.0, g);
    tn_state_config_t sc = tn_state_config_default();
    sc.max_bond_dim = 64;
    tn_mps_state_t *mps = dmrg_init_random_mps((uint32_t)N, 8, &sc);
    if (!mpo || !mps) { NC_INFO("tdvp-real: setup unavailable (skip)"); if (mpo) mpo_free(mpo); if (mps) tn_mps_free(mps); return; }

    tdvp_config_t cfg = tdvp_config_default();
    cfg.evolution_type = TDVP_REAL_TIME;
    cfg.dt = 0.02;
    cfg.max_bond_dim = 64;
    cfg.normalize = true;

    tdvp_engine_t *eng = tdvp_engine_create(mps, mpo, &cfg);
    if (!eng) { NC_INFO("tdvp-real: engine unavailable (skip)"); mpo_free(mpo); tn_mps_free(mps); return; }

    double e0 = dmrg_compute_energy(mps, mpo);
    nc_finite("tdvp-real E0", e0);

    const int nsteps = 200;
    double emin = e0, emax = e0; int nan_seen = 0;
    for (int s = 0; s < nsteps; s++) {
        tdvp_result_t r; memset(&r, 0, sizeof r);
        int rc = tdvp_step(eng, &r);
        if (rc != 0) { NC_INFO("tdvp-real step %d rc=%d (stop)", s, rc); break; }
        if (nc_is_bad(r.energy) || nc_is_bad(r.norm)) { nan_seen = 1; NC_MISS("tdvp-real NaN at step %d (E=%.6g norm=%.6g)", s, r.energy, r.norm); break; }
        if (r.energy < emin) emin = r.energy;
        if (r.energy > emax) emax = r.energy;
        tdvp_result_clear(&r);
    }
    if (!nan_seen) {
        double drift = emax - emin;
        /* real-time energy conservation: lenient bound catches blow-ups,
         * tight enough to flag gross instability over 200 steps. */
        double bound = 0.05 * (fabs(e0) + 1.0);
        g_checks++;
        if (drift > bound)
            NC_MISS("tdvp-real N=%d g=%.2f: energy drift=%.4e over %d steps (bound %.4e, E0=%.4f)",
                    N, g, drift, nsteps, bound, e0);
        else
            NC_INFO("tdvp-real N=%d g=%.2f: drift=%.3e over %d steps (E0=%.4f)", N, g, drift, nsteps, e0);
    }
    tdvp_engine_free(eng);
    mpo_free(mpo);
    tn_mps_free(mps);
}

static void tdvp_imagtime_underflow(int N, double g) {
    mpo_t *mpo = mpo_tfim_create((uint32_t)N, 1.0, g);
    tn_state_config_t sc = tn_state_config_default();
    sc.max_bond_dim = 48;
    tn_mps_state_t *mps = dmrg_init_random_mps((uint32_t)N, 8, &sc);
    if (!mpo || !mps) { NC_INFO("tdvp-imag: setup unavailable (skip)"); if (mpo) mpo_free(mpo); if (mps) tn_mps_free(mps); return; }

    tdvp_config_t cfg = tdvp_config_default();
    cfg.evolution_type = TDVP_IMAGINARY_TIME;
    cfg.dt = 0.05;
    cfg.max_bond_dim = 48;
    cfg.normalize = true;

    tdvp_engine_t *eng = tdvp_engine_create(mps, mpo, &cfg);
    if (!eng) { NC_INFO("tdvp-imag: engine unavailable (skip)"); mpo_free(mpo); tn_mps_free(mps); return; }

    double e_first = 0.0, e_last = 0.0; int have_first = 0, nan_seen = 0;
    const int nsteps = 400; /* long run: log_norm_factor accumulation */
    for (int s = 0; s < nsteps; s++) {
        tdvp_result_t r; memset(&r, 0, sizeof r);
        int rc = tdvp_step(eng, &r);
        if (rc != 0) { NC_INFO("tdvp-imag step %d rc=%d (stop)", s, rc); break; }
        if (nc_is_bad(r.energy) || nc_is_bad(r.norm)) { nan_seen = 1; NC_MISS("tdvp-imag NaN at step %d (E=%.6g norm=%.6g)", s, r.energy, r.norm); break; }
        if (!have_first) { e_first = r.energy; have_first = 1; }
        e_last = r.energy;
        tdvp_result_clear(&r);
    }
    if (have_first && !nan_seen) {
        double nrm = tn_mps_norm(mps);
        nc_finite("tdvp-imag final norm", nrm);
        g_checks++;
        /* imaginary time projects toward the ground state: energy must not
         * INCREASE (allow small numerical slack). */
        if (e_last > e_first + 1e-3 * (fabs(e_first) + 1.0))
            NC_MISS("tdvp-imag N=%d g=%.2f: energy rose %.6f -> %.6f (should decrease)", N, g, e_first, e_last);
        else
            NC_INFO("tdvp-imag N=%d g=%.2f: E %.6f -> %.6f, final norm=%.4g", N, g, e_first, e_last, nrm);
    }
    tdvp_engine_free(eng);
    mpo_free(mpo);
    tn_mps_free(mps);
}

int main(void) {
    nc_begin("dmrg_tdvp");

    int Ns[] = {6, 8, 10};
    double gs[] = {0.5, 1.0, 1.5, 2.0};
    for (size_t ni = 0; ni < 3; ni++)
        for (size_t gi = 0; gi < 4; gi++)
            dmrg_vs_exact(Ns[ni], gs[gi]);

    tdvp_realtime_drift(8, 1.0);
    tdvp_realtime_drift(8, 1.5);
    tdvp_imagtime_underflow(8, 1.0);

    return nc_end();
}
