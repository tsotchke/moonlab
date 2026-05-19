/**
 * @file    libirrep_bridge.h
 * @brief   Optional bridge between moonlab and libirrep
 *          (https://github.com/tsotchke/libirrep).
 *
 * libirrep ships a production-grade QEC substrate (toric, surface, color,
 * bivariate-bicycle, hypergraph + lifted product, honeycomb + CSS Floquet,
 * 3D toric, X-cube fracton, HaPPY, single-shot, Bacon-Shor, Steane*Steane,
 * BdG-skyrmion), rep-theory primitives (SO(3) / SU(2) / O(3) / SE(3)
 * Clebsch-Gordan + reduction tables), and a verified spin-1/2 Heisenberg
 * sector-ED stack the kagome / triangular / honeycomb benchmarks already
 * use as the ground-truth reference.  Moonlab has historically embedded
 * a hardcoded `LIBIRREP_KAGOME12_E0 = -5.44487522` value lifted from
 * libirrep's `PHYSICS_RESULTS.md`; this bridge replaces that copy-pasted
 * constant with a live call when libirrep is available at build time.
 *
 * @section build  Build-time activation
 *
 * Pass `-DQSIM_ENABLE_LIBIRREP=ON` to CMake.  The CMake glue tries, in
 * order, `find_package(libirrep CONFIG)`, then `pkg-config libirrep`,
 * then `-DQSIM_LIBIRREP_ROOT=<path>` / `$LIBIRREP_ROOT` pointing at a
 * source tree with `build/lib/liblibirrep.{a,dylib,so}` (matching SbNN's
 * vendored-submodule pattern).  When libirrep is found CMake sets
 * `MOONLAB_HAS_LIBIRREP=1` and the real implementation links in;
 * otherwise the bridge compiles to lightweight stubs that return
 * `MOONLAB_LIBIRREP_NOT_BUILT` so consumers can call the API
 * unconditionally.
 *
 * @section status  Status convention
 *
 * Every bridge entry point returns an `int` status: `0` on success and
 * a negative code on failure.  Codes are in the
 * `MOONLAB_STATUS_ERR_MODULE_BASE - 0..` range so they coexist with
 * other moonlab subsystems without collision.
 */

#ifndef MOONLAB_LIBIRREP_BRIDGE_H
#define MOONLAB_LIBIRREP_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Bridge-specific status codes.  Drawn from the
 *         `MOONLAB_STATUS_ERR_MODULE_BASE` band so callers can dispatch
 *         on them without colliding with other modules. */
#define MOONLAB_LIBIRREP_OK              ( 0)
#define MOONLAB_LIBIRREP_NOT_BUILT       (-201) /**< Built without libirrep linkage. */
#define MOONLAB_LIBIRREP_BAD_ARG         (-202) /**< Caller passed a NULL output. */
#define MOONLAB_LIBIRREP_INTERNAL        (-203) /**< libirrep returned a non-OK status. */
#define MOONLAB_LIBIRREP_OOM             (-204) /**< Bridge-side allocation failure. */

/**
 * @brief  Live-compute the ground-state energy of the 12-site kagome
 *         spin-1/2 antiferromagnetic Heisenberg model in libirrep's
 *         `J = 1` spin convention.
 *
 * Builds the (Lx, Ly) = (2, 2) kagome lattice via
 * `irrep_lattice_build(IRREP_LATTICE_KAGOME, 2, 2)` -> 12 sites,
 * harvests the nearest-neighbour bond list via
 * `irrep_lattice_fill_bonds_nn`, constructs
 * `H = J sum_<i,j> S_i.S_j` via `irrep_heisenberg_new` (J = 1), and
 * extracts E_0 with `irrep_lanczos_eigvals_reorth` over the full
 * 4096-dim Hilbert space (no symmetry sector restriction; the same
 * setup `examples/kagome12_ed.c` ships in libirrep).
 *
 * Expected value: E_0 ~ -5.44487522 J (PRB 83, 212401; matches what
 * moonlab's MPO + dense-zheev pipeline reports for the Pauli-operator
 * Heisenberg with `J' = 0.25`).
 *
 * @param[out] out_energy  Receives E_0 on success.  Untouched on failure.
 * @return MOONLAB_LIBIRREP_OK on success, MOONLAB_LIBIRREP_NOT_BUILT
 *         when moonlab was compiled without libirrep, or a negative
 *         status code when a libirrep call fails.
 */
int moonlab_libirrep_kagome12_e0(double *out_energy);

/**
 * @brief  Indicates whether moonlab was compiled with the libirrep
 *         linkage path active.
 *
 * @return `1` if `MOONLAB_HAS_LIBIRREP` was defined during the
 *         compilation of the bridge TU; `0` otherwise.  Callers can
 *         gate "use libirrep when available, else fall back" logic on
 *         this without parsing build configuration.
 */
int moonlab_libirrep_available(void);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_LIBIRREP_BRIDGE_H */
