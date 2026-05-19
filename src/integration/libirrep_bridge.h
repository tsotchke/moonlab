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

/* ============================================================================
 * Sector-ED: spin-1/2 Heisenberg on a 2D lattice with Sz conservation.
 *
 * The full 2^N Hilbert space blows up past N = 14 for dense diagonalisation
 * via `mpo_to_matrix + zheev_`.  libirrep's space-group + rep-table machinery
 * factors out translation symmetry, restricting Lanczos to the orbit-
 * representative basis at fixed `popcount = N/2 - sz_total_2x/2`.  The
 * resulting sector dim scales like `C(N, N/2) / |G|` instead of `2^N`,
 * which makes N = 18 / 24 / 27 ground-state ED a workstation problem.
 * ========================================================================= */

/** @brief Lattice geometry handled by @ref moonlab_libirrep_heisenberg_sector_e0.
 *  Numeric values match `irrep_lattice_kind_t` so callers can use them
 *  interchangeably without re-numbering.  Confirmed at libirrep 1.5.0. */
typedef enum {
    MOONLAB_LIBIRREP_LATTICE_SQUARE     = 0,
    MOONLAB_LIBIRREP_LATTICE_TRIANGULAR = 1,
    MOONLAB_LIBIRREP_LATTICE_HONEYCOMB  = 2,
    MOONLAB_LIBIRREP_LATTICE_KAGOME     = 3
} moonlab_libirrep_lattice_kind_t;

/** @brief Wallpaper-group choice for the symmetry projection.
 *  P1 (translation-only) is always available; the higher-symmetry choices
 *  require `Lx == Ly` (square / kagome) or specific cluster shapes.  When
 *  unsure, pass `P1` -- it gives the strongest sector compression that
 *  still keeps the ground state in the (Gamma, A1) sector for AFM
 *  Heisenberg on bipartite + frustrated 2D lattices. */
typedef enum {
    MOONLAB_LIBIRREP_WALLPAPER_P1    = 0,  /**< Translation only. */
    MOONLAB_LIBIRREP_WALLPAPER_P6MM  = 1,  /**< Full C_6v (kagome / triangular). */
    MOONLAB_LIBIRREP_WALLPAPER_P4MM  = 2,  /**< Full C_4v (square). */
    MOONLAB_LIBIRREP_WALLPAPER_P3M1  = 3,  /**< Hex 3-fold + 3 vertex mirrors. */
    MOONLAB_LIBIRREP_WALLPAPER_P2    = 4,  /**< 2-fold rotation only. */
    MOONLAB_LIBIRREP_WALLPAPER_P6    = 5,  /**< Chiral hex: 6 rotations, no mirrors. */
    MOONLAB_LIBIRREP_WALLPAPER_P4    = 6,  /**< Chiral square: 4 rotations, no mirrors. */
    MOONLAB_LIBIRREP_WALLPAPER_P31M  = 7   /**< Hex 3-fold + 3 edge mirrors. */
} moonlab_libirrep_wallpaper_t;

/**
 * @brief  Low-eigenvalues of the spin-1/2 AFM Heisenberg model on a
 *         lattice cluster, restricted to the totally-symmetric (Gamma, A1)
 *         sector at fixed total-Sz.
 *
 * Builds the lattice, the space group, the rep table at the requested
 * popcount, the Heisenberg Hamiltonian, and runs full-reorth Lanczos on
 * the orbit-representative basis using `irrep_heisenberg_apply_in_sector`
 * as the matvec.  Unlocks N > 14 (the dense-ED brute-force ceiling).
 *
 * Concrete dim reductions at the AFM-friendly Sz = 0 sector:
 *
 * | N  | popcount | full 2^N  | C(N, N/2)  | (Gamma, A1) cluster size |
 * |----|----------|-----------|------------|--------------------------|
 * | 12 |    6     |    4 096  |       924  | tens (kagome 2x2)        |
 * | 18 |    9     |  262 144  |    48 620  | hundreds (kagome 3x2)    |
 * | 24 |   12     | 16 777 216|  2 704 156 | thousands (kagome 4x2)   |
 * | 27 |   13.5   |  ~1.3e8   |  ~2.0e7    | ~190k (kagome 3x3)       |
 *
 * @param[in]  lattice_kind   See @ref moonlab_libirrep_lattice_kind_t.
 * @param[in]  Lx, Ly         Cluster unit-cell count along each axis.
 *                            Total site count `N = sites_per_cell * Lx * Ly`
 *                            (e.g. kagome 2x2 -> 12, 3x2 -> 18, 4x2 -> 24).
 * @param[in]  wallpaper      Symmetry-group choice.  Pass
 *                            `MOONLAB_LIBIRREP_WALLPAPER_P1` for "any
 *                            cluster shape, translation-only".  Higher
 *                            symmetries require `Lx == Ly` (square / kagome).
 * @param[in]  sz_total_2x    Twice the target total Sz.  Pass `0` for the
 *                            Sz = 0 (singlet-containing) sector.  Range
 *                            is `[-N, N]` in even steps.
 * @param[in]  k_wanted       Number of lowest eigenvalues to return
 *                            (typically 1 for ground state only).
 * @param[in]  max_iters      Lanczos iteration count.  200 is plenty for
 *                            ground-state convergence on kagome 12 / 18;
 *                            500-1000 for N >= 24.  Memory is
 *                            `max_iters * sector_dim * 16 B` so the
 *                            full-reorth path caps practical N at ~30 on
 *                            workstation memory.
 * @param[out] eigvals_out    Caller-allocated, length `k_wanted`.
 *                            Filled ascending on success; untouched on
 *                            failure.
 * @param[out] sector_dim_out Optional (may be NULL).  Receives the
 *                            actual sector dimension (orbit-rep count)
 *                            on success -- useful for sanity checks +
 *                            memory budgeting.
 * @return  MOONLAB_LIBIRREP_OK on success, MOONLAB_LIBIRREP_NOT_BUILT
 *          when libirrep isn't linked, MOONLAB_LIBIRREP_BAD_ARG when the
 *          inputs are inconsistent (e.g. `sz_total_2x` parity mismatch
 *          with `N`), MOONLAB_LIBIRREP_OOM on allocation failure, or
 *          MOONLAB_LIBIRREP_INTERNAL for libirrep call failures.
 */
int moonlab_libirrep_heisenberg_sector_e0(
    moonlab_libirrep_lattice_kind_t lattice_kind,
    int Lx, int Ly,
    moonlab_libirrep_wallpaper_t    wallpaper,
    int sz_total_2x,
    int k_wanted,
    int max_iters,
    double *eigvals_out,
    long long *sector_dim_out);

/* ============================================================================
 * CSS-code bridge: opaque handle backed by `irrep_css_code_t`.
 *
 * `moonlab_libirrep_qec_t` is a handle to a fully-built CSS code -- physical
 * qubit count, X / Z parity-check matrices, derived logical-qubit count.
 * Phase B of v0.6.1 ships the surface-code constructor; v0.6.2+ adds toric,
 * color, BB qLDPC, hypergraph-product, etc., behind the same opaque type so
 * the JS / Python / Rust bindings shipped in v0.5.12-14 pick them up for
 * free in a follow-on cycle.
 * ========================================================================= */

/** @brief Opaque handle for a CSS code built via libirrep's QEC zoo. */
typedef struct moonlab_libirrep_qec moonlab_libirrep_qec_t;

/** @brief Construct a distance-`d` rotated surface code via
 *         `irrep_surface_init` + `irrep_surface_build`.
 *
 *  The rotated surface code at distance `d` has `n = d^2` physical
 *  qubits, `d^2 - 1` total stabilisers (split evenly into X and Z),
 *  one logical qubit, and code distance `d`.
 *
 *  @param[in]  distance  Code distance; must be >= 2.
 *  @param[out] out       Receives an owning handle on success.  Caller
 *                        must release with @ref moonlab_libirrep_qec_free.
 *  @return     MOONLAB_LIBIRREP_OK on success or a negative status code. */
int moonlab_libirrep_surface_code_new(int distance, moonlab_libirrep_qec_t **out);

/** @brief Construct the Kitaev 2D toric code on an `Lx x Ly` torus.
 *
 *  The toric code at `(Lx, Ly)` has `n = 2 Lx Ly` physical qubits
 *  (one per torus edge), `Lx Ly` X-stabilisers (one per vertex),
 *  `Lx Ly` Z-stabilisers (one per plaquette), `k = 2` logical qubits
 *  (the two homology classes of the torus), and code distance
 *  `d = min(Lx, Ly)`. */
int moonlab_libirrep_toric_code_new(int Lx, int Ly, moonlab_libirrep_qec_t **out);

/** @brief Steane [[7, 1, 3]] color code -- the smallest 2D color code,
 *         self-dual (X-stabilisers = Z-stabilisers in support). */
int moonlab_libirrep_color_steane_new(moonlab_libirrep_qec_t **out);

/** @brief [[15, 7, 3]] Hamming code recast as a CSS code. */
int moonlab_libirrep_color_hamming_15_7_3_new(moonlab_libirrep_qec_t **out);

/** @brief IBM bivariate-bicycle [[72, 12, 6]] qLDPC code.
 *  Bravyi et al. 2024 (Nature 627, 778), Table 3 instance 1. */
int moonlab_libirrep_bb_72_12_6_new(moonlab_libirrep_qec_t **out);

/** @brief IBM bivariate-bicycle [[144, 12, 12]] qLDPC code. */
int moonlab_libirrep_bb_144_12_12_new(moonlab_libirrep_qec_t **out);

/** @brief IBM bivariate-bicycle [[288, 12, 18]] qLDPC code. */
int moonlab_libirrep_bb_288_12_18_new(moonlab_libirrep_qec_t **out);

/** @brief Hypergraph-product CSS code from a `[d, 1, d]` repetition
 *  code on each side.  Yields:
 *    d = 3 -> [[13, 1, 3]]
 *    d = 4 -> [[25, 1, 4]]
 *    d = 5 -> [[41, 1, 5]]
 *  Only these three published instances are supported in v0.6.2;
 *  arbitrary `d` falls back to MOONLAB_LIBIRREP_BAD_ARG. */
int moonlab_libirrep_hgp_repetition_new(int d, moonlab_libirrep_qec_t **out);

/** @brief Release a handle returned by any
 *         `moonlab_libirrep_<family>_code_new` factory.  Safe to call
 *         with `NULL`. */
void moonlab_libirrep_qec_free(moonlab_libirrep_qec_t *q);

/** @brief Number of physical qubits `n`. */
int moonlab_libirrep_qec_n_qubits(const moonlab_libirrep_qec_t *q);

/** @brief Number of X-stabiliser generators. */
int moonlab_libirrep_qec_n_x_stabs(const moonlab_libirrep_qec_t *q);

/** @brief Number of Z-stabiliser generators. */
int moonlab_libirrep_qec_n_z_stabs(const moonlab_libirrep_qec_t *q);

/** @brief Number of logical qubits, `k = n - rank(H_X) - rank(H_Z)`.  */
int moonlab_libirrep_qec_logical_qubits(const moonlab_libirrep_qec_t *q);

/** @brief Code distance (brute-force over the stabiliser group up to
 *         the natural distance bound).  Cached after the first call. */
int moonlab_libirrep_qec_distance(moonlab_libirrep_qec_t *q);

/** @brief Read the qubit supports of an X-stabiliser row into a flat
 *         length-`n_qubits` byte buffer (0 / 1 per qubit).
 *
 *  @param[in]  q         CSS handle.
 *  @param[in]  row       Row index in `[0, n_x_stabs)`.
 *  @param[out] support   Caller-allocated, length `n_qubits`.  Filled
 *                        with 0 / 1 per qubit on success.
 *  @return     MOONLAB_LIBIRREP_OK or a negative status code. */
int moonlab_libirrep_qec_get_x_check_row(const moonlab_libirrep_qec_t *q,
                                         int row, unsigned char *support);

/** @brief Z-row counterpart of @ref moonlab_libirrep_qec_get_x_check_row. */
int moonlab_libirrep_qec_get_z_check_row(const moonlab_libirrep_qec_t *q,
                                         int row, unsigned char *support);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_LIBIRREP_BRIDGE_H */
