/**
 * @file topological.h
 * @brief Anyon models, fusion trees, braiding, and stabilizer codes.
 *
 * OVERVIEW
 * --------
 * Topological quantum computation (TQC), proposed by Kitaev (2003)
 * and reviewed in Nayak, Simon, Stern, Freedman and Das Sarma (2008),
 * encodes quantum information in *non-local* degrees of freedom of
 * anyonic systems: the ground-space multiplicity of a system of
 * localised anyons depends on their global topology (braid type,
 * fusion channel), not on local perturbations.  Gates are
 * implemented by adiabatically *braiding* anyon worldlines in
 * spacetime; the resulting unitary depends only on the braid group
 * element, which makes the computation intrinsically protected
 * against local noise.
 *
 * Two anyon models are implemented:
 *
 *   - @em Fibonacci anyons.  The fusion rule @f$\tau \times \tau =
 *     1 + \tau@f$ generates a universal gate set via braiding alone
 *     (Freedman-Kitaev-Larsen-Wang proved that braid-group
 *     representations of this model are computationally universal).
 *     Single-qubit gates require braid words of specific lengths
 *     acting on the fusion-tree Hilbert space (dimension 2 in the
 *     anyon sector with total charge @f$\tau@f$).  The canonical
 *     braid-based approximation of standard gates is due to
 *     Bonesteel, Hormozi, Zikos and Simon (2005).
 *   - @em Ising anyons.  Fusion rule @f$\sigma\times\sigma = 1 +
 *     \psi@f$, with braiding realising the Clifford group (but NOT
 *     universal on its own without a magic-state supply).  The F
 *     and R matrices are the standard ones reviewed in Nayak et al.
 *     sections III.B-C.
 *
 * IMPLEMENTATION STATUS
 * ---------------------
 *   - The F- and R-symbol tables (get_F_symbol, get_R_symbol) of every built-in
 *     model are @em verified: anyon_verify_coherence() reports the maximum
 *     violation of the MacLane pentagon equation, both hexagon equations and
 *     F-matrix unitarity, and it is ~1e-15 for Fibonacci, Ising and SU(2)_k.
 *   - braid_anyons() is a @em faithful unitary representation of the Artin
 *     braid group @f$B_n@f$ on the fusion-path basis (see FUSION-TREE BASIS
 *     below).  Yang-Baxter @f$\sigma_i\sigma_{i+1}\sigma_i =
 *     \sigma_{i+1}\sigma_i\sigma_{i+1}@f$ and far commutation
 *     @f$\sigma_i\sigma_j = \sigma_j\sigma_i, |i-j|\ge 2@f$ hold to ~1e-16 for
 *     Fibonacci, Ising and SU(2)_k; every generator is unitary and
 *     @f$\sigma_i\sigma_i^{-1} = I@f$ to ~1e-16.  Asserted by
 *     `unit_topological`.
 *   - apply_F_move() implements the change of fusion basis at a vertex using
 *     the tabulated F-symbols.
 *   - Ising braiding realises the single-qubit Clifford group @em exactly:
 *     ising_compile_clifford() returns a braid word whose logical action equals
 *     the requested Clifford to ~1e-16 (no approximation anywhere).
 *   - Fibonacci braiding is dense but not exhaustive in PSU(2).  Two answers
 *     are provided: fibonacci_compile_su2() is a Solovay-Kitaev compiler that
 *     takes a caller-supplied @f$\epsilon@f$ and returns a braid word whose
 *     @em measured distance to the target is below it, and
 *     fibonacci_exact_phase_gate() returns exact (~1e-16) words for the
 *     gates that are exactly realisable.  Which gates those are is settled,
 *     not guessed -- see EXACT REALISABILITY below.
 *   - Topological charge measurement (anyon_measure_pair_charge) and
 *     measurement-only braiding (anyon_forced_measurement_braid) reproduce the
 *     braid transformations exactly without moving anyons.
 *
 * FUSION-TREE BASIS
 * -----------------
 * A fusion_tree_t over external charges @f$a_0 \ldots a_{n-1}@f$ with total
 * charge @f$Q@f$ carries the state of the anyons in the @em standard
 * (left-linear / "comb") fusion basis:
 *
 * @verbatim
 *      a0   a1   a2   a3        vertex v fuses e_{v-1} x a_v -> e_v
 *       \   /    /    /         e_0 := a_0,  e_{n-1} := Q
 *        \ /    /    /
 *      e_1 \   /    /           basis label of a path:
 *           \ /    /                (e_1, e_2, ..., e_{n-1})
 *         e_2 \   /
 *              \ /
 *             e_3 = Q
 * @endverbatim
 *
 * Each basis vector is one admissible tuple @f$(e_1,\ldots,e_{n-1})@f$, i.e.
 * @f$N^{e_v}_{e_{v-1} a_v} \ne 0@f$ at every vertex and @f$e_{n-1} = Q@f$.
 * fusion_tree_t::labels stores those tuples, one row of num_vertices entries
 * per path, in lexicographic order; fusion_tree_t::amplitudes[p] is the
 * amplitude of row p.  All built-in models are multiplicity free
 * (@f$N^c_{ab}\in\{0,1\}@f$), so a path is fully specified by its edge labels.
 *
 * Braiding @f$\sigma_i@f$ (braid_anyons at @p position = i) exchanges @f$a_i@f$
 * and @f$a_{i+1}@f$.  For i = 0 the pair meets at vertex 1 and the action is
 * diagonal, @f$R^{a_0 a_1}_{e_1}@f$ per path.  For i >= 1 the pair does not
 * meet at a vertex of the standard tree, so the operator is the F-conjugated
 * R-matrix
 * @f[
 *   [\sigma_i]_{e'_i e_i} = \sum_f \overline{[F^{e_{i-1} a_{i+1} a_i}_{e_{i+1}}]_{e'_i f}}
 *                            \; R^{a_i a_{i+1}}_f \;
 *                            [F^{e_{i-1} a_i a_{i+1}}_{e_{i+1}}]_{e_i f},
 * @f]
 * which is what makes @f$\sigma_1,\sigma_2,\sigma_3,\ldots@f$ genuinely
 * different operators.
 *
 * EXACT REALISABILITY (Fibonacci)
 * -------------------------------
 * The image @f$G = \langle \sigma_1,\sigma_2\rangle@f$ of the braid group in
 * U(2) is countable and dense in PSU(2), so most targets are reachable only in
 * the limit.  Which ones are reachable exactly is decidable here because every
 * @f$B \in G@f$ has the form
 * @f$\begin{pmatrix} p & \varphi^{-1/2} r\\ \varphi^{-1/2} s & t\end{pmatrix}@f$
 * with @f$p,r,s,t \in K=\mathbb{Q}(\zeta_5)@f$ (immediate from
 * @f$\sigma_1 = \mathrm{diag}(e^{4\pi i/5}, e^{-3\pi i/5})@f$,
 * @f$F = \bigl(\begin{smallmatrix}\varphi^{-1} & \varphi^{-1/2}\\
 * \varphi^{-1/2} & -\varphi^{-1}\end{smallmatrix}\bigr)@f$, and closure of that
 * shape under multiplication).  Consequences, each proved in
 * tests/unit/test_topological.c and MATH.md:
 *
 *   - @b Exact: @f$\sigma_1^k@f$ gives every @f$R_z(m\pi/5)@f$, m = 0..9, in
 *     particular the logical Pauli @f$Z = \sigma_1^5@f$.  Also every conjugate
 *     of those by a braid word.
 *   - @b Impossible: no braid word is proportional to H, X or T.
 *     H would force @f$\varphi^{-1/2} = p/r \in K@f$, false since
 *     @f$\mathbb{Q}(\sqrt\varphi)@f$ is not CM while @f$K@f$ is.
 *     X would force @f$r^2 = \varphi\mu@f$ for a 10th root of unity @f$\mu@f$;
 *     applying the Galois element @f$\sqrt5 \mapsto -\sqrt5@f$ to
 *     @f$r\bar r = \varphi@f$ gives @f$|\sigma(r)|^2 = \varphi' < 0@f$.
 *     T would force @f$\mathrm{tr}^2/\det = 2+\sqrt2 \in K@f$, but the only
 *     quadratic subfield of @f$\mathbb{Q}(\zeta_5)@f$ is
 *     @f$\mathbb{Q}(\sqrt5)@f$.
 *
 * Measurement-only protocols (Bonderson, Freedman and Nayak, PRL 101, 010501
 * (2008)) reproduce braid transformations exactly -- see
 * anyon_forced_measurement_braid() -- so charge measurement is at least as
 * powerful as braiding.  The proofs above concern finite braid words; no exact
 * adaptive construction for the Fibonacci H, X or T is known, and none is
 * implemented here.  For those three the answer is fibonacci_compile_su2()'s
 * checked @f$\epsilon@f$ guarantee, not a fixed-fidelity gate.
 *
 * Surface and toric codes complement the anyon models by encoding
 * logical qubits in the ground space of a commuting stabilizer
 * Hamiltonian.  Kitaev (2003) introduced the toric code; Fowler,
 * Mariantoni, Martinis and Cleland (2012) gave the canonical
 * treatment of the planar surface code and its error-correction
 * protocol, both of which this module implements (see the
 * `surface_code_t` and `surface_code_clifford_t` sections below for
 * dense-state and Clifford-tableau variants respectively).
 *
 * REFERENCES
 * ----------
 *  - A. Yu. Kitaev, "Fault-tolerant quantum computation by anyons",
 *    Ann. Phys. 303, 2 (2003), arXiv:quant-ph/9707021.  Foundational
 *    TQC paper; toric-code + anyon-braiding framework.
 *  - C. Nayak, S. H. Simon, A. Stern, M. Freedman and S. Das Sarma,
 *    "Non-Abelian Anyons and Topological Quantum Computation",
 *    Rev. Mod. Phys. 80, 1083 (2008), arXiv:0707.1889.  Canonical
 *    review; F-symbols, R-symbols, fusion trees, braid group
 *    representations.
 *  - N. E. Bonesteel, L. Hormozi, G. Zikos and S. H. Simon, "Braid
 *    Topologies for Quantum Computation", Phys. Rev. Lett. 95,
 *    140503 (2005), arXiv:quant-ph/0505065.  Explicit braid-word
 *    compilation of single-qubit gates for Fibonacci anyons.
 *  - A. G. Fowler, M. Mariantoni, J. M. Martinis and A. N. Cleland,
 *    "Surface codes: Towards practical large-scale quantum
 *    computation", Phys. Rev. A 86, 032324 (2012), arXiv:1208.0928.
 *    Canonical reference for the planar surface code.
 *
 * @stability evolving
 * @since v0.1.2
 */

#ifndef TOPOLOGICAL_H
#define TOPOLOGICAL_H

#include "../../quantum/state.h"
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>
#include "../../applications/moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif


// ============================================================================
// ANYON MODELS
// ============================================================================

/**
 * @brief Anyon type enumeration
 */
typedef enum {
    ANYON_MODEL_FIBONACCI,  // Fibonacci anyons (τ×τ = 1+τ)
    ANYON_MODEL_ISING,      // Ising anyons (σ×σ = 1+ψ)
    ANYON_MODEL_SU2_K       // SU(2)_k anyons
} anyon_model_t;

/**
 * @brief Anyon charge labels
 * For Fibonacci: 1 (vacuum), τ (tau)
 * For Ising: 1 (vacuum), σ (sigma), ψ (psi)
 */
typedef uint32_t anyon_charge_t;

// Fibonacci anyon charges
#define FIB_VACUUM 0
#define FIB_TAU    1

// Ising anyon charges
#define ISING_VACUUM 0
#define ISING_SIGMA  1
#define ISING_PSI    2

/**
 * @brief Anyon model specification
 */
typedef struct {
    anyon_model_t type;
    uint32_t num_charges;           // Number of distinct charges
    uint32_t level;                 // Level k for SU(2)_k
    double complex **F_matrices;    // F-symbols (6j-symbols)
    double complex **R_matrices;    // R-symbols (braiding phases)
    uint32_t ***fusion_rules;       // N^c_{ab} fusion multiplicities
} anyon_system_t;

/**
 * @brief Initialize Fibonacci anyon system
 *
 * Fibonacci anyons have fusion rule τ×τ = 1+τ and are universal
 * for quantum computation via braiding alone.
 *
 * @return Fibonacci anyon system
 * @stability evolving
 */
MOONLAB_API anyon_system_t *anyon_system_fibonacci(void);

/**
 * @brief Initialize Ising anyon system
 *
 * Ising anyons have fusion rules:
 * σ×σ = 1+ψ, σ×ψ = σ, ψ×ψ = 1
 *
 * @return Ising anyon system
 * @stability evolving
 */
MOONLAB_API anyon_system_t *anyon_system_ising(void);

/**
 * @brief Initialize SU(2)_k anyon system
 *
 * Returns the genuine SU(2)_k model on k+1 charges labelled by 2j = 0..k, with
 * F/R symbols generated from the quantum 6j-symbols.  k=2 is the Ising model
 * (3 charges: 1, σ, ψ) and is what anyon_system_ising() wraps.
 *
 * k=3 is @em not Fibonacci: SU(2)_3 has four charges (2j = 0,1,2,3).  Fibonacci
 * is the even-integer-spin subcategory of SU(2)_3 and is a separate hand-coded
 * model; use anyon_system_fibonacci() for it.
 *
 * @param k Level parameter (k >= 1); k=2 gives Ising
 * @return SU(2)_k anyon system
 * @stability evolving
 */
MOONLAB_API anyon_system_t *anyon_system_su2k(uint32_t k);

/**
 * @brief Free anyon system
 * @stability evolving
 */
MOONLAB_API void anyon_system_free(anyon_system_t *sys);

/**
 * @brief Get quantum dimension of anyon charge
 *
 * For Fibonacci: d_1 = 1, d_τ = φ (golden ratio)
 * For Ising: d_1 = 1, d_σ = √2, d_ψ = 1
 *
 * @param sys Anyon system
 * @param charge Anyon charge
 * @return Quantum dimension
 * @stability evolving
 */
MOONLAB_API double anyon_quantum_dimension(const anyon_system_t *sys, anyon_charge_t charge);

/**
 * @brief Get total quantum dimension D = √(Σ d_a²)
 * @stability evolving
 */
MOONLAB_API double anyon_total_dimension(const anyon_system_t *sys);

// ============================================================================
// FUSION TREES
// ============================================================================

/**
 * @brief Fusion tree node
 *
 * Represents the fusion of two anyons into a third:
 * a × b → c (with multiplicity N^c_{ab})
 */
typedef struct fusion_node {
    anyon_charge_t left;         // Left incoming charge
    anyon_charge_t right;        // Right incoming charge
    anyon_charge_t result;       // Outgoing fused charge
    struct fusion_node *parent;  // Parent in tree
    struct fusion_node *left_child;
    struct fusion_node *right_child;
} fusion_node_t;

/**
 * @brief Sentinel for fusion_tree_t::recoupled_vertex: standard (comb) basis.
 */
#define FUSION_TREE_STANDARD_BASIS 0xFFFFFFFFu

/**
 * @brief Fusion tree state
 *
 * A fusion tree represents a specific way of fusing n anyons
 * to obtain a total charge. The state is a superposition over
 * valid intermediate fusion outcomes.
 *
 * The basis is the standard left-linear one documented in the FUSION-TREE
 * BASIS section at the top of this header: vertex v (v = 1..n-1) fuses
 * @f$e_{v-1}\times a_v \to e_v@f$ with @f$e_0 = a_0@f$ and @f$e_{n-1} = Q@f$.
 * ::labels holds one row of ::num_vertices edge charges per path, in
 * lexicographic order, and row p is the basis vector whose amplitude is
 * ::amplitudes[p].  Row p's entry j is @f$e_{j+1}@f$.
 *
 * apply_F_move() can put the tree into the basis recoupled at one vertex; then
 * ::recoupled_vertex names that vertex and its column of ::labels holds the
 * recoupled channel f of @f$(a_v \times a_{v+1}) \to f@f$ instead of
 * @f$e_v@f$.  Otherwise ::recoupled_vertex is #FUSION_TREE_STANDARD_BASIS.
 */
typedef struct {
    anyon_system_t *anyon_sys;     // Anyon model
    anyon_charge_t *external;       // External (physical) anyon charges
    uint32_t num_anyons;            // Number of external anyons
    anyon_charge_t total_charge;    // Total fused charge
    fusion_node_t *root;            // Root of fusion tree
    double complex *amplitudes;     // Amplitudes for each fusion path
    uint32_t num_paths;             // Number of valid fusion paths
    anyon_charge_t *labels;         // num_paths x num_vertices internal charges
    uint32_t num_vertices;          // num_anyons - 1 (row stride of `labels`)
    uint32_t recoupled_vertex;      // FUSION_TREE_STANDARD_BASIS, or the vertex
                                    // currently carrying a recoupled label
} fusion_tree_t;

/**
 * @brief Create fusion tree from external charges
 *
 * Enumerates all valid fusion paths and initializes amplitudes.
 *
 * @param sys Anyon system
 * @param charges External anyon charges
 * @param num_anyons Number of anyons
 * @param total_charge Required total charge
 * @return Fusion tree state
 * @stability evolving
 */
MOONLAB_API fusion_tree_t *fusion_tree_create(anyon_system_t *sys,
                                   const anyon_charge_t *charges,
                                   uint32_t num_anyons,
                                   anyon_charge_t total_charge);

/**
 * @brief Free fusion tree
 * @stability evolving
 */
MOONLAB_API void fusion_tree_free(fusion_tree_t *tree);

/**
 * @brief Count fusion paths
 *
 * Returns the dimension of the fusion space for given charges.
 *
 * @param sys Anyon system
 * @param charges External charges
 * @param num_anyons Number of anyons
 * @param total_charge Total charge
 * @return Number of distinct fusion paths
 * @stability evolving
 */
MOONLAB_API uint32_t fusion_count_paths(const anyon_system_t *sys,
                            const anyon_charge_t *charges,
                            uint32_t num_anyons,
                            anyon_charge_t total_charge);

/**
 * @brief Edge labels of one fusion path.
 *
 * @param tree Fusion tree
 * @param path Path index (< tree->num_paths)
 * @return Pointer to tree->num_vertices charges, or NULL on error.  The
 *         storage belongs to @p tree.
 * @stability evolving
 */
MOONLAB_API const anyon_charge_t *fusion_tree_path_labels(const fusion_tree_t *tree,
                                                          uint32_t path);

/**
 * @brief Index of the path with the given edge labels.
 *
 * @param tree Fusion tree
 * @param labels tree->num_vertices charges
 * @return Path index, or -1 if the label tuple is not admissible.
 * @stability evolving
 */
MOONLAB_API int32_t fusion_tree_find_path(const fusion_tree_t *tree,
                                          const anyon_charge_t *labels);

/**
 * @brief Set the tree to a single basis state (amplitude 1 on @p path).
 * @stability evolving
 */
MOONLAB_API qs_error_t fusion_tree_set_basis_state(fusion_tree_t *tree, uint32_t path);

// ============================================================================
// BRAIDING OPERATIONS
// ============================================================================

/**
 * @brief Braid two adjacent anyons: the braid generator @f$\sigma_i@f$.
 *
 * Exchanges the anyons at @p position and @p position + 1 and applies the
 * corresponding unitary to tree->amplitudes in place: the R-matrix phase of
 * each fusion path's own intermediate charge, together with the F-matrix basis
 * change needed when the exchanged pair does not meet at a vertex of the
 * standard tree.  See FUSION-TREE BASIS at the top of this header for the
 * explicit matrix.
 *
 * The map @f$i \mapsto \sigma_i@f$ is a unitary representation of the Artin
 * braid group @f$B_n@f$: Yang-Baxter and far commutation hold to ~1e-16, and
 * @f$\sigma_i \sigma_i^{-1} = I@f$ exactly.  Total charge and norm are
 * preserved.
 *
 * @param tree Fusion tree (modified in place); must be in the standard basis
 * @param position Position of left anyon to braid (< num_anyons - 1)
 * @param clockwise Direction of braid (true = σ, false = σ⁻¹)
 * @return QS_SUCCESS, QS_ERROR_INVALID_QUBIT if @p position is out of range,
 *         QS_ERROR_INVALID_STATE if an F-move is outstanding on @p tree
 * @stability evolving
 */
MOONLAB_API qs_error_t braid_anyons(fusion_tree_t *tree, uint32_t position, bool clockwise);

/**
 * @brief Apply an F-move (change of fusion basis) at a vertex.
 *
 * Changes the fusion order at @p vertex, (a×b)×c ↔ a×(b×c): the subtree
 * @f$((e_{v-1}\, a_v)_{e_v}\, a_{v+1})_{e_{v+1}}@f$ becomes
 * @f$(e_{v-1}\, (a_v\, a_{v+1})_f)_{e_{v+1}}@f$, and the amplitudes are
 * transformed by @f$[F^{e_{v-1} a_v a_{v+1}}_{e_{v+1}}]_{e_v f}@f$.  The
 * column of tree->labels at @p vertex then holds f instead of @f$e_v@f$ and
 * tree->recoupled_vertex is set to @p vertex.
 *
 * Calling it again on the same vertex applies @f$F^\dagger@f$ and returns the
 * tree to the standard basis, so it is an involution.  Only one vertex may be
 * recoupled at a time.  This is the basis change that lets braids of anyons
 * that are not adjacent in the tree be composed from F- and R-moves.
 *
 * @param tree Fusion tree (modified in place)
 * @param vertex Vertex to apply the F-move at, 1 <= vertex <= num_anyons - 2
 * @return QS_SUCCESS, QS_ERROR_INVALID_PARAM if @p vertex is out of range,
 *         QS_ERROR_INVALID_STATE if a *different* vertex is already recoupled
 * @stability evolving
 */
MOONLAB_API qs_error_t apply_F_move(fusion_tree_t *tree, uint32_t vertex);

/**
 * @brief Measure the total topological charge of an adjacent anyon pair.
 *
 * Projects the state onto the sector in which the anyons at @p position and
 * @p position + 1 fuse to @p outcome, renormalising if the projection is
 * nonzero.  This is the exact projector built from the F-symbols: the tree is
 * recoupled so that the pair meets at a vertex, the rows with a different
 * channel are annihilated, and the tree is recoupled back.
 *
 * @param tree Fusion tree (modified in place if the outcome is possible)
 * @param position Left anyon of the pair (< num_anyons - 1)
 * @param outcome Charge to project onto
 * @return Probability of the outcome in [0,1], or -1.0 on argument error.  The
 *         state is left untouched when the probability is 0.
 * @stability evolving
 */
MOONLAB_API double anyon_measure_pair_charge(fusion_tree_t *tree, uint32_t position,
                                             anyon_charge_t outcome);

/**
 * @brief Probability distribution of the total charge of an adjacent pair.
 *
 * Non-destructive: @p out receives one probability per charge of the model and
 * the state is unchanged.
 *
 * @param tree Fusion tree
 * @param position Left anyon of the pair (< num_anyons - 1)
 * @param out Array of at least tree->anyon_sys->num_charges doubles
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t anyon_pair_charge_distribution(const fusion_tree_t *tree,
                                                      uint32_t position,
                                                      double *out);

/**
 * @brief Measurement-only realisation of a braid generator (forced measurement).
 *
 * Reproduces braid_anyons(@p tree, @p position, @p clockwise) exactly, using
 * only topological charge measurements of anyon pairs plus the R-phases that
 * the measurement outcomes herald -- no anyon is transported.  This is the
 * forced-measurement construction of Bonderson, Freedman and Nayak, PRL 101,
 * 010501 (2008): the braid transformation is decomposed in the eigenbasis of
 * the pair charge, which topological charge measurement resolves exactly.
 *
 * The resulting amplitudes agree with braid_anyons() to ~1e-16.  Because it is
 * a decomposition of the same unitary, it inherits the same exactly-realisable
 * set -- measurement adds no unitary that braiding cannot already produce.
 *
 * @param tree Fusion tree (modified in place)
 * @param position Position of left anyon (< num_anyons - 1)
 * @param clockwise Direction of the braid being simulated
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t anyon_forced_measurement_braid(fusion_tree_t *tree,
                                                      uint32_t position,
                                                      bool clockwise);

/**
 * @brief Get F-matrix element
 *
 * F^{abc}_d[e,f] relates different fusion orderings:
 * (a×b→e)×c→d ↔ a×(b×c→f)→d
 *
 * @param sys Anyon system
 * @param a,b,c,d External charges
 * @param e,f Intermediate channels
 * @return F-matrix element
 * @stability evolving
 */
MOONLAB_API double complex get_F_symbol(const anyon_system_t *sys,
                            anyon_charge_t a, anyon_charge_t b,
                            anyon_charge_t c, anyon_charge_t d,
                            anyon_charge_t e, anyon_charge_t f);

/**
 * @brief Get R-matrix element
 *
 * R^{ab}_c is the phase acquired when exchanging a and b
 * that fuse to c.
 *
 * @param sys Anyon system
 * @param a,b Exchanged charges
 * @param c Fusion outcome
 * @return R-matrix element (phase)
 * @stability evolving
 */
MOONLAB_API double complex get_R_symbol(const anyon_system_t *sys,
                            anyon_charge_t a, anyon_charge_t b,
                            anyon_charge_t c);

/**
 * @brief Maximum residual of the fusion-category coherence conditions.
 *
 * Returns the largest absolute violation, over all charge configurations, of
 *   - the MacLane pentagon equation for the F-symbols,
 *   - both hexagon equations relating the F- and R-symbols, and
 *   - unitarity of every F-matrix.
 * A consistent braided fusion category (anyon model) returns ~0 (machine
 * precision); a nonzero value certifies that the tabulated F/R symbols are not
 * mutually consistent.  Intended as an exact verification / regression guard
 * for the built-in models (Fibonacci, Ising, SU(2)_k).
 *
 * @param sys Anyon system
 * @return max coherence residual (>= 0), or -1 on error
 * @stability evolving
 */
MOONLAB_API double anyon_verify_coherence(const anyon_system_t *sys);

// ============================================================================
// BRAID WORDS AND BRAID-WORD COMPILATION
// ============================================================================

/**
 * @brief One braid generator: @f$\sigma_{position}^{\pm1}@f$.
 */
typedef struct {
    uint32_t position;   // left anyon of the exchanged pair
    uint8_t  clockwise;  // 1 = sigma, 0 = sigma^{-1}
} braid_gen_t;

/**
 * @brief A word in the braid generators, applied left to right.
 */
typedef struct {
    braid_gen_t *gens;
    uint32_t length;
    uint32_t capacity;
} braid_word_t;

/**
 * @brief Allocate an empty braid word.
 *
 * Zero length, zero capacity, no generator storage yet; the first
 * ::braid_word_append reserves 16 generators and doubles from there.
 *
 * @return Owned braid word, or NULL on allocation failure.
 * @stability evolving
 */
MOONLAB_API braid_word_t *braid_word_create(void);

/**
 * @brief Release a braid word and its generator array.
 *
 * @param w Word to release; NULL is a no-op.
 * @stability evolving
 */
MOONLAB_API void braid_word_free(braid_word_t *w);

/**
 * @brief Append one generator @f$\sigma_{position}^{\pm1}@f$ to the word.
 *
 * Grows the generator array by doubling when it is full.
 *
 * @param w         Word to extend.
 * @param position  Index of the left anyon of the exchanged pair.
 * @param clockwise true for @f$\sigma_{position}@f$, false for
 *                  @f$\sigma_{position}^{-1}@f$.
 * @return QS_SUCCESS; QS_ERROR_INVALID_STATE if @p w is NULL;
 *         QS_ERROR_OUT_OF_MEMORY if the array cannot grow.
 * @stability evolving
 */
MOONLAB_API qs_error_t braid_word_append(braid_word_t *w, uint32_t position, bool clockwise);

/**
 * @brief Concatenate @p src onto the end of @p dst.
 *
 * Copies @p src's generators in order, so the result applies @p dst
 * first and then @p src.  @p src is not modified and may be reused;
 * appending a word to itself is not supported.
 *
 * @param dst Word to extend.
 * @param src Word to append.
 * @return QS_SUCCESS; QS_ERROR_INVALID_STATE if either pointer is NULL;
 *         QS_ERROR_OUT_OF_MEMORY if @p dst cannot grow.
 * @stability evolving
 */
MOONLAB_API qs_error_t braid_word_append_word(braid_word_t *dst, const braid_word_t *src);

/**
 * @brief Append the group inverse @f$\mathrm{src}^{-1}@f$ to @p dst.
 *
 * Copies @p src's generators in reverse order with each handedness
 * flipped, so @c dst followed by @c src followed by this appended
 * block reduces to @c dst.  @p src is not modified.
 *
 * @param dst Word to extend.
 * @param src Word whose inverse is appended.
 * @return QS_SUCCESS; QS_ERROR_INVALID_STATE if either pointer is NULL;
 *         QS_ERROR_OUT_OF_MEMORY if @p dst cannot grow.
 * @stability evolving
 */
MOONLAB_API qs_error_t braid_word_append_inverse(braid_word_t *dst, const braid_word_t *src);
/**
 * @brief Allocate an independent copy of @p w.
 *
 * The clone owns its own generator array, so the two words can be
 * extended or freed independently.
 *
 * @param w Word to copy.
 * @return Owned copy; NULL if @p w is NULL or an allocation fails.
 * @stability evolving
 */
MOONLAB_API braid_word_t *braid_word_clone(const braid_word_t *w);

/**
 * @brief Number of generators currently in the word.
 *
 * This is the braid length, not the allocated capacity.
 *
 * @param w Word to query.
 * @return Generator count; 0 if @p w is NULL.
 * @stability evolving
 */
MOONLAB_API uint32_t braid_word_length(const braid_word_t *w);

/**
 * @brief Free reduction: cancel adjacent @f$\sigma_i\sigma_i^{-1}@f$ pairs.
 * @return the reduced length
 * @stability evolving
 */
MOONLAB_API uint32_t braid_word_reduce(braid_word_t *w);

/**
 * @brief Apply a braid word to a fusion tree, generator by generator.
 * @stability evolving
 */
MOONLAB_API qs_error_t braid_word_apply(const braid_word_t *w, fusion_tree_t *tree);

/**
 * @brief Matrix of a braid word on the fusion space of the given anyons.
 *
 * Column j of @p out is the image of basis path j, so
 * @f$out[i \cdot d + j] = \langle i | W | j\rangle@f$ with d the number of
 * fusion paths.  @p out must hold at least d*d complex values; call with
 * @p out = NULL to query d.
 *
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t braid_word_matrix(const braid_word_t *w,
                                         anyon_system_t *sys,
                                         const anyon_charge_t *charges,
                                         uint32_t num_anyons,
                                         anyon_charge_t total_charge,
                                         double complex *out,
                                         uint32_t *out_dim);

/**
 * @brief Distance between two 2x2 unitaries, modulo global phase.
 *
 * @f$d(U,V) = \min_\phi \|U - e^{i\phi} V\|_{op} = \sqrt{2 - |\mathrm{tr}(U^\dagger V)|}@f$
 * after normalising both to SU(2).  0 iff U and V agree up to a phase, and at
 * most 2.  This is the metric every epsilon in this header is measured in.
 * @stability evolving
 */
MOONLAB_API double su2_projective_distance(const double complex a[4],
                                           const double complex b[4]);

/**
 * @brief Exact single-qubit Clifford braid word for Ising anyons.
 *
 * The Ising qubit is 4 σ anyons of total charge 1 (dimension 2, no leakage).
 * The image of @f$\langle\sigma_1,\sigma_2\rangle@f$ in PSU(2) is exactly the
 * 24-element single-qubit Clifford group, so every Clifford has an @em exact
 * braid word.  This returns the shortest one found by breadth-first search of
 * the group's Cayley graph, and the achieved distance is ~1e-16, not an
 * approximation.
 *
 * @param sys Ising anyon system (ANYON_MODEL_ISING, or SU(2)_2)
 * @param target 2x2 target unitary, row-major; must be Clifford up to phase
 * @param achieved_error Receives the measured su2_projective_distance (optional)
 * @return Braid word on positions {0,1} of a 4-anyon tree, or NULL if the
 *         target is not a single-qubit Clifford
 * @stability evolving
 */
MOONLAB_API braid_word_t *ising_compile_clifford(anyon_system_t *sys,
                                                 const double complex target[4],
                                                 double *achieved_error);

/**
 * @brief Exact two-qubit Clifford braid word for Ising anyons.
 *
 * Uses the @em dense encoding: 6 σ anyons of total charge 1 carry exactly two
 * qubits (dimension 4), so unlike the 4-anyons-per-qubit register there is no
 * subspace to leak out of -- the whole fusion space is computational.  Qubit 0
 * is the charge of the pair (0,1) and qubit 1 the charge of the pair (2,3),
 * with 1 -> |0> and ψ -> |1>; basis order is |q0 q1> = 00, 01, 10, 11.
 *
 * The image of @f$\langle\sigma_1..\sigma_5\rangle@f$ in PU(4) is finite and
 * is enumerated exhaustively, so a reachable target -- CNOT among them -- gets
 * an @em exact braid word (~1e-16), and an unreachable one gets NULL rather
 * than a silent approximation.
 *
 * @param sys Ising anyon system
 * @param target 4x4 target unitary, row-major
 * @param achieved_error Receives the measured distance (optional)
 * @return Braid word on positions 0..4 of a 6-anyon tree, or NULL
 * @stability evolving
 */
MOONLAB_API braid_word_t *ising_compile_clifford2(anyon_system_t *sys,
                                                  const double complex target[16],
                                                  double *achieved_error);

/**
 * @brief Order of the projective image of the Ising braid group.
 *
 * @param sys Ising anyon system
 * @param num_anyons 4 (one qubit) or 6 (dense two-qubit encoding)
 * @return Number of distinct elements up to global phase, or 0 on error
 * @stability evolving
 */
MOONLAB_API uint32_t ising_braid_group_order(anyon_system_t *sys, uint32_t num_anyons);

/**
 * @brief Solovay-Kitaev braid-word compiler for Fibonacci anyons.
 *
 * The Fibonacci qubit here is 3 τ anyons of total charge τ (dimension 2), with
 * generators @f$\sigma_1,\sigma_2@f$ at positions 0 and 1.  Given any 2x2
 * unitary and any @p epsilon, returns a braid word whose logical action is
 * within @p epsilon of the target in su2_projective_distance -- the distance
 * is *measured* on the returned word before it is handed back, so the bound is
 * a guarantee and not an asymptotic statement.  Word length grows
 * polylogarithmically in 1/epsilon (~@f$\log^{3.97}(1/\epsilon)@f$).
 *
 * The first call builds the base approximation net (a few seconds); it is
 * cached for the process lifetime.
 *
 * @param sys Fibonacci anyon system
 * @param target 2x2 target unitary, row-major
 * @param epsilon Requested accuracy (> 0)
 * @param achieved_error Receives the measured distance (optional)
 * @return Braid word, or NULL if @p epsilon could not be met or on error
 * @stability evolving
 */
MOONLAB_API braid_word_t *fibonacci_compile_su2(anyon_system_t *sys,
                                                const double complex target[4],
                                                double epsilon,
                                                double *achieved_error);

/**
 * @brief Exact Fibonacci phase gate @f$R_z(m\pi/5)@f$, m = 0..9.
 *
 * @f$\sigma_1 = \mathrm{diag}(R^{\tau\tau}_1, R^{\tau\tau}_\tau)@f$ is, up to
 * phase, @f$\mathrm{diag}(1, e^{3\pi i/5})@f$, and 3 is invertible mod 10, so
 * @f$\langle\sigma_1\rangle@f$ is exactly the order-10 group of
 * @f$\mathrm{diag}(1, e^{i m\pi/5})@f$.  m = 5 is the logical Pauli Z.  These
 * are the exactly realisable diagonal gates; H, X and T provably are not (see
 * EXACT REALISABILITY at the top of this header).
 *
 * @param m Phase index, 0..9
 * @return Braid word @f$\sigma_1^{k}@f$ with 3k ≡ m (mod 10), or NULL
 * @stability evolving
 */
MOONLAB_API braid_word_t *fibonacci_exact_phase_gate(uint32_t m);

/**
 * @brief Size and covering radius of the Solovay-Kitaev base net.
 *
 * Builds the net if it is not built yet.  The covering radius is the worst
 * su2_projective_distance from a sampled SU(2) element to the net, i.e. the
 * @f$\epsilon_0@f$ the recursion starts from; the recursion converges when
 * @f$c^2\epsilon_0 < 1@f$.
 *
 * @param sys Fibonacci anyon system
 * @param covering_radius Receives the measured covering radius (optional)
 * @return Number of net elements, or 0 on error
 * @stability evolving
 */
MOONLAB_API uint32_t fibonacci_braid_net_size(anyon_system_t *sys,
                                              double *covering_radius);

// ============================================================================
// ANYONIC QUANTUM GATES
// ============================================================================

/**
 * @brief Anyonic qubit encoding
 *
 * For Fibonacci anyons, a qubit is encoded in 4 anyons
 * with total charge 1: |0⟩ ~ (τ,τ)→1, |1⟩ ~ (τ,τ)→τ
 */
typedef struct {
    fusion_tree_t *tree;
    anyon_system_t *sys;
    uint32_t num_logical_qubits;
} anyonic_register_t;

/**
 * @brief Create anyonic qubit register
 *
 * @param sys Anyon system
 * @param num_qubits Number of logical qubits
 * @return Anyonic register
 * @stability evolving
 */
MOONLAB_API anyonic_register_t *anyonic_register_create(anyon_system_t *sys,
                                             uint32_t num_qubits);

/**
 * @brief Free anyonic register
 * @stability evolving
 */
MOONLAB_API void anyonic_register_free(anyonic_register_t *reg);

/* ----------------------------------------------------------------------------
 * ANYONIC GATES.
 *
 * Every gate below is a braid word applied to the register's fusion tree by
 * braid_anyons(), so each is a genuine unitary on the encoded qubit.  Qubit q
 * occupies anyons 4q..4q+3 and its logical bit is the charge @f$e_{4q+1}@f$ of
 * the pair (4q, 4q+1); braids inside a block preserve every other block's
 * charge, so the gates act as U (x) I on the register.
 *
 * How exact a gate is depends on the model, and the difference is not glossed:
 *
 *   - Ising: X, Z, H and every other single-qubit Clifford are EXACT
 *     (~1e-16), compiled by ising_compile_clifford().  T is not a Clifford and
 *     is therefore not reachable by Ising braiding at all -- anyonic_T_gate()
 *     returns QS_ERROR_NOT_SUPPORTED on an Ising register rather than
 *     pretending otherwise.
 *   - Fibonacci: Z is EXACT (@f$\sigma_1^5@f$, see fibonacci_exact_phase_gate);
 *     X, H and T are provably not exactly realisable by any finite braid (see
 *     EXACT REALISABILITY at the top of this header) and are compiled by
 *     fibonacci_compile_su2() to a caller-specified epsilon, with the achieved
 *     error measured, guaranteed below epsilon, and reported.
 * ------------------------------------------------------------------------- */

/** @brief Default accuracy for the anyonic gates that take no epsilon. */
#define ANYONIC_GATE_DEFAULT_EPSILON 1e-6


/**
 * @brief Apply a logical NOT (Pauli X) via braiding.
 *
 * Ising: exact Clifford braid word (~1e-16).  Fibonacci: compiled to
 * #ANYONIC_GATE_DEFAULT_EPSILON; use anyonic_apply_unitary() to choose
 * another epsilon.
 *
 * @param reg Anyonic register
 * @param qubit Target qubit
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t anyonic_not(anyonic_register_t *reg, uint32_t qubit);

/**
 * @brief Apply a Hadamard via braiding
 *
 * Ising: exact (H is Clifford).  Fibonacci: compiled to
 * #ANYONIC_GATE_DEFAULT_EPSILON -- exactly realisable Fibonacci braid words
 * for H do not exist, and the reason is a proof, not a limitation of this
 * implementation.
 *
 * @param reg Anyonic register
 * @param qubit Target qubit
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t anyonic_hadamard(anyonic_register_t *reg, uint32_t qubit);

/**
 * @brief Apply a T gate (π/8 phase gate) via braiding.
 *
 * Fibonacci only: T is compiled by Solovay-Kitaev to within @p precision, and
 * the returned state is guaranteed to be that of a braid word whose measured
 * distance to T is below @p precision.  Ising braiding generates only the
 * Clifford group, which does not contain T, so an Ising register returns
 * QS_ERROR_NOT_SUPPORTED.
 *
 * @param reg Anyonic register
 * @param qubit Target qubit
 * @param precision Requested accuracy (> 0); pass 0 for the default
 * @return QS_SUCCESS, QS_ERROR_NOT_SUPPORTED for Ising, or error
 * @stability evolving
 */
MOONLAB_API qs_error_t anyonic_T_gate(anyonic_register_t *reg, uint32_t qubit,
                          double precision);

/**
 * @brief Apply an arbitrary single-qubit unitary via braiding.
 *
 * Compiles @p target with ising_compile_clifford() or fibonacci_compile_su2()
 * as appropriate and applies the resulting braid word.
 *
 * @param reg Anyonic register
 * @param qubit Target qubit
 * @param target 2x2 unitary, row-major
 * @param epsilon Requested accuracy (> 0); pass 0 for the default
 * @param achieved Receives the measured distance actually attained (optional)
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t anyonic_apply_unitary(anyonic_register_t *reg, uint32_t qubit,
                                             const double complex target[4],
                                             double epsilon, double *achieved);

/**
 * @brief Apply a two-qubit entangling gate by weaving between blocks.
 *
 * A unitary inter-qubit weave in the geometry of Bonesteel, Hormozi, Zikos and
 * Simon (PRL 95, 140503 (2005)).  It entangles: from the product state
 * @f$|{+}0\rangle@f$ it produces concurrence 0.414242.
 *
 * @warning It also leaks 8.3043e-2 of the amplitude out of the logical
 * subspace, and that is unavoidable rather than a defect of this particular
 * weave.  Qubit q's logical bit is the charge of the pair (4q, 4q+1), and the
 * block's vacuum constraint forces the pair (4q+2, 4q+3) to carry the same
 * charge; a braid that entangles two blocks must change one of those pair
 * charges without changing its partner, taking the block out of the vacuum
 * channel.  An exhaustive search over every braid word of length <= 8 on the
 * two-block register finds no word that is both entangling and leakage-free,
 * and none whose logical block is proportional to a unitary while entangling.
 * For an @em exact, leakage-free two-qubit gate use ising_compile_clifford2(),
 * whose dense 6-anyon encoding has no non-computational subspace.
 *
 * Requires adjacent qubits (qubit2 == qubit1 + 1) and a Fibonacci register.
 *
 * @param reg Anyonic register
 * @param qubit1 Control qubit
 * @param qubit2 Target qubit (must be qubit1 + 1)
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t anyonic_entangle(anyonic_register_t *reg,
                            uint32_t qubit1, uint32_t qubit2);

/**
 * @brief Logical 2x2 matrix of the encoded qubit's state (for verification).
 *
 * Writes the amplitudes of the register's logical basis states into @p out,
 * which must hold 2^num_logical_qubits complex values, and returns the total
 * probability that remains inside the logical subspace (1 - leakage).
 *
 * @param reg Anyonic register
 * @param out Receives the logical amplitudes (optional)
 * @return Probability in the logical subspace, or -1.0 on error
 * @stability evolving
 */
MOONLAB_API double anyonic_register_logical_state(const anyonic_register_t *reg,
                                                  double complex *out);

// ============================================================================
// SURFACE CODE
// ============================================================================

/**
 * @brief Surface code lattice
 *
 * Implements the 2D surface code on a square lattice with
 * distance d (d×d data qubits, (d-1)² syndrome qubits).
 */
typedef struct {
    uint32_t distance;           // Code distance
    uint32_t num_data_qubits;    // d²
    uint32_t num_ancilla_qubits; // (d-1)² for each type (X and Z)
    quantum_state_t *state;      // Full quantum state
    uint8_t *x_syndrome;         // X-type syndrome measurements
    uint8_t *z_syndrome;         // Z-type syndrome measurements
} surface_code_t;

/**
 * @brief Create surface code
 *
 * @param distance Code distance (odd, ≥3)
 * @return Surface code structure
 * @stability evolving
 */
MOONLAB_API surface_code_t *surface_code_create(uint32_t distance);

/**
 * @brief Free surface code
 * @stability evolving
 */
MOONLAB_API void surface_code_free(surface_code_t *code);

/**
 * @brief Initialize surface code in logical |0⟩
 *
 * @param code Surface code
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_init_logical_zero(surface_code_t *code);

/**
 * @brief Initialize surface code in logical |+⟩
 *
 * @param code Surface code
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_init_logical_plus(surface_code_t *code);

/**
 * @brief Apply logical X gate
 *
 * String of X operators along a path from left to right edge.
 *
 * @param code Surface code
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_logical_X(surface_code_t *code);

/**
 * @brief Apply logical Z gate
 *
 * String of Z operators along a path from top to bottom edge.
 *
 * @param code Surface code
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_logical_Z(surface_code_t *code);

/**
 * @brief Measure X-type stabilizers
 *
 * Measures all face (plaquette) stabilizers.
 *
 * @param code Surface code (syndrome updated)
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_measure_X_stabilizers(surface_code_t *code);

/**
 * @brief Measure Z-type stabilizers
 *
 * Measures all vertex (star) stabilizers.
 *
 * @param code Surface code (syndrome updated)
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_measure_Z_stabilizers(surface_code_t *code);

/**
 * @brief Apply single-qubit error
 *
 * @param code Surface code
 * @param qubit Data qubit index
 * @param error_type 'X', 'Y', or 'Z'
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_apply_error(surface_code_t *code,
                                     uint32_t qubit, char error_type);

/**
 * @brief Decode syndrome and apply correction
 *
 * Uses minimum weight perfect matching decoder.
 *
 * @param code Surface code (corrected in place)
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_decode_correct(surface_code_t *code);

// ============================================================================
// SURFACE CODE (Clifford-backed)
// ============================================================================

/**
 * @brief Surface code simulated on the Aaronson-Gottesman tableau.
 *
 * The 2D surface code (Kitaev 2003; Fowler-Mariantoni-Martinis-Cleland
 * 2012) is a stabilizer code that encodes one logical qubit into a
 * planar array of @f$d \times d@f$ physical data qubits protected by
 * @f$(d-1)^2@f$ X-type and @f$(d-1)^2@f$ Z-type stabilizer generators.
 * Each plaquette stabilizer is a four-body operator (two-body on the
 * boundary).  Logical operators are strings spanning the lattice; the
 * code distance @f$d@f$ equals the minimum support of a non-trivial
 * logical operator.
 *
 * Because every stabilizer and every gate in the surface-code syndrome-
 * extraction circuit is Clifford, the entire protocol can be simulated
 * in polynomial time on the Aaronson-Gottesman tableau.  This variant
 * -- unlike the dense `surface_code_t` in the same header, which is
 * capped near @f$d = 5@f$ by statevector memory -- scales to arbitrary
 * distance; @f$d = 15@f$ uses 617 qubits and runs comfortably on a
 * laptop.  Syndrome extraction is ancilla-mediated (one ancilla per
 * stabilizer), matching the usual fault-tolerant protocol layout so the
 * simulation reflects the same gate count a hardware implementation
 * would execute.
 *
 * @verbatim
 *   qubits [0 .. d^2 - 1]                          = data qubits
 *   qubits [d^2 .. d^2 + (d-1)^2 - 1]              = Z-syndrome ancillas
 *   qubits [d^2 + (d-1)^2 .. d^2 + 2(d-1)^2 - 1]   = X-syndrome ancillas
 * @endverbatim
 *
 * The initial state is @f$|0\rangle^{\otimes N}@f$, which is already a
 * +1 eigenstate of every Z-stabilizer; X-stabilizer outcomes start
 * undefined (@f$|0\rangle@f$ has no X-string eigenvalue).  Error
 * detection against X-errors is nevertheless correct: an X injected on
 * a data qubit flips exactly the Z-stabilizers whose support contains
 * that qubit (verified in the d=7, 9, 15 unit tests).  Decoders --
 * minimum-weight perfect matching in practice, e.g. Fowler et al.
 * §VII -- can be layered on top without changes to the tableau.
 *
 * REFERENCES
 * ----------
 *  - A. Yu. Kitaev, "Fault-tolerant quantum computation by anyons",
 *    Ann. Phys. 303, 2 (2003), arXiv:quant-ph/9707021.  Origin of the
 *    toric / surface code.
 *  - A. G. Fowler, M. Mariantoni, J. M. Martinis and A. N. Cleland,
 *    "Surface codes: Towards practical large-scale quantum computation",
 *    Phys. Rev. A 86, 032324 (2012), arXiv:1208.0928.  The canonical
 *    reference for the planar surface code, syndrome extraction
 *    protocol, and the MWPM decoder.
 */
typedef struct {
    uint32_t distance;
    uint32_t num_data_qubits;
    uint32_t num_ancilla_qubits;   /* (d-1)² each for X and Z */
    struct clifford_tableau_t* tableau;
    uint8_t* x_syndrome;           /* (d-1)² bits, row-major over faces */
    uint8_t* z_syndrome;           /* (d-1)² bits, row-major over vertices */
    uint64_t rng_state;
} surface_code_clifford_t;

/**
 * @brief Build a distance-@p distance surface code on a Clifford tableau.
 *
 * Allocates a tableau of @f$d^2 + 2(d-1)^2@f$ qubits in
 * @f$|0\rangle^{\otimes N}@f$ using the layout documented above (data
 * qubits first, then the Z-syndrome ancillas, then the X-syndrome
 * ancillas), plus one syndrome byte per stabilizer of each type.
 *
 * @param distance Code distance @f$d@f$; must be odd and at least 3.
 * @param rng_seed Seed for the splitmix64 stream driving syndrome
 *                 measurement.  0 selects a fixed nonzero default so
 *                 the stream never degenerates.
 * @return Owned code handle, or NULL if @p distance is even or below 3,
 *         or if any allocation fails.
 * @stability evolving
 */
MOONLAB_API surface_code_clifford_t* surface_code_clifford_create(uint32_t distance,
                                                       uint64_t rng_seed);

/**
 * @brief Release a Clifford-backed surface code.
 *
 * Frees the underlying tableau, both syndrome buffers, and the handle.
 *
 * @param code Code handle to release; NULL is a no-op.
 * @stability evolving
 */
MOONLAB_API void surface_code_clifford_free(surface_code_clifford_t* code);

/**
 * Data-qubit index from (row, col).
 * @stability evolving
 */
MOONLAB_API uint32_t surface_code_clifford_data_index(const surface_code_clifford_t* code,
                                          uint32_t row, uint32_t col);

/**
 * Apply a Pauli error to a data qubit. Type is 'X', 'Y' or 'Z'.
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_clifford_apply_error(surface_code_clifford_t* code,
                                             uint32_t data_qubit,
                                             char error_type);

/**
 * Measure all Z-type stabilizers (ZZZZ on four data qubits around each
 * interior vertex). Populates `z_syndrome`. Ancilla-mediated: for each
 * vertex, reset its ancilla to |0⟩, CNOT each data qubit onto it, then
 * measure the ancilla in Z basis.
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_clifford_measure_z_syndromes(surface_code_clifford_t* code);

/**
 * Measure all X-type stabilizers (XXXX on four data qubits around each
 * interior face). Populates `x_syndrome`. Ancilla-mediated: H on
 * ancilla, CNOT(ancilla → each data), H on ancilla, measure.
 * @stability evolving
 */
MOONLAB_API qs_error_t surface_code_clifford_measure_x_syndromes(surface_code_clifford_t* code);

/**
 * Sum of set bits across both syndromes (diagnostic).
 * @stability evolving
 */
MOONLAB_API uint32_t surface_code_clifford_syndrome_weight(const surface_code_clifford_t* code);

// ============================================================================
// TORIC CODE
// ============================================================================

/**
 * @brief Toric code on a torus
 *
 * Similar to surface code but on periodic boundary conditions,
 * encoding 2 logical qubits.
 */
typedef struct {
    uint32_t L;                  // Linear size (L×L torus)
    uint32_t num_qubits;         // 2L² edge qubits
    quantum_state_t *state;      // Full quantum state
    uint8_t *vertex_syndrome;    // A_v eigenvalues
    uint8_t *plaquette_syndrome; // B_p eigenvalues
} toric_code_t;

/**
 * @brief Create toric code
 *
 * @param L Linear size
 * @return Toric code structure
 * @stability evolving
 */
MOONLAB_API toric_code_t *toric_code_create(uint32_t L);

/**
 * @brief Free toric code
 * @stability evolving
 */
MOONLAB_API void toric_code_free(toric_code_t *code);

/**
 * @brief Initialize toric code ground state
 *
 * Projects onto the +1 eigenspace of all stabilizers.
 *
 * @param code Toric code
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t toric_code_init_ground_state(toric_code_t *code);

/**
 * @brief Create anyon pair
 *
 * Creates an e-anyon (vertex) or m-anyon (plaquette) pair
 * by applying a string of Pauli operators.
 *
 * @param code Toric code
 * @param type 'e' for electric (Z-string), 'm' for magnetic (X-string)
 * @param x1,y1 Start position
 * @param x2,y2 End position
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t toric_code_create_anyon_pair(toric_code_t *code,
                                         char type,
                                         uint32_t x1, uint32_t y1,
                                         uint32_t x2, uint32_t y2);

/**
 * @brief Move anyon
 *
 * @param code Toric code
 * @param type Anyon type
 * @param from_x,from_y Current position
 * @param to_x,to_y New position
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t toric_code_move_anyon(toric_code_t *code, char type,
                                  uint32_t from_x, uint32_t from_y,
                                  uint32_t to_x, uint32_t to_y);

/**
 * @brief Braid anyons in toric code
 *
 * @param code Toric code
 * @param anyon1_x,anyon1_y First anyon position
 * @param anyon2_x,anyon2_y Second anyon position
 * @return QS_SUCCESS or error
 * @stability evolving
 */
MOONLAB_API qs_error_t toric_code_braid(toric_code_t *code,
                            uint32_t anyon1_x, uint32_t anyon1_y,
                            uint32_t anyon2_x, uint32_t anyon2_y);

// ============================================================================
// TOPOLOGICAL ENTANGLEMENT ENTROPY
// ============================================================================

/**
 * @brief Compute topological entanglement entropy
 *
 * S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
 * where A, B, C are regions forming an annulus.
 *
 * For topologically ordered states, S_topo = log(D) where
 * D is the total quantum dimension.
 *
 * @param state Quantum state
 * @param region_A Qubits in region A
 * @param num_A Size of region A
 * @param region_B Qubits in region B
 * @param num_B Size of region B
 * @param region_C Qubits in region C
 * @param num_C Size of region C
 * @return Topological entanglement entropy
 * @stability evolving
 */
MOONLAB_API double topological_entanglement_entropy(const quantum_state_t *state,
                                         const uint32_t *region_A, uint32_t num_A,
                                         const uint32_t *region_B, uint32_t num_B,
                                         const uint32_t *region_C, uint32_t num_C);

/**
 * @brief Compute Kitaev-Preskill topological entropy
 *
 * Alternative formula using disk and annulus regions.
 *
 * @param state Quantum state
 * @param center_qubits Central disk qubits
 * @param num_center Number of center qubits
 * @param ring_qubits Surrounding ring qubits
 * @param num_ring Number of ring qubits
 * @return Topological entropy γ = log(D)
 * @stability evolving
 */
MOONLAB_API double kitaev_preskill_entropy(const quantum_state_t *state,
                                const uint32_t *center_qubits, uint32_t num_center,
                                const uint32_t *ring_qubits, uint32_t num_ring);

// ============================================================================
// MODULAR S AND T MATRICES
// ============================================================================

/**
 * @brief Compute modular S-matrix
 *
 * S_{ab} = (1/D) Σ_c N^c_{ab} d_c e^{2πi θ_c}
 * where θ_c is the topological spin.
 *
 * @param sys Anyon system
 * @param S_matrix Output S-matrix (num_charges × num_charges)
 * @stability evolving
 */
MOONLAB_API void compute_modular_S_matrix(const anyon_system_t *sys,
                               double complex *S_matrix);

/**
 * @brief Compute modular T-matrix
 *
 * T_{ab} = δ_{ab} e^{2πi θ_a}
 *
 * @param sys Anyon system
 * @param T_matrix Output T-matrix (num_charges × num_charges)
 * @stability evolving
 */
MOONLAB_API void compute_modular_T_matrix(const anyon_system_t *sys,
                               double complex *T_matrix);

/**
 * @brief Compute topological spin
 *
 * θ_a = e^{2πi h_a} where h_a is the conformal weight.
 *
 * @param sys Anyon system
 * @param charge Anyon charge
 * @return Topological spin e^{2πi θ}
 * @stability evolving
 */
MOONLAB_API double complex topological_spin(const anyon_system_t *sys,
                                 anyon_charge_t charge);

#ifdef __cplusplus
}
#endif

#endif /* TOPOLOGICAL_H */
