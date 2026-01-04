/**
 * @file chemistry.c
 * @brief Implementation of quantum chemistry algorithms
 */

#include "chemistry.h"
#include "../../quantum/gates.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// JORDAN-WIGNER TRANSFORMATION
// ============================================================================

jw_operator_t jw_transform_single(fermion_op_t op, uint32_t num_orbitals) {
    jw_operator_t result = {0};

    // a†_j = (1/2)(X_j - iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
    // a_j  = (1/2)(X_j + iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0

    // This produces 2 Pauli strings (X and Y terms)
    result.num_terms = 2;
    result.terms = calloc(2, sizeof(pauli_string_t));
    if (!result.terms) return result;

    for (int t = 0; t < 2; t++) {
        result.terms[t].num_qubits = num_orbitals;
        result.terms[t].ops = calloc(num_orbitals, sizeof(pauli_type_t));
        if (!result.terms[t].ops) {
            jw_operator_free(&result);
            return (jw_operator_t){0};
        }

        // Initialize all to identity
        for (uint32_t q = 0; q < num_orbitals; q++) {
            result.terms[t].ops[q] = PAULI_I;
        }

        // Z string on qubits 0 to j-1
        for (uint32_t q = 0; q < op.orbital; q++) {
            result.terms[t].ops[q] = PAULI_Z;
        }

        // X or Y on qubit j
        if (t == 0) {
            result.terms[t].ops[op.orbital] = PAULI_X;
            result.terms[t].coeff = 0.5;  // (1/2) for X term
        } else {
            result.terms[t].ops[op.orbital] = PAULI_Y;
            // For creation: -i/2, for annihilation: +i/2
            if (op.type == FERMION_CREATE) {
                result.terms[t].coeff = -0.5 * I;
            } else {
                result.terms[t].coeff = 0.5 * I;
            }
        }
    }

    return result;
}

// Multiply two Pauli operators at same site
static pauli_type_t pauli_multiply(pauli_type_t a, pauli_type_t b, double complex *phase) {
    // Pauli multiplication table with phases
    // I*X = X, X*I = X, etc.
    // X*Y = iZ, Y*X = -iZ, etc.

    if (a == PAULI_I) { *phase *= 1; return b; }
    if (b == PAULI_I) { *phase *= 1; return a; }
    if (a == b) { *phase *= 1; return PAULI_I; }

    // Non-trivial cases
    if (a == PAULI_X && b == PAULI_Y) { *phase *= I; return PAULI_Z; }
    if (a == PAULI_Y && b == PAULI_X) { *phase *= -I; return PAULI_Z; }
    if (a == PAULI_Y && b == PAULI_Z) { *phase *= I; return PAULI_X; }
    if (a == PAULI_Z && b == PAULI_Y) { *phase *= -I; return PAULI_X; }
    if (a == PAULI_Z && b == PAULI_X) { *phase *= I; return PAULI_Y; }
    if (a == PAULI_X && b == PAULI_Z) { *phase *= -I; return PAULI_Y; }

    return PAULI_I;  // Should not reach here
}

// Multiply two Pauli strings
static pauli_string_t pauli_string_multiply(const pauli_string_t *a,
                                             const pauli_string_t *b) {
    pauli_string_t result = {0};
    if (a->num_qubits != b->num_qubits) return result;

    result.num_qubits = a->num_qubits;
    result.ops = calloc(result.num_qubits, sizeof(pauli_type_t));
    if (!result.ops) return result;

    result.coeff = a->coeff * b->coeff;

    for (uint32_t q = 0; q < result.num_qubits; q++) {
        result.ops[q] = pauli_multiply(a->ops[q], b->ops[q], &result.coeff);
    }

    return result;
}

jw_operator_t jw_transform_product(const fermion_op_t *ops, uint32_t num_ops,
                                   uint32_t num_orbitals) {
    if (num_ops == 0) {
        jw_operator_t result = {0};
        result.num_terms = 1;
        result.terms = calloc(1, sizeof(pauli_string_t));
        if (result.terms) {
            result.terms[0].num_qubits = num_orbitals;
            result.terms[0].ops = calloc(num_orbitals, sizeof(pauli_type_t));
            result.terms[0].coeff = 1.0;
        }
        return result;
    }

    // Start with first operator
    jw_operator_t result = jw_transform_single(ops[0], num_orbitals);

    // Multiply by remaining operators
    for (uint32_t i = 1; i < num_ops; i++) {
        jw_operator_t next_op = jw_transform_single(ops[i], num_orbitals);

        // Multiply all terms: (Σ P_a)(Σ Q_b) = Σ P_a Q_b
        uint32_t new_num_terms = result.num_terms * next_op.num_terms;
        pauli_string_t *new_terms = calloc(new_num_terms, sizeof(pauli_string_t));

        if (new_terms) {
            uint32_t idx = 0;
            for (uint32_t a = 0; a < result.num_terms; a++) {
                for (uint32_t b = 0; b < next_op.num_terms; b++) {
                    new_terms[idx] = pauli_string_multiply(&result.terms[a],
                                                           &next_op.terms[b]);
                    idx++;
                }
            }
        }

        // Free old terms
        for (uint32_t t = 0; t < result.num_terms; t++) {
            free(result.terms[t].ops);
        }
        free(result.terms);

        result.terms = new_terms;
        result.num_terms = new_num_terms;

        jw_operator_free(&next_op);
    }

    return result;
}

void jw_operator_free(jw_operator_t *op) {
    if (!op) return;
    for (uint32_t i = 0; i < op->num_terms; i++) {
        free(op->terms[i].ops);
    }
    free(op->terms);
    op->terms = NULL;
    op->num_terms = 0;
}

void pauli_string_free(pauli_string_t *ps) {
    if (!ps) return;
    free(ps->ops);
    ps->ops = NULL;
    ps->num_qubits = 0;
}

// ============================================================================
// MOLECULAR HAMILTONIAN
// ============================================================================

molecular_hamiltonian_t *molecular_hamiltonian_create(uint32_t num_orbitals,
                                                       uint32_t num_electrons,
                                                       double nuclear_repulsion) {
    molecular_hamiltonian_t *h = calloc(1, sizeof(molecular_hamiltonian_t));
    if (!h) return NULL;

    h->num_orbitals = num_orbitals;
    h->num_electrons = num_electrons;
    h->nuclear_repulsion = nuclear_repulsion;

    // Allocate space for integrals (will grow as needed)
    h->h1 = NULL;
    h->h2 = NULL;
    h->num_h1 = 0;
    h->num_h2 = 0;

    return h;
}

void molecular_hamiltonian_add_h1(molecular_hamiltonian_t *h,
                                   uint32_t p, uint32_t q, double value) {
    if (!h || fabs(value) < 1e-15) return;

    h->num_h1++;
    h->h1 = realloc(h->h1, h->num_h1 * sizeof(one_electron_integral_t));
    if (h->h1) {
        h->h1[h->num_h1 - 1] = (one_electron_integral_t){p, q, value};
    }
}

void molecular_hamiltonian_add_h2(molecular_hamiltonian_t *h,
                                   uint32_t p, uint32_t q,
                                   uint32_t r, uint32_t s, double value) {
    if (!h || fabs(value) < 1e-15) return;

    h->num_h2++;
    h->h2 = realloc(h->h2, h->num_h2 * sizeof(two_electron_integral_t));
    if (h->h2) {
        h->h2[h->num_h2 - 1] = (two_electron_integral_t){p, q, r, s, value};
    }
}

qubit_hamiltonian_t *molecular_to_qubit_hamiltonian(const molecular_hamiltonian_t *mol_h) {
    if (!mol_h) return NULL;

    qubit_hamiltonian_t *h = calloc(1, sizeof(qubit_hamiltonian_t));
    if (!h) return NULL;

    h->num_qubits = mol_h->num_orbitals;

    // Start with nuclear repulsion as identity term
    h->num_terms = 1;
    h->terms = calloc(1, sizeof(pauli_string_t));
    if (!h->terms) {
        free(h);
        return NULL;
    }
    h->terms[0].num_qubits = h->num_qubits;
    h->terms[0].ops = calloc(h->num_qubits, sizeof(pauli_type_t));
    h->terms[0].coeff = mol_h->nuclear_repulsion;

    // Add one-electron terms: h_pq a†_p a_q
    for (uint32_t i = 0; i < mol_h->num_h1; i++) {
        uint32_t p = mol_h->h1[i].p;
        uint32_t q = mol_h->h1[i].q;
        double val = mol_h->h1[i].value;

        fermion_op_t ops[2] = {
            {FERMION_CREATE, p},
            {FERMION_ANNIHILATE, q}
        };

        jw_operator_t jw = jw_transform_product(ops, 2, mol_h->num_orbitals);

        // Add terms to Hamiltonian
        h->terms = realloc(h->terms, (h->num_terms + jw.num_terms) * sizeof(pauli_string_t));
        for (uint32_t t = 0; t < jw.num_terms; t++) {
            h->terms[h->num_terms + t] = jw.terms[t];
            h->terms[h->num_terms + t].coeff *= val;
        }
        h->num_terms += jw.num_terms;

        free(jw.terms);  // Don't free individual ops as they're now owned by h
    }

    // Add two-electron terms: (1/2) h_pqrs a†_p a†_q a_r a_s
    for (uint32_t i = 0; i < mol_h->num_h2; i++) {
        uint32_t p = mol_h->h2[i].p;
        uint32_t q = mol_h->h2[i].q;
        uint32_t r = mol_h->h2[i].r;
        uint32_t s = mol_h->h2[i].s;
        double val = mol_h->h2[i].value * 0.5;  // Factor of 1/2

        fermion_op_t ops[4] = {
            {FERMION_CREATE, p},
            {FERMION_CREATE, q},
            {FERMION_ANNIHILATE, r},
            {FERMION_ANNIHILATE, s}
        };

        jw_operator_t jw = jw_transform_product(ops, 4, mol_h->num_orbitals);

        h->terms = realloc(h->terms, (h->num_terms + jw.num_terms) * sizeof(pauli_string_t));
        for (uint32_t t = 0; t < jw.num_terms; t++) {
            h->terms[h->num_terms + t] = jw.terms[t];
            h->terms[h->num_terms + t].coeff *= val;
        }
        h->num_terms += jw.num_terms;

        free(jw.terms);
    }

    return h;
}

void molecular_hamiltonian_free(molecular_hamiltonian_t *h) {
    if (!h) return;
    free(h->h1);
    free(h->h2);
    free(h);
}

void qubit_hamiltonian_free(qubit_hamiltonian_t *h) {
    if (!h) return;
    for (uint32_t i = 0; i < h->num_terms; i++) {
        free(h->terms[i].ops);
    }
    free(h->terms);
    free(h);
}

// Apply Pauli string to state and return coefficient for expectation value
static double complex apply_pauli_string_expectation(const pauli_string_t *ps,
                                                      const quantum_state_t *state) {
    // <ψ|P|ψ> for Pauli string P
    // For each basis state |i>, P|i> = phase * |j> where j has bits flipped by X,Y

    double complex result = 0.0;
    uint64_t dim = state->state_dim;

    for (uint64_t i = 0; i < dim; i++) {
        double complex amp_i = state->amplitudes[i];
        if (cabs(amp_i) < 1e-15) continue;

        // Compute P|i>
        uint64_t j = i;
        double complex phase = 1.0;

        for (uint32_t q = 0; q < ps->num_qubits; q++) {
            int bit = (i >> q) & 1;

            switch (ps->ops[q]) {
                case PAULI_I:
                    break;
                case PAULI_X:
                    j ^= (1ULL << q);  // Flip bit
                    break;
                case PAULI_Y:
                    j ^= (1ULL << q);  // Flip bit
                    phase *= bit ? I : -I;  // Y|0> = i|1>, Y|1> = -i|0>
                    break;
                case PAULI_Z:
                    phase *= bit ? -1.0 : 1.0;
                    break;
            }
        }

        // Contribution: conj(amp_i) * phase * amp_j
        result += conj(amp_i) * phase * state->amplitudes[j];
    }

    return result;
}

double qubit_hamiltonian_expectation(const qubit_hamiltonian_t *h,
                                      const quantum_state_t *state) {
    if (!h || !state) return 0.0;
    if (state->num_qubits != h->num_qubits) return 0.0;

    double complex total = 0.0;

    for (uint32_t t = 0; t < h->num_terms; t++) {
        double complex term_exp = apply_pauli_string_expectation(&h->terms[t], state);
        total += h->terms[t].coeff * term_exp;
    }

    return creal(total);  // Hamiltonian is Hermitian, expectation is real
}

// ============================================================================
// UCCSD ANSATZ
// ============================================================================

uccsd_config_t *uccsd_config_create(uint32_t num_orbitals, uint32_t num_electrons) {
    uccsd_config_t *config = calloc(1, sizeof(uccsd_config_t));
    if (!config) return NULL;

    config->num_orbitals = num_orbitals;
    config->num_electrons = num_electrons;

    // Count excitations
    // Occupied: 0 to num_electrons-1
    // Virtual: num_electrons to num_orbitals-1
    uint32_t n_occ = num_electrons;
    uint32_t n_virt = num_orbitals - num_electrons;

    // Singles: n_occ * n_virt
    config->num_singles = n_occ * n_virt;

    // Doubles: C(n_occ,2) * C(n_virt,2)
    config->num_doubles = (n_occ * (n_occ - 1) / 2) * (n_virt * (n_virt - 1) / 2);

    config->num_amplitudes = config->num_singles + config->num_doubles;
    config->amplitudes = calloc(config->num_amplitudes, sizeof(double));

    return config;
}

void uccsd_config_free(uccsd_config_t *config) {
    if (!config) return;
    free(config->amplitudes);
    free(config);
}

/**
 * @brief Compute Jordan-Wigner parity between two orbitals
 *
 * In Jordan-Wigner transformation, fermionic operators pick up a sign
 * based on the occupation of orbitals between i and j.
 * parity = (-1)^(sum of occupations between lo and hi)
 */
static int jw_parity(uint64_t basis, uint32_t orbital_i, uint32_t orbital_j) {
    uint32_t lo = (orbital_i < orbital_j) ? orbital_i : orbital_j;
    uint32_t hi = (orbital_i < orbital_j) ? orbital_j : orbital_i;

    int parity = 0;
    for (uint32_t q = lo + 1; q < hi; q++) {
        parity ^= (basis >> q) & 1;
    }
    return parity ? -1 : 1;
}

/**
 * @brief Apply single excitation exp(θ(a†_a a_i - a†_i a_a))
 *
 * This implements the exact unitary for a single fermionic excitation
 * using the Jordan-Wigner transformation. The operator:
 *
 *   T1 - T1† = a†_a a_i - a†_i a_a
 *
 * In Jordan-Wigner representation becomes:
 *   (1/2)(X_a - iY_a)(Z_{a-1}...Z_{i+1})(X_i + iY_i)
 * - (1/2)(X_i - iY_i)(Z_{i-1}...Z_{a+1})(X_a + iY_a)   [for a > i]
 *
 * The exponential acts as a rotation in the 2D subspace spanned by
 * |...1_i...0_a...⟩ and |...0_i...1_a...⟩, with the Jordan-Wigner
 * string providing the correct fermionic sign.
 *
 * Matrix form in this subspace:
 *   exp(θ(|01⟩⟨10| - |10⟩⟨01|)) = [cos(θ), -sin(θ)]
 *                                  [sin(θ),  cos(θ)]
 */
qs_error_t uccsd_apply_single(quantum_state_t *state,
                               uint32_t i, uint32_t a, double t,
                               uint32_t num_orbitals) {
    if (!state) return QS_ERROR_INVALID_STATE;
    if (i >= num_orbitals || a >= num_orbitals) return QS_ERROR_INVALID_QUBIT;
    if (fabs(t) < 1e-15) return QS_SUCCESS;

    uint64_t dim = state->state_dim;
    double complex *new_amps = malloc(dim * sizeof(double complex));
    if (!new_amps) return QS_ERROR_OUT_OF_MEMORY;

    memcpy(new_amps, state->amplitudes, dim * sizeof(double complex));

    double cos_t = cos(t);
    double sin_t = sin(t);

    // Process each basis state
    for (uint64_t basis = 0; basis < dim; basis++) {
        int occ_i = (basis >> i) & 1;
        int occ_a = (basis >> a) & 1;

        // Only act on states where exactly one of i,a is occupied
        if (occ_i == 1 && occ_a == 0) {
            // This state couples to its partner with i<->a swapped
            uint64_t partner = basis ^ (1ULL << i) ^ (1ULL << a);

            // Compute fermionic sign from Jordan-Wigner string
            int jw_sign = jw_parity(basis, i, a);

            double complex psi_basis = state->amplitudes[basis];
            double complex psi_partner = state->amplitudes[partner];

            // Apply rotation: |10⟩ -> cos(t)|10⟩ + jw_sign*sin(t)|01⟩
            //                 |01⟩ -> -jw_sign*sin(t)|10⟩ + cos(t)|01⟩
            new_amps[basis] = cos_t * psi_basis + jw_sign * sin_t * psi_partner;
            new_amps[partner] = -jw_sign * sin_t * psi_basis + cos_t * psi_partner;
        }
    }

    memcpy(state->amplitudes, new_amps, dim * sizeof(double complex));
    free(new_amps);

    return QS_SUCCESS;
}

/**
 * @brief Compute fermionic sign for double excitation operator
 *
 * For the operator a†_a a†_b a_j a_i acting on a Fock state, we must
 * account for the anticommutation of fermionic operators. The sign
 * depends on the number of occupied orbitals that each operator must
 * "hop over" to reach its target position in normal-ordered form.
 *
 * The operator sequence a†_a a†_b a_j a_i applied to |...1_i 1_j...0_a 0_b...⟩:
 * 1. a_i annihilates at position i: sign from electrons to the right of i
 * 2. a_j annihilates at position j: sign from electrons to the right of j (after a_i acted)
 * 3. a†_b creates at position b: sign from electrons to the right of b
 * 4. a†_a creates at position a: sign from electrons to the right of a (after a†_b acted)
 *
 * @param basis The basis state being acted upon
 * @param i,j Occupied orbitals (will be annihilated)
 * @param a,b Virtual orbitals (will be created)
 * @return The fermionic sign (+1 or -1)
 */
static int fermionic_sign_double(uint64_t basis, uint32_t i, uint32_t j,
                                  uint32_t a, uint32_t b) {
    // Sort indices for consistent ordering
    if (i > j) { uint32_t tmp = i; i = j; j = tmp; }
    if (a > b) { uint32_t tmp = a; a = b; b = tmp; }

    int sign = 1;

    // Count occupied orbitals between each pair of indices
    // This accounts for the Jordan-Wigner strings

    // For annihilation operators (right to left in normal ordering):
    // a_i picks up sign from occupied orbitals in [0, i)
    for (uint32_t k = 0; k < i; k++) {
        if ((basis >> k) & 1) sign = -sign;
    }

    // a_j picks up sign from occupied orbitals in [0, j), excluding i (already annihilated)
    for (uint32_t k = 0; k < j; k++) {
        if (k != i && ((basis >> k) & 1)) sign = -sign;
    }

    // After annihilation, the state has electrons removed from i and j
    // For creation operators:
    // a†_b creates at b, picking up sign from [0, b) in the intermediate state
    uint64_t intermediate = basis ^ (1ULL << i) ^ (1ULL << j);
    for (uint32_t k = 0; k < b; k++) {
        if ((intermediate >> k) & 1) sign = -sign;
    }

    // a†_a creates at a, picking up sign from [0, a) after b is created
    intermediate ^= (1ULL << b);
    for (uint32_t k = 0; k < a; k++) {
        if ((intermediate >> k) & 1) sign = -sign;
    }

    return sign;
}

/**
 * @brief Apply double excitation exp(θ(a†_a a†_b a_j a_i - h.c.))
 *
 * This implements the exact unitary for a double fermionic excitation.
 * The operator T2 - T2† = a†_a a†_b a_j a_i - a†_i a†_j a_b a_a
 * generates rotations in the 2D subspace:
 *   |...1_i 1_j...0_a 0_b...⟩ ↔ |...0_i 0_j...1_a 1_b...⟩
 *
 * The exponential has the form:
 *   exp(θ(|ijab⟩⟨abij| - |abij⟩⟨ijab|)) = cos(θ)I + sin(θ)(|ijab⟩⟨abij| - |abij⟩⟨ijab|)
 *
 * where |ijab⟩ denotes the state with orbitals i,j occupied and a,b empty.
 */
qs_error_t uccsd_apply_double(quantum_state_t *state,
                               uint32_t i, uint32_t j,
                               uint32_t a, uint32_t b, double t,
                               uint32_t num_orbitals) {
    if (!state) return QS_ERROR_INVALID_STATE;
    if (i >= num_orbitals || j >= num_orbitals ||
        a >= num_orbitals || b >= num_orbitals) return QS_ERROR_INVALID_QUBIT;
    if (i == j || a == b) return QS_ERROR_INVALID_QUBIT;  // Must be distinct
    if (fabs(t) < 1e-15) return QS_SUCCESS;

    uint64_t dim = state->state_dim;
    double complex *new_amps = malloc(dim * sizeof(double complex));
    if (!new_amps) return QS_ERROR_OUT_OF_MEMORY;

    memcpy(new_amps, state->amplitudes, dim * sizeof(double complex));

    double cos_t = cos(t);
    double sin_t = sin(t);

    for (uint64_t basis = 0; basis < dim; basis++) {
        int occ_i = (basis >> i) & 1;
        int occ_j = (basis >> j) & 1;
        int occ_a = (basis >> a) & 1;
        int occ_b = (basis >> b) & 1;

        // Only act on states in the excitation subspace
        if (occ_i == 1 && occ_j == 1 && occ_a == 0 && occ_b == 0) {
            // Partner state with i,j <-> a,b
            uint64_t partner = basis ^ (1ULL << i) ^ (1ULL << j) ^
                               (1ULL << a) ^ (1ULL << b);

            // Compute exact fermionic sign
            int sign = fermionic_sign_double(basis, i, j, a, b);

            double complex psi_basis = state->amplitudes[basis];
            double complex psi_partner = state->amplitudes[partner];

            // Apply rotation in the 2D subspace
            new_amps[basis] = cos_t * psi_basis + sign * sin_t * psi_partner;
            new_amps[partner] = -sign * sin_t * psi_basis + cos_t * psi_partner;
        }
    }

    memcpy(state->amplitudes, new_amps, dim * sizeof(double complex));
    free(new_amps);

    return QS_SUCCESS;
}

qs_error_t uccsd_apply(quantum_state_t *state, const uccsd_config_t *config) {
    if (!state || !config) return QS_ERROR_INVALID_STATE;

    uint32_t n_occ = config->num_electrons;
    uint32_t n_virt = config->num_orbitals - config->num_electrons;

    uint32_t amp_idx = 0;

    // Apply single excitations
    for (uint32_t i = 0; i < n_occ; i++) {
        for (uint32_t a = n_occ; a < config->num_orbitals; a++) {
            qs_error_t err = uccsd_apply_single(state, i, a,
                                                 config->amplitudes[amp_idx],
                                                 config->num_orbitals);
            if (err != QS_SUCCESS) return err;
            amp_idx++;
        }
    }

    // Apply double excitations
    for (uint32_t i = 0; i < n_occ; i++) {
        for (uint32_t j = i + 1; j < n_occ; j++) {
            for (uint32_t a = n_occ; a < config->num_orbitals; a++) {
                for (uint32_t b = a + 1; b < config->num_orbitals; b++) {
                    qs_error_t err = uccsd_apply_double(state, i, j, a, b,
                                                         config->amplitudes[amp_idx],
                                                         config->num_orbitals);
                    if (err != QS_SUCCESS) return err;
                    amp_idx++;
                }
            }
        }
    }

    return QS_SUCCESS;
}

// ============================================================================
// HARTREE-FOCK STATE PREPARATION
// ============================================================================

qs_error_t hartree_fock_state(quantum_state_t *state,
                               uint32_t num_electrons,
                               uint32_t num_orbitals) {
    if (!state) return QS_ERROR_INVALID_STATE;
    if (num_electrons > num_orbitals) return QS_ERROR_INVALID_QUBIT;

    // Initialize state if needed
    if (state->num_qubits != num_orbitals) {
        quantum_state_free(state);
        qs_error_t err = quantum_state_init(state, num_orbitals);
        if (err != QS_SUCCESS) return err;
    }

    // Reset to |0...0>
    quantum_state_reset(state);

    // Apply X gates to first num_electrons qubits to get |1...10...0>
    for (uint32_t q = 0; q < num_electrons; q++) {
        gate_pauli_x(state, q);
    }

    return QS_SUCCESS;
}

// ============================================================================
// MOLECULAR GEOMETRY
// ============================================================================

// Nuclear charges for common elements
static int element_charge(const char *symbol) {
    if (strcmp(symbol, "H") == 0) return 1;
    if (strcmp(symbol, "He") == 0) return 2;
    if (strcmp(symbol, "Li") == 0) return 3;
    if (strcmp(symbol, "Be") == 0) return 4;
    if (strcmp(symbol, "B") == 0) return 5;
    if (strcmp(symbol, "C") == 0) return 6;
    if (strcmp(symbol, "N") == 0) return 7;
    if (strcmp(symbol, "O") == 0) return 8;
    if (strcmp(symbol, "F") == 0) return 9;
    if (strcmp(symbol, "Ne") == 0) return 10;
    return 0;
}

molecule_t *molecule_create(const atom_t *atoms, uint32_t num_atoms,
                             int charge, int multiplicity) {
    molecule_t *mol = calloc(1, sizeof(molecule_t));
    if (!mol) return NULL;

    mol->atoms = calloc(num_atoms, sizeof(atom_t));
    if (!mol->atoms) {
        free(mol);
        return NULL;
    }

    memcpy(mol->atoms, atoms, num_atoms * sizeof(atom_t));
    mol->num_atoms = num_atoms;
    mol->charge = charge;
    mol->multiplicity = multiplicity;

    return mol;
}

void molecule_free(molecule_t *mol) {
    if (!mol) return;
    free(mol->atoms);
    free(mol);
}

double molecule_nuclear_repulsion(const molecule_t *mol) {
    if (!mol) return 0.0;

    double energy = 0.0;
    const double BOHR_PER_ANGSTROM = 1.8897259886;

    for (uint32_t i = 0; i < mol->num_atoms; i++) {
        for (uint32_t j = i + 1; j < mol->num_atoms; j++) {
            int Zi = element_charge(mol->atoms[i].symbol);
            int Zj = element_charge(mol->atoms[j].symbol);

            double dx = mol->atoms[i].x - mol->atoms[j].x;
            double dy = mol->atoms[i].y - mol->atoms[j].y;
            double dz = mol->atoms[i].z - mol->atoms[j].z;
            double r_angstrom = sqrt(dx*dx + dy*dy + dz*dz);
            double r_bohr = r_angstrom * BOHR_PER_ANGSTROM;

            if (r_bohr > 1e-10) {
                energy += (double)(Zi * Zj) / r_bohr;
            }
        }
    }

    return energy;  // In Hartree
}

molecule_t *molecule_h2(double bond_length) {
    atom_t atoms[2] = {
        {"H", 0.0, 0.0, 0.0},
        {"H", bond_length, 0.0, 0.0}
    };
    return molecule_create(atoms, 2, 0, 1);
}

molecule_t *molecule_lih(double bond_length) {
    atom_t atoms[2] = {
        {"Li", 0.0, 0.0, 0.0},
        {"H", bond_length, 0.0, 0.0}
    };
    return molecule_create(atoms, 2, 0, 1);
}

molecule_t *molecule_h2o(double oh_length, double angle) {
    double angle_rad = angle * M_PI / 180.0;
    double half_angle = angle_rad / 2.0;

    atom_t atoms[3] = {
        {"O", 0.0, 0.0, 0.0},
        {"H", oh_length * sin(half_angle), oh_length * cos(half_angle), 0.0},
        {"H", -oh_length * sin(half_angle), oh_length * cos(half_angle), 0.0}
    };
    return molecule_create(atoms, 3, 0, 1);
}
