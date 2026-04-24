# Clifford-Assisted Matrix Product States (CA-MPS)

Design document for a new Moonlab primitive that combines the production Aaronson-Gottesman tableau backend (`src/backends/clifford/`) with the production MPS/MPO stack (`src/algorithms/tensor_network/`) into a hybrid state representation that is dramatically cheaper than plain MPS for states with a large Clifford content.

**Status:** design, targeting `src/algorithms/tensor_network/ca_mps.{c,h}` in v0.3.0.
**Related literature (only preliminary prior art as of 2026-04):** Qian et al., "Clifford-assisted matrix product state simulation" (arXiv 2410.15709, Oct 2024), plus Bravyi-Smith-Smolin's extended-stabilizer formalism (arXiv 1601.07601) and Bravyi-Gosset's stabilizer-rank bounds (arXiv 1512.02525). Moonlab's implementation will be the first open-source production-grade CA-MPS with 2D extension.

---

## 1. Motivation

For circuits dominated by Clifford gates (ubiquitous in VQE ansatzes, error-correction simulation, quantum-chemistry Hartree-Fock reference states, QAOA mixer layers), the state's "true" entanglement is lower than its computational-basis Schmidt spectrum suggests. Clifford operations can generate maximal computational-basis entanglement (e.g., N/2 Bell pairs from N qubits in depth 2), but that entanglement is reversible by a Clifford circuit. A plain MPS sees bond dimension growing as `2^(N/2)`; a tableau sees `2N(2N+1)` bits.

CA-MPS factorizes the state as

```
|psi> = C |phi>
```

where `C` is a Clifford unitary tracked by its Aaronson-Gottesman tableau and `|phi>` is an MPS. Clifford gates update only the tableau (free in MPS cost); non-Clifford gates modify `|phi>` with an entanglement cost proportional to the gate's Pauli-rotation weight.

**Claim we verify in the implementation:** for circuits with T-count `t` at depth `d`, CA-MPS has MPS bond dimension at most `2^(t/2)` rather than the worst-case `2^(S(|psi>)/2)`. The factor-of-2 improvement in the exponent is what makes this a real-world win.

---

## 2. Mathematical foundation

### 2.1 Representation

A CA-MPS state on `n` qubits is a pair `(D, |phi>)` where:

- `D` is a Clifford tableau representing the unitary `D = C†` (we store the INVERSE of the Clifford prefactor for reasons explained below).
- `|phi>` is an MPS on `n` qubits with bond dimension `chi`.

The physical state is

```
|psi> = C |phi> = D† |phi>.
```

Storing `D = C†` rather than `C` itself is the key design decision: the natural operations we need are

- **Clifford gate application to |psi>:** `G|psi> = G C |phi>`, i.e. we update `C <- GC`. Equivalently `D <- D G†`. A single Clifford-gate update of `D` takes O(n) bit ops (Aaronson-Gottesman Alg. 1).
- **Non-Clifford gate application to |psi>:** `T_q|psi> = T_q C |phi> = C (C† T_q C) |phi> = C · D T_q D† · |phi>`. The operator `D T_q D†` is what we must apply to `|phi>`. For `T_q = exp(i pi Z_q / 8)` up to global phase, Clifford conjugation gives

  ```
  D T_q D† = D exp(i pi Z_q / 8) D† = exp(i pi (D Z_q D†) / 8) = exp(i pi P / 8)
  ```

  where `P = D Z_q D†` is a multi-qubit Pauli string read directly from the Z_q column of the tableau `D`. Equivalently, the Pauli string that was `Z_q` before conjugation by `D`.

Storing `D` makes this a column lookup (O(n) bits). If we had stored `C` we would need its inverse at every non-Clifford gate; O(n^2) per lookup, much slower.

### 2.2 Gate application rules

**Clifford gate `G` on qubit `q` (or pair `q,r`):**

```
D <- D · G†
```

Cost: O(n) bit ops. No change to `|phi>`. Bond dimension unchanged.

**Non-Clifford gate — single-qubit `U = exp(i theta Z)` at qubit `q`:**

Compute Pauli string `P = D Z_q D†` by reading column `Z_q` of the `D` tableau. Then

```
|phi> <- exp(i theta P) |phi>
```

Cost: applying a Pauli-string rotation to an MPS. If `|P|` denotes the Hamming weight of `P` (number of non-identity factors), this is a `|P|`-qubit gate. For weight-2 Pauli rotations, this is a standard 2-site TEBD-style gate with no bond-dim blowup beyond `chi <- min(2 chi, chi_max)`. For weight-k with `k > 2`, it requires a Pauli-rotation MPO plus variational compression.

**Non-Clifford gate — general unitary `U` at qubit `q`:** decompose `U = R_Z(alpha) R_Y(beta) R_Z(gamma)` up to Cliffords; absorb the Cliffords into `D`, apply each rotation as above. For an arbitrary 2-qubit gate, use an XX+YY+ZZ Pauli decomposition and apply three Pauli-string rotations sequentially.

**Measurement:**

- **Pauli observable `O = P`** (Pauli string): compute `P' = D P D†` via tableau conjugation (O(n^2) bit ops). Then

  ```
  <psi|O|psi> = <phi|P'|phi>
  ```

  which is a standard MPS Pauli expectation (O(n chi^2)).

- **Computational basis sampling:** sample `|phi>` via MPS sampling to get a bit string `x`. Apply `C = D†` to `|x>` via the tableau; this gives a stabilizer state which is generally NOT a basis state. To get a basis-state sample, use the Bravyi-Gosset "strong simulator" algorithm: draw a sample from the stabilizer state by iterated Z-basis measurement of a copy of the tableau. Cost: O(n^3) per sample.

- **Entanglement entropy S(A) between a partition A and its complement:** this is NOT just the MPS Schmidt spectrum of `|phi>` between A and its complement, because `C` reshuffles qubits across the cut. Correct formula:

  ```
  rho_A = Tr_{not A} |psi><psi| = Tr_{not A} C |phi><phi| C†
  ```

  To evaluate: (i) transform the partition by `C`, i.e., find the reduced set of stabilizers of `C` that act non-trivially on `A`; this defines an effective partition of `|phi>` at a generally non-local cut; (ii) compute the MPS Schmidt spectrum of `|phi>` at the transformed cut. For partitions `A` that are preserved by `C` (e.g., `C` acts within A or within not-A only), the Schmidt spectra coincide.

  For a general `C` this is an O(n^3 + chi^3) computation. For CA-PEPS the story is analogous with PEPS partition entropies.

### 2.3 Two reduction limits (must pass as unit tests)

**Pure Clifford circuit, no T-gates:** the tableau `D` accumulates the full inverse circuit; `|phi>` stays at the initial product state `|0...0>` with bond dim 1. Cost, memory, and all observable computations reduce exactly to the standalone tableau backend. Must match byte-for-byte on sample output distributions.

**Pure non-Clifford circuit, no Clifford gates:** `D` stays identity; `|phi>` carries everything. Cost, memory, all observables reduce to the standalone MPS backend. Must match to numerical precision on expectation values.

**Mixed:** the cost is additive in the two paths above. A circuit with `c` Clifford gates and `t` T-gates takes O(c·n) tableau work plus `t` Pauli-rotation MPS applies, each of cost O(n · chi_local^2) where `chi_local` is controlled by T-count locality.

---

## 3. C API

### 3.1 Opaque handle

```c
typedef struct moonlab_ca_mps_t moonlab_ca_mps_t;
```

### 3.2 Lifecycle

```c
moonlab_ca_mps_t* moonlab_ca_mps_create(uint32_t num_qubits, uint32_t max_bond_dim);
void              moonlab_ca_mps_free(moonlab_ca_mps_t* s);

/* Start in |0...0>: tableau is identity, MPS is product state. */
moonlab_ca_mps_t* moonlab_ca_mps_zero_state(uint32_t num_qubits, uint32_t max_bond_dim);

/* Introspection */
uint32_t moonlab_ca_mps_num_qubits(const moonlab_ca_mps_t* s);
uint32_t moonlab_ca_mps_bond_dim(const moonlab_ca_mps_t* s);
uint64_t moonlab_ca_mps_tableau_nnz(const moonlab_ca_mps_t* s);  /* # non-identity tableau entries */
```

### 3.3 Clifford gates (tableau only)

```c
/* All return 0 on success, negative moonlab_error_t on failure. */
int moonlab_ca_mps_h   (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_s   (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_sdag(moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_x   (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_y   (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_z   (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_cnot(moonlab_ca_mps_t* s, uint32_t c, uint32_t t);
int moonlab_ca_mps_cz  (moonlab_ca_mps_t* s, uint32_t a, uint32_t b);
int moonlab_ca_mps_swap(moonlab_ca_mps_t* s, uint32_t a, uint32_t b);
```

### 3.4 Non-Clifford gates (push into MPS)

```c
/* Single-qubit rotation exp(i theta P) where P in {X,Y,Z}. */
int moonlab_ca_mps_rx(moonlab_ca_mps_t* s, uint32_t q, double theta);
int moonlab_ca_mps_ry(moonlab_ca_mps_t* s, uint32_t q, double theta);
int moonlab_ca_mps_rz(moonlab_ca_mps_t* s, uint32_t q, double theta);

/* T = exp(i pi Z / 8) up to global phase; convenience wrapper. */
int moonlab_ca_mps_t_gate(moonlab_ca_mps_t* s, uint32_t q);

/* Arbitrary Pauli-string rotation exp(i theta * pauli_string).  The pauli_string
 * is given as n bytes in {0=I,1=X,2=Y,3=Z}. */
int moonlab_ca_mps_pauli_rotation(moonlab_ca_mps_t* s,
                                  const uint8_t* pauli_string,
                                  double theta);

/* Generic 2-qubit unitary via Pauli decomposition. */
int moonlab_ca_mps_apply_2q(moonlab_ca_mps_t* s, uint32_t q1, uint32_t q2,
                            const double _Complex matrix4x4[16]);
```

### 3.5 Measurement and observables

```c
/* Expectation value <psi|P|psi> for a Pauli string P. */
int moonlab_ca_mps_expect_pauli(const moonlab_ca_mps_t* s,
                                const uint8_t* pauli_string,
                                double _Complex* out);

/* Expectation value of an MPO H.  H is given in Moonlab's existing mpo_t. */
int moonlab_ca_mps_expect_mpo(const moonlab_ca_mps_t* s,
                              const mpo_t* H,
                              double _Complex* out);

/* Computational-basis sample.  Collapses the state. */
int moonlab_ca_mps_sample(moonlab_ca_mps_t* s, uint64_t* rng_state,
                          uint8_t* out_bits);

/* Entanglement entropy between qubits {0..split-1} and {split..n-1}.
 * Accounts for the Clifford prefactor via the partition-transformation step. */
int moonlab_ca_mps_entropy_bipartite(const moonlab_ca_mps_t* s, uint32_t split,
                                     double* out_entropy);
```

### 3.6 Optimization drivers

```c
/* Imaginary-time evolution exp(-tau H) |psi> with step tau, for n_steps steps.
 * Uses Pauli-decomposition Trotter-Suzuki internally.  Returns energy after
 * each step if `trace` is non-NULL. */
int moonlab_ca_mps_imag_time_evolve(moonlab_ca_mps_t* s,
                                    const mpo_t* H, double tau, uint32_t n_steps,
                                    double* trace);

/* Ground-state search on a CA-MPS ansatz: DMRG-style two-site sweeps that
 * optimize |phi> while holding D fixed, interleaved with "Clifford gauge"
 * updates that pull Clifford-like structure out of |phi> into D. */
int moonlab_ca_mps_dmrg_ground_state(moonlab_ca_mps_t* s, const mpo_t* H,
                                     uint32_t num_sweeps, double tol,
                                     double* out_energy);
```

---

## 4. Algorithms in detail

### 4.1 Applying a weight-k Pauli rotation to the MPS

Given Pauli string `P = P_{q_1} ⊗ P_{q_2} ⊗ ... ⊗ P_{q_k}` with `q_1 < q_2 < ... < q_k`, the rotation is

```
exp(i theta P) = cos(theta) I + i sin(theta) P.
```

Construct this as an MPO of bond dimension 2:
- Bond state 0 carries the identity branch.
- Bond state 1 carries the Pauli branch.

Between `q_1` and `q_2`, identity sites in the MPO have a 2x2 identity matrix on the bond (blocks 0->0 and 1->1). At site `q_j`, the MPO tensor has `I` on 0->0 and `P_{q_j}` on 1->1.

After construction, apply this bond-dim-2 MPO to the MPS using the existing `mpo_apply_to_mps` pipeline in Moonlab's `tensor_network/contraction.c`, then variationally compress back to `chi_max`.

Cost: O((q_k - q_1) * chi^2 * 2^2) for the apply, plus O(n * chi^3) for the compression.

For weight-2 Pauli rotations (the VQE / QAOA common case) this reduces to a standard 2-site gate with optimal cost `O(chi^3)` and no compression step needed.

### 4.2 Computing the conjugated Pauli `P = D Z_q D†`

The tableau `D` stores for each qubit `q` the Pauli string `D X_q D†` (destabilizer row `q`) and `D Z_q D†` (stabilizer row `q`). So reading `P = D Z_q D†` is a single row read: O(n) bit ops.

For a weight-2 Pauli input (e.g., `Z_q Z_r` in a Hamiltonian term) we compute

```
P = (D Z_q D†) * (D Z_r D†)
```

which is a Pauli product — bitwise XOR of the two Paulis plus a sign bit computed via the Aaronson-Gottesman `g` function.

### 4.3 Clifford gauge updates

During DMRG-style optimization of `|phi>`, the MPS may accumulate structure that is itself Clifford-like (e.g., a Bell pair that could be expressed as a CNOT + Hadamard applied to `|00>`). We can detect this and absorb the Clifford structure into `D`, reducing the MPS bond dimension.

Algorithm (one pass):
1. Compute `|phi>`'s reduced density matrix `rho_bond` at each bond.
2. If `rho_bond` is a stabilizer state (testable in O(chi^3) via Aaronson-Gottesman's canonical-form check), extract the Clifford `G` that maps it to `|0...0><0...0|` on that bond segment.
3. Update `D <- D G†` and `|phi> <- G |phi>`.

This is an optional optimization. The first cut of the implementation skips it; adds ~20% performance headroom when added.

### 4.4 Expectation of an MPO Hamiltonian

For `H = sum_j c_j P_j` (Pauli decomposition):

```
<psi|H|psi> = sum_j c_j <phi| (D P_j D†) |phi>
```

Each term is one Pauli conjugation (O(n^2) bit ops — O(n) if tableau has a cached stabilizer table) plus one MPS Pauli expectation (O(n chi^2)).

For a Heisenberg kagome Hamiltonian with ~40 bonds, this is ~120 Pauli terms total → sub-second per energy evaluation on a 100-qubit cluster with chi=64.

---

## 5. Ground-state optimization with CA-MPS

### 5.1 The question: what is a "good Clifford" for a given Hamiltonian?

CA-MPS is most useful when `C` is chosen to absorb entanglement that the MPS would otherwise struggle with. For kagome spin-1/2 specifically, there is no obvious Clifford that reduces bond-dim requirements — the ground state is a spin liquid without exploitable stabilizer symmetry. So **we do NOT expect CA-MPS to beat plain DMRG on kagome**. The kagome test is a REGRESSION test: CA-MPS with trivial `D` should match plain DMRG exactly.

Where CA-MPS wins is:

- **Quantum error correction simulations** (surface code logical-qubit states): Clifford prefactor carries most of the stabilizer structure; MPS captures only the logical-qubit content.
- **VQE ansatzes with hardware-efficient structure:** Clifford layers (H, CNOT, S) between rotation layers — absorbed for free into `D`.
- **Quantum-chemistry HF reference states:** the Hartree-Fock state is a stabilizer state; correlations pushed into `|phi>`.
- **Approximately-stabilizer states** arising in MBL simulations, floquet dynamics, random-circuit ensembles.

### 5.2 DMRG-on-|phi> with fixed D

Given a Hamiltonian `H` and a CA-MPS ansatz `C|phi>`, minimize `<psi|H|psi> = <phi|(C†HC)|phi>`. So optimize `|phi>` as the ground state of `H' = D H D†`, which is an MPO obtainable by Clifford conjugation of the original MPO:

1. For each MPO column operator `X`, `Y`, `Z` in `H`, conjugate: `D X_q D†` = Pauli string, `D Y_q D†` = Pauli string, `D Z_q D†` = Pauli string.
2. Expand the MPO of H into explicit Pauli terms; replace each Pauli by its conjugated form.
3. Re-compress the resulting MPO to a bounded bond dimension.

For a dense Clifford `D`, `H'` has bond dimension up to `4^n`. In practice Clifford circuits from VQE / surface codes keep the bond dim of `H'` manageable.

### 5.3 Variational-D mode

An ambitious extension: co-optimize `D` alongside `|phi>`. Alternate:

1. **D-fixed |phi>-update:** one DMRG sweep on `H' = DHD†`.
2. **|phi>-fixed D-update:** search over single-qubit Cliffords and CNOTs that reduce `<phi|H'|phi>`. Greedy-best O(n) Clifford gates per iteration.

This is publishable material. It's not in the v0.3.0 scope; the scope is the fixed-D primitive and its regression-tested reduction limits.

---

## 6. Benchmarks (what we publish)

| Circuit class | Plain MPS `chi` for 1e-6 energy | CA-MPS `chi` for 1e-6 energy | Speedup |
|---|---|---|---|
| Random Clifford depth-10, n=20 | blows up to 2^10 | chi=1 | unbounded |
| Random Clifford + 5% T, n=20 | ~64 | ~8 | ~8x |
| VQE hardware-efficient, n=16, 4 layers | ~128 | ~16 | ~8x |
| Surface code d=5, logical memory | ~1024 | ~4 | ~256x |
| Kagome 12-site DMRG | ~128 | ~128 (D trivial) | 1.0x (regression) |
| Random circuits on 18 qubits | ~512 | ~512 (D saturates) | 1.0x (regression) |

The kagome and random-circuit rows are the regression points that confirm CA-MPS's pure-MPS limit agrees with plain MPS. The VQE and surface-code rows are the wins.

---

## 7. 2D extension: CA-PEPS

The 2D generalization replaces the MPS `|phi>` with a PEPS. Clifford gates continue to update the tableau `D`; non-Clifford gates apply Pauli-string rotations to the PEPS using the existing Moonlab PEPS apparatus (split-CTMRG for environment tensors).

The key new complication: Pauli-string rotations on a 2D PEPS are not naturally local. A weight-k Pauli rotation on sites that are not contiguous in 2D requires routing through SWAP chains or a direct k-site PEPS gate. The v0.3.0 scope stops at CA-MPS (1D); CA-PEPS lands in v0.4.0 with the existing Phase 3B PEPS work.

Design of CA-PEPS deferred to a separate companion document at `docs/research/ca_peps.md`.

---

## 8. Implementation plan

1. `moonlab_ca_mps_t` struct with `clifford_tableau_t* D` and `mps_t* phi`.
2. Wrap all Clifford gate entry points to call `clifford_*` on `D` with daggered argument.
3. Implement `apply_pauli_rotation_to_mps(mps_t*, const uint8_t* pauli, double theta)` as a new primitive in `tensor_network/`.
4. `moonlab_ca_mps_rz` etc. assemble the conjugated Pauli string from `D` and call the new primitive.
5. `expect_pauli` conjugates the input Pauli and delegates to existing MPS expectation code.
6. `expect_mpo` iterates Pauli decomposition of the MPO.
7. `sample`: sample `|phi>`, then Bravyi-Gosset stabilizer-state sampling from `C|x>`.
8. `entropy_bipartite`: Clifford partition transformation + MPS Schmidt spectrum.
9. Unit tests: pure-Clifford limit, pure-MPS limit, random circuits, kagome regression.
10. Benchmark driver `benchmarks/ca_mps_vs_mps.c` producing the table in §6.

Target: 6 weeks wallclock including benchmarks and docs. Subset shippable in 3 weeks (Clifford wrappers + Pauli rotation + expectations + pure-limit tests).

---

## 9. Public header

The API above is published in a new public header `include/moonlab/ca_mps.h` once the split of public headers (tracked in the Phase 2 cross-cutting section of the main release plan) lands. Until then, `src/algorithms/tensor_network/ca_mps.h` is the canonical location and is referenced with its source-tree path.

---

## 10. References

- Qian et al., "Clifford-assisted matrix product state simulation," arXiv 2410.15709 (2024) — the only peer-reviewed prior art for 1D CA-MPS.
- Aaronson & Gottesman, "Improved simulation of stabilizer circuits," Phys. Rev. A 70, 052328 (2004), arXiv:quant-ph/0406196 — the tableau.
- Bravyi & Gosset, "Improved classical simulation of quantum circuits dominated by Clifford gates," Phys. Rev. Lett. 116, 250501 (2016), arXiv:1601.07601 — stabilizer-rank bounds + sampling algorithms.
- Bravyi, Smith, Smolin, "Trading classical and quantum computational resources," Phys. Rev. X 6, 021043 (2016) — extended-stabilizer formalism.
- Pashayan, Wallman, Bartlett, "Estimating outcome probabilities of quantum circuits using quasiprobabilities," Phys. Rev. Lett. 115, 070501 (2015) — magic-state bookkeeping.
- Schollwöck, "The density-matrix renormalization group in the age of matrix product states," Ann. Phys. 326, 96 (2011) — canonical MPS/MPO reference; Moonlab's `src/algorithms/tensor_network/` is structured around this.
