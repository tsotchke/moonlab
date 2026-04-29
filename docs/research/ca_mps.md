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

### 5.3 Variational-D mode (v0.4.0 — gating item for VQE/QAOA paper)

The §6 benchmarks falsify the §5.1 prediction that fixed-D CA-MPS wins on VQE-HEA / QAOA. The mechanism is: when D is the "free" Clifford accumulated from the entangling layers of the workload, conjugating the workload's non-Clifford rotations through it produces weight-near-n Pauli strings that maximally entangle |phi>. The implicit assumption -- "the natural D is Clifford-aligned with the workload's entanglement" -- is wrong for canonical VQE and QAOA.

The fix is to **search for D** rather than accept the workload-supplied one. Co-optimize `D` and `|phi>` by alternating:

1. **D-fixed |phi>-update:** standard DMRG sweep on the conjugated MPO `H' = D H D†`. `H'` is built once per outer iteration via `clifford_conjugate_pauli` on each Pauli term of H (see §4.4 for the per-term cost).

2. **|phi>-fixed D-update:** greedy local-Clifford search. For each candidate gate `G` in `{H_q, S_q, S_dag_q, X_q, Y_q, Z_q}` (single-qubit) and `{CNOT_{q,r}, CZ_{q,r}, SWAP_{q,r}}` (nearest-neighbor 2-qubit), compute the energy change

       E(GD, phi) - E(D, phi) = sum_k c_k * (<phi|G Q_k G^dag|phi> - <phi|Q_k|phi>)

   where `Q_k = D P_k D†` are the Paulis we already track. For a single-qubit `G_q`, only Pauli terms with non-trivial support on `q` contribute, so the per-candidate cost is O(non-identity terms with support q) cached single-Pauli expectations -- not a fresh DMRG run. Pick the gate that decreases energy the most, apply, repeat until no improvement. O(n^2) candidates per pass, O(passes-until-saturated * n^2 * MPS-expect) total.

#### Minimum-viable validating experiment ("oracle proof") -- COMPLETED 2026-04-28

`examples/tensor_network/ca_mps_oracle_proof.c` runs the gating experiment:
take the TFIM ground state |psi> via DMRG, apply two candidate Clifford
oracles (CNOT-chain, and H_all then CNOT-chain), measure the half-cut von
Neumann entanglement entropy of |phi> = D|psi>.

Result (full table in `benchmarks/results/ca_mps_oracle_proof_2026-04-28.json`):

| n | g | S(psi) | S(phi, h_then_cnot) | reduction |
|---|---|---|---|---|
| 8 | 0.25 | 1.000 | 0.038 | 26x |
| 8 | 1.00 (critical) | 0.515 | 0.080 | 6.5x |
| 8 | 2.50 | 0.086 | 0.0017 | 50x |
| 12 | 0.50 | 1.004 | 0.128 | 8x |
| 12 | 1.00 (critical) | 0.572 | 0.116 | 5x |
| 12 | 2.50 | 0.086 | 0.0017 | 50x |

A *single* hand-supplied Clifford (H on every qubit, then a left-to-right
CNOT chain) drops the half-cut entanglement of the TFIM ground state by
5-50x across the entire phase diagram, including criticality at g = 1.
The CNOT-chain alone is the wrong oracle in every regime tested -- it
*increases* entropy.  The dual transformation (H first, then CNOT chain)
is the right one; the operative entanglement is in the X basis, not the
Z basis.

**Conclusion: variational-D is well-motivated.** A greedy local-Clifford
search starting from D = I should find at least this much, since each
gate evaluation can detect when applying H or CNOT decreases <phi|H'|phi>.
The discovered Clifford may not match h_then_cnot exactly, but it should
reach within a small factor.

Side observation worth recording: `tn_mps_bond_dim` reports the *allocated*
bond dimension, not the post-truncation rank.  For DMRG with chi_max=32,
chi_psi == 32 across the whole sweep regardless of actual entanglement.
The publishable benchmark must use entropy as the primary metric; bond-dim
ratios are unreliable.  This is a separate task list item for v0.4.0
(see #77 below) -- the bench harness in `tests/performance/bench_ca_mps.c`
also affects this.

#### Implementation cost

Full var-D (greedy local-Clifford search + DMRG-on-H' alternating loop): ~2 weeks.

Oracle proof: completed (~3 hours including bench rewrite + entropy diagnostic).

---

## 6. Benchmarks (measured)

The numbers below are from `tests/performance/bench_ca_mps.c` run on Apple M2 Ultra; raw JSON in `benchmarks/results/ca_mps_v3_2026-04-28.json`.

**Important:** the primary metric is the maximum half-cut von Neumann
entanglement entropy `S` (in nats), not the bond dimension. DMRG / TEBD
keep the allocated bond at `chi_max` even when actual rank is far smaller,
so chi numbers are a poor proxy for "how compactly the state can be
represented." `S` is representation-independent and is the right
yardstick for any CA-MPS-vs-plain-MPS comparison.

### 6.1 Where CA-MPS wins (clean)

| Circuit | n | depth | S_plain | S_ca | wallclock speedup |
|---|---|---|---|---|---|
| Random Clifford | 10 | 120 | 4.35 | 0.00 | 4.2e5x |
| Random Clifford | 12 | 150 | 0.00 | 0.00 | 1.1e6x |
| Surface-stabilizer cycle | 10 | 10 cycles | 4.00 | 0.00 | 451x |
| Surface-stabilizer cycle | 12 | 12 cycles | 4.28 | 0.00 | 4.4e6x |
| Clifford-heavy (5% T) | 10 | 120 gates | 3.00 | 0.60 | 56x |
| Clifford-heavy (5% T) | 12 | 150 gates | 3.00 | 0.60 | 10x |

Stabilizer-rich workloads drive `S_plain` to a substantial fraction of
the n/2 ceiling while `D` absorbs all of it; `S_ca` stays at 0 (or 0.6
for the 5% T case, where the residual entanglement is the magic content).
Wallclock follows the entropy gap.

(One curiosity in the table: pure_clifford n=12 with depth 150 and our
fixed RNG seed lands on a stabilizer state with S=0 at every half-cut,
which is why both rows are 0; chi_plain=16 confirms the state isn't
trivial -- it's just unentangled at the half-cuts we sample.)

### 6.2 Where CA-MPS LOSES (the falsified §5.1 prediction, more honestly)

| Circuit | n | layers | S_plain | S_ca | S_ratio | wallclock |
|---|---|---|---|---|---|---|
| VQE hardware-efficient | 8 | 4 | 2.34 | 2.73 | 0.86x | CA-MPS 367x slower |
| VQE hardware-efficient | 10 | 4 | 2.56 | 3.43 | 0.75x | CA-MPS 936x slower |
| VQE hardware-efficient | 12 | 4 | 3.17 | 3.75 | 0.85x | CA-MPS 12793x slower |
| QAOA ring (p=4) | 12 | 4 | 0.72 | 1.44 | 0.50x | CA-MPS 180x slower |
| Random Pauli rotation | 12 | 150 gates | 4.38 | 5.24 | 0.84x | CA-MPS wins 15x |

CA-MPS's |phi> has 1.2-2x more entanglement than plain MPS's |psi> on
VQE-HEA / QAOA -- a modest penalty on the metric that matters. The
chi-based "16-32x worse" headline from a previous draft of this section
was a measurement artifact (chi reports allocated storage, not
post-truncation rank).

The wallclock penalty is real and large -- driven by per-gate cost,
not entropy. Each non-Clifford rotation in CA-MPS becomes a multi-qubit
Pauli rotation in |phi>, applied as a bond-2 MPO. Even with the small
entropy gap, repeated MPO-applications are expensive vs plain MPS's
single-site rotations.

The §5.1 prediction "Clifford layers absorbed for free into D" is
falsified for canonical VQE / QAOA -- CA-MPS at fixed-D simply doesn't
beat plain MPS on these workloads, so there's no reason to use it.
But the failure mode is "modest entropy penalty + bad per-gate cost,"
not "catastrophic entanglement blow-up." Variational-D (§5.3) addresses
both: the oracle proof (2026-04-28) shows a hand-supplied Clifford can
drop |phi> entropy by 5-50x on TFIM, which would close the entropy gap
and justify the per-gate cost.

### 6.3 What's publishable today

The fixed-D 1D CA-MPS in v0.3.0 is a clean win on a specific workload class:
- **stabilizer-rich circuits** (surface-code stabilizer extraction, error-correction simulation, Clifford-tableau benchmarks): S_plain to S_ca ratio of infinity (S_ca = 0); 451x to 4.4M x wallclock speedup;
- **circuits with sparse non-Clifford content** (clifford_heavy at 5% T-density wins 5x on entropy + 10-56x wallclock);
- the **regression points** (random Pauli rotation) confirm correctness in the pure-MPS limit (entropy ratio 0.84x, but wallclock 15x faster at n=12 because Clifford gates are cheap).

Variational-D (§5.3) shipped in 2026-04-29: alternating imag-time |phi>-update + greedy Clifford D-update over a four-warmstart basin pool (I, H_all, dual=H_all+CNOT_chain, ferro=H_0+CNOT_chain).  Headline result: TFIM phase sweep, var-D matches plain DMRG energy convergence (dE_rel < 0.7%) with **5x to 1430x lower |phi> half-cut entropy than plain DMRG's |psi>** across the entire phase diagram including the quantum critical point.

### 6.4 Model dependence -- XXZ Heisenberg (negative result)

Generalising the var-D head-to-head to the 1D XXZ Heisenberg model
(`examples/tensor_network/ca_mps_var_d_heisenberg.c`) shows a clear
model-dependent split.  Sweep result at n=6 (full sweep at n=8 didn't
finish in budget; raw text in `benchmarks/results/ca_mps_var_d_xxz_n6_2026-04-29.txt`):

| Delta | regime | dE_rel | S_psi | S_phi | reduction |
|---|---|---|---|---|---|
| 0.0 | XY (gapless) | 13% | 1.04 | 1.03 | **1.0x (no reduction)** |
| 0.5 | gapless XXZ | 0.02% | 1.03 | 1.02 | **1.0x** |
| 1.0 | SU(2) Heisenberg | 0.04% | 1.03 | 1.02 | **1.0x** |
| 1.5 | Ising-anisotropy | 0.06% | 1.03 | 0.25 | **4x** |
| 2.0 | gapped Ising-like | 0.10% | 1.03 | 0.20 | **5x** |

For Delta <= 1 (the gapless XXZ regime including the SU(2)-symmetric
Heisenberg point), var-D gives essentially no entropy reduction.  This
is consistent with the underlying mechanism: var-D only helps when the
target state has stabilizer-structured entanglement that some Clifford
can absorb.  Heisenberg's GS in the gapless regime has SU(2)-symmetric
fluctuations with no analogous Clifford basis transformation.

For Delta >= 1.5 (Ising-anisotropy regime), the Z-coupling dominates
and the GS becomes Ising-like; the dual Clifford warmstart (which
worked for TFIM) starts to help, giving 4-5x reduction.

This is a useful negative result for the paper: var-D is
**model-dependent** -- a powerful method for systems with stabilizer-
adjacent ground states (TFIM, Ising-like, surface codes), not a
generic compression for all Hamiltonians.  Generalising further would
require either model-specific Clifford warmstart catalogues or a
deeper basin-search algorithm (e.g. multi-gate composite moves, beam
search, simulated annealing) that we don't currently have.

CA-PEPS (§7) is the gating item for venue-uplift to a physics-novelty journal.

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
