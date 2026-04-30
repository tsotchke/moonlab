# Variational-D CA-MPS for 1+1D Z₂ lattice gauge theory

Connecting the var-D method to lattice gauge theory: under what conditions does the Clifford prefactor `D` absorb the Gauss-law constraints, leaving |φ⟩ with only the dynamical (gauge-invariant) entanglement?

## Setup

The 1+1D Z₂ lattice gauge theory with staggered fermions, written as a qubit Hamiltonian on an interleaved chain (matter at even indices, link gauge field at odd):

$$
H = -t \sum_{x=0}^{N-2} \bigl[X_{2x} X_{2x+1} X_{2x+2} + Y_{2x} X_{2x+1} Y_{2x+2}\bigr]
    - h \sum_{x=0}^{N-2} Z_{2x+1}
    + \tfrac{m}{2} \sum_{x=0}^{N-1} (-1)^x Z_{2x}
    + \lambda \sum_{x=1}^{N-2} (I - G_x)
$$

with the Gauss-law operator at interior matter site `x`

$$
G_x \;=\; X_{2x-1}\, X_{2x+1}\, Z_{2x}.
$$

Gauss-law-respecting (physical) states satisfy $G_x|\psi\rangle = |\psi\rangle$ for all `x`. The penalty term `λ ∑ (I - G_x)` enforces this energetically as `λ → ∞`.

## Why this should be a clean var-D testbed

The {G_x} for interior `x` are mutually-commuting weight-3 Pauli operators — i.e. they generate a stabilizer subgroup of the n-qubit Pauli group (n = 2N - 1). Standard Aaronson-Gottesman tableau theory says: there exists a Clifford `D` such that

$$
D^\dagger G_x D \;=\; Z_{q(x)}
$$

for some choice of "stabilizer-representative" qubits `q(x)`. Under that `D`, the gauge-invariant subspace becomes precisely the subspace where `Z_{q(x)} = +1`, i.e. those qubits are pinned to `|0⟩`. The gauge-invariant content lives only on the remaining `n - (N-2) = N + 1` qubits.

If var-D *finds* such a `D`, then `|φ⟩ = D†|ψ⟩` has support only on the dynamical qubits and `S(|φ⟩) ≤ \log(2^{(N+1)/2}) = (N+1)/2 \cdot \log 2` — strictly less than the bound `(2N-1)/2 \cdot \log 2` on the plain |ψ⟩ entropy. That's the predicted entropy-reduction theorem.

## Theorem (provisional)

**Claim.** For the gauge-invariant ground state $|\psi_{\text{phys}}\rangle$ of the Z₂ LGT Hamiltonian above, there exists a Clifford `D ∈ C_n` such that

$$
S\bigl(D^\dagger |\psi_{\text{phys}}\rangle\bigr) \;\leq\; \log_2 \dim(\text{gauge-invariant subspace}) / 2
\;=\; \tfrac{1}{2}(N+1)
$$

while the plain Schmidt entropy of `|ψ_phys⟩` itself can saturate `(2N-1)/2`.

**Proof sketch.** The (N - 2) Gauss-law generators are independent commuting Paulis. By the Aaronson-Gottesman normal-form theorem, there is a Clifford `D` mapping them to N - 2 single-qubit Z's. Under that mapping, gauge-invariant states factorise as `|0…0⟩_{q_1...q_{N-2}} ⊗ |\widetilde{\phi}⟩` on the remaining N + 1 qubits. Half-cut entropy of a tensor product is the entropy of the non-trivial factor, which is bounded by the smaller half's dimension — `⌊(N+1)/2⌋ \cdot \log 2`. ∎

## The empirical gap (2026-04-30)

Implementation in `src/applications/hep/lattice_z2_1d.{c,h}` + driver in `examples/hep/z2_gauge_var_d.c`. Initial run at `N = 4` matter sites (7 qubits total), `t = 1`, `m = 0.5`, `λ = 5`, with the standard four-warmstart var-D pool we use for TFIM:

| h | E_varD | S(|φ⟩) | S(|ψ⟩) plain | ratio | Gauss violation |
|---|---|---|---|---|---|
| 0.25 | -10.87 | 1.92 | 1.00 | 1.93 | 2.5e-02 |
| 0.50 | -11.58 | 1.11 | 1.00 | 1.12 | 4.0e-02 |

(Raw output: `benchmarks/results/z2_lgt_var_d_partial_2026-04-30.txt`. The full `h` sweep didn't finish in the session's compute budget; n=7 with composite-2-gate moves is wallclock-expensive at ~5 min/point.)

**The empirical ratio is > 1**: `|φ⟩` has *more* entropy than plain |ψ⟩. This is the same model-dependence pattern documented in §6.4 (XXZ Heisenberg gapless regime): the existing four warmstarts (I, H_all, dual=H_all+CNOT-chain, ferro=H_0+CNOT-chain) are TFIM-tailored, and none of them is the gauge-fixing Clifford for Z₂ LGT. The greedy local search starting from any of those warmstarts cannot reach the AG-normal-form Clifford because reaching it requires `O(N)` consecutive accepts that look bad in step 1.

The Gauss-violation 2-4% indicates the penalty term λ = 5 isn't sharply enforcing the constraint; pushing λ higher would help but doesn't change the entropy story.

## Open research direction: gauge-aware warmstarts

The math says `D_{\text{gauge}}` exists and gives the entropy compression. The implementation says greedy search can't find it. The obvious bridge: write `D_{\text{gauge}}` analytically.

For Z₂ LGT in 1D, the AG-normal-form Clifford that diagonalises {G_x} is constructible:
1. View {G_x} as rows of a binary symplectic matrix.
2. Apply Gaussian elimination (over `F₂`) using elementary Clifford row operations (Hadamard on a column, CNOT on column pairs, phase swaps).
3. The accumulated Clifford circuit is `D_{\text{gauge}}`.

This is `O(n^3)` classical work per Hamiltonian — cheap. An automated builder takes the `{G_x}` Pauli strings and produces the gate sequence. This becomes a 5th warmstart in `ca_mps_warmstart_t`: `CA_MPS_WARMSTART_GAUGE_FIX` parameterised by a stabilizer subgroup.

Predicted result with `D_{\text{gauge}}`: `S(|φ⟩) → ` constant `+ O(\sqrt{h^2 + m^2})` correction, *not* growing with N. This would be the strong claim — Z₂ LGT becomes essentially classical-MPS-tractable in the gauge-invariant subspace once the right `D` is supplied.

## Strategic alignment

This bridges into the program's existing theory work in three concrete ways:

- **`paper_T1_bps_qgt_dictionary`**: gauge-fixing Clifford ↔ BPS-reduction map on moduli space. Both are "dimensionally reducing" the configuration space to the physical (constrained) subspace via a discrete (Clifford) or continuous (BPS) projection.
- **`paper_T2_homotopy_unification`**: the {G_x} stabilizer subgroup at fixed boundary parities is the discrete remnant of the Gauss-law constraint algebra; its `H¹` cohomology classifies the inequivalent gauge sectors. Connecting this to π₁ of the configuration space is direct.
- **Track A SaaS demo vignette**: "load this lattice gauge theory Hamiltonian into Moonlab's cloud, get the gauge-invariant ground state with a single line." Currently we ship the math + the Pauli-sum builder + the (failing) greedy-search demo. The gauge-fix warmstart turns this into a flagship feature no competitor offers.

## Status

| Component | Status |
|---|---|
| Pauli-sum builder (`lattice_z2_1d.c`) | ✅ shipped, unit-tested |
| Gauss-law operator accessor | ✅ shipped, unit-tested |
| Wilson-line operator accessor | ✅ shipped, unit-tested |
| Demo driver (var-D vs plain MPS) | ✅ shipped, partial results |
| Math write-up | ✅ this document |
| Gauge-aware warmstart `D_{\text{gauge}}` | ⏳ open research item |
| Theorem proof (full) | ⏳ open theory item |
| Cross-check vs exact diagonalisation | ⏳ pending |
| Larger N (N=6, N=8) | ⏳ needs gauge-aware warmstart first |

## Appendix — the math, slightly more carefully

The Hilbert space of `n = 2N - 1` qubits has `2^n` dimensions. The Gauss-law projector
$$
P_{\text{phys}} = \prod_{x=1}^{N-2} \frac{I + G_x}{2}
$$
projects onto the gauge-invariant subspace. Since the `G_x` are independent commuting stabilizers, `P_phys` has rank `2^{n - (N-2)} = 2^{N+1}`.

A Clifford `D` is an isomorphism of the Pauli group up to phases; it preserves the symplectic structure. Aaronson-Gottesman theorem: any commuting subgroup of the Pauli group can be mapped by Clifford conjugation to a subgroup of the Z-Paulis, so

$$
\exists D : D^\dagger G_x D = Z_{q(x)}, \quad x = 1, \ldots, N-2,
$$

with `q(1), ..., q(N-2)` distinct qubit indices. Then

$$
P_{\text{phys}} = D \prod_x \frac{I + Z_{q(x)}}{2} D^\dagger = D |0\rangle\langle 0|_{q(\cdot)} \otimes I_{\text{rest}} D^\dagger.
$$

Any gauge-invariant state factorises as
$$
|\psi_{\text{phys}}\rangle = D \bigl(|0\rangle^{\otimes (N-2)}_{q(\cdot)} \otimes |\widetilde\phi\rangle_{\text{rest}}\bigr)
$$
which means `D†|ψ_phys⟩` has support only on the `N + 1` "non-frozen" qubits. The half-cut entropy of `D†|ψ_phys⟩` is bounded by `\log_2 2^{\lfloor (N+1)/2 \rfloor}` = `\lfloor (N+1)/2 \rfloor \cdot \log 2`, in nats.

Compare to the plain |ψ_phys⟩'s half-cut entropy, which can saturate `\log_2 2^{\lfloor n/2 \rfloor}` = `\lfloor (2N-1)/2 \rfloor \cdot \log 2` — almost twice as large for `N = O(1)` and growing twice as fast in `N`.

This is the rigorous statement. Implementing `D_{\text{gauge}}` is the open task.
