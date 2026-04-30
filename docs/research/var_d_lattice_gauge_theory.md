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
| Gauge-aware warmstart `D_gauge` | ✅ shipped, unit-tested (`ca_mps_var_d_stab_warmstart.{c,h}`, `tests/unit/test_gauge_warmstart.c`) |
| Exactly gauge-invariant kinetic terms | ✅ shipped — `lattice_z2_1d.c` now uses `K_x = (-t/2) X_{2x} Y_{2x+1} Y_{2x+2} + (+t/2) Y_{2x} Y_{2x+1} X_{2x+2}` with term-by-term `[K_x, G_y] = 0` pinned by `tests/unit/test_z2_lgt_pauli_sum.c`. |
| Theorem proof (full) | ⏳ open theory item |
| Cross-check vs exact diagonalisation | ⏳ pending |
| Larger N (N=6, N=8) | ⏳ tractable now that warmstart ships |

## Gauge-aware warmstart `D_gauge` (shipped)

The construction `D_gauge` is implemented in `src/algorithms/tensor_network/ca_mps_var_d_stab_warmstart.{c,h}` as `moonlab_ca_mps_apply_stab_subgroup_warmstart` and exposed through the var-D alternating-loop config as the new enum value `CA_MPS_WARMSTART_STABILIZER_SUBGROUP`.

Algorithm: symplectic Gauss-Jordan on the Pauli tableau (Aaronson-Gottesman 2004). The k commuting generators are encoded as a `(k, 2n+1)` F2 matrix `[X | Z | r]`. Column operations apply Clifford gates (H, S, CNOT) to the tableau and accumulate in a gate log; row operations multiply generators (free). After elimination the tableau is `{ Z_{q_0}, ..., Z_{q_{k-1}} }` with phase bits `r_p`. The state `|b>` with `b_{q_p} = r_p` is in the +1 eigenspace of every transformed generator. The preparation circuit is the inverse-and-reversed gate log applied to `|b>` (which is reached from `|0^n>` by X gates on the negative-phase pivots). Total cost: `O(n^2)` gates, applied to `state->D` (the CA-MPS Clifford prefactor) so they are absorbed into the Clifford layer for free.

The unit test `tests/unit/test_gauge_warmstart.c` pins:
1. Bell stabilizers `{XX, ZZ}` -> `<XX> = <ZZ> = +1`.
2. GHZ-3 stabilizers `{XXX, ZZI, IZZ}` -> all three expectations = +1.
3. The four interior Gauss-law operators of an N=4 1+1D Z2 LGT (7 qubits) -> all four expectations = +1.
4. Anti-commuting input `{X_0, Z_0}` is rejected with `CA_MPS_ERR_INVALID`.
5. The full var-D entry path with `CA_MPS_WARMSTART_STABILIZER_SUBGROUP` produces a state in the gauge sector immediately after the warmstart phase.

## Hamiltonian gauge invariance — shipped

The headline empirical run (`examples/hep/z2_gauge_var_d.c`) reports four Gauss-law-violation columns per `h`:
- `Gviol_plain`: plain-MPS final state.
- `Gviol_id`: var-D with IDENTITY warmstart, final state.
- `Gviol_gw_init`: var-D with `STABILIZER_SUBGROUP` warmstart, **immediately** after warmstart, **before** any imag-time evolution.
- `Gviol_gw_fin`: same run, after the full alternating loop.

`Gviol_gw_init` is at machine zero by construction — the warmstart projects exactly into the gauge sector.  Earlier prereleases of this example showed `Gviol_gw_fin` drifting to O(1e-2) under imag-time evolution because the kinetic terms in the LGT Hamiltonian (`X_{2x} X_{2x+1} X_{2x+2}` and `Y_{2x} X_{2x+1} Y_{2x+2}`) anti-commuted with `G_x = X_{2x-1} Z_{2x} X_{2x+1}` term-by-term: `X_{2x}` and `Z_{2x}` anti-commute, and that was the only non-trivial overlap, giving an odd anti-commute parity.  The lambda penalty `lambda * (I - G_x)` enforced gauge invariance only **energetically**, not as an exact symmetry of the Hamiltonian.

The fix that ships in `lattice_z2_1d.c` replaces those kinetic Pauli strings with the gauge-invariant pair derived from inserting the Z2 link operator `U_{2x+1} = X_{2x+1}` into the JW expansion (the link operator combined with the JW string `Z_{2x+1}` becomes `XZ = -iY` on the link qubit; the Hermitian symmetrisation of the resulting hop yields):

```
K_x = -(t/2) X_{2x} Y_{2x+1} Y_{2x+2} + (t/2) Y_{2x} Y_{2x+1} X_{2x+2}
```

For each piece the anti-commute count with `G_x = X Z X` (qubits 2x-1, 2x, 2x+1) is:
- `XYY` vs `G_x`: qubit 2x has `X` vs `Z` (anti) and qubit 2x+1 has `Y` vs `X` (anti) → 2 → even → commute.
- `YYX` vs `G_x`: qubit 2x has `Y` vs `Z` (anti) and qubit 2x+1 has `Y` vs `X` (anti) → 2 → even → commute.

And similarly with `G_{x+1} = X Z X` on qubits 2x+1, 2x+2, 2x+3 — both pieces commute term-by-term.  The full Hamiltonian therefore preserves the gauge sector exactly, the lambda penalty becomes redundant for physics in the +1 sector, and `Gviol_gw_fin` stays at machine zero throughout the alternating loop.

`tests/unit/test_z2_lgt_pauli_sum.c` pins the term-by-term commutativity check directly in the Pauli-byte representation; the absence of `XXX` / `YXY` and the presence of `XYY` / `YYX` with the half-amplitude coefficients are also pinned.

Future option for any Hamiltonian where the kinetic terms can't be rewritten this way: add a gauge-aware var-D inner loop that constrains the greedy Clifford gate search to gates `g` for which `g D g^\dagger` still commutes with every generator.  Open research item; not blocking for Z2 LGT.

## Generalisation beyond Z2

The warmstart algorithm itself is independent of the model: it takes any list of commuting Pauli generators on n qubits and emits an O(n^2) Clifford circuit. Concrete next applications:
- **Surface / toric code stabilizers.** Plaquette and star operators are 4-qubit Pauli strings that commute by construction.
- **Repetition / colour codes.** All standard CSS codes.
- **ZN gauge theories with N > 2.** Need single-qudit Cliffords (clock + shift gates), not just qubit Cliffords. The symplectic-elimination skeleton stays the same.
- **Continuous-symmetry sectors via stabilizer projection.** Pick an abelian commuting subgroup of the symmetry algebra; the warmstart then projects into a fixed irrep.

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
