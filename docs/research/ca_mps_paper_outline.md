# Paper outline: variational-D Clifford-augmented MPS

Working draft for the arXiv preprint covered by task #51.  Bench
results in `benchmarks/results/ca_mps_*_2026-04-29.json`.

## Title
**Variational-D Clifford-augmented matrix product states for stabilizer-adjacent ground states**

(More candid than "for ground-state problems" -- the §6.4 negative
result on Heisenberg makes the model-dependence load-bearing for
the claim, and trying to hide it would invite reviewer pushback.)

## Abstract (draft)

Clifford-augmented matrix product states (CA-MPS) factorize a state
as |ψ⟩ = D|φ⟩ where D is a Clifford unitary stored as an Aaronson-
Gottesman tableau and |φ⟩ is an MPS.  Existing fixed-D CA-MPS works
well for stabilizer-rich circuits but is no better than plain MPS
on ground-state problems where the natural D is unaligned with the
target state's entanglement structure.  We introduce variational-D
CA-MPS: alternating imaginary-time evolution on |φ⟩ with greedy
local-Clifford search on D, optionally seeded with a model-specific
basis-rotation warmstart.  On the 1D transverse-field Ising model,
variational-D matches plain DMRG energy convergence within 0.7%
while reducing |φ⟩'s half-cut entanglement entropy by 4× to 1430×
across the entire phase diagram, including the quantum critical
point.  The method is genuinely model-dependent: it does not reduce
entropy on the SU(2)-symmetric XXZ Heisenberg point or in the
gapless XXZ regime.  We characterize the regimes where variational-D
helps and document the failure modes honestly.

## 1. Introduction

- TN methods + bond-dim wall (cite DMRG, MPS, PEPS reviews)
- Clifford-stabilizer structure can be tracked separately (cite
  Bravyi-Smith-Smolin, Stim)
- Combining the two: extended-stabilizer formalism, then CA-MPS
  (cite Qian et al. 2024)
- Our contribution: variational-D extension that *finds* the right
  Clifford rather than relying on a hand-supplied one

## 2. Background

### 2.1 MPS, DMRG, half-cut entanglement
### 2.2 Clifford tableau formalism (one-paragraph review)
### 2.3 Fixed-D CA-MPS (cite Qian)

## 3. Variational-D CA-MPS

### 3.1 The state |ψ⟩ = D|φ⟩ and the bilinear cost ⟨ψ|H|ψ⟩

### 3.2 Alternating optimization
- |φ⟩-update: imaginary-time evolution under conjugated Pauli sum
- D-update: greedy local-Clifford gate search at fixed |φ⟩
- Convergence proof (energy descent in both updates)

### 3.3 Warmstart catalog
- IDENTITY (D=I)
- H_ALL (basis rotation)
- DUAL (TFIM dual transformation: H_all + CNOT-chain)
- FERRO (cat-state encoder: H_0 + CNOT-chain)

### 3.4 Composite-move extension
- 2-gate compositions for escape from 1-gate local minima
- Runtime cost analysis

## 4. Implementation

- Open-source C library (Moonlab/`src/algorithms/tensor_network/ca_mps_var_d.{c,h}`)
- ABI-stable C exports for downstream integration (QGTL etc.)
- Unit tests + multi-warmstart benchmark harness
- License: MIT

## 5. Results

### 5.1 TFIM phase sweep (Figure 1)
Table from `benchmarks/results/ca_mps_var_d_vs_plain_dmrg_2026-04-29.json`:
- 5 g values × 2-3 N values
- Plain DMRG E vs var-D E (energy convergence within 0.7%)
- S_psi vs S_phi (entropy reduction 4× to 1430×)
- Entropy ratio bar chart with regime labels

### 5.2 Direct Clifford disentangler (post-DMRG)
- Validates that the entropy reduction matches the oracle proof
  (5x-50x at N=6,8) when using the dual warmstart Clifford
- Documents the n=10 truncation-blowup failure mode

### 5.3 XXZ Heisenberg (negative result)
- Shows the method is model-dependent
- Δ ≤ 1: ratio ≈ 1.0 (gapless SU(2))
- Δ ≥ 1.5: ratio 4-5× (Ising-anisotropy)

### 5.4 [If kagome results positive] kagome 12-site application
### 5.5 [If kagome results negative] discussion of what's needed

## 6. Discussion

- When does variational-D help?
  - Stabilizer-adjacent GS (TFIM, surface code, Ising-anisotropy)
- When does it fail?
  - Gapless SU(2) (Heisenberg point)
- What would extend the reach?
  - CA-PEPS for 2D
  - Multi-gate composite search for non-local Clifford basins

## 7. Conclusion

- Method is a real, validated tool for a specific class of
  ground-state problems
- Open-source library; downstream integration ABI-stable
- Caveats honestly characterized

## Appendices

### A. Convergence parameters used in each experiment
### B. Hardware + wallclock times
### C. Reproducibility: exact commands + commit SHAs
### D. Composite-move algorithm pseudocode

## Reproducibility

- Repo: `https://github.com/tsotchke/quantum_simulator`
- Pinning commit SHA: ${THIS_COMMIT}
- Reproduction:
  ```
  cmake -S . -B build -G Ninja -DQSIM_BUILD_EXAMPLES=ON
  cmake --build build --target example_ca_mps_var_d_vs_plain_dmrg
  ./build/example_ca_mps_var_d_vs_plain_dmrg
  ```
- All bench JSON output in `benchmarks/results/`

---

Status as of 2026-04-29: outline + abstract drafted, results section
populated through 5.3 with shipped numbers, kagome (5.4) result
pending experiment completion.  Next steps are to run the kagome
test to completion, write the introduction + background, and
prepare the figures.  Estimated 2-3 days to convert to arXiv-ready
draft.
