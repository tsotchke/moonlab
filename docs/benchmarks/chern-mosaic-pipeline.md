# Chern mosaic pipeline: where we are, what's left

Tracks the Antao-Sun-Fumega-Lado (PRL 136, 156601 (2026)) Chern-
mosaic reproduction end-to-end.  The Bianco-Resta local marker for
site alpha,

    C(alpha) = 2 pi i <alpha| (Q X Y P - P X Y Q) |alpha>,

decomposes into a short sequence of primitives that Moonlab either
already has or still needs.

## Primitives

| Primitive                                              | Status                           |
|--------------------------------------------------------|----------------------------------|
| MPS linear combination  alpha A + beta B               | shipped (`mpo_kpm_mps_combine`)  |
| MPS x MPO application                                  | shipped (`mpo_kpm_apply_mpo`)    |
| Chebyshev moments <phi|T_n(H)|psi>                     | shipped (`mpo_kpm_chebyshev_moments`) |
| Scalar <phi|sign(H)|psi>                               | shipped (`mpo_kpm_sign_matrix_element`) |
| MPS-level sign(H)|ket> and P|ket>                      | shipped (`mpo_kpm_apply_sign`, `_apply_projector`) |
| Diagonal single-site-sum MPO  sum_i f_i O_i            | shipped (`mpo_kpm_diagonal_sum_mpo`) |
| Identity MPO                                           | shipped (`mpo_kpm_identity_mpo`) |
| MPO x MPO multiplication                               | shipped (`mpo_kpm_mpo_multiply`) |
| MPO linear combination                                 | shipped (`mpo_kpm_mpo_combine`)  |
| MPO-level sign(H_tilde)                                | shipped (`mpo_kpm_sign_mpo`)     |
| Filled-band projector as MPO                           | shipped (`mpo_kpm_projector_mpo`) |
| QWZ (or any 2D tight-binding) H as an MPO              | **todo**                         |
| Binary-quantics / QTCI compression for X, Y operators  | **todo** (P5.08 step 5)          |
| Full lattice Chern-marker loop                         | **todo** (step 6)                |

## What landed this sprint

Twelve commits on master took the pipeline from "MPS states only" to
"MPS plus MPO" with bidirectional validation:

- `apply_sign` / `apply_projector` at the MPS level produce P|ket> to
  a Jackson-KPM error of ~3e-6 against dense LAPACK at N_c = 1000.
- `mpo_kpm_diagonal_sum_mpo` gives X-hat, Y-hat, number operator as
  bond-2 MPOs.
- `mpo_kpm_sign_mpo` and `mpo_kpm_projector_mpo` return sign(H) and
  P as MPOs via the MPO x MPO Chebyshev recurrence; Frobenius-norm
  error 2.1e-4 at N_c = 300, chi_max = 32 on L=3 TFIM.
- Cross-route check: mpo_kpm_apply_projector(H, ket) vs
  mpo_kpm_apply_mpo(mpo_kpm_projector_mpo(H), ket) agree on
  <ket|P|ket> to 4e-7.
- `<alpha|Q X P|alpha>` composed from three primitives matches dense
  to 3.7e-7.

## What the final two steps need

### Step 5: QWZ Hamiltonian as an MPO

For the 1D (trivial-first) and 2D (paper-faithful) QWZ models the
bond dimension of the Hamiltonian MPO scales with the linear lattice
size via snake-ordering (y-hops are long-range when flattened to 1D).
Moonlab already has the finite-automaton builder for
Heisenberg + DMI (`mpo_2d_create`) and a TFIM builder; neither fits
QWZ's hop matrices `T_x = (sigma_z + i sigma_x) / 2`.  Either:

- Build a new `mpo_qwz_create` finite-automaton MPO following the
  pattern of `mpo_tfim_create`: on-site  `m sigma_z`, forward and
  adjoint hops between adjacent MPS sites, long-range y-hops
  spanning one row of the snake.
- Or cast QWZ into the existing bond-list interface
  (`mpo_from_bond_list`) after extending it to accept general 2x2
  hop matrices rather than Pauli-pair interactions.

Either approach is ~200 LOC.  A verification test diagonalises the
resulting MPO dense and compares to a direct construction of QWZ on
a 2x2 or 3x3 lattice.

### Step 6: QTCI for X, Y and final lattice loop

With step 5, the full pipeline is literally:

    H_mpo  = mpo_qwz_create(L, m, params);
    X_mpo  = mpo_kpm_diagonal_sum_mpo(N_sites, Z, x_per_site);
    Y_mpo  = mpo_kpm_diagonal_sum_mpo(N_sites, Z, y_per_site);
    P_mpo  = mpo_kpm_projector_mpo(H_mpo, params);
    Q_mpo  = mpo_kpm_mpo_combine(I_mpo, 1.0, P_mpo, -1.0, chi);
    for each alpha in lattice:
        |alpha>    = computational-basis MPS
        |Q X Y P|a = apply_mpo(Q, apply_mpo(X, apply_mpo(Y, apply_mpo(P, alpha))))
        |P X Y Q|a = apply_mpo(P, apply_mpo(X, apply_mpo(Y, apply_mpo(Q, alpha))))
        C(alpha)   = 2 pi i (<alpha|Q X Y P|alpha> - <alpha|P X Y Q|alpha>)

Without QTCI the MPS chain length is the full site count -- fine
for 16-64 sites, breaks past that.  QTCI compresses the chain to
log_2(N_sites) qubits with bond dimension determined by the function
structure (rank-linear for separable f(x, y); rank-O(1) for smooth
modulations).  That's where the 10^6-to-10^8 capability the paper
headlines actually lives.

`src/algorithms/topology_realspace/mpo_kpm.*` is structured so the
QTCI position operators become drop-in replacements for
`mpo_kpm_diagonal_sum_mpo` -- same call site, smaller chain length,
bigger bond dim per tensor.

## Reproducing the current state

Every bench in this module writes a provenance manifest (see
`docs/benchmarks/reproducible-benchmarks.md`).  The sparse-stencil
Chern mosaic is already reproducible:

```
MOONLAB_CHERN_OUT_PPM=/tmp/chern.ppm \
MOONLAB_MANIFEST_OUT=/tmp/chern_manifest.json \
./build/bench_chern_mosaic_hq \
  --L 64 --n 4 --V0 0.3 --Q 0.8976 --n-cheby 200
```

yields a 56 x 56 PPM image, CSV per-site values, and a JSON manifest
tying the numbers back to a specific git SHA and host config.
