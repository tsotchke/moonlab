# Numerical-edge / uninitialised-memory findings

Bug-hunt of the numerical-correctness and uninitialised-memory surface that the
differential / oracle / fuzz lanes do not probe. Harnesses live in
`tests/numerical/`, driven by `scripts/run_numerical.sh`; the CI legs
(valgrind + MSan on Linux) are in `.github/workflows/numerical.yml`.

Environment for the local run: macOS 15 / arm64 (Apple Silicon), Homebrew
clang 21, library at v1.1.0 (ABI 0.4.0), CPU-only Release build.

---

## BUG 1 (HIGH) -- one-sided Jacobi SVD fallback produces a non-orthonormal U on any rank-deficient matrix

`tensor_svd`'s no-LAPACK fallback (one-sided Jacobi) returns a left factor `U`
that is **not** an isometry (`U^H U != I`) whenever the input has one or more
zero / negligible singular values. The columns are individually finite and the
`A = U S Vh` reconstruction still holds (the offending columns carry `S = 0`),
so any check that only verifies reconstruction misses it -- but `U` is not
unitary, which the code's own comment says "breaks DMRG's left-canonical form".

- Suspect code: `src/algorithms/tensor_network/tensor.c:1356-1366`, the U-column
  extraction of the `m >= n` one-sided Jacobi path. The `S_work[i] <= 1e-15`
  branch (line 1363) fills the left vector with the standard basis vector `e_i`:

  ```c
  if (S_work[i] > 1e-15) {
      for (uint32_t row = 0; row < m; row++)
          U_work[row * min_mn + i] = A_work[row * n + i] / S_work[i];
  } else {
      for (uint32_t row = 0; row < m; row++)
          U_work[row * min_mn + i] = (row == i) ? 1.0 : 0.0;   /* <-- BUG */
  }
  ```

  `e_i` is not orthogonalised against the computed rank-space columns (which are
  dense), so `U` loses orthonormality. LAPACK's `zgesvd` completes the null space
  to a proper orthonormal basis; this fallback does not.

- Minimal deterministic repro -- rank-1 3x3 real `A = u v^T`,
  `u = (1,2,3)`, `v = (1,1,1)` (harness case `rank1_3x3`):

  | path                         | singular values             | max abs(U^H U - I) |
  |------------------------------|-----------------------------|--------------------|
  | LAPACK zgesvd (shipping mac) | `6.48074, 6.7e-16, 5.6e-32` | `6.66e-16` (ok)    |
  | one-sided Jacobi (fallback)  | `6.48074, 6.6e-17, 0`       | `8.02e-01` (BUG)   |

  Second repro, rank-2 6x6 (harness case `rank2_6x6`): fallback max abs(U^H U - I) = 6.40e-01.

- Reachability: the fallback is compiled when the platform has no LAPACK
  (per the tensor.c comments, "primarily Windows clang-cl without an OpenBLAS
  package"), and is force-selectable with `-DQSIM_FORCE_SVD_FALLBACK`. The
  shipping macOS (Accelerate) and Linux (OpenBLAS) paths are unaffected.

- Impact: rank-deficient bond matrices are the common case in MPS/DMRG at low
  entanglement -- product states, post-truncation bonds, boundary tensors. On a
  no-LAPACK build, every SVD of such a bond yields a non-isometric `U`, silently
  corrupting left/right-canonical form and therefore DMRG/TDVP energies, norms,
  and observables. The failure is invisible to reconstruction-only tests.

- Reproduce:
  ```
  cmake -S tests/numerical -B build/num -DMOONLAB_LIB_DIR=build
  cmake --build build/num -j2 --target t_svd_lapack t_svd_fallback
  ./build/num/t_svd_lapack     # 0 fails
  ./build/num/t_svd_fallback   # 2 fails: rank1_3x3, rank2_6x6 (U not orthonormal)
  ```

- Fix sketch (NOT applied -- reporting only): after extracting the rank-space U
  columns, complete `U` to an orthonormal basis over the zero-singular-value
  columns (Gram-Schmidt / a QR of the orthogonal complement), exactly as
  `zgesvd` returns a full orthonormal `U`.

---

## Clean subsystems (verified, no defect)

All harnesses below report `fails=0` on the shipping paths.

- **Extreme rotation angles** (`t_rotations`, 217 checks). `gate_rx/ry/rz/u3/phase`
  and a 4-qubit VQE/QAOA-style ansatz at `theta in {0, 1e-300, 1e-16, pi-1e-15,
  2pi, 1e6, -1e6, pi, pi/4}`: analytic single-qubit amplitudes match a
  long-double reference to 1e-13, norm preserved to 1e-13, no NaN/Inf.
- **Denormal handling**: `rx(1e-300)` on `|0>` yields `a1.im = -5.000e-301` --
  a subnormal, **not** flushed to zero. FTZ/DAZ is confirmed OFF for the library
  (as intended; `-ffast-math` is not enabled).
- **Hermitian eigendecomposition** -- both the shipping Accelerate `zheev`
  (`t_eigen_lapack`) and the forced complex-Hermitian Jacobi fallback
  (`t_eigen_jacobi`, `matrix_math.c` built `-U__APPLE__`), 76 checks each. Pauli
  X/Y/Z, exactly-degenerate `diag(3,3,1)`, near-degenerate split, rank-1
  projector, `diag(1e14, 1, 1e-14)` spread, all-`1e-160` near-zero-norm matrix,
  seeded random Hermitian `n in {2,4,8,16}`: eigenvalues descending, reconstruction
  `H = V D V^H`, orthonormality, and residual `||Hv - lambda v||` all within tol.
- **LAPACK SVD** (`t_svd_lapack`, 50 checks): square / wide / tall / rank-1 /
  rank-2 / near-degenerate / `diag(1e14,1,1e-14)` / identity / near-zero /
  all-zero. Reconstruction, orthonormal U and Vh, descending nonneg singular
  values. (This is the differential control for BUG 1.)
- **DMRG vs exact diagonalization** (`t_dmrg`): open-chain TFIM
  `H = -sum Z_iZ_{i+1} - g sum X_i` for `N in {6,8,10}`, `g in {0.5, 1.0
  (critical), 1.5, 2.0}`, compared to a dense eigensolve of the full `2^N`
  Hamiltonian. All 12 cases agree to < 1e-3 (typically < 1e-6), including the
  small-gap critical point `g=1`.
- **TDVP** (`t_dmrg`): real-time energy conservation drift `2.5e-12` and
  `1.6e-12` over 200 steps (N=8, g in {1.0,1.5}); imaginary-time projection
  converges toward the ground state (`E: -8.548 -> -9.838` at N=8 g=1, matching
  the exact ground `-9.83795`) with `log_norm_factor` accumulation staying
  finite (final norm = 1).
- **QFT/IQFT round-trip** (`t_qft`): random product states, 4..22 qubits,
  round-trip returns the input to a dimension-scaled tolerance; GHZ at 22 qubits
  exact; no NaN.
- **Measurement / normalization near machine epsilon** (`t_measure`, 15 checks):
  the `< 1e-15` collapse guards trip cleanly with no NaN written to the buffer;
  just above the guard, collapse renormalizes to unit norm; `quantum_state_normalize`
  on a `1e-200` (norm-underflowing) state returns an error rather than dividing
  by ~0.

## Observations (not bugs)

- **Deep-circuit accumulation**: 200k unitary ops (T/T-dagger, RZ +/- pairs)
  drift the norm by `~2.8e-12` (observed) with no inter-gate renormalization --
  ordinary FP rounding, well within tolerance. Callers running very deep
  circuits and needing bit-tight norm should renormalize periodically.
- **Measurement guard asymmetry** (`measurement.c`): on a below-floor outcome
  `measurement_single_qubit` returns `-1` (and leaves the state un-renormalized),
  while `measurement_partial` silently *skips* renormalization and returns the
  outcome. Neither NaNs, but the two APIs leave a broken (unnormalized) state on
  the same condition with different return contracts.
- **POVM vs weak-measurement guard threshold**: `measurement_povm` guards with
  `p <= 0.0` whereas `measurement_weak_z` guards with `p <= 1e-15`
  (MEASUREMENT_SMALL_NORM). Both avoid a divide-by-zero NaN; the POVM path admits
  tiny-but-positive `p` and renormalizes a numerically fragile post-measurement
  state.

## Uninitialised memory

valgrind memcheck and MemorySanitizer are both **unavailable on macOS arm64**
(no valgrind port; `-fsanitize=memory` is unsupported for `arm64-apple-darwin`).
Local best-effort probes, all clean:

- **Heap poisoning** (`MallocPreScribble`=0xAA fresh / `MallocScribble`=0x55
  freed / `MallocGuardEdges`): every harness matches its clean-run baseline --
  no uninitialised-read propagation surfaced across gates, measurement, state,
  tensor SVD/eigen, DMRG, and TDVP allocations.
- **AddressSanitizer** on the shipping library paths (gates / measurement /
  DMRG / TDVP / LAPACK SVD+eigen / QFT): no heap-buffer-overflow, use-after-free,
  stack-buffer-overflow, or global-buffer-overflow.
- **ASan + UBSan** on the directly-compiled fallback algorithms (one-sided
  Jacobi SVD, Householder QR, complex-Hermitian Jacobi eigen): no memory errors
  and no undefined behaviour (the 2 SVD-fallback failures are the algorithmic
  BUG 1, not a memory error).

The authoritative uninitialised-VALUE sweep of the `core|tn|ca_mps|algorithms`
test binaries and the numerical harnesses runs under valgrind memcheck (and MSan
on the pure-C fallbacks) in the Linux CI jobs `valgrind-uninit` /
`msan-fallback` in `.github/workflows/numerical.yml`.
