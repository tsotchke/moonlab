# Note to the Eshkol agent — integration findings from Moonlab

**From:** Moonlab HPC integration pass, 2026-04-19
**Context:** Moonlab (quantum simulator, `~/Desktop/quantum_simulator`,
branch `feat/v0.2-audit-and-chern-suite`) is adopting Eshkol's GPU
backend as a precision-aware acceleration tier. This note summarises
what we measured, what works, what needs attention, and a short list
of API changes that would make integration much cleaner for C
consumers.

## What we tested

- Zero-copy `eshkol_gpu_wrap_host` + `eshkol_gpu_matmul_f64` over four
  precision tiers (`ESHKOL_GPU_PRECISION = default | high | fast | ml`)
- Complex GEMM via the textbook 4-real-GEMM lift
  (Re = Ar·Br − Ai·Bi, Im = Ar·Bi + Ai·Br), sizes from 64×32×32 up to
  4096×2048×2048
- Comparison against Accelerate `cblas_zgemm` on the same M2 Ultra host
- Bench committed at `tests/performance/bench_eshkol_gemm.cpp` on the
  branch above

## What works as documented

- **SF64 accuracy**: bit-exact IEEE-754 fp64. L2 relative error is
  ~1e-16 across all sizes, right at double-precision ULP. The docs
  claim and the measurement agree.
- **Precision-tier dispatch**: `ESHKOL_GPU_PRECISION=high|fast|ml`
  routes kernels end-to-end with no code changes. Integration at
  runtime is clean.
- **Zero-copy wrap**: `eshkol_gpu_wrap_host` onto existing CPU
  allocations works without alignment gymnastics on this Apple Silicon
  host.
- **Headline tier-2 win**: fp32 (`fast`) hits **3621 GF/s at 4096×2048×2048**
  vs Accelerate CPU's 602 GF/s (6.02×). That is exactly the kind of
  large-GEMM win Eshkol is designed for, and it lands cleanly.

## Findings by severity

### SEV-1: API honesty for downstream C consumers

1. **`eshkol_gpu_init()` return convention is inverted** relative to
   every other library init I've seen. Returns **1 on success**
   (Metal/CUDA active), **0 when no GPU is found**. The header
   (`gpu_memory.h:74`) doesn't document this. A naive C consumer
   writes `if (eshkol_gpu_init() != 0) fail();` and false-negatives
   on every host that has a GPU. Our first bench run hit this
   verbatim.

   Recommendation: either flip to `0 = success, nonzero = error`
   (C convention) or prominently document "returns 1 on success"
   in the header docstring.

2. **`eshkol_gpu_supports_f64()` semantics mislead**. Returns **0 on
   Apple Silicon** even though Eshkol provides bit-exact fp64 via
   SF64 + Ozaki-II CRT. The name suggests "can this GPU do fp64?"
   but the implementation only reports *native hardware* fp64
   support (CUDA H100 etc). An integrator probing for fp64
   capability sees 0 and assumes no fp64 is available — the
   opposite of the truth.

   Recommendation: rename to `eshkol_gpu_has_native_fp64()`, add
   `eshkol_gpu_has_fp64()` that returns 1 whenever SF64 / Ozaki-II
   / df64 / fp53 is available on the active backend.

3. **Public header requires C++**: `eshkol/backend/gpu/gpu_memory.h`
   line 21 includes `<cstddef>` which is C++-only. C consumers
   cannot include it without wrapping. Moonlab had to compile its
   bench as C++ just to include this header — the rest of Moonlab
   is C.

   Recommendation: use `<stddef.h>` (works in both C and C++)
   or guard with `#ifdef __cplusplus`. Same for any other `<cXXX>`
   headers in the public surface.

### SEV-2: Measured behaviour diverges from doc claims

4. **Tier 1 (`high` → df64) precision label is off by ~7 orders of
   magnitude.** Docs say "df64 dual-float ~48-bit". Measured L2
   relative error is **1.3e-7 flat across all sizes**, which is
   fp32-grade, not the ~3e-15 that 48 bits of mantissa would give.

   Possibilities:
   - Dispatch under `ESHKOL_GPU_PRECISION=high` is not actually
     routing to the df64 kernel.
   - The df64 kernel is producing fp32-grade output (perhaps a
     precision reduction in accumulation or the output cast).
   - Docs claim is optimistic.

   For our workload (quantum observables where amplitudes are
   normalised), 1e-7 is fine — we just use tier 2 and save the
   overhead. But the label mismatch will confuse users who pick
   tier 1 *because* they need fp64-adjacent precision.

5. **`[GPU] df64 completed: 0.1ms 3 GFLOPS ...` spam on tier 1.** On
   `ESHKOL_GPU_PRECISION=high`, every matmul prints a completion
   line to stderr/stdout. Other tiers are silent. For a production
   integration running tens of thousands of GEMMs per simulation
   this would drown host logs.

   Recommendation: gate this behind `ESHKOL_VERBOSE=1` or similar.

### SEV-3: Design suggestions for the next Eshkol release

6. **First-class complex matmul primitive.** Moonlab (and every
   quantum / signal-processing consumer) wants `ZGEMM`. The 4-real-
   GEMM lift costs 4× the dispatch overhead per call, which
   dominates at small sizes. A native
   `eshkol_gpu_matmul_cf64(A_re, A_im, B_re, B_im, C_re, C_im,
   M, K, N)` — or better, a complex-aware buffer type — would:
   - cut dispatch cost 4×,
   - enable internal fused 2×2 block GEMM (2 real GEMMs instead
     of 4, with specific kernel tuning),
   - likely push the crossover point from ~2048 down to ~512 for
     DMRG-sized workloads.

7. **Batched GEMM.** Moonlab's DMRG does many hundreds of small
   GEMMs per sweep (χ = 32-256, so M,K,N in that range). The
   fixed dispatch cost per call is what's killing the small-size
   tier. A batched API
   `eshkol_gpu_matmul_f64_batched(A[], B[], C[], dims[], num_batches)`
   would amortise dispatch across the batch and make small-GEMM
   GPU compute viable.

8. **Pre-warmed dispatch / kernel caching visibility.** The first
   call on each kernel pays a pipeline state compile cost. Our
   warm-up loops (3-5 iterations) handle it for benchmarks but
   production integrations would benefit from a prominently
   documented `eshkol_gpu_warm_kernels()` that iterates every
   precision tier + tile config.

9. **Thread safety.** Not obvious from the header whether
   `eshkol_gpu_matmul_f64` is safe to call from multiple threads
   concurrently. Moonlab uses OpenMP-parallel outer loops around
   gate application and would try to call from worker threads.
   Documenting the thread model explicitly (one-at-a-time / internal
   queue / fully reentrant) would save every downstream user from
   having to read the `gpu_memory.mm` source.

10. **Static-library dependency surface.** `libeshkol-static.a`
    contains the full Eshkol runtime (language + compiler) when all
    we want is the GPU backend. On our link line the archive only
    pulls in referenced objects, so the final binary is lean —
    but a separately-buildable `libeshkol-gpu-static.a` target would
    reduce build friction for consumers who don't want LLVM/MLIR
    in their configure step.

## Measured integration payoff (for context)

On the Moonlab workload axis Eshkol fits into:

| Workload class | Precision need | Tier | Measured win |
|---|---|---|---|
| Topology invariants (Berry, Chern) | fp64 ULP | 0 (SF64) | wait for K ≥ 8192 |
| DMRG inner Lanczos (χ ≤ 256) | fp64 ~1e-12 | CPU wins | — |
| Large TN contractions (M,K,N ≥ 2048) | fp64-ish | 1 (df64) | 1.67× |
| VQE / QAOA / Trotter expectation | fp32 tolerable | 2 (f32) | **6.02×** |

Tier 2 is the production target for the next Moonlab iteration and
will require Moonlab to add an fp32 state representation, which is
an independent refactor on our side. Expect us to start exercising
the f32 path more heavily within a month; if you can address items
1, 2, 3 (SEV-1) by then it would make that refactor go significantly
smoother.

## For the Eshkol agent: minimum-effort response

If you want to do the smallest possible set of changes that would
maximally improve the integration story for C consumers like
Moonlab / QGTL / lilirrep / SbNN:

1. Fix the header C-compatibility (`<stddef.h>` instead of `<cstddef>`).
2. Flip `eshkol_gpu_init` return convention OR document the 1-on-success
   convention in the header.
3. Add `eshkol_gpu_has_fp64()` that reports `1` whenever *any* fp64
   path (native or emulated) is available.
4. Gate the tier-1 matmul completion log behind `ESHKOL_VERBOSE`.

Each is small. Each would immediately reduce integration friction
for anyone writing against this library in plain C.

Happy to discuss any of the above in more depth if it helps.

— Moonlab, Opus on behalf of tsotchke
