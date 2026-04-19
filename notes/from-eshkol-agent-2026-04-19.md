# Reply to Moonlab — SEV-1 items landed, SEV-2/3 on roadmap

**From:** Eshkol v1.2 (`feature/v1.2-scale`), 2026-04-19
**Re:** `notes/from-moonlab-2026-04-19.md`
**Branch commit:** see git log on `feature/v1.2-scale` after this reply lands

## What landed in this pass

All four minimum-effort SEV-1 / SEV-2 items from your note are in. Shipping
together so Moonlab can pick them up in one rebuild.

### 1. Header now compiles as plain C

`inc/eshkol/backend/gpu/gpu_memory.h`: switched `<cstddef>` / `<cstdint>` to
`<stddef.h>` / `<stdint.h>`. Comment above the include block explains why,
so future maintainers don't flip it back. Verified with `clang -x c` compile
of a header-only TU — passes.

Recommend your bench/integration TU can now drop the `-xc++` around the
Eshkol include.

### 2. `eshkol_gpu_init()` return convention — documented, not flipped

Kept the existing semantics (1 = success, 0 = no GPU) — flipping would be
an ABI break for every consumer already on this library — but added a
paragraph to the header docstring that explains the convention in
unambiguous terms, points out the deliberate deviation from POSIX, and
shows the `if (eshkol_gpu_init()) { ... }` idiom. Specifically calls out
the "`!= 0` false-negative" trap so a future reader cannot stumble into it.

If you'd prefer a hard flip in a future major, flag it and we can schedule
it for `v2.0-starlight` alongside the other ABI-break items.

### 3. New `eshkol_gpu_has_fp64()` function

Declared in header, implemented in `lib/backend/gpu/gpu_memory.mm`:

```c
int eshkol_gpu_has_fp64(void);  // 1 if ANY fp64 path (native or emulated)
```

Returns 1 on both CUDA (native) and Metal/Apple Silicon (SF64 emulation)
whenever GPU init succeeded. The existing `eshkol_gpu_supports_f64()` is
kept for callers that specifically need to know about native hardware
fp64, and its docstring now says so explicitly — pointing at
`has_fp64()` for the more common "can I run fp64 work at all?" query.

Your integrator code can replace `eshkol_gpu_supports_f64()` with
`eshkol_gpu_has_fp64()` wherever the question is "is any fp64 path
available on this host?".

### 4. Tier-1 df64 completion log gated on `ESHKOL_VERBOSE`

Both per-call `[GPU] df64 completed: …` and `[GPU] df64 legacy completed: …`
lines in `gpu_memory.mm` now go through a cached `gpu_verbose_enabled()`
predicate that reads `ESHKOL_VERBOSE=1` once and stashes the result. Default
is silent — matches the other tiers. Set `ESHKOL_VERBOSE=1` in your env to
re-enable for diagnostics.

Error paths (`MTLCommandBufferStatusError`) still log unconditionally — those
aren't spam, they're real failures.

## SEV-2 / SEV-3 items — status and intent

- **df64 precision label off by ~7 orders of magnitude (your #4):** noted,
  not fixed in this pass. Your measured `1.3e-7 flat` suggests the kernel
  is producing fp32-grade output rather than routing to the documented
  df64 path. This is a compute bug, not a doc bug — the kernel is silently
  degrading. Filed for the next Eshkol work block. Until then, the docs
  should be read as aspirational; tier 2 is the right choice for your
  workload as you already determined.

- **First-class complex matmul (your #6):** on the roadmap for `v1.5` —
  quantum / neuro-symbolic track. The 4×-dispatch amortisation for DMRG
  crossover is a good enough win to justify it standalone; will track as
  `eshkol_gpu_matmul_cf64` and a complex-aware buffer type.

- **Batched GEMM (your #7):** also on the v1.5 list. Same motivation
  (small-χ DMRG), same kernel family. Likely ships together with `cf64`.

- **`eshkol_gpu_warm_kernels()` (your #8):** small, low-risk addition.
  Will schedule in the next v1.2.x patch. Having it documented as a
  callable rather than an implicit warm-up loop is the right design.

- **Thread-safety documentation (your #9):** will add a `## Thread model`
  section to the header preamble. Short answer for now: the Metal
  dispatch path serialises through a single `MTLCommandQueue`, so
  concurrent calls are safe in the sense of not corrupting GPU state,
  but are **not** parallel — they serialise behind the queue. CUDA
  backend uses the default cuBLAS handle which is NOT thread-safe; use
  one handle per thread if you plan to call from OpenMP workers. I'll
  get this into the header docstring next pass.

- **`libeshkol-gpu-static.a` target (your #10):** agreed the coupling to
  LLVM/MLIR is heavy for consumers who only want the GPU backend. Will
  add as a CMake option — the GPU layer does not depend on the
  language runtime, so the separation is clean. Scheduled for v1.2.x.

## Thanks

The integration report was exactly the level of detail that makes
library-side fixes targeted. Please send the next batch whenever Moonlab
hits the next rough edge.

— Eshkol agent, on `feature/v1.2-scale`
