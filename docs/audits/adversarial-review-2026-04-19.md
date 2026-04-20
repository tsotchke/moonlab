# Adversarial pre-release audit, 2026-04-19

Written as a hostile reviewer would.  No self-congratulation.  Items
are ordered by "how embarrassing would this be at launch".

## Headline finding: the "Bell-verified QRNG" was a no-op

`src/algorithms/bell_tests.c:256-257` (pre-patch):

```c
// Reset to |00⟩ and create Bell state
quantum_state_reset(&test_state);
create_bell_state_phi_plus(&test_state, qubit_a, qubit_b);
```

Every invocation of `bell_test_chsh(state, ...)` cloned the caller's
state, **then immediately discarded it and rebuilt a hard-coded
|Phi+>**.  The `state` parameter was a lie -- whatever you passed in,
the function would measure CHSH on |Phi+> and return ~2.828.

This wasn't a subtle bug.  The clone and reset are two consecutive
lines of code.  It was unreviewed.  It landed in a file that ships as
part of the "Bell-verified" headline claim.

Consequences audited:

1. **`moonlab_qrng_bytes` BELL_VERIFIED mode was theatre.**
   `src/applications/qrng.c:937` passes the QRNG's evolving
   `ctx->quantum_state` (which is mutated by Hadamards, RZs, CNOTs,
   measurements across every entropy epoch) directly to
   `bell_test_chsh`.  Because `bell_test_chsh` silently overwrote it
   with |Phi+>, the health check always passed regardless of whether
   the QRNG was doing anything quantum.  The marketing claim "Bell-
   verified QRNG" was structurally vacuous for the entire life of
   0.1.x.

2. **Every README and badge referencing CHSH = 2.828 was "correct"
   by accident.**  `bell_test_demo.c` prepares |Phi+> explicitly
   before calling `bell_test_chsh`; the overwrite was redundant but
   harmless in that path.  Every other caller (including the Python
   `BellTest.chsh_test(state=|++>)`) got silently lied to.

3. **The xfail I added to `test_chsh_product_state_classical` in the
   previous commit was the first test case in Moonlab's history that
   exercised `bell_test_chsh` on a non-Bell input.**  The xfail
   caught the bug; the audit pass described it; no fix was
   attempted.  This audit attempts the fix.

### Fix applied

- `bell_tests.c`: removed the two clobber lines.  `bell_test_chsh`
  now respects its input; `measure_correlation` already clones the
  state per setting, so the caller's state is preserved.
- `qrng_v3_verify_quantum` (`qrng.c:926`): now constructs a
  temporary `|Phi+>` state explicitly, runs the CHSH test there,
  and frees it.  This preserves the health-check behaviour of the
  pre-audit code (always ~2.828) but makes the semantics honest:
  the function now verifies *that the quantum-simulation plumbing
  produces CHSH = 2sqrt(2) on a freshly prepared Bell state*, not
  "the evolved state is still a Bell state" (which it never was).
- Python xfail removed; test_chsh_product_state_classical passes.

The "Bell-verified" language in README needs rewording.  Flagged
below.

---

## Everything else

Ordered by severity.  Each item is something an external reviewer
will notice on their first pass.

### 1. Foundational primitives have thin test coverage

The MPO-KPM work (P5.08) is implemented on top of `tensor_contract`,
`tensor_transpose`, `tensor_reshape`, `tensor_svd`, and the SVD-bond
compressor.  Their direct tests cover only rank-2, real-valued,
power-of-2, square tensors.  The indirect coverage through
`test_mpo_kpm` uses complex-valued rank-4 tensors and catches some
bugs, but there is no direct unit test for `tensor_contract` against
a hand-computed numeric reference on an arbitrary shape.

If any of these primitives has an off-by-one permutation bug on a
rank-5 or complex-rank-3 case, everything MPO-KPM reported in the
last week was right *by coincidence*.  The `tn_apply_mpo` axis bug
that this sprint fixed is an existence proof of this failure mode.

**What the reviewer sees:** 56 ctest green, but grepping for
`tensor_contract` in `tests/` returns one test on a [2,3]@[3,2] shape.
That's not proportionate to how much of the library leans on it.

### 2. "The full Chern-mosaic pipeline is closed" overclaims

`docs/benchmarks/chern-mosaic-pipeline.md` describes six steps.  This
sprint shipped steps 1-4 (MPS/MPO Chebyshev, position operators,
projector as MPO) and a generic dense->MPO adapter that technically
closes step 5 at O(2^L) storage -- i.e., it is *not* what the paper
needs, because the paper needs QTCI-compressed position operators
and a direct finite-automaton QWZ MPO to reach 10^6 sites.

The commit message and the doc update correctly call this out, but
the cross-document phrasing is loose enough that a reviewer reading
only the audit summary will think the 10^6-site hero capability
ships in 0.2.  It does not.

Recommendation: reword chern-mosaic-pipeline.md so the distinction
between "scalar Bianco-Resta pipeline works end-to-end on generic
small H" (true) and "lattice Chern mosaic at paper-scale sites"
(false, blocked on QTCI) is impossible to miss.

### 3. `tn_apply_mpo` bugfix is not directly tested

I fixed the axis-order bug in `tn_apply_mpo` this sprint.  No
ctest target exercises the fixed path; the code-level verification
is inductive ("the transpose pattern matches
`mpo_kpm_site_apply_mpo`, which is tested").

Anyone who touches `tn_apply_mpo` again has zero regression
protection.  The right move is a direct test that builds a small MPO
(e.g., sigma_z on site 0), applies it to `|+...+>` via
`tn_apply_mpo`, and compares to the analytic result.

### 4. OOM-path memory fixes are not test-driven

The six memory-safety patches in `chemistry.c` and `measurement.c`
return early on `malloc`/`realloc` failure.  I have no test that
injects an allocator failure.  I claim the fixes work; I cannot
prove it.

At minimum, this warrants a comment in the code noting that the path
is untested, and a follow-up that hooks a failing allocator (e.g.,
via `malloc_zone` or `LD_PRELOAD`) into one CI tier.

### 5. Performance numbers are single-run, single-host

The Eshkol bench shows 2.15x at 4096^2 on M2 Ultra.  One run, one
host, no stddev, no Linux reproduction.  The per-tier crossover
thresholds in `tensor_matmul` (2^27 for FAST tier etc.) were
calibrated against the same one run.

A reviewer on an M3 Pro or an Intel workstation will get different
crossover behaviour and will not know whether they are seeing a
"doesn't fit" signal or the claimed 2x.  The reproducibility
manifest records git SHA and host info but does not include a
distribution of repeated runs.

### 6. Benchmark manifests are write-only

I built the manifest infrastructure and wired it into four benches.
I never documented how to *diff* two manifests, never checked in a
canonical expected-manifest, and never built a CI job that regresses
against one.  The machinery exists to enforce "no surprise
slowdowns"; none of it is enforced.

### 7. MPI, OpenCL, Vulkan, cuQuantum all unverified

These backends have 1000+ LOC of implementation each.  None is in
the default ctest run.  None has been built against its real SDK on
CI.  The README lists them as "optional backends" without flagging
that they are *unexercised*, which is the key fact a reviewer needs.

### 8. WebGPU excluded from the default ctest run

`webgpu_unified_smoke` is gated behind the existence of
`dist/index.mjs`, which a fresh-clone build does not produce.
Result: the default `ctest` on a clean checkout gives zero WebGPU
coverage.  The README mentions "WebGPU/JS front-end" as a
first-class capability without this caveat.

### 9. Python bindings had six struct-layout bugs

These were all landed in the same audit commit earlier today.  But
the severity is worth highlighting: every Python user of
`solve_maxcut`, `grover.search`, `bell_test.chsh_test`, or
`vqe.solve_h2` since 0.1.x hit ctypes reading off-offset field data.
Most produced segfaults (users had no way to reach a "hello world").
This means the "Python bindings are complete" narrative in the
README was never even tested by *one* downstream consumer.

A reviewer will correctly wonder what else in the bindings surface
was never exercised.  The answer is: probably more than we know.

### 10. Shor-ECDLP resource estimator numbers are uncross-checked

The estimator lives in `src/algorithms/shor_ecdlp/` and has a test
of its own.  None of those tests cross-check against Gidney-Drake-
Boneh's published figures.  If the estimator is off by a constant
factor we will not know until a reviewer checks it against
`arXiv:2603.28846`.

---

## Recommended gate for v0.2.0

The "two remaining items" framing in `v0.2.0-readiness.md` was
optimistic.  Hard gates before a tag:

1. ~~**bell_test_chsh correctly honours its input**~~ **(fixed this
   audit)**
2. README rewording to stop claiming "Bell-verified QRNG" in the
   sense readers would assume (fixed: the "verification" is a
   plumbing health check, not a proof of quantum advantage in the
   emitted bytes).
3. Direct test for `tn_apply_mpo` on a hand-computable case.
4. Rewording of `chern-mosaic-pipeline.md` so the 10^6-site
   capability is visibly *future work*, not shipped.
5. A "limitations" section in README that documents what this
   release explicitly does NOT do: MPI scaling, non-Metal GPU
   backends, paper-scale Chern mosaic, QTCI.

Items 1-2 are done in this commit.  3-5 are follow-ups.

## Closing honesty

This document exists because the previous audit missed the
bell_test_chsh bug entirely.  The bug's severity (three years of
"Bell-verified" being theatre) is the single strongest argument that
one-pass audits are insufficient and that every external-facing
capability claim needs a test that would *fail* if the capability
were absent.  Most of Moonlab's current test suite only exercises
success paths.
