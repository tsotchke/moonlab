# Response to the Eshkol PR #299 note — from the Moonlab side, 2026-07-17

_Untracked pickup file (kept out of Moonlab's public git history, same as the note)._

## Technical confirmation — everything #299 needs is present and stable on master

Verified on current master:

- `vqe_compute_qgt` is declared `MOONLAB_API` in `src/algorithms/vqe.h:583`.
- Implementation `src/algorithms/vqe_qng.c:51`:
  `int vqe_compute_qgt(vqe_solver_t *solver, const double *params, double *qgt_out)`
  — exactly your signature; returns `0` on success, `-1` on error.
- `qgt_out` is written **row-major n×n**, `n = solver->ansatz->num_parameters`, symmetric.
- `src/algorithms/vqe_qng.c` is in the `libquantumsim` source list (`CMakeLists.txt:738`);
  the symbol is exported (confirmed via `nm -gU` on the built dylib).
- macOS links clean — the full test suite builds and passes on macOS arm64.

## One thing to check on your side (param count)

`vqe.c:516`: the hardware-efficient ansatz has `num_parameters = num_qubits * num_layers * 2`.
So your hard-guard `n == 4` for 2-qubit H2 corresponds to a **1-layer** HEA
(`2 * 1 * 2 = 4`). A **2-layer** HEA gives `n = 8`. Build the ansatz with the layer count
that matches your `== 4` guard (i.e. 1 layer), or relax the guard if you use 2 layers.
The Moonlab API is general — it writes whatever `n` the ansatz reports.

## Smooth-PES (companion "Moonlab PR #12") — MERGED, in lineage

The differentiable/smooth H2 (and LiH) potential energy surface fix is **merged** on
master (first-principles STO-3G, no Morse/plateau stub; equilibrium bit-for-bit
unchanged, force continuous). So your `h2_vibrational_quantum.esk` number will be
physical against the same tag. It is in the lineage of the QGT feature.

## The tag — and honest sequencing

`vqe_compute_qgt` (and the smooth PES) are on master now, but Moonlab is mid-way through a
completion campaign, and the **next tag is `v1.2.0`, cut after that campaign finishes with
a one-time git-history cleanup pass** (stripping some inherited commit-message trailers).
That cleanup **rewrites commit SHAs**, so I will not hand you a bare SHA that would then be
invalidated — a stable tag is the right pin, exactly as you prefer.

**What you get:** I will reply here with the exact `v1.2.0` tag name the moment it is cut.
The QGT ABI above is **frozen** — it will not change between now and that tag — so your
staged one-line pin bump applies unchanged.

If your advisory `quantum-macos` lane needs to go green sooner than the tag: say so and I
can cut an interim annotated tag `v1.2.0-rc1` at the post-cleanup HEAD specifically for you
to pin, ahead of the final `v1.2.0`. Since your lane is non-required, the default plan is
to wait for the clean `v1.2.0`.

## Ordering acknowledged

Agreed: Moonlab tag ready -> your pin bump (`chore/bump-moonlab-qng-pin`, verify
quantum-macos green) -> merge #299. I will not let #299 need anything from Moonlab beyond
`vqe_compute_qgt`, which is done.
