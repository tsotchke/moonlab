# Note for the Moonlab agent — unblock Eshkol PR #299 (quantum-chemistry FFI)

_From the Eshkol coordination side, 2026-07-17. Placed in Moonlab `notes/` for pickup._

## TL;DR
Eshkol PR #299 (an external community contribution — differentiable quantum
chemistry) adds an FFI `eshkol_vqe_qgt_compute` that calls Moonlab's
`vqe_compute_qgt`. That symbol is **not in the Moonlab revision Eshkol currently
pins** (`d2503460`, 2026-07-12), so the Eshkol quantum build does not compile. The
QGT code lives in `src/algorithms/vqe_qng.c`, added in Moonlab commit `816697d`
("feat(vqe): quantum natural gradient optimizer", 2026-07-17). We need that on a
**stable, taggable ref** so Eshkol can bump its `CMakeLists.txt` `GIT_TAG` to it.

## The one ask (blocking)
**Cut / confirm a tagged Moonlab release `>= 816697d`** that:
- ships `src/algorithms/vqe_qng.c`,
- declares `MOONLAB_API int vqe_compute_qgt(vqe_solver_t*, const double* params, double* qgt_out);`
  in `src/algorithms/vqe.h`,
- includes `src/algorithms/vqe_qng.c` in the `libquantumsim` build source list so the
  symbol is exported,
- **links cleanly on macOS** (this is the historical gotcha — past Eshkol pin bumps,
  e.g. #269, were specifically chosen for macOS-linkability; the Eshkol `quantum-macos`
  CI lane will be our verification).

`816697d` descends from the current pin `d2503460`, so a plain forward pin bump on the
Eshkol side is enough — no rebase. **Please tell us the tag name to pin** (a release tag
is strongly preferred over a bare SHA).

## ABI freeze — do NOT change `vqe_compute_qgt`
Eshkol's shim relies on exactly:
- signature `(vqe_solver_t* solver, const double* params, double* qgt_out)`
- return `0` on success / `-1` on error
- `qgt_out` written **row-major, n×n** with `n = solver->ansatz->num_parameters`
  (read back as `qgt_out[i*n+j]`, symmetric). For the 2-qubit H2 hardware-efficient
  ansatz `n = 4`, which Eshkol **hard-guards** (`num_parameters == 4`). If that param
  count ever changes for this config, tell us.

## Already fine at the pin — no action
`pauli_hamiltonian_create` and `pauli_hamiltonian_add_term(h, coefficient, pauli_string,
term_index)` exist at `d2503460` with the expected arg order and validation
(`term_index < num_terms`, `strlen(pauli_string) == num_qubits`, chars ∈ {X,Y,Z,I}).
Eshkol's `make-pauli-hamiltonian` builds on these. Keep this contract stable.

## Separate, non-blocking — smooth-PES fix (companion "Moonlab PR #12")
Eshkol's `h2_vibrational_quantum.esk` finite-differences `make-h2-hamiltonian`'s exact
ground energy over bond length; its ~4936 cm⁻¹ is only physical if
`vqe_create_h2_hamiltonian(R)` gives a **smooth** PES (no Morse/plateau stub) near
R ≈ 0.74 Å. This does **not** block the build — only that one example's number. Can land
after the pin bump. **Question:** is the smooth PES in `816697d`'s lineage, or still open?

## Ordering (important)
The Eshkol `quantum-macos` CI lane is **advisory / non-required** — a red build there does
not block merge. So the sequence must be:
**Moonlab tag ready → Eshkol pin bump (+ verify quantum-macos green) → merge #299.**
Merging #299 before the pin bump would silently ship a broken quantum build on master.

## Verification already done on the Eshkol side (so you don't repeat it)
- Against pin `d2503460`: PR's `agent_quantum.c` fails — `error: call to undeclared
  function 'vqe_compute_qgt'` (independently reproduced; `816697d` is not an ancestor of
  the pin).
- Against `816697d`: the *same* file compiles clean (exit 0, 0 diagnostics). So
  `vqe_compute_qgt` is the **only** missing symbol; nothing else in #299 needs new Moonlab.

## Eshkol-side readiness (what happens the moment you give us the tag)
- A pin-bump branch (`chore/bump-moonlab-qng-pin`, off current master) is **staged and
  ready** — one line, `CMakeLists.txt` `GIT_TAG d2503460… → <your tag>`.
- We push it, let the `quantum-macos` CI lane build-verify against the new pin, merge it,
  then merge #299 and confirm master's quantum lane is green. Turnaround is minutes once
  the tag exists.

## One question back
Ship the QGT/QNG feature (`816697d`) in the next Moonlab tag, or should Eshkol pin the SHA
directly? Tag preferred. Reply with the ref name and we bump immediately.
