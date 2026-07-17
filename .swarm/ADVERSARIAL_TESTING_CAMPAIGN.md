# Moonlab Adversarial Testing Campaign

This is the doctrine for Moonlab's adversarial generative gate matrix: the
machinery that makes an incomplete or silently-wrong release impossible to
ship. It mirrors the Eshkol completion-oracle campaign
(`~/Desktop/eshkol/.icc/completion-oracles.yaml`) and adapts its pillar model
to a quantum simulator whose failure modes are *silent numerical wrongness*,
not crashes.

The single governing idea: a Moonlab claim ("the MPS backend is exact at these
sizes", "this gradient is analytic", "sampling is Born-rule correct") is only
believed once a *generated, seed-deterministic* family of programs has
cross-checked it against an independent reference and turned the readiness
oracle red when it disagreed. A fixed hand-written test pins one point; an
adversarial generator sweeps a growing space and finds the point the author
did not think of.

## Non-negotiables

- **No stubs.** Every probe exercises a real production path. A probe that
  cannot reach the code it claims to certify is deleted, not shipped green.
- **No tolerance-loosening to make HEAD pass.** When an oracle catches a real
  bug, the tolerance stays; the failure is recorded in
  `tests/oracle/KNOWN_FAILURES.txt` with its owning lane and exact seed, and
  the sibling lane that owns the fix flips it back to required-pass. Widening
  `1e-10` to `1e-3` to get a green check is the exact dishonesty this campaign
  exists to prevent.
- **Determinism.** Every corpus is a pure function of its seed. No wall-clock,
  no `/dev/urandom`, no unspecified iteration order enters generation. Two runs
  of `gen_circuit_corpus.py --seed S` are byte-identical. A probe that samples
  uses a seeded splitmix64 entropy source, never the OS RNG.
- **Replayability.** Every mismatch prints the seed and the circuit (or the
  ansatz + Hamiltonian) that produced it, so a failure is reproduced by copying
  one line, not by re-running a stochastic search.

## Pillars

Each pillar owns a trace-event name. `scripts/run_moonlab_oracles.sh` runs the
pillar, applies the `KNOWN_FAILURES.txt` allowlist, and emits one JSON-L line
per pillar to `scripts/icc_traces/moonlab_oracles.jsonl` of the form

```json
{"kind":"moonlab_oracle","name":"<pillar_event>","value":"PASS","total":N,"failed":0,"xfail":K,"snippet":"..."}
```

The `.icc/completion-oracles.yaml` `moonlab-adversarial-matrix` target requires
`runtime_event` PASS for each of these names.

### P1 -- backend differential oracle (`backend_differential_oracle`)

`scripts/gen_circuit_corpus.py` emits a seed-deterministic corpus of circuit
families over qubit counts {2,4,6,8,10} and depths {4,16,64}, each tagged with
its class:

- `clifford` -- pure-Clifford chains (H, S, S-dagger, X, Y, Z, forward CNOT,
  CZ, SWAP).
- `rot_cnot_ladder` -- single-qubit rotations interleaved with forward CNOT
  brickwork ladders.
- `random_universal` -- random universal circuits (rotations + T + forward
  CNOT/CZ/SWAP over adjacent pairs).
- `param_layers` -- hardware-efficient parameterized layers (rotation layer +
  entangling layer).
- `reversed_2q` -- rotations + reversed adjacent CNOTs (`control > target`),
  the exact shape that exposes a two-qubit-gate transpose bug in the MPS apply
  path. Isolating the reversed direction in its own class keeps the four
  forward classes a clean green signal and makes the known bug one quarantinable
  family instead of a corpus-wide red-out.

The 2q gates are adjacent (a single local SVD) so the exact MPS stays within the
< 2 min budget; adjacent brickwork is universal, so full entanglement is still
reached at depth.

`tests/oracle/test_backend_differential.c` runs each circuit on the dense state
vector and on `tn_mps` (bond cap `2^ceil(n/2)`, exact at these sizes; the lazy
`log_norm_factor` is committed with `tn_mps_normalize` before the amplitudes are
read) and compares the full probability vector -- reversing the qubit index for
tn_mps's big-endian basis -- and every `<Z_q>` / `<Z_q Z_{q+1}>` expectation to
`1e-10`. The exact bond-`2^ceil(n/2)` MPS at n=10 costs seconds per deep
circuit, so the MPS comparison runs where affordable (n<=8 all depths, plus
n=10 at shallow depth); the dense-vs-tableau exact check below runs on every
circuit, so n=10 is never left uncovered. Clifford-only circuits are
additionally cross-checked against the Aaronson-Gottesman tableau, exactly: each
tableau-derived `<Z_q>` (via clone-and-measure determinism) must equal the dense
value.

### P2 -- gradient oracle (`gradient_oracle`)

`tests/oracle/test_gradient_oracle.c` builds seeded hardware-efficient ansaetze
(2-6 qubits, 1-3 layers) over random Pauli-sum Hamiltonians and cross-checks
three independent gradient routes:

- adjoint autograd (`vqe_compute_gradient`, fast path) vs analytic
  parameter-shift, agreement `1e-7`;
- adjoint vs central finite differences, agreement `1e-5`.

It also pins the quantum geometric tensor (`vqe_compute_qgt`, landed in the QNG
PR): the metric must be symmetric and positive-semidefinite (no negative
eigenvalue below `-1e-8`).

### P3 -- measurement statistics oracle (`measurement_statistics_oracle`)

`tests/oracle/test_measurement_oracle.c` takes corpus circuits, computes exact
Born probabilities from the dense amplitudes, and draws `N` shots from a
deterministic splitmix64 entropy source. It asserts:

- a chi-square goodness-of-fit statistic below the upper-tail critical value at
  significance `1e-3` (deterministic at the fixed seed and `N`, so a correct
  sampler always clears it and a biased sampler fails);
- collapse consistency -- after a projective single-qubit measurement the state
  is renormalized (norm 1 to `1e-10`) and a repeated measurement of the same
  qubit is idempotent (returns the same outcome with probability 1).

### P4 -- crypto / QRNG conformance (sibling lanes, not this runner)

Moonlab's FIPS 203 ML-KEM / AES-DRBG / SHA-3 / QRNG conformance is owned by the
crypto and statistical sibling lanes and their ACVP/KAT/statistical batteries.
This campaign does **not** duplicate or re-emit that work. The shared
`moonlab-adversarial-matrix` target (`.icc/completion-oracles.yaml`, owned by the
integrator) folds those sibling events -- `qrng_statistical_battery`,
`qrng_bias_positive_control`, `mlkem_negative_fuzz`, `mlkem_avalanche`,
`entropy_health_rejects_bad` (kind `moonlab_statistical`), and the differential/
fuzz lanes' `cross_backend_differential` / `fuzz_corpus_clean` -- alongside this
runner's five `moonlab_oracle` events. `run_moonlab_oracles.sh` emits only the
five it owns; the crypto/QRNG rows are the sibling lanes' responsibility.

### P5 -- edge composition matrix (`edge_matrix_oracle`)

`tests/oracle/test_edge_matrix.c` sweeps feature *pairs* -- the compositions
that unit tests of single features miss:

- gate then measure then gate (collapse followed by further evolution stays
  normalized and respects the collapsed subspace);
- fusion on/off parity (`fuse_execute` on a raw circuit vs on its
  `fuse_compile` output must agree to `1e-10`);
- MPS canonical-form transitions (an expectation is invariant under
  left/right/mixed canonicalization and under lossless apply/truncate/measure
  interleavings);
- noise-channel + measurement composition (a `p=0` channel is a measurement
  no-op; a `p=1` bit-flip flips the outcome deterministically; every channel
  preserves the trace).

## KNOWN_FAILURES policy

`tests/oracle/KNOWN_FAILURES.txt` is an **allowlist**, never a tolerance knob.
Each line is

```
<probe_id>   owner=<lane>   seed=<seed>   # justification
```

Rules:

1. An entry documents a *real* bug that a *named sibling lane* owns and is
   fixing. `owner=` names that lane; `seed=` records the exact corpus seed (and
   the probe id already encodes qubit count, depth, and instance) so the
   failure is reproduced deterministically.
2. Allowlisting a probe is the **only** sanctioned response to a current-HEAD
   failure. Loosening a tolerance, deleting a probe, or narrowing the corpus to
   route around the bug is prohibited.
3. An allowlisted probe that *passes* is a stale entry (the fix landed). The
   runner reports it as XPASS; the entry is removed at the next integration
   pass so the probe returns to required-pass.
4. The oracle binaries are allowlist-aware: a run whose only failures are
   allowlisted exits 0 (XFAIL), so the default ctest lane stays green while the
   bug is open. A non-allowlisted failure exits nonzero and turns the pillar
   event -- and thus the release readiness oracle -- red.
5. An entry ending in `*` is a prefix, so one line can quarantine a whole
   known-bug family (e.g. `reversed_2q_*__diff_mps`) instead of enumerating
   every probe. Prefer an explicit per-probe list when the set is small and
   fixed, so each fix shows up as an individual XPASS.

Current quarantine (corpus seed 20260717): the 13 `reversed_2q_*__diff_mps`
probes, owned by the tensor-network lane -- tn_mps applies a reversed adjacent
CNOT (`control > target`) as the wrong unitary, so the state diverges from the
dense reference (dP ~ 0.25) even though every forward-direction 2q gate is
exact. Nothing else is quarantined; P2/P3/P5/P6 are fully required-pass.

## How the oracles gate release

1. `cmake --build build` builds the corpus header (checked-in default) and the
   oracle targets under the `oracle` ctest label (fast: full run < 2 min).
2. `scripts/run_moonlab_oracles.sh` regenerates the corpus, rebuilds the oracle
   targets, runs `ctest -L oracle`, applies the allowlist, and writes
   `scripts/icc_traces/moonlab_oracles.jsonl`. It exits nonzero on any
   non-allowlisted FAIL.
3. `icc readiness --repo moonlab --target moonlab-adversarial-matrix
   --trace-dir scripts/icc_traces` consumes those events. The
   integrator-owned target (`.icc/completion-oracles.yaml`) requires all five
   `moonlab_oracle` events PASS plus the sibling fuzz/differential/statistical
   events; `moonlab-release-readiness` mirrors the matrix. An incomplete or
   silently-wrong release cannot reach `ready`.

The five events this runner owns: `backend_differential_oracle` (P1),
`gradient_oracle` (P2), `measurement_statistics_oracle` (P3),
`edge_matrix_oracle` (P5), `property_invariants_oracle` (P6).
