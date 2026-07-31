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
runner's seven `moonlab_oracle` events. `run_moonlab_oracles.sh` emits only the
seven it owns; the crypto/QRNG rows are the sibling lanes' responsibility.

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

### P7 -- parametric analyticity oracle (`analyticity_oracle`)

The expectation value of any observable after a parameterized circuit is an
*exact* trigonometric polynomial in each rotation angle. A parameter appearing
in q rotation gates -- each of the form `exp(-i(theta + phi_g) P / 2)` with `P`
a Pauli -- gives

```
f(theta) = <psi(theta)| O |psi(theta)> = sum_{k=-q..q} c_k e^{ik theta}
```

because each gate contributes `e^{+-i theta/2}` to the ket and its conjugate to
the bra, so the harmonic content spans exactly `-q..q`. The identity is exact;
only floating-point accumulation intervenes. That is what makes it adversarial:
a backend wrong in a way no single-point comparison catches -- a mis-signed
generator, a dropped cross term, a truncation leaking a spurious harmonic --
violates the degree bound or fails to reproduce `f` at angles it never saw.

`tests/oracle/test_analyticity_oracle.c` builds seed-deterministic circuits on
4-8 qubits (rx/ry/rz among Cliffords, adjacent forward-direction 2q gates only,
so the `reversed_2q` bug P1 quarantines cannot leak into this lane) carrying a
shared parameter at q = 1, 2, 3 occurrences, over a random weighted Pauli-Z sum
normalized so `sum|w| = 1`. Generation *rejects* placements the observable
cannot see -- a parameter outside the light cone of the measured operator would
make the probe vacuous -- by re-rolling deterministically from the same seeded
stream until the harmonic content clears `1e-3`.

- **trig fit** -- the `2q+1` equally spaced angles determine `c_k` exactly by
  DFT; the fitted polynomial must then reproduce `f` at 16 independent random
  angles it was never given, to `1e-11 * (1 + gates/16)`. Run on the dense state
  vector and on `tn_mps` at bond cap `2^ceil(n/2)` (exact at these sizes).
- **degree bound** -- an over-sampled DFT at `2q+3` nodes pins the *degree*: the
  harmonics at `+-(q+1)` must vanish. A backend that interpolates its own
  samples but carries an extra parameter dependence fails here.
- **Clifford closure** -- an all-Clifford family plus q shared `rz` gates at
  pi/2-multiple offsets closes the loop on the Aaronson-Gottesman tableau: the
  polynomial fitted from *non-Clifford* angles on the dense backend must predict
  the tableau's exact algebraic value at `theta = m pi/2`.
- **noise continuation** -- with a depolarizing channel at rate p on k sites the
  expectation is a polynomial in p of degree <= k. Fitted at `k+1`
  Chebyshev-spaced rates in `[0, 0.1]` it must reproduce other rates in the same
  regime *and continue to a different one* (p = 0.3, 0.5, 0.75, the last being
  where depolarizing saturates the maximally mixed state). Two independent noise
  paths are gated: the deterministic channel path `moonlab_mpdo_*` (a
  matrix-product density operator, exact here because a 1q Kraus map leaves the
  bond dimension untouched), and the state-vector path, whose channels are a
  Monte-Carlo unravelling rather than a channel -- so the polynomial check runs
  on the *exact per-Pauli-branch expectation*, all `4^k` Kraus branches
  enumerated through the production `noise_depolarizing_single` with
  deterministic branch-selecting uniforms and weighted by the exact Kraus
  weights (`1-p` for I, `p/3` for each of X, Y, Z). That reaches the entangled,
  multi-qubit-observable regime MPDO cannot represent. A third probe pins the
  two paths against each other where both are exact.

Extrapolating a fit outside its sampling window amplifies floating-point error
by exactly the Lebesgue sum `A(t) = sum_j |L_j(t)|` of the Lagrange basis at the
evaluation point. That factor is **computed, not guessed**, and the gate is
`1e-13 * (1 + A(t))`. Observed residuals sit at `2.2e-16 * (1 + A(t))` -- machine
epsilon times the exact conditioning -- so the gate is tight, not slack.

### P8 -- Wick-rotation consistency oracle (`wick_rotation_oracle`)

The Loschmidt amplitude `L(z) = <psi0| e^{-zH} |psi0> = sum_j |<psi0|E_j>|^2
e^{-z E_j}` is *entire*. Real time is the imaginary axis `z = it`; imaginary time
is the real axis `z = tau`. Moonlab reaches those two axes through near-disjoint
code -- real-time TDVP and the dense spectral propagator on one side, CA-MPS
imaginary-time Trotter, imaginary-time TDVP and DMRG on the other -- yet they are
constrained to be values of one analytic function.

`tests/oracle/test_wick_rotation_oracle.c` materializes each Hamiltonian as a
dense `2^n x 2^n` matrix through the production path (`mpo_to_matrix` for the
TFIM MPOs, the public CSR of `xxz_build_sparse` for the disordered chain --
densified in the test, so the oracle shares no code with what it gates),
diagonalizes it with `hermitian_eigen_decomposition` (LAPACK `zheev`), and forms
the spectral weights. The oracle is itself gated first -- eigen residual,
completeness `sum_j w_j = 1`, `L(0) = 1` -- so a broken oracle cannot bless a
broken path. Systems: TFIM chains n = 2..8 at h = 0.8 plus the critical point
h = 1.0, and a disordered (hence non-integrable) XXZ chain at n = 6, 8, W = 3.

- **real-time TDVP** on `tn_mps` at an exact bond dimension. A two-site TDVP
  whose manifold is the full Hilbert space has an identity tangent-space
  projector, so the projector-splitting integrator carries no Trotter error. The
  probe *proves* that rather than assuming it: it runs at dt and dt/2 and
  requires step-size independence before gating `L(it)` and `<Z_0>(t)`.
- **dense spectral propagator** `mbl_evolve_exact`, and the **dense Krylov
  propagator** `mbl_evolve_krylov` at two Krylov dimensions, gated on the
  converged value.
- **imaginary-time TDVP**, gated on two functionals of the same analytic
  function: the normalized overlap `L(tau)/sqrt(L(2tau))` and the instantaneous
  energy `E(tau) = -d/dz log L(z)` at `z = 2tau`.
- **CA-MPS imaginary-time Trotter**, the one path that exposes the bare
  amplitude: `exp(-tau P)` is applied non-unitarily and `moonlab_ca_mps_norm`
  reports the decayed norm, so `||e^{-tau H}|psi0>||^2 = L(2tau)` directly. This
  path *is* Trotterized, so it gets the treatment the doctrine requires: the same
  evolution is run at three step counts S, 2S, 4S, the convergence order is
  measured **from the samples alone** (successive differences must ratio to 4,
  i.e. second order), and the twice-Richardson-extrapolated value is what gets
  gated -- at `1e-8` relative, against an observed `8.6e-11`. The tolerance is
  not relaxed to absorb Trotter error; the extrapolation removes it.
- **DMRG ground energy**, the `tau -> infinity` endpoint of the imaginary-time
  axis, against the oracle's lowest eigenvalue, with the variational bound
  `E_dmrg >= E_0` checked separately.

Then the loop closes. Taking the *frequencies* `E_j` from the oracle and nothing
else, the spectral *weights* are recovered from the repo's imaginary-time samples
alone: `E(tau) = sum_j w_j E_j e^{-2 tau E_j} / sum_j w_j e^{-2 tau E_j}` is
linear in w, so K-1 imaginary-time energy samples plus `sum_j w_j = 1` determine
every weight. Those weights, fitted on the imaginary axis, must then reproduce
the repo's *real-time* TDVP amplitudes. That is the literal analytic
continuation, and it makes the two implementations answer to one function.

Recovering weights from imaginary-time data is an inverse-Laplace problem, so its
conditioning is **computed** (the 1-norm condition number of the fit matrix) and
the closure gate is `1e-13 * cond`. At K = 4 distinct levels the condition number
is ~`9e1` and at K = 8 it is ~`1e6`, both far inside double precision; by K = 16
it exceeds `1e15` and no gate would mean anything. So the closure runs where it
resolves (n = 2, 3) and this document says so rather than pretending otherwise.
Every other P8 probe runs the full n = 4..8 range.

Every P8 tolerance is `base * 2^n` with the base set from the *measured* worst
residual of that path, not from what happens to pass: `1e-13` for the dense
propagators (algebraically exact, one spectral projection, observed `2.3e-15`)
and `1e-10` for the MPS propagators (Lanczos and SVD roundoff over O(100)
sweeps, observed `2.9e-10`). Blending them into one constant would slacken the
dense gate by four orders to the MPS path's noise floor.

**Recorded limitation.** Imaginary-time TDVP renormalizes each two-site block to
unit Frobenius norm inside the local update (`tdvp.c`, the `TDVP_IMAGINARY_TIME`
rescale before the SVD split) and never accumulates the discarded factor, so the
norm decay is lost: `tdvp_result_t.norm` is 1.0 and `tn_mps_true_norm` is 1.0
even with `config.normalize = false`, and `||e^{-tau H}|psi0>||` is not
extractable from that path. P8 therefore gates imaginary-time TDVP on the
overlap and energy functionals, which are exact identities on the same `L(z)`,
and gates the bare amplitude on CA-MPS, which does expose it. This is a
limitation of the TDVP API, not of the identity.

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
exact. Nothing else is quarantined; P2/P3/P5/P6/P7/P8 are fully required-pass.

## How the oracles gate release

1. `cmake --build build` builds the corpus header (checked-in default) and the
   oracle targets under the `oracle` ctest label (fast: full run < 2 min).
2. `scripts/run_moonlab_oracles.sh` regenerates the corpus, rebuilds the oracle
   targets, runs `ctest -L oracle`, applies the allowlist, and writes
   `scripts/icc_traces/moonlab_oracles.jsonl`. It exits nonzero on any
   non-allowlisted FAIL.
3. `icc readiness --repo moonlab --target moonlab-adversarial-matrix
   --trace-dir scripts/icc_traces` consumes those events. The
   integrator-owned target (`.icc/completion-oracles.yaml`) requires all seven
   `moonlab_oracle` events PASS plus the sibling fuzz/differential/statistical
   events; `moonlab-release-readiness` mirrors the matrix. An incomplete or
   silently-wrong release cannot reach `ready`.

The seven events this runner owns: `backend_differential_oracle` (P1),
`gradient_oracle` (P2), `measurement_statistics_oracle` (P3),
`edge_matrix_oracle` (P5), `property_invariants_oracle` (P6),
`analyticity_oracle` (P7), `wick_rotation_oracle` (P8).
