# MoonLab Platform Dominance Campaign

ICC-led program to take MoonLab to measured, reproducible leadership on every
front it is intended for. This is a multi-phase engineering campaign, not a
marketing claim: "dominance" on a front means a head-to-head benchmark against
that front's strongest incumbent, run under the ICC runtime-evidence gate, that
MoonLab wins or ties on the front's defining metric -- on the record.

The governing principle is the credibility moat: MoonLab already ships provable
correctness (fingerprint-bound runtime evidence). Every dominance claim in this
campaign must be one the ICC gate can prove. A claim the gate cannot back does
not ship.

## How dominance is defined and gated

For each front: (1) name the incumbent SOTA tool, (2) fix the defining metric,
(3) build a reproducible head-to-head benchmark harness, (4) bind the result to
a clean source fingerprint via the ICC gate, (5) publish the number. "Parity"
= within noise of the incumbent; "lead" = decisively past it on the metric with
the same correctness guarantees. No front is declared won on a claim; only on a
gate-bound benchmark.

## Phases

- **Phase 0 -- Foundation (in progress).** The v1.2.0 completion campaign: make
  every documented surface real (bindings, examples), ABI 0.6.0 -> 0.7.0,
  full evidence rebind, ICC readiness green. You cannot lead a front while the
  API is aspirational. Prerequisite for everything below.
- **Phase 1 -- Measurement.** Build the dominance benchmark harness: one
  reproducible, ICC-bound head-to-head per front vs its incumbent. This defines
  and instruments "dominance" before any front is worked. Without it every
  later claim is unmeasured.
- **Phase 2+ -- Per-front campaigns.** Take each front to measured
  parity-then-lead, in leverage order, each gated by its Phase 1 benchmark.

## The fronts

Grounded in the ICC feature map (real product surfaces named per front).

| # | Front | Incumbent SOTA | Defining metric | MoonLab today | Gate |
|---|---|---|---|---|---|
| F1 | Dense state vector (1+multi GPU) | cuQuantum/cuStateVec, Qiskit Aer GPU | gate throughput, max qubits, time-to-solution | `src/optimization` (SIMD/AVX512/SVE, fusion), `src/backends/moonlab` CUDA, N=33 MPI | competitive, not ahead |
| F2 | Tensor networks | quimb+cotengra, ITensor, TeNPy | contraction time, max bond dim, 2D system size | `tensor.c`/`dmrg.c`/`tn_measurement.c`, MPS/MPO/PEPS/TDVP, CA-MPS hybrid | strong, needs cotengra-class contraction |
| F3 | Clifford / stabilizer / QEC | Stim + PyMatching | Clifford qubit scale, decoder accuracy/speed, threshold | `src/backends/clifford` pauli-frame, surface codes, `libirrep_bridge` | present, must match Stim scale + best decoders |
| F4 | Open quantum systems | QuTiP | Lindblad/MPDO system size | noise channels, ZNE/PEC, MPDO | wedge via MPDO; needs full Lindblad |
| F5 | Variational / QML / autodiff | PennyLane, TFQ | gradient speed, expressibility, convergence | VQE/QAOA/autograd/QNG/UCCSD, SbNN sibling | strong base, needs GPU gradients |
| F6 | Chemistry | OpenFermion, Qiskit Nature, PySCF | molecule size, basis sets, chemical accuracy | first-principles STO-3G H2/LiH, `chemistry.c` | narrow, needs arbitrary molecules + mappings |
| F7 | Compilation / transpilation | tket, PyZX, Qiskit transpiler | gate/T-count reduction, routing quality | gate fusion | largest gap to platform |
| F8 | Hardware execution | Qiskit, Cirq, Braket, tket | vendor coverage, on-hardware fidelity | QGTL (IBM/Rigetti/D-Wave), maturing | bridge exists, needs depth |
| F9 | Post-quantum crypto + QRNG | liboqs, standalone RNGs | KAT coverage, throughput, honest characterization | ML-KEM FIPS 203, SHA-3, conditioned QRNG w/ SP 800-90B | already leads on characterization honesty |
| F10 | Correctness / provenance | (none ship this) | evidence-bound release, fingerprint binding | ICC runtime-evidence gate | categorical lead today; extend it |
| F11 | Portability / deployment | (fragmented) | platform matrix breadth | C core, WASM, Jetson, mesh, cloud | already leads breadth |

## Leverage order (why this sequence)

1. **F10 + F11 are already lead** -- do not chase, extend and defend. They are
   the moat: provable correctness and a compounding shared substrate are the two
   things the six specialist incumbents cannot replicate without becoming
   MoonLab.
2. **F2 (tensor networks)** first among contested fronts: highest scientific
   payoff (beyond-classical-supremacy simulation) and it compounds into F4, F6,
   F3. A world-class contraction engine lifts multiple fronts.
3. **F3 (QEC/decoders via libirrep)** next: the geometry/representation-theory
   thesis (QGT <-> QEC node) can produce a decoder structurally new rather than
   a faster Stim copy -- a real wedge, not a catch-up.
4. **F1 (multi-GPU/multi-node)** benchmarked head-to-head vs cuQuantum on real
   frontier hardware and published.
5. **F5, F6, F4** -- bounded, well-understood; play to the SbNN/chemistry siblings.
6. **F7, F8** -- turn the simulator into the platform (compilation + hardware).

## Standing rules

- Every "beyond SOTA" statement is gate-provable or it does not ship.
- Benchmark in the open against the named incumbent per front; no self-referential wins.
- One improvement to the shared substrate must lift every front that consumes it
  -- prefer substrate wins over per-front point fixes.
- No stubs, no fixtures, no loosened tolerances to manufacture a win.
