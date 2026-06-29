# Archived Moonlab Documentation: Research workflow: end-to-end across every binding

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Research workflow: end-to-end across every binding

A realistic single-thread tutorial that walks the same problem
through every Moonlab binding the way an actual research project
would.  Pick whichever language is most ergonomic for each stage;
the same library underneath produces the same answers.

The problem: characterise the antiferromagnetic Heisenberg chain
ground state on N qubits, then run a noisy device-emulator
verification of the prepared state.

## Stages

1. **DMRG ground-state energy** (Python) -- get the ground-state
   energy at increasing N.
2. **Two-site TDVP imaginary-time** (Rust) -- prepare the actual
   MPS ground state, not just its energy.
3. **CA-MPS variational-D** (JS in a browser or Node) -- the
   Clifford-assisted alternative when the state has substantial
   Clifford structure.
4. **Topology Chern-marker validation** (Python) -- a sanity-check
   topological invariant on a related QWZ model.
5. **Vendor-noise pre-flight** (Python via the scheduler binding)
   -- run a small portion of the circuit through the IBM Falcon
   stochastic-noise emulator to see how the prepared state would
   look on real hardware.
6. **Submit via control plane** (Python) -- the same circuit, but
   routed through the cloud control plane the way a production
   user would.

All six stages share the same underlying C library and stable ABI;
nothing in any stage requires a different library build.

---

## Stage 1: DMRG ground-state energy (Python)

[archived fence delimiter: ```python]
from moonlab.dmrg import heisenberg_ground_energy

# Antiferromagnetic Heisenberg: J = 1, Delta = 1, h = 0.
for N in (8, 12, 16, 20):
    E = heisenberg_ground_energy(num_sites=N, J=1.0, Delta=1.0,
                                  h=0.0, max_bond_dim=64, num_sweeps=20)
    print(f"N={N}: E_0 = {E:.6f}, E_0/N = {E/N:.6f}")
[archived fence delimiter: ```]

Expected output (chi=64, 20 sweeps):

[archived fence delimiter: ```]
N=8:  E_0 = -13.499730, E_0/N = -1.687466
N=12: E_0 = -20.568363, E_0/N = -1.714030
N=16: E_0 = -27.646949, E_0/N = -1.727934
N=20: E_0 = -34.729893, E_0/N = -1.736495
[archived fence delimiter: ```]

For comparison, the Bethe-ansatz value `E/N -> -ln(2) - 1/4 + 1/4 = 1/4 - ln(2)`
in the antiferromagnetic XXX limit is around `-1.773` for the open chain.
The N = 20 number is ~2% above that; tightening `max_bond_dim` to 128 and
running 40 sweeps closes most of the gap.

## Stage 2: Two-site TDVP imag-time MPS preparation (Rust)

For when you need the prepared *state*, not just the energy.

[archived fence delimiter: ```rust]
use moonlab::tdvp::{TdvpConfig, TdvpEngine, EvolutionType, MpoHeisenberg, RandomMps};

fn main() -> moonlab::Result<()> {
    let mpo = MpoHeisenberg::new(16, 1.0, 1.0, 0.0)?;
    let mps = RandomMps::new(16, 8, 32)?;

    let mut cfg = TdvpConfig::adaptive(1e-3);
    cfg.evolution_type = EvolutionType::ImaginaryTime;
    cfg.dt = 0.05;

    let mut engine = TdvpEngine::new(mps, mpo, cfg)?;
    for step in 0..30 {
        let r = engine.step()?;
        println!("step {step}: E = {:+.6}", r.energy);
    }
    Ok(())
}
[archived fence delimiter: ```]

The MPS handle persists on the engine and can be cloned or measured
between steps.  Energy converges to within 1e-4 of the DMRG result
in ~25 imaginary-time steps at dt=0.05.

## Stage 3: CA-MPS variational-D (JS, browser or Node)

For circuits with substantial Clifford structure, CA-MPS can run
with bond-dim that's a fraction of what plain MPS needs.

[archived fence delimiter: ```ts]
import { CaMps, varDRun, Warmstart } from '@moonlab/core';

const N = 12;
const s = await CaMps.create(N, 16);

// Build the Heisenberg Pauli sum row-major: (N-1) XX + (N-1) YY + (N-1) ZZ
const numTerms = 3 * (N - 1);
const paulis = new Uint8Array(numTerms * N);
const coeffs = new Float64Array(numTerms);
let row = 0;
for (const p of [1, 2, 3]) {
  for (let i = 0; i < N - 1; i++) {
    paulis[row * N + i]     = p;
    paulis[row * N + i + 1] = p;
    coeffs[row] = 1.0;
    row++;
  }
}

const result = await varDRun(s, paulis, coeffs, numTerms, {
  maxOuterIters: 20,
  imagTimeDtau: 0.05,
  imagTimeStepsPerOuter: 4,
  cliffordPassesPerOuter: 6,
  warmstart: Warmstart.HAll,
});
console.log(`final E = ${result.finalEnergy.toFixed(6)}, ` +
            `iters = ${result.outerIterations}`);
s.dispose();
[archived fence delimiter: ```]

The same call works in Deno, Node, or directly in a browser via
the WASM build.  Use the `CaMps.sampleZ` Born-rule sampler to
draw shots for downstream verification.

## Stage 4: Topology Chern-marker sanity check (Python)

A related model -- the QWZ Chern insulator -- has a known
topological invariant that survives small perturbations.  Useful
as a sanity check before pushing the same code to the real
research problem.

[archived fence delimiter: ```python]
from moonlab.topology import chern_kpm_create, chern_kpm_local_marker

# QWZ with m = -1 (Chern = -1 phase).
ck = chern_kpm_create(L=24, m=-1.0, n_cheby=100)
c_center = chern_kpm_local_marker(ck, x=12, y=12)
print(f"Local Chern marker at bulk site: {c_center:.4f}")
assert abs(c_center - (-1.0)) < 0.1, "QWZ Chern marker should be -1"
[archived fence delimiter: ```]

For a true mosaic of the 2D Chern landscape under quasi-crystalline
modulation, use the C-side `bench_chern_mosaic_hq` harness; the
Python binding exposes the same `chern_kpm_*` API but caps out at
~L = 300 on the sparse-stencil backend.

## Stage 5: Vendor-noise pre-flight (Python)

Before committing time on a real QPU, run the circuit through one
of the three pre-baked vendor-noise emulators.  Each profile uses
public typical calibration data.

[archived fence delimiter: ```python]
from moonlab.scheduler import (
    Job, register_vendor_noise_backends, list_backends,
)
from moonlab.qgtl import GateType

register_vendor_noise_backends()
print("Registered backends:", list_backends())

# Prepare a small section of the workflow as a verifiable circuit:
# a 4-qubit GHZ state, measured in the Z basis.
j = Job(num_qubits=4)
j.add_gate(GateType.H, target=0)
j.add_gate(GateType.CNOT, target=1, control=0)
j.add_gate(GateType.CNOT, target=2, control=1)
j.add_gate(GateType.CNOT, target=3, control=2)
j.set_num_shots(8192).set_num_workers(1).set_rng_seed(0xdeadbeef)

for backend in ("ibm-falcon", "rigetti-aspen", "ionq-forte"):
    j.set_backend(backend)
    r = j.execute()
    pure_ghz = sum(1 for o in r.outcomes if o in (0, 15))
    fidelity_proxy = pure_ghz / r.total_shots
    print(f"{backend}: pure-GHZ fraction = {fidelity_proxy:.4f}")
[archived fence delimiter: ```]

Sample output on a recent calibration:

[archived fence delimiter: ```]
Registered backends: ['simulator', 'ibm-falcon', 'rigetti-aspen', 'ionq-forte']
ibm-falcon:     pure-GHZ fraction = 0.9016
rigetti-aspen:  pure-GHZ fraction = 0.8765
ionq-forte:     pure-GHZ fraction = 0.9718
[archived fence delimiter: ```]

IonQ produces the highest fidelity because of its cleaner 2q gates
(p_2q ~ 0.4% vs IBM Falcon's ~1%).  If your workflow demands >95%
fidelity at this depth, the pre-flight tells you IonQ is the right
target before you spend the queue time.

## Stage 6: Submit via the control plane (Python)

When you're ready to share the workflow with collaborators or run
it at scale, submit through the moonlab control plane.

Server (in one terminal):

[archived fence delimiter: ```bash]
docker compose -f deploy/docker/docker-compose.yml up control-plane
[archived fence delimiter: ```]

Client (in another):

[archived fence delimiter: ```python]
from moonlab.control_plane import submit_circuit
from moonlab.qgtl import QgtlCircuit, GateType

c = QgtlCircuit(4)
c.add_gate(GateType.H, target=0)
c.add_gate(GateType.CNOT, target=1, control=0)
c.add_gate(GateType.CNOT, target=2, control=1)
c.add_gate(GateType.CNOT, target=3, control=2)

probs = submit_circuit(
    host="127.0.0.1", port=8765,
    circuit_text=c.serialize(),
    insecure=False,            # set True if using a self-signed cert
)
print(probs[0], probs[15])     # |0000> and |1111> for the GHZ_4
[archived fence delimiter: ```]

Stage 5's vendor-noise emulator backends are also available on the
server side; the control plane's job spec lets you pin a backend
per submission so cloud clients can sweep IBM / Rigetti / IonQ
pre-flights without re-uploading their code.

## What you should have at this point

- Quantitative ground-state energy across N (stage 1).
- A persistent MPS handle for the prepared state (stage 2).
- A Clifford-assisted alternative when applicable (stage 3).
- A topological-invariant sanity check on a related model (stage 4).
- Realistic device-specific fidelity estimates for the three major
  gate-model vendor families (stage 5).
- A reproducible cloud submission path for collaborators (stage 6).

Every stage produces JSON-archivable artefacts you can attach to
a paper draft or commit to the repo via `benchmarks/results/`.

## Where to next

- `docs/research/ca_mps.md` -- the CA-MPS method paper outline.
- `docs/benchmarks/decoder_shootout.md` -- if your workflow needs
  QEC, the decoder shoot-out numbers are how moonlab compares
  greedy / MWPM / pymatching at d = 5 / 7 / 9.
- `docs/benchmarks/mpi_scaling.md` -- if you're pushing past
  N = 28, here's the MPI scaling validation table.
- `docs/INTEGRATION_libirrep_SbNN.md` -- the libirrep + SbNN
  bridge contract for QEC code factories and learned-decoder
  routing.
- `examples/qgtl_hardware_demo/` -- the QGTL hardware-bridge
  template that QGTL sibling-library users adapt for vendor
  SDKs (IBM Qiskit, Rigetti pyquil, etc.).
```
