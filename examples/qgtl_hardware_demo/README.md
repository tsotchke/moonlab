# QGTL hardware-bridge demo

End-to-end demo of how a moonlab circuit reaches real quantum
hardware via QGTL.

```
moonlab Python (this demo)
    |
    +-- moonlab.qgtl.QgtlCircuit (build)
    |       |
    |       +-- .serialize() -> moonlab-circuit v1 text
    |             |
    |             +------> moonlab control plane  -> noise-free simulation
    |             |
    |             +------> qgtl.hardware.{ibm,rigetti,ionq,dwave}.submit_circuit_text
    |                          -> physical QPU job + outcome poll
```

The simulation half is fully in this repo:

```
docker compose -f deploy/docker/docker-compose.yml up   # spin up control plane
python3 examples/qgtl_hardware_demo/qgtl_bridge_demo.py
```

The hardware half lives in
[QGTL](https://github.com/tsotchke/quantum_geometric_tensor); install
it separately, then change the `run_hardware_path_stub` body in
`qgtl_bridge_demo.py` to call the QGTL backend you've authenticated
against.

## Why this matters

The moonlab-circuit-v1 text format is the **contract** between
moonlab (simulation, design, validation) and QGTL (deployment to
physical hardware).  Both sides accept the same text; the contract
is the v1.0 stable surface for cross-platform circuits.

Workflow:

1. **Design + simulate** in moonlab (this side).  Validate
   correctness, study noise sensitivity via MPDO, compute the QGT,
   etc.
2. **Serialize** to the canonical text format via `circuit.serialize()`.
3. **Submit** the *same text* via QGTL's hardware backend to the QPU
   of your choice.
4. **Compare** the simulated probability vector to the empirical
   QPU outcome distribution -- the gap is the noise budget your
   error-correction code needs to handle.

## What this demo does NOT do

- Submit to a real QPU.  That requires QGTL installed + an IBM /
  Rigetti / IonQ / D-Wave account.
- Apply error correction.  Use `moonlab.libirrep_qec` for the QEC
  code zoo (see `docs/INTEGRATION_libirrep_SbNN.md`).
- Apply error mitigation.  See `src/mitigation/` for ZNE / PEC /
  CDR / virtual distillation.

## See also

- `docs/CONTROL_PLANE.md`   -- the line protocol the simulation path uses
- `docs/PARITY_MATRIX.md`   -- which capabilities are in each binding
- QGTL repo                  -- <https://github.com/tsotchke/quantum_geometric_tensor>
