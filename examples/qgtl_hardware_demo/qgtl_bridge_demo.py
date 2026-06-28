#!/usr/bin/env python3
"""End-to-end Moonlab + QGTL hardware-bridge demo.

Shows the full path:

    Python code            -> moonlab.qgtl: build a circuit
                                |
                                v
                              serialise() -> moonlab-circuit-v1 text
                                |
                                +----+
                                v    v
                          moonlab    QGTL backend
                          control    (IBM / Rigetti / IonQ / D-Wave)
                          plane      [out-of-scope of this demo;
                                      requires QGTL repo installed]

The Bell circuit text is identical at both branches.  Pointing the
submit at QGTL is a one-line change once QGTL's Python wrapper is
installed (see https://github.com/tsotchke/quantum_geometric_tensor).

Run:
    python3 examples/qgtl_hardware_demo/qgtl_bridge_demo.py \\
        --host 127.0.0.1 --port 7070
"""

from __future__ import annotations

import argparse
import json
import sys

from moonlab.qgtl import GateType, QgtlCircuit
from moonlab.control_plane import submit_circuit


def build_bell_circuit() -> QgtlCircuit:
    c = (QgtlCircuit(num_qubits=2)
         .add_gate(GateType.H,    target=0)
         .add_gate(GateType.CNOT, target=1, control=0))
    return c


def run_simulation_path(host: str, port: int, circuit: QgtlCircuit) -> dict:
    """Submit the circuit to a local moonlab control plane (ideal,
    noise-free simulation)."""
    text = circuit.serialize()
    probs = submit_circuit(host, port, text)
    return {
        "transport": f"line-protocol://{host}:{port}",
        "backend":   "moonlab-simulator",
        "probabilities": list(probs),
    }


def run_hardware_path_stub(circuit: QgtlCircuit) -> dict:
    """Where this demo would hand off to QGTL for real-hardware
    execution.  Print the serialised circuit -- the exact same text
    QGTL would accept via its `qgt_circuit_submit_ibm()` / Rigetti /
    IonQ / D-Wave entry points.

    Once QGTL's Python wrapper is installed, replace this stub with:

        from qgtl.hardware.ibm import submit_circuit_text
        outcomes = submit_circuit_text(
            circuit.serialize(),
            backend="ibmq_brisbane",
            shots=1024)
    """
    return {
        "transport":     "qgtl-hardware (stub)",
        "backend":       "ibm / rigetti / ionq / d-wave (your pick)",
        "circuit_text":  circuit.serialize(),
        "note":          "install QGTL to actually submit",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7070)
    args = p.parse_args()

    circuit = build_bell_circuit()

    print("=== moonlab circuit (Bell pair) ===")
    print(circuit.serialize())

    print("=== simulation path (via moonlab control plane) ===")
    sim = run_simulation_path(args.host, args.port, circuit)
    print(json.dumps(sim, indent=2))

    print("=== hardware path (QGTL bridge) ===")
    hw = run_hardware_path_stub(circuit)
    print(json.dumps(hw, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
