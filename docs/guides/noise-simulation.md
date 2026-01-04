# Noise Simulation Guide

Add realistic noise to quantum simulations.

## Overview

Real quantum computers experience various types of noise and errors. Moonlab can simulate these effects to help design noise-resilient algorithms and understand how noise impacts quantum computations.

## Noise Models

### Depolarizing Channel

Randomly applies X, Y, or Z with probability p/3 each:

```python
from moonlab import QuantumState
from moonlab.noise import DepolarizingChannel

state = QuantumState(2)
noise = DepolarizingChannel(error_rate=0.01)

state.h(0)
noise.apply(state, qubit=0)  # 1% chance of X, Y, or Z error
```

### Amplitude Damping

Models energy relaxation (T1 decay):

```python
from moonlab.noise import AmplitudeDamping

# T1 = 50 μs, gate time = 20 ns
gamma = 1 - np.exp(-20e-9 / 50e-6)
damping = AmplitudeDamping(gamma=gamma)

state.x(0)  # Prepare |1⟩
damping.apply(state, 0)  # Some decay to |0⟩
```

### Phase Damping

Models dephasing (T2 decay):

```python
from moonlab.noise import PhaseDamping

# T2 = 70 μs, gate time = 20 ns
gamma = 1 - np.exp(-20e-9 / 70e-6)
dephasing = PhaseDamping(gamma=gamma)

state.h(0)  # Create superposition
dephasing.apply(state, 0)  # Lose phase coherence
```

### Bit Flip

Classical bit-flip error:

```python
from moonlab.noise import BitFlip

bit_flip = BitFlip(probability=0.001)
state.h(0)
bit_flip.apply(state, 0)
```

### Phase Flip

Random phase error:

```python
from moonlab.noise import PhaseFlip

phase_flip = PhaseFlip(probability=0.001)
state.h(0)
phase_flip.apply(state, 0)
```

## Noise After Gates

### Per-Gate Noise

```python
from moonlab import QuantumState
from moonlab.noise import DepolarizingChannel

state = QuantumState(4)
noise_1q = DepolarizingChannel(error_rate=0.001)  # 0.1% single-qubit
noise_2q = DepolarizingChannel(error_rate=0.01)   # 1% two-qubit

# Apply gates with noise
state.h(0)
noise_1q.apply(state, 0)

state.cnot(0, 1)
noise_2q.apply(state, 0)
noise_2q.apply(state, 1)
```

### Automatic Noise Insertion

```python
from moonlab import NoisyCircuit
from moonlab.noise import DepolarizingChannel

circuit = NoisyCircuit(4)
circuit.set_noise_model(
    single_qubit=DepolarizingChannel(0.001),
    two_qubit=DepolarizingChannel(0.01)
)

# Noise applied automatically after each gate
circuit.h(0)
circuit.cnot(0, 1)
circuit.h(2)
circuit.cnot(2, 3)

state = circuit.run()
```

## Kraus Operators

### Custom Noise Channels

Define noise via Kraus operators:

```python
from moonlab.noise import KrausChannel
import numpy as np

# Custom channel: sqrt(0.99)*I + sqrt(0.01)*Z
K0 = np.sqrt(0.99) * np.array([[1, 0], [0, 1]])
K1 = np.sqrt(0.01) * np.array([[1, 0], [0, -1]])

channel = KrausChannel([K0, K1])
channel.apply(state, qubit=0)
```

### Verify Channel

```python
# Check trace preservation: sum(K†K) = I
from moonlab.noise import verify_channel

is_valid = verify_channel(channel)
print(f"Valid CPTP channel: {is_valid}")
```

## Measurement Errors

### Readout Errors

```python
from moonlab.noise import ReadoutError

# Confusion matrix: P(measured | actual)
# P(0|0)=0.99, P(1|0)=0.01, P(0|1)=0.02, P(1|1)=0.98
readout = ReadoutError([[0.99, 0.01], [0.02, 0.98]])

state = QuantumState(1)
state.x(0)  # Prepare |1⟩

# Measure with readout error
result = readout.measure(state, 0)  # May incorrectly return 0
```

### Correlated Readout Errors

```python
from moonlab.noise import CorrelatedReadoutError

# Full 2^n x 2^n confusion matrix
confusion = np.array([
    [0.95, 0.02, 0.02, 0.01],  # Actual |00⟩
    [0.02, 0.93, 0.03, 0.02],  # Actual |01⟩
    [0.02, 0.03, 0.93, 0.02],  # Actual |10⟩
    [0.01, 0.02, 0.02, 0.95]   # Actual |11⟩
])

readout = CorrelatedReadoutError(confusion)
```

## Device-Based Noise

### IBM Backend Model

```python
from moonlab.noise import IBMNoiseModel

# Create noise model from device properties
noise = IBMNoiseModel(
    t1={'0': 50e-6, '1': 55e-6},  # T1 times in seconds
    t2={'0': 70e-6, '1': 65e-6},  # T2 times
    gate_errors={
        'sx': {'0': 0.001, '1': 0.0012},
        'cx': {('0', '1'): 0.01}
    },
    readout_errors={'0': 0.02, '1': 0.025}
)

# Apply to circuit
circuit = NoisyCircuit(2, noise_model=noise)
```

### Custom Device Model

```python
from moonlab.noise import DeviceNoiseModel

device = DeviceNoiseModel(
    num_qubits=5,
    topology=[(0,1), (1,2), (2,3), (3,4), (0,4)],  # Ring
    t1=[50e-6] * 5,
    t2=[70e-6] * 5,
    single_qubit_error=0.001,
    two_qubit_error=0.01,
    readout_error=0.02
)
```

## Thermal Relaxation

### Combined T1/T2

```python
from moonlab.noise import ThermalRelaxation

relaxation = ThermalRelaxation(
    t1=50e-6,        # T1 in seconds
    t2=70e-6,        # T2 in seconds
    gate_time=20e-9  # Gate duration
)

state = QuantumState(1)
state.h(0)
relaxation.apply(state, 0)
```

### Temperature Effects

```python
from moonlab.noise import ThermalRelaxation

# Excited state population at finite temperature
temperature = 20e-3  # 20 mK
frequency = 5e9      # 5 GHz qubit

import scipy.constants as const
excited_pop = 1 / (1 + np.exp(const.h * frequency / (const.k * temperature)))

relaxation = ThermalRelaxation(
    t1=50e-6,
    t2=70e-6,
    gate_time=20e-9,
    excited_population=excited_pop
)
```

## Error Mitigation

### Zero-Noise Extrapolation

```python
from moonlab.error_mitigation import ZNE

def circuit_with_noise(noise_scale):
    state = QuantumState(4, backend='cpu')
    noise = DepolarizingChannel(0.01 * noise_scale)

    for q in range(4):
        state.h(q)
        noise.apply(state, q)

    return state.expectation_z(0)

# Extrapolate to zero noise
zne = ZNE(noise_factors=[1, 2, 3])
mitigated = zne.extrapolate(circuit_with_noise)
print(f"Mitigated expectation: {mitigated:.4f}")
```

### Measurement Error Mitigation

```python
from moonlab.error_mitigation import MeasurementMitigation

# Calibrate
calibrator = MeasurementMitigation(num_qubits=4)
calibrator.calibrate(shots=10000)

# Apply correction
raw_counts = {'0000': 450, '0001': 50, ...}
corrected_counts = calibrator.correct(raw_counts)
```

## Noise Analysis

### Fidelity Under Noise

```python
from moonlab import QuantumState
from moonlab.noise import DepolarizingChannel

def circuit_fidelity(error_rate, depth):
    """Measure fidelity vs circuit depth."""
    ideal = QuantumState(4)
    noisy = QuantumState(4)
    noise = DepolarizingChannel(error_rate)

    for _ in range(depth):
        for q in range(4):
            ideal.h(q)
            noisy.h(q)
            noise.apply(noisy, q)

    return ideal.fidelity(noisy)

# Scan error rates
for rate in [0.001, 0.01, 0.1]:
    fidelity = circuit_fidelity(rate, depth=10)
    print(f"Error rate {rate}: Fidelity = {fidelity:.4f}")
```

### Error Accumulation

```python
import matplotlib.pyplot as plt

depths = range(1, 51)
fidelities = []

for depth in depths:
    fidelities.append(circuit_fidelity(0.01, depth))

plt.plot(depths, fidelities)
plt.xlabel('Circuit Depth')
plt.ylabel('Fidelity')
plt.title('Fidelity Decay with Depth (1% error rate)')
```

## Example: VQE with Noise

```python
from moonlab.algorithms import VQE
from moonlab.noise import DeviceNoiseModel

# Define realistic device
device = DeviceNoiseModel(
    num_qubits=4,
    t1=[50e-6] * 4,
    t2=[70e-6] * 4,
    single_qubit_error=0.001,
    two_qubit_error=0.01
)

# VQE with noise
vqe = VQE(
    num_qubits=4,
    noise_model=device,
    shots=10000,
    error_mitigation='zne'
)

# Compare with ideal
result_noisy = vqe.compute_ground_state(H2_hamiltonian)
vqe.noise_model = None
result_ideal = vqe.compute_ground_state(H2_hamiltonian)

print(f"Ideal energy: {result_ideal.energy:.6f}")
print(f"Noisy energy: {result_noisy.energy:.6f}")
print(f"Error: {abs(result_noisy.energy - result_ideal.energy):.6f}")
```

## See Also

- [Noise Models Concepts](../concepts/noise-models.md) - Theory background
- [C API: Noise](../api/c/noise.md) - Low-level noise API
- [Tutorial: Noise Effects](../tutorials/06-vqe-molecular-simulation.md#noise-effects) - Noise in VQE

