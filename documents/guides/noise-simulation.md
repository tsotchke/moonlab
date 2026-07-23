# Noise Simulation Guide

Add realistic noise to quantum simulations with the `moonlab.noise` module.

## Overview

Real quantum computers experience decoherence and gate errors. Moonlab
models these with the Kraus-operator noise engine in the C core
(`src/quantum/noise.c`). Every class in `moonlab.noise` is a thin wrapper
over a real C channel -- there is no Python re-implementation of the
physics.

The C channels are **Monte-Carlo trajectory unravellings**: one
`apply(...)` call realises a single Kraus branch drawn from a uniform
random sample, exactly as a real device produces one sample per shot.
Averaging an observable (or the reduced density operator) over many shots
recovers the completely-positive trace-preserving (CPTP) map. Every draw
comes from a per-object `numpy` generator; pass `seed=` for
reproducibility.

## Noise Models

### Depolarizing Channel

Randomly applies X, Y, or Z with probability `p/3` each:

```python
from moonlab import QuantumState
from moonlab.noise import DepolarizingChannel

state = QuantumState(2)
noise = DepolarizingChannel(error_rate=0.01)   # or DepolarizingChannel(0.01)

state.h(0)
noise.apply(state, qubit=0)  # 1% chance of an X, Y, or Z error
```

The C convention is
`E_p(rho) = (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)`, which shrinks
the Bloch vector by a factor `1 - 4p/3`. It reaches the maximally mixed
state at `p = 3/4` (not `p = 1`).

### Amplitude Damping

Models energy relaxation (T1 decay), `|1> -> |0>`:

```python
import numpy as np
from moonlab import QuantumState
from moonlab.noise import AmplitudeDamping

# T1 = 50 us, gate time = 20 ns
gamma = 1 - np.exp(-20e-9 / 50e-6)
damping = AmplitudeDamping(gamma=gamma)

state = QuantumState(1)
state.x(0)              # prepare |1>
damping.apply(state, 0) # some probability of decay to |0>
```

At `gamma = 1` a populated `|1>` deterministically decays to `|0>`.

### Phase Damping

Models dephasing (T2 decay) with no energy loss:

```python
import numpy as np
from moonlab import QuantumState
from moonlab.noise import PhaseDamping

# T2 = 70 us, gate time = 20 ns
gamma = 1 - np.exp(-20e-9 / 70e-6)
dephasing = PhaseDamping(gamma=gamma)

state = QuantumState(1)
state.h(0)                # create superposition |+>
dephasing.apply(state, 0) # lose phase coherence (averaged <X> shrinks)
```

### Bit Flip / Phase Flip / Bit-Phase Flip

The elementary Pauli channels (X, Z, Y applied with probability `p`):

```python
from moonlab import QuantumState
from moonlab.noise import BitFlip, PhaseFlip, BitPhaseFlip

state = QuantumState(1)
state.h(0)

BitFlip(probability=0.001).apply(state, 0)       # X with prob 0.001
PhaseFlip(probability=0.001).apply(state, 0)      # Z with prob 0.001
BitPhaseFlip(probability=0.001).apply(state, 0)   # Y with prob 0.001
```

## Noise After Gates

### Per-Gate Noise

```python
from moonlab import QuantumState
from moonlab.noise import DepolarizingChannel

state = QuantumState(4)
noise_1q = DepolarizingChannel(error_rate=0.001)  # 0.1% single-qubit
noise_2q = DepolarizingChannel(error_rate=0.01)   # 1% two-qubit

# Apply gates, then insert noise after each
state.h(0)
noise_1q.apply(state, 0)

state.cnot(0, 1)
noise_2q.apply(state, 0)
noise_2q.apply(state, 1)
```

### Automatic Per-Gate Noise with a Device Model

`DeviceNoiseModel` carries a per-qubit noise profile and applies it via
the C `noise_apply_model` / `noise_apply_model_two_qubit`. Insert
`apply_single` after single-qubit gates and `apply_two` after two-qubit
gates:

```python
from moonlab import QuantumState
from moonlab.noise import DeviceNoiseModel

device = DeviceNoiseModel(
    num_qubits=4,
    single_qubit_error=0.001,
    two_qubit_error=0.01,
)

state = QuantumState(4)

state.h(0);        device.apply_single(state, 0)
state.cnot(0, 1);  device.apply_two(state, 0, 1)
state.h(2);        device.apply_single(state, 2)
state.cnot(2, 3);  device.apply_two(state, 2, 3)
```

## Measurement Errors

### Readout Errors

`ReadoutError` wraps the C `noise_readout_error` classical bit-flip on
top of the real projective-measurement engine. Construct it from the two
flip probabilities or from a 2x2 assignment (confusion) matrix
`[[P(0|0), P(1|0)], [P(0|1), P(1|1)]]`:

```python
from moonlab import QuantumState
from moonlab.noise import ReadoutError

# P(0|0)=0.99, P(1|0)=0.01, P(0|1)=0.02, P(1|1)=0.98
readout = ReadoutError([[0.99, 0.01], [0.02, 0.98]])

state = QuantumState(1)
state.x(0)  # prepare |1>

# Projectively measure, then apply the classical readout flip:
result = readout.measure(state, 0)  # may incorrectly return 0
```

You can also flip a raw classical outcome directly:

```python
from moonlab.noise import ReadoutError

readout = ReadoutError(error_0_to_1=0.3, error_1_to_0=0.1)
noisy_bit = readout.apply_outcome(0)  # 1 with probability 0.3
```

## Device-Based Noise

### Custom Device Model

`DeviceNoiseModel` accepts per-qubit T1/T2 relaxation times and an
optional `topology` (informational), alongside the depolarizing and
readout rates:

```python
from moonlab import QuantumState
from moonlab.noise import DeviceNoiseModel

device = DeviceNoiseModel(
    num_qubits=5,
    topology=[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],  # ring
    t1=[50e-6] * 5,
    t2=[70e-6] * 5,
    gate_time=20e-9,
    single_qubit_error=0.001,
    two_qubit_error=0.01,
    readout_error=0.02,
)

# apply_single now also applies thermal relaxation for the gate time.
state = QuantumState(5)
state.x(0)
device.apply_single(state, 0)

# Build a matching readout-error channel from the device readout rate:
readout = device.readout()
```

## Thermal Relaxation

### Combined T1/T2

`ThermalRelaxation` wraps the C `noise_thermal_relaxation`, which derives
an amplitude-damping rate `gamma1 = 1 - exp(-t/T1)` and a residual
pure-dephasing rate from the T2 budget (T2 is clamped to `<= 2*T1`):

```python
from moonlab import QuantumState
from moonlab.noise import ThermalRelaxation

relaxation = ThermalRelaxation(
    t1=50e-6,        # T1
    t2=70e-6,        # T2
    gate_time=20e-9  # gate duration (same time unit as T1/T2)
)

state = QuantumState(1)
state.h(0)
relaxation.apply(state, 0)
```

### Temperature Effects

A nonzero excited-state population (finite temperature) is modelled as
generalized amplitude damping: on the excitation branch the relaxation is
conjugated by X, so the qubit relaxes toward `|1>` instead of `|0>`.

```python
import numpy as np
from moonlab.noise import ThermalRelaxation

# Excited-state population at finite temperature
temperature = 20e-3  # 20 mK
frequency = 5e9      # 5 GHz qubit

import scipy.constants as const
excited_pop = 1 / (1 + np.exp(const.h * frequency / (const.k * temperature)))

relaxation = ThermalRelaxation(
    t1=50e-6,
    t2=70e-6,
    gate_time=20e-9,
    excited_population=excited_pop,
)
```

(If SciPy is not installed, set `excited_pop` to a literal such as `0.02`.)

## Error Mitigation

### Zero-Noise Extrapolation

`ZNE` takes a user callable `circuit_fn(noise_scale) -> expectation`. You run your
circuit at each amplified noise scale and return a real expectation value; ZNE sweeps
the requested `noise_factors` and extrapolates the result back to the zero-noise limit
with the C estimator (`method='linear'`, `'richardson'` (default), or `'exponential'`).

```python
import numpy as np
from moonlab import QuantumState
from moonlab.error_mitigation import ZNE

def expectation_z0(state):
    """<Z_0> from the measured probability distribution."""
    probs = state.probabilities()
    signs = np.where(np.arange(len(probs)) & 1, -1.0, 1.0)  # bit 0 sets the sign
    return float(np.dot(signs, probs))

def circuit_with_noise(noise_scale):
    # Ideal circuit: a definite <Z_0> = cos(0.7).
    state = QuantumState(4)
    state.ry(0, 0.7)
    ideal = expectation_z0(state)
    # Model a depolarizing channel whose strength grows with the noise scale:
    # each unit of noise shrinks the expectation toward 0 by (1 - p).
    p = 0.05
    return ideal * (1.0 - p) ** noise_scale

# Extrapolate to zero noise (lambda = 1 is the native noise level).
zne = ZNE(noise_factors=[1, 2, 3], method='richardson')
mitigated = zne.extrapolate(circuit_with_noise)
print(f"Mitigated <Z_0>: {mitigated:.4f}   (ideal cos(0.7) = {np.cos(0.7):.4f})")
print(f"Fit residual std: {zne.last_stderr:.2e}")
```

### Measurement Error Mitigation

`MeasurementMitigation` builds the assignment (confusion) matrix by preparing each
computational-basis state on the real simulator and sampling its measured distribution,
then corrects raw shot counts with the Tikhonov-regularised inverse. With an ideal
readout the assignment matrix is the identity and `correct()` is a no-op; under a
readout-noise model it removes the bias. The matrix is `2**num_qubits` square, so this
targets small registers (the default cap is 12 qubits).

```python
from moonlab.error_mitigation import MeasurementMitigation

# Calibrate the assignment matrix (shots per prepared basis state).
calibrator = MeasurementMitigation(num_qubits=2)
calibrator.calibrate(shots=2000)

# Correct raw shot counts (dict of bitstring -> count).
raw_counts = {'00': 450, '01': 30, '10': 25, '11': 495}
corrected_counts = calibrator.correct(raw_counts)
print(corrected_counts)
```

## Noise Analysis

### Fidelity Under Noise

Because each `apply` realises one trajectory, the noisy state stays pure,
so its fidelity to the ideal state is `|<ideal|noisy>|^2`. Average over
many trajectories to estimate the channel-averaged fidelity:

```python
import numpy as np
from moonlab import QuantumState
from moonlab.noise import DepolarizingChannel

def statevector_fidelity(a, b):
    """|<a|b>|^2 for two pure statevectors."""
    return float(abs(np.vdot(a.get_statevector(), b.get_statevector())) ** 2)

def circuit_fidelity(error_rate, depth, shots=200):
    """Trajectory-averaged fidelity vs circuit depth."""
    noise = DepolarizingChannel(error_rate)
    total = 0.0
    for _ in range(shots):
        ideal = QuantumState(4)
        noisy = QuantumState(4)
        for _ in range(depth):
            for q in range(4):
                ideal.h(q)
                noisy.h(q)
                noise.apply(noisy, q)
        total += statevector_fidelity(ideal, noisy)
    return total / shots

# Scan error rates
for rate in [0.001, 0.01, 0.1]:
    fidelity = circuit_fidelity(rate, depth=10)
    print(f"Error rate {rate}: Fidelity = {fidelity:.4f}")
```

### Error Accumulation

```python
import matplotlib.pyplot as plt

depths = range(1, 21)
fidelities = [circuit_fidelity(0.01, depth) for depth in depths]

plt.plot(list(depths), fidelities)
plt.xlabel('Circuit Depth')
plt.ylabel('Fidelity')
plt.title('Fidelity Decay with Depth (1% error rate)')
```

## Example: Energy Estimation Under Noise

A VQE energy is an expectation value `<psi(theta)|H|psi(theta)>`. Noise
biases that estimate. Here we prepare a trial state, add device noise
after each gate, and compare the trajectory-averaged energy of a simple
Hamiltonian `H = Z_0` against the ideal value.

```python
import numpy as np
from moonlab import QuantumState
from moonlab.noise import DeviceNoiseModel

def energy_z0(state):
    """<Z_0> from the probability distribution."""
    probs = state.probabilities()
    signs = np.where(np.arange(len(probs)) & 1, -1.0, 1.0)
    return float(np.dot(signs, probs))

theta = 0.7  # ansatz parameter

# Ideal energy.
ideal_state = QuantumState(4)
ideal_state.ry(0, theta)
ideal_energy = energy_z0(ideal_state)

# Noisy energy: trajectory-average over many shots.
device = DeviceNoiseModel(
    num_qubits=4,
    t1=[50e-6] * 4,
    t2=[70e-6] * 4,
    gate_time=20e-9,
    single_qubit_error=0.01,
)

shots = 400
acc = 0.0
for _ in range(shots):
    s = QuantumState(4)
    s.ry(0, theta)
    device.apply_single(s, 0)
    acc += energy_z0(s)
noisy_energy = acc / shots

print(f"Ideal energy:  {ideal_energy:.6f}   (cos(theta) = {np.cos(theta):.6f})")
print(f"Noisy energy:  {noisy_energy:.6f}")
print(f"Bias:          {abs(noisy_energy - ideal_energy):.6f}")
```

Feed the noisy estimator into `moonlab.error_mitigation.ZNE` to
extrapolate the bias away, as shown in the Zero-Noise Extrapolation
section above.

## See Also

- [Noise Models Concepts](../concepts/noise-models.md) - Theory background
- [C API: Noise](../api/c/noise.md) - Low-level noise API
- [Tutorial: Noise Effects](../tutorials/06-vqe-molecular-simulation.md#noise-effects) - Noise in VQE
