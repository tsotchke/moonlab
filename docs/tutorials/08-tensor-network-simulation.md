# Tutorial 08: Tensor Network Simulation

Simulate large quantum systems with MPS and DMRG.

**Duration**: 60 minutes
**Prerequisites**: [Tutorial 07](07-qaoa-optimization.md)
**Difficulty**: Advanced

## Learning Objectives

By the end of this tutorial, you will:

- Understand Matrix Product States (MPS)
- Implement basic MPS operations
- Run DMRG to find ground states
- Simulate systems beyond state vector limits

## The Scaling Problem

State vector simulation memory scales as $2^n$:

| Qubits | Memory |
|--------|--------|
| 30 | 16 GB |
| 40 | 16 TB |
| 50 | 16 PB |

**Tensor networks** exploit structure to represent states compactly.

## Matrix Product States

An **MPS** represents an $n$-qubit state as:

$$|\psi\rangle = \sum_{i_1,\ldots,i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} \cdots A^{[n]}_{i_n} |i_1 \ldots i_n\rangle$$

where each $A^{[k]}$ is a $\chi \times \chi$ matrix (bond dimension $\chi$).

**Storage**: $O(n\chi^2)$ instead of $O(2^n)$

## Step 1: Creating an MPS

```python
from moonlab.tensor_network import MPS, MPO
import numpy as np

# Create a 50-qubit MPS with bond dimension 32
n_qubits = 50
bond_dim = 32

mps = MPS(n_qubits, bond_dim)

# MPS starts in |00...0⟩ state
print(f"Number of sites: {mps.num_sites}")
print(f"Bond dimension: {mps.bond_dim}")
print(f"Total parameters: {mps.num_parameters}")
```

**Output**:
```
Number of sites: 50
Bond dimension: 32
Total parameters: 51200
```

Compare: Full state vector would need $2^{50} \approx 10^{15}$ parameters!

## Step 2: Applying Gates

### Single-Qubit Gates

```python
# Apply Hadamard to first qubit
mps.h(0)

# Apply rotation
mps.ry(10, np.pi/4)

# Apply X gate
mps.x(25)
```

### Two-Qubit Gates

Two-qubit gates can increase bond dimension:

```python
# Apply CNOT (may increase bond dimension)
mps.cnot(5, 6)

# Apply CZ
mps.cz(20, 21)

# Check new bond dimension
print(f"Bond dimension after gates: {mps.bond_dim}")
```

### Compression

After operations, compress to control bond dimension:

```python
# Compress to maximum bond dimension 64
truncation_error = mps.compress(max_bond_dim=64, tolerance=1e-10)
print(f"Truncation error: {truncation_error:.2e}")
```

## Step 3: GHZ State with MPS

```python
def create_ghz_mps(n):
    """Create n-qubit GHZ state using MPS."""
    mps = MPS(n, bond_dim=2)

    # H on first qubit
    mps.h(0)

    # CNOT chain
    for i in range(n - 1):
        mps.cnot(i, i + 1)

    return mps

# Create 100-qubit GHZ state!
ghz = create_ghz_mps(100)

# Verify: should have 50/50 probability for |00...0⟩ and |11...1⟩
probs = ghz.get_probabilities([0, 99])  # Check first and last qubit
print(f"P(00) = {probs['00']:.4f}")
print(f"P(11) = {probs['11']:.4f}")
```

**Output**:
```
P(00) = 0.5000
P(11) = 0.5000
```

A 100-qubit state would need $10^{30}$ amplitudes with state vector!

## Step 4: Expectation Values

```python
# Single-site expectation
exp_z = ghz.expectation_z(50)
print(f"⟨Z_50⟩ = {exp_z:.4f}")

# Two-site correlation
corr_zz = ghz.correlation_zz(0, 99)
print(f"⟨Z_0 Z_99⟩ = {corr_zz:.4f}")

# Entanglement entropy
entropy = ghz.entanglement_entropy(50)  # Bipartition at site 50
print(f"Entanglement entropy: {entropy:.4f} bits")
```

**Output**:
```
⟨Z_50⟩ = 0.0000
⟨Z_0 Z_99⟩ = 1.0000
Entanglement entropy: 1.0000 bits
```

## Step 5: DMRG Algorithm

**DMRG** (Density Matrix Renormalization Group) finds ground states of local Hamiltonians.

### Heisenberg Spin Chain

$$H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1} = \frac{J}{2} \sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})$$

```python
from moonlab.tensor_network import DMRG, HeisenbergMPO

# Create Heisenberg Hamiltonian as MPO
n_sites = 50
J = 1.0
hamiltonian = HeisenbergMPO(n_sites, J)

# Configure DMRG
dmrg = DMRG(
    max_sweeps=20,
    max_bond_dim=100,
    energy_tolerance=1e-10,
    truncation_weight=1e-12
)

# Run DMRG
result = dmrg.find_ground_state(hamiltonian)

print(f"Ground state energy: {result.energy:.10f}")
print(f"Energy per site: {result.energy / n_sites:.10f}")
print(f"Final bond dimension: {result.bond_dim}")
print(f"Converged: {result.converged}")
print(f"Sweeps performed: {result.num_sweeps}")
```

**Output**:
```
Ground state energy: -22.3178435912
Energy per site: -0.4463568718
Final bond dimension: 87
Converged: True
Sweeps performed: 12
```

### Analyzing the Ground State

```python
# Get ground state MPS
ground_state = result.state

# Compute local magnetization
magnetization = [ground_state.expectation_z(i) for i in range(n_sites)]

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(magnetization, 'o-')
plt.xlabel('Site')
plt.ylabel('⟨Sz⟩')
plt.title('Local Magnetization in Heisenberg Ground State')
plt.axhline(y=0, color='r', linestyle='--')
plt.savefig('heisenberg_magnetization.png')
plt.show()

# Compute spin-spin correlation
correlations = []
for i in range(n_sites):
    corr = ground_state.correlation_zz(0, i)
    correlations.append(corr)

plt.figure(figsize=(12, 4))
plt.semilogy(range(n_sites), np.abs(correlations), 'o-')
plt.xlabel('Distance from site 0')
plt.ylabel('|⟨Z_0 Z_i⟩|')
plt.title('Spin-Spin Correlation (semi-log)')
plt.savefig('heisenberg_correlation.png')
plt.show()
```

## Step 6: Transverse Field Ising Model

$$H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i$$

```python
from moonlab.tensor_network import IsingMPO

# Phase transition at h/J = 1
J = 1.0
h_values = np.linspace(0.1, 2.0, 20)

energies = []
magnetizations = []

for h in h_values:
    # Create Hamiltonian
    H = IsingMPO(n_sites=30, J=J, h=h)

    # Find ground state
    dmrg = DMRG(max_bond_dim=50)
    result = dmrg.find_ground_state(H)

    # Compute observables
    gs = result.state
    mag = np.mean([abs(gs.expectation_z(i)) for i in range(30)])

    energies.append(result.energy)
    magnetizations.append(mag)

    print(f"h={h:.2f}: E={result.energy:.4f}, |⟨Z⟩|={mag:.4f}")

# Plot phase transition
plt.figure(figsize=(10, 6))
plt.plot(h_values, magnetizations, 'o-')
plt.axvline(x=1.0, color='r', linestyle='--', label='Critical point')
plt.xlabel('h/J')
plt.ylabel('|⟨Z⟩|')
plt.title('Ising Model Phase Transition')
plt.legend()
plt.savefig('ising_phase_transition.png')
plt.show()
```

## Step 7: Time Evolution with TEBD

**TEBD** (Time-Evolving Block Decimation) simulates time evolution:

```python
from moonlab.tensor_network import TEBD

# Initial state: domain wall
mps = MPS(30, bond_dim=50)
for i in range(15):
    mps.x(i)  # |111...000...⟩

# Hamiltonian
H = HeisenbergMPO(30, J=1.0)

# TEBD configuration
tebd = TEBD(
    time_step=0.05,
    trotter_order=2,
    max_bond_dim=100
)

# Evolve and track magnetization
times = []
profiles = []

for t in np.arange(0, 10, 0.5):
    tebd.evolve(mps, H, 0.5)  # Evolve for 0.5 time units

    # Record magnetization profile
    mag_profile = [mps.expectation_z(i) for i in range(30)]
    times.append(t)
    profiles.append(mag_profile)

# Visualize
plt.figure(figsize=(12, 8))
for t, profile in zip(times, profiles):
    plt.plot(profile, alpha=0.5, label=f't={t:.1f}')
plt.xlabel('Site')
plt.ylabel('⟨Sz⟩')
plt.title('Domain Wall Dynamics in Heisenberg Chain')
plt.savefig('tebd_dynamics.png')
plt.show()
```

## MPS Limitations

MPS works well when:
- System is 1D or quasi-1D
- Entanglement follows area law
- Ground states of gapped Hamiltonians

MPS struggles with:
- Highly entangled states (random circuits)
- 2D+ systems (need PEPS)
- Volume-law entanglement

### Bond Dimension Growth

```python
# Random circuit: bond dimension grows exponentially
mps = MPS(20, bond_dim=2)

for depth in range(10):
    for i in range(0, 19, 2):
        # Random 2-qubit gate
        angle = np.random.uniform(0, 2*np.pi)
        mps.cnot(i, i+1)
        mps.ry(i, angle)

    mps.compress(max_bond_dim=200)
    print(f"Depth {depth}: Bond dim = {mps.bond_dim}")
```

## Exercises

### Exercise 1: XXZ Model

Implement DMRG for the XXZ model:
$$H = J \sum_i (X_i X_{i+1} + Y_i Y_{i+1} + \Delta Z_i Z_{i+1})$$

Explore the phase diagram as $\Delta$ varies.

### Exercise 2: Entanglement Entropy Scaling

For the critical Ising model ($h = J$), verify the logarithmic scaling of entanglement entropy.

### Exercise 3: 100-Qubit Grover

Can you run Grover's algorithm on 100 qubits using MPS? What limits the depth you can simulate?

### Exercise 4: TDVP Evolution

Implement TDVP (Time-Dependent Variational Principle) for more accurate time evolution.

## Key Takeaways

1. **MPS** represent states with limited entanglement efficiently
2. **Bond dimension** controls accuracy vs. cost
3. **DMRG** finds ground states of 1D systems
4. **TEBD** simulates time evolution
5. **Compression** keeps bond dimension manageable

## Next Steps

Accelerate simulations with GPU:

**[09. GPU Acceleration →](09-gpu-acceleration.md)**

## Further Reading

- [Tensor Networks](../concepts/tensor-networks.md) - Theory background
- [DMRG Algorithm](../algorithms/dmrg-algorithm.md) - Full mathematical treatment
- [C API: Tensor Network](../api/c/tensor-network.md) - Low-level implementation
- Schollwöck (2011). "The density-matrix renormalization group in the age of matrix product states." Annals of Physics, 326, 96-192.

