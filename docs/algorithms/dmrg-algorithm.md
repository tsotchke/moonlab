# DMRG Algorithm

Complete guide to the Density Matrix Renormalization Group.

## Overview

The Density Matrix Renormalization Group (DMRG) is the most powerful numerical method for finding ground states of one-dimensional quantum systems. It works by iteratively optimizing a Matrix Product State (MPS) representation.

**Discovered**: Steven White, 1992

**Applications**:
- Spin chain ground states
- Strongly correlated electrons
- Quantum phase transitions
- Entanglement studies
- Quantum chemistry (DMRG-CASSCF)

## Mathematical Foundation

### Matrix Product States

An $n$-site MPS with bond dimension $\chi$:

$$|\psi\rangle = \sum_{s_1,\ldots,s_n} A^{[1]}_{s_1} A^{[2]}_{s_2} \cdots A^{[n]}_{s_n} |s_1 s_2 \ldots s_n\rangle$$

where each $A^{[i]}_{s_i}$ is a $\chi_{i-1} \times \chi_i$ matrix.

**Storage**: $O(n d \chi^2)$ vs $O(d^n)$ for full state vector

### Canonical Forms

#### Left-canonical
$$\sum_{s} A^{[i]\dagger}_s A^{[i]}_s = I$$

#### Right-canonical
$$\sum_{s} B^{[i]}_s B^{[i]\dagger}_s = I$$

#### Mixed-canonical (center at site $k$)
$$|\psi\rangle = A^{[1]} \cdots A^{[k-1]} C^{[k]} B^{[k+1]} \cdots B^{[n]}$$

### Matrix Product Operators

Hamiltonians are represented as MPOs:

$$H = \sum_{s_1,s'_1,\ldots} W^{[1]}_{s_1,s'_1} \cdots W^{[n]}_{s_n,s'_n} |s_1\ldots s_n\rangle\langle s'_1\ldots s'_n|$$

## DMRG Algorithm

### Single-Site DMRG

**Goal**: Minimize $E = \langle\psi|H|\psi\rangle$

**Procedure**:

1. Start with random MPS in mixed-canonical form
2. Sweep left-to-right, then right-to-left
3. At each site, solve local eigenvalue problem
4. Repeat until energy converges

### Sweep Procedure

```
Right sweep: → → → → → → → → →
Site:        1 2 3 4 5 6 7 8 9

Left sweep:  ← ← ← ← ← ← ← ← ←
Site:        9 8 7 6 5 4 3 2 1
```

At each site:
1. Form effective Hamiltonian $H_{eff}$
2. Solve $H_{eff} |v\rangle = E |v\rangle$ for ground state
3. Update local tensor
4. Move canonical center

### Two-Site DMRG

Optimize pairs of adjacent sites for better convergence:

1. Merge tensors $A^{[i]}$ and $A^{[i+1]}$ into $\Theta$
2. Solve eigenvalue problem
3. SVD to split back: $\Theta = U S V^\dagger$
4. Truncate to bond dimension $\chi$

## Implementation

### Basic DMRG

```c
#include "dmrg.h"

int main() {
    // Heisenberg chain: H = J Σ S_i · S_{i+1}
    int n_sites = 50;
    double J = 1.0;

    // Create Heisenberg MPO
    mpo_t* H = heisenberg_mpo_create(n_sites, J);

    // Configure DMRG
    dmrg_config_t config = {
        .max_bond_dim = 100,
        .max_sweeps = 20,
        .energy_tolerance = 1e-10,
        .truncation_cutoff = 1e-12,
        .two_site = true  // Two-site algorithm
    };

    // Run DMRG
    dmrg_result_t result = dmrg_find_ground_state(H, &config);

    printf("Ground state energy: %.10f\n", result.energy);
    printf("Energy per site: %.10f\n", result.energy / n_sites);
    printf("Final bond dimension: %d\n", result.final_bond_dim);
    printf("Converged: %s\n", result.converged ? "yes" : "no");
    printf("Sweeps: %d\n", result.num_sweeps);

    // Access ground state MPS
    mps_t* ground_state = result.state;

    mpo_destroy(H);
    mps_destroy(ground_state);
    return 0;
}
```

### Python Interface

```python
from moonlab.tensor_network import DMRG, HeisenbergMPO, MPS

# Create Heisenberg Hamiltonian
n_sites = 50
J = 1.0
H = HeisenbergMPO(n_sites, J)

# Configure DMRG
dmrg = DMRG(
    max_bond_dim=100,
    max_sweeps=20,
    energy_tolerance=1e-10,
    two_site=True,
    verbose=True
)

# Find ground state
result = dmrg.find_ground_state(H)

print(f"Energy: {result.energy:.10f}")
print(f"Energy/site: {result.energy / n_sites:.10f}")
print(f"Exact (Bethe): {-0.4431471805599 * n_sites:.10f}")
```

## Effective Hamiltonian

### Construction

The effective Hamiltonian at site $k$ is formed by contracting the environment:

```
        ┌─────────────────────────────────┐
        │         MPO (H)                 │
Left    │  ┌───┐ ┌───┐ ... ┌───┐ ┌───┐    │    Right
Env  ─○─┤  │W_1│─│W_2│─...─│W_k│─│...│────├─○─ Env
  L     │  └─┬─┘ └─┬─┘     └─┬─┘ └───┘    │     R
        │    │     │    θ    │            │
        │  ┌─┴─┐ ┌─┴─┐ ┌───┐ ┌───┐        │
        │  │A_1│─│A_2│─│ ? │─│...│────────┤
        │  └───┘ └───┘ └───┘ └───┘        │
        └─────────────────────────────────┘
```

The effective Hamiltonian acts on the local tensor $\theta$.

### Sparse Eigensolver

Use iterative methods for large effective Hamiltonians:

```python
from scipy.sparse.linalg import eigsh

def solve_local_eigenvalue(H_eff, initial_guess):
    """Solve effective eigenvalue problem."""
    def matvec(v):
        return apply_effective_hamiltonian(H_eff, v)

    eigenvalue, eigenvector = eigsh(
        LinearOperator((dim, dim), matvec=matvec),
        k=1,
        which='SA',  # Smallest algebraic
        v0=initial_guess,
        tol=1e-10
    )

    return eigenvalue[0], eigenvector[:, 0]
```

## SVD and Truncation

### Singular Value Decomposition

After optimization, split the two-site tensor:

$$\Theta_{s_k, s_{k+1}} = \sum_{\alpha} U_{s_k, \alpha} S_\alpha V^\dagger_{\alpha, s_{k+1}}$$

### Truncation Strategies

#### Fixed Bond Dimension
Keep largest $\chi$ singular values:

```python
def truncate_fixed(S, U, V, chi):
    """Truncate to fixed bond dimension."""
    S_trunc = S[:chi]
    U_trunc = U[:, :chi]
    V_trunc = V[:chi, :]
    return U_trunc, S_trunc, V_trunc
```

#### Truncation Error Threshold
Keep singular values above threshold:

```python
def truncate_threshold(S, U, V, epsilon):
    """Truncate by singular value threshold."""
    chi = np.sum(S / S[0] > epsilon)
    return truncate_fixed(S, U, V, chi)
```

#### Discarded Weight
Monitor truncation error:

$$\epsilon_{trunc} = \sum_{\alpha > \chi} S_\alpha^2$$

## Hamiltonians as MPOs

### Nearest-Neighbor Interactions

For $H = \sum_i h_{i,i+1}$:

```python
def nearest_neighbor_mpo(n_sites, interaction):
    """
    Create MPO for nearest-neighbor Hamiltonian.

    interaction: 4x4 matrix for 2-site interaction
    """
    d = 2  # Local dimension
    D = 5  # MPO bond dimension

    # Decompose interaction
    # h = sum_a L_a ⊗ R_a
    L_ops, R_ops = decompose_interaction(interaction)

    W = np.zeros((D, D, d, d))

    # Bulk tensor structure:
    # W = | I    0    0    0    0  |
    #     | L_1  0    0    0    0  |
    #     | L_2  0    0    0    0  |
    #     | ...  0    0    0    0  |
    #     | h_0  R_1  R_2  ...  I  |

    return construct_mpo(W, n_sites)
```

### Long-Range Interactions

Use larger MPO bond dimension or approximate:

```python
def coulomb_mpo(n_sites, decay_cutoff=1e-8):
    """
    Create MPO for 1/r Coulomb interaction.

    Uses exponential sum approximation.
    """
    # Approximate 1/r as sum of exponentials
    # 1/r ≈ Σ_k c_k exp(-λ_k r)

    coefficients, exponents = fit_coulomb(decay_cutoff)
    return exponential_sum_mpo(n_sites, coefficients, exponents)
```

## Advanced Features

### Excited States

Find multiple eigenstates:

```python
# Project out ground state to find first excited
dmrg_excited = DMRG(max_bond_dim=100)

# First, find ground state
gs_result = dmrg_excited.find_ground_state(H)
ground_state = gs_result.state

# Find first excited state
result_1 = dmrg_excited.find_excited_state(
    H,
    orthogonal_to=[ground_state],
    penalty_weight=10.0
)

print(f"Ground state energy: {gs_result.energy:.6f}")
print(f"First excited energy: {result_1.energy:.6f}")
print(f"Gap: {result_1.energy - gs_result.energy:.6f}")
```

### Finite Temperature

Use purification for thermal states:

```python
from moonlab.tensor_network import PurificationDMRG

# Thermal state at temperature T
beta = 1.0 / T  # Inverse temperature

thermal_dmrg = PurificationDMRG(max_bond_dim=200)
result = thermal_dmrg.find_thermal_state(H, beta)

# Expectation values at finite T
magnetization = result.state.expectation_z(n_sites // 2)
energy = result.state.expectation(H)
```

### Time Evolution (TEBD)

Time-evolving block decimation:

```python
from moonlab.tensor_network import TEBD

# Initial state
mps = MPS.product_state(n_sites, [0] * (n_sites // 2) + [1] * (n_sites // 2))

# Time evolution
tebd = TEBD(
    time_step=0.05,
    trotter_order=2,
    max_bond_dim=100
)

for t in np.arange(0, 10, 0.1):
    tebd.evolve(mps, H, 0.1)

    # Measure observables
    mag = [mps.expectation_z(i) for i in range(n_sites)]
    print(f"t={t:.1f}: ⟨Sz⟩ = {mag}")
```

## Observables

### Local Expectation Values

```python
# Single-site
sz = mps.expectation_z(site=25)
sx = mps.expectation_x(site=25)

# Two-site correlation
szz = mps.correlation_zz(site1=10, site2=40)

# General operator
op = np.array([[1, 0], [0, -1]])  # Z
exp_val = mps.expectation_local(op, site=25)
```

### Entanglement Entropy

```python
# Von Neumann entropy at bond
entropy = mps.entanglement_entropy(bond=25)

# Full entanglement profile
profile = [mps.entanglement_entropy(i) for i in range(n_sites - 1)]

import matplotlib.pyplot as plt
plt.plot(profile)
plt.xlabel('Bond')
plt.ylabel('Entanglement entropy')
plt.title('Entanglement profile')
```

### Correlation Functions

```python
def correlation_function(mps, op1, op2, max_distance):
    """Compute ⟨O_i O_j⟩ as function of distance."""
    center = mps.num_sites // 2
    correlations = []

    for d in range(max_distance):
        corr = mps.two_point_correlation(op1, center, op2, center + d)
        correlations.append(corr)

    return correlations

# Spin-spin correlation
zz_corr = correlation_function(ground_state, 'Z', 'Z', 20)
```

## Performance Optimization

### Bond Dimension Scheduling

```python
# Start with small χ, increase gradually
dmrg = DMRG(
    bond_dim_schedule=[20, 40, 60, 80, 100],
    sweeps_per_bond_dim=2
)
```

### Noise for Local Minima

Add noise to escape local minima:

```python
dmrg = DMRG(
    max_bond_dim=100,
    noise_schedule=[1e-3, 1e-4, 1e-5, 0, 0]  # Decrease noise
)
```

### Parallelization

```python
# Enable OpenMP parallelization
dmrg = DMRG(
    max_bond_dim=200,
    num_threads=8  # Parallelize tensor contractions
)
```

## Convergence Analysis

### Energy Convergence

```python
# Monitor energy during sweeps
dmrg = DMRG(max_bond_dim=100, verbose=True)
result = dmrg.find_ground_state(H)

# Plot convergence
plt.semilogy(result.energy_history[1:] - result.energy, 'o-')
plt.xlabel('Sweep')
plt.ylabel('Energy error')
plt.title('DMRG Convergence')
```

### Bond Dimension Scaling

```python
# Extrapolate to χ → ∞
bond_dims = [20, 40, 60, 80, 100, 120]
energies = []
truncation_errors = []

for chi in bond_dims:
    dmrg = DMRG(max_bond_dim=chi)
    result = dmrg.find_ground_state(H)
    energies.append(result.energy)
    truncation_errors.append(result.truncation_error)

# Plot E vs truncation error for extrapolation
plt.plot(truncation_errors, energies, 'o-')
plt.xlabel('Truncation error')
plt.ylabel('Energy')
```

## Example: Ising Phase Transition

```python
from moonlab.tensor_network import DMRG, IsingMPO
import numpy as np

# Transverse field Ising model
# H = -J Σ Z_i Z_{i+1} - h Σ X_i
# Critical point at h/J = 1

J = 1.0
h_values = np.linspace(0.5, 1.5, 11)

n_sites = 100
dmrg = DMRG(max_bond_dim=50)

energies = []
magnetizations = []
entropies = []

for h in h_values:
    H = IsingMPO(n_sites, J=J, h=h)
    result = dmrg.find_ground_state(H)

    gs = result.state
    mag = np.mean([abs(gs.expectation_z(i)) for i in range(n_sites)])
    entropy = gs.entanglement_entropy(n_sites // 2)

    energies.append(result.energy / n_sites)
    magnetizations.append(mag)
    entropies.append(entropy)

    print(f"h/J = {h:.2f}: E/N = {result.energy/n_sites:.6f}, "
          f"|⟨Z⟩| = {mag:.4f}, S = {entropy:.4f}")

# Phase transition visible in magnetization and entropy peak
```

## Complexity Analysis

| Component | Complexity |
|-----------|------------|
| Two-site update | $O(d^2 \chi^3 D + d^3 \chi^3)$ |
| Sweep | $O(n \cdot \text{update})$ |
| Eigensolver | $O(\chi^3 d^2)$ per iteration |
| Full DMRG | $O(n_{sweeps} \cdot n \cdot d^3 \chi^3)$ |

Where:
- $d$ = local Hilbert space dimension
- $\chi$ = bond dimension
- $D$ = MPO bond dimension
- $n$ = number of sites

## See Also

- [Tutorial: Tensor Networks](../tutorials/08-tensor-network-simulation.md) - Step-by-step tutorial
- [Tensor Networks Concepts](../concepts/tensor-networks.md) - Theoretical background
- [C API: Tensor Network](../api/c/tensor-network.md) - Complete C API reference

## References

**Foundational Papers**:
1. White, S.R. (1992). "Density matrix formulation for quantum renormalization groups." *Phys. Rev. Lett.* 69, 2863.
2. White, S.R. (1993). "Density-matrix algorithms for quantum renormalization groups." *Phys. Rev. B* 48, 10345.
3. Schollwöck, U. (2005). "The density-matrix renormalization group." *Rev. Mod. Phys.* 77, 259.

**Matrix Product States**:
4. Schollwöck, U. (2011). "The density-matrix renormalization group in the age of matrix product states." *Ann. Phys.* 326, 96-192.
5. Orús, R. (2014). "A practical introduction to tensor networks." *Ann. Phys.* 349, 117-158.
6. Vidal, G. (2003). "Efficient classical simulation of slightly entangled quantum computations." *Phys. Rev. Lett.* 91, 147902.

**Quantum Chemistry Applications**:
7. Chan, G.K.-L. & Head-Gordon, M. (2002). "Highly correlated calculations with a polynomial cost algorithm: A study of the density matrix renormalization group." *J. Chem. Phys.* 116, 4462.

