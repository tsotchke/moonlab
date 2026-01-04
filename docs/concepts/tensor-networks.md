# Tensor Networks

Efficient representation of quantum many-body states.

## Introduction

Tensor networks provide an efficient way to represent and manipulate quantum states that would otherwise require exponential resources. By exploiting the structure of entanglement in physical states, tensor networks enable simulation of systems far beyond the reach of full state-vector methods.

## Motivation

### The Exponential Wall

A full state vector for $n$ qubits requires $2^n$ complex amplitudes:

| Qubits | Amplitudes | Memory |
|--------|-----------|--------|
| 30 | ~10^9 | 16 GB |
| 50 | ~10^15 | 16 PB |
| 100 | ~10^30 | Impossible |

### Area Law States

Physical ground states often satisfy an **area law**: entanglement entropy scales with boundary size, not volume:

$$S(A) \sim |\partial A|$$

Such states can be efficiently represented using tensor networks.

## Tensor Basics

### Tensors as Multi-Index Arrays

A **tensor** is a multi-dimensional array. Index notation:

- Scalar: $c$ (rank 0)
- Vector: $v_i$ (rank 1)
- Matrix: $M_{ij}$ (rank 2)
- 3-tensor: $T_{ijk}$ (rank 3)

### Tensor Diagrams

Graphical notation for tensor operations:

```
Scalar:       ●

Vector:       ●──i

Matrix:    j──●──i

3-tensor:     i
              │
            j─●─k
```

Each leg represents an index. Connected legs = contraction (summation).

### Tensor Contraction

**Contraction** sums over shared indices:

$$C_{ik} = \sum_j A_{ij} B_{jk}$$

Diagrammatically:

```
i──[A]──j──[B]──k   →   i──[C]──k
```

## Matrix Product States (MPS)

### Definition

A **Matrix Product State** represents an $n$-qubit state as:

$$|\psi\rangle = \sum_{i_1,\ldots,i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} \cdots A^{[n]}_{i_n} |i_1 i_2 \cdots i_n\rangle$$

where:
- $A^{[k]}_{i_k}$ are matrices (tensors) at each site
- Matrix dimensions: $\chi \times \chi$ (**bond dimension**)
- Physical indices $i_k \in \{0, 1\}$

### Diagram

```
|i₁⟩    |i₂⟩    |i₃⟩         |iₙ⟩
  │       │       │             │
[A¹]───[A²]───[A³]───...───[Aⁿ]
```

### Storage

MPS storage scales as $O(n \chi^2 d)$ where:
- $n$ = number of sites
- $\chi$ = bond dimension
- $d$ = physical dimension (2 for qubits)

| State Type | Required $\chi$ |
|-----------|-----------------|
| Product states | 1 |
| GHZ state | 2 |
| Random state | $2^{n/2}$ |
| Ground state (gapped) | $O(1)$ |
| Ground state (critical) | $O(\text{poly}(n))$ |

### MPS in Moonlab

```c
// Create MPS with bond dimension 32
mps_state_t* mps = mps_create(num_qubits, 32);

// Apply gates
mps_apply_gate_1q(mps, H_gate, qubit);
mps_apply_gate_2q(mps, CNOT_gate, control, target);

// Compress to limit bond dimension growth
mps_compress(mps, max_bond_dim, tolerance);

// Measure expectation value
double exp_z = mps_expectation_z(mps, qubit);

// Get full state vector (if small enough)
complex_t* amplitudes = mps_to_statevector(mps);
```

## Canonical Forms

### Left-Canonical Form

Each tensor satisfies:

$$\sum_{i_k} (A^{[k]}_{i_k})^\dagger A^{[k]}_{i_k} = I$$

Tensors are left-orthonormal.

### Right-Canonical Form

Each tensor satisfies:

$$\sum_{i_k} A^{[k]}_{i_k} (A^{[k]}_{i_k})^\dagger = I$$

Tensors are right-orthonormal.

### Mixed-Canonical Form

Left-canonical to site $k$, right-canonical from site $k+1$:

```
[L]──[L]──[C]──[R]──[R]
  ↑         ↑        ↑
 left    center    right
 canonical       canonical
```

Center matrix $C$ contains the Schmidt values for bipartition at $k$.

### SVD-Based Canonicalization

```c
// Put MPS in left-canonical form
mps_canonicalize_left(mps);

// Put in right-canonical form
mps_canonicalize_right(mps);

// Put in mixed-canonical form with center at site k
mps_canonicalize_mixed(mps, k);
```

## MPS Operations

### Gate Application

**Single-qubit gate**: Absorb into local tensor

```c
// O(χ²d²) complexity
mps_apply_gate_1q(mps, gate, site);
```

**Two-qubit gate**: Contract with tensors, decompose via SVD

```
[A]──[B]         [A']──[B']
 │    │    →      │     │
  ╲  ╱             ╲   ╱
   [G]             SVD
```

Bond dimension may increase after 2-qubit gates.

### Compression

After operations, compress to limit bond dimension:

1. Put in canonical form
2. SVD at each bond
3. Truncate to keep largest $\chi$ singular values
4. Error bounded by discarded weight

```c
// Compress with relative tolerance
double truncation_error = mps_compress(mps, max_chi, 1e-10);
```

### Expectation Values

Local observables computed in $O(n\chi^2)$:

```c
double exp = mps_expectation_local(mps, observable, site);
```

Two-point correlators:

```c
double corr = mps_correlation(mps, obs1, site1, obs2, site2);
```

## Matrix Product Operators (MPO)

### Definition

An **MPO** represents operators as tensor networks:

$$\hat{O} = \sum_{i,j} W^{[1]}_{i_1 j_1} W^{[2]}_{i_2 j_2} \cdots W^{[n]}_{i_n j_n} |i\rangle\langle j|$$

### Diagram

```
|i₁⟩    |i₂⟩    |i₃⟩
  │       │       │
[W¹]───[W²]───[W³]───...
  │       │       │
⟨j₁|    ⟨j₂|    ⟨j₃|
```

### Hamiltonians as MPO

Local Hamiltonians have efficient MPO representations:

**Transverse-field Ising**:
$$H = -J\sum_i Z_i Z_{i+1} - h\sum_i X_i$$

MPO bond dimension: 3

**Heisenberg**:
$$H = J\sum_i \vec{S}_i \cdot \vec{S}_{i+1}$$

MPO bond dimension: 5

```c
// Create Ising Hamiltonian MPO
mpo_t* H = mpo_create_ising(num_sites, J, h);

// Compute energy
double energy = mps_expectation_mpo(mps, H);
```

## DMRG Algorithm

### Overview

**Density Matrix Renormalization Group** finds ground states by variationally optimizing MPS tensors.

### Algorithm

1. Initialize MPS in canonical form
2. Sweep left-to-right, then right-to-left
3. At each site, solve local eigenvalue problem
4. Update tensor and move canonical center
5. Repeat until convergence

### Local Optimization

At site $k$, minimize:

$$E = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}$$

with respect to tensor $A^{[k]}$, keeping others fixed.

This is a generalized eigenvalue problem solvable by Lanczos/Davidson.

### Implementation

```c
// Create DMRG solver
dmrg_config_t config = {
    .max_sweeps = 20,
    .max_bond_dim = 100,
    .energy_tolerance = 1e-10,
    .truncation_weight = 1e-12
};

dmrg_result_t result = dmrg_ground_state(H_mpo, initial_mps, &config);

printf("Ground state energy: %.10f\n", result.energy);
printf("Converged: %s\n", result.converged ? "yes" : "no");
printf("Final bond dimension: %d\n", result.final_bond_dim);
```

### Convergence

DMRG converges rapidly for 1D gapped systems. Convergence indicators:
- Energy change per sweep
- Variance: $\langle H^2 \rangle - \langle H \rangle^2$
- Truncation error

## Time Evolution

### TEBD (Time-Evolving Block Decimation)

For time evolution under local Hamiltonian:

1. Trotter decompose: $e^{-iHt} \approx \prod_k e^{-ih_k t}$
2. Apply 2-site gates sequentially
3. Compress after each gate

```c
// Time evolve with TEBD
tebd_config_t config = {
    .time_step = 0.01,
    .num_steps = 100,
    .trotter_order = 2,
    .max_bond_dim = 200
};

tebd_evolve(mps, H_local_terms, &config);
```

### TDVP (Time-Dependent Variational Principle)

Project dynamics onto MPS manifold:

$$i\frac{d|\psi\rangle}{dt} = P_{T_\psi} H |\psi\rangle$$

where $P_{T_\psi}$ projects onto tangent space of MPS manifold.

**Advantages over TEBD**:
- Works for long-range Hamiltonians
- Better conservation of energy
- Single-site variant preserves bond dimension

```c
// TDVP evolution
tdvp_config_t config = {
    .time_step = 0.01,
    .method = TDVP_TWO_SITE  // or TDVP_ONE_SITE
};

tdvp_evolve(mps, H_mpo, total_time, &config);
```

## Higher-Dimensional Networks

### PEPS (Projected Entangled Pair States)

Generalization of MPS to 2D:

```
    │       │       │
──[T]───[T]───[T]──
    │       │       │
──[T]───[T]───[T]──
    │       │       │
```

**Challenges**:
- Contraction is #P-hard
- Approximate contraction methods needed

### MERA (Multi-scale Entanglement Renormalization Ansatz)

Hierarchical structure for critical systems:

```
        [top]
       /     \
    [U]       [U]
   /   \     /   \
 [U]   [U] [U]   [U]
  │     │   │     │
 ...   ...  ...  ...
```

Captures logarithmic entanglement scaling.

## Performance

### Complexity Comparison

| Operation | State Vector | MPS ($\chi$) |
|-----------|-------------|--------------|
| Storage | $O(2^n)$ | $O(n\chi^2)$ |
| 1-qubit gate | $O(2^n)$ | $O(\chi^2)$ |
| 2-qubit gate | $O(2^n)$ | $O(\chi^3)$ |
| Expectation | $O(2^n)$ | $O(n\chi^3)$ |
| DMRG sweep | N/A | $O(n\chi^3 D)$ |

$D$ = MPO bond dimension

### When to Use Tensor Networks

**Good for**:
- 1D systems with area-law entanglement
- Ground states of gapped Hamiltonians
- Thermal states at low temperature
- Shallow circuits

**Not good for**:
- Highly entangled states
- Deep random circuits
- 2D systems (PEPS tractable but expensive)

## References

**Review Articles**:
1. Schollwöck, U. (2011). "The density-matrix renormalization group in the age of matrix product states." *Ann. Phys.* 326, 96-192.
2. Orús, R. (2014). "A practical introduction to tensor networks." *Ann. Phys.* 349, 117-158.
3. Verstraete, F., Murg, V., & Cirac, J.I. (2008). "Matrix product states, projected entangled pair states, and variational renormalization group methods." *Adv. Phys.* 57, 143-224.

**Foundational Papers**:
4. White, S.R. (1992). "Density matrix formulation for quantum renormalization groups." *Phys. Rev. Lett.* 69, 2863.
5. Vidal, G. (2003). "Efficient classical simulation of slightly entangled quantum computations." *Phys. Rev. Lett.* 91, 147902.
6. Vidal, G. (2004). "Efficient simulation of one-dimensional quantum many-body systems." *Phys. Rev. Lett.* 93, 040502.

**Entanglement Renormalization**:
7. Vidal, G. (2007). "Entanglement renormalization." *Phys. Rev. Lett.* 99, 220405.
8. Evenbly, G. & Vidal, G. (2009). "Algorithms for entanglement renormalization." *Phys. Rev. B* 79, 144108.

## See Also

- [DMRG Algorithm](../algorithms/dmrg-algorithm.md) - Detailed DMRG guide
- [State Vector Simulation](state-vector-simulation.md) - Comparison approach
- [Entanglement Measures](entanglement-measures.md) - When tensor networks work
- [C API: Tensor Network](../api/c/tensor-network.md) - Function reference

