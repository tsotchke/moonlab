# Skyrmion Quantum Computing Module

Production-grade tensor network simulation for topological skyrmion-based quantum computing.

## Architecture Overview

```
skyrmion/
├── lattice/
│   ├── lattice_2d.h          # 2D lattice infrastructure
│   ├── snake_ordering.h      # Optimal 2D→1D mapping
│   └── neighbor_cache.h      # Precomputed neighbor structures
├── hamiltonian/
│   ├── mpo_2d.h              # Long-range MPO construction
│   ├── heisenberg.h          # Exchange interactions
│   ├── dmi.h                 # Dzyaloshinskii-Moriya interaction
│   └── zeeman_aniso.h        # External field & anisotropy
├── dynamics/
│   ├── tdvp.h                # Time-dependent variational principle
│   ├── tebd_2d.h             # 2D TEBD with swap gates
│   └── stt.h                 # Spin-transfer torque dynamics
├── observables/
│   ├── topological_charge.h  # Skyrmion number Q
│   ├── spin_structure.h      # S(q) structure factor
│   └── skyrmion_tracking.h   # Position & size tracking
├── operations/
│   ├── skyrmion_create.h     # Skyrmion creation
│   ├── skyrmion_move.h       # Current-driven motion
│   └── braiding.h            # Topological qubit operations
└── qubits/
    ├── skyrmion_qubit.h      # Qubit encoding
    ├── gates.h               # Topological gates
    └── readout.h             # Measurement protocols
```

## Physical Model

### Hamiltonian

```
H = H_exchange + H_DMI + H_Zeeman + H_anisotropy + H_dipolar

H_exchange = -J Σ_{<i,j>} S_i · S_j

H_DMI = D Σ_{<i,j>} d_ij · (S_i × S_j)
        where d_ij depends on bond direction and DMI type:
        - Bulk DMI: d_ij || r_ij
        - Interfacial DMI: d_ij ⊥ r_ij (in-plane)

H_Zeeman = -B · Σ_i S_i

H_anisotropy = -K Σ_i (S_i · n̂)²

H_dipolar = (μ₀/4π) Σ_{i≠j} [S_i·S_j/r³ - 3(S_i·r)(S_j·r)/r⁵]
```

### Topological Charge

```
Q = (1/4π) ∫ n · (∂n/∂x × ∂n/∂y) d²r

Discretized on lattice:
Q = (1/4π) Σ_plaquettes Ω_ijk

where Ω_ijk is the solid angle subtended by spins at vertices i,j,k:
Ω = 2 atan2(S_i · (S_j × S_k), 1 + S_i·S_j + S_j·S_k + S_k·S_i)
```

## Implementation Details

### Long-Range MPO Construction

For 2D systems with snake ordering, nearest-neighbor 2D interactions become
long-range 1D interactions with maximum range R = Lx (lattice width).

**Finite Automaton MPO:**
- Bond dimension scales as O(R × n_interaction_types)
- For Heisenberg + DMI: bond dim ≈ 4 × Lx
- Uses "carry" indices to track open interactions

```
MPO[site] structure:
- Track which interactions are "open" (started but not closed)
- Close interaction when partner site is reached
- Bond indices: [identity, open_XX, open_YY, open_ZZ, open_XY, ...]
```

### TDVP Time Evolution

Two-site TDVP for dynamics:
```
|ψ(t+dt)⟩ = exp(-iHdt)|ψ(t)⟩

Implemented via:
1. Sweep left-to-right, updating two sites at a time
2. Solve local TDSE: i∂_t|θ⟩ = H_eff|θ⟩
3. Use Lanczos for matrix exponential
4. SVD truncation to maintain bond dimension
```

### Skyrmion Braiding

Braiding two skyrmions exchanges their positions while accumulating
a geometric phase. For Ising anyons (Majorana-like):

```
Braid matrix: B = exp(iπ/4 σ_x)

Implementation:
1. Create two skyrmions at positions (x1,y1) and (x2,y2)
2. Apply time-dependent field gradient to move skyrmions
3. Execute braiding path (e.g., one skyrmion circles the other)
4. Measure accumulated phase via interference
```

## Performance Targets

| System Size | Bond Dim | Memory | DMRG Time | TDVP Step |
|-------------|----------|--------|-----------|-----------|
| 8×8 = 64    | 256      | 2 GB   | 10 min    | 1 sec     |
| 12×12 = 144 | 512      | 16 GB  | 2 hr      | 30 sec    |
| 16×16 = 256 | 1024     | 64 GB  | 24 hr     | 5 min     |

## References

1. N. Nagaosa & Y. Tokura, "Topological properties and dynamics of magnetic
   skyrmions", Nature Nanotech. 8, 899-911 (2013)

2. A. Fert, N. Reyren, V. Cros, "Magnetic skyrmions: advances in physics and
   potential applications", Nature Reviews Materials 2, 17031 (2017)

3. C. Psaroudaki & C. Panagopoulos, "Skyrmion Qubits: A New Class of Quantum
   Logic Gates", Phys. Rev. Lett. 127, 067201 (2021)

4. X. Zhang et al., "Skyrmion-skyrmion and skyrmion-edge repulsions in skyrmion-
   based racetrack memory", Sci. Rep. 5, 7643 (2015)

5. S.R. White, "Density matrix formulation for quantum renormalization groups",
   Phys. Rev. Lett. 69, 2863 (1992)

6. J. Haegeman et al., "Time-dependent variational principle for quantum
   lattices", Phys. Rev. Lett. 107, 070601 (2011)
