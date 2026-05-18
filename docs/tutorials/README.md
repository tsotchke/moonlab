# Moonlab tutorials

A curated set of self-contained walkthroughs covering the principal
subsystems of the Moonlab quantum simulator.  Each tutorial states
its prerequisites explicitly and pins every numerical claim to a
unit test or benchmark in the source tree.

## Index

| # | Tutorial | Subject |
|---|---|---|
| 1 | [getting_started.md](getting_started.md) | Build configuration, first single-qubit program, Bell-pair preparation with CHSH verification. |
| 2 | [mpdo_noise.md](mpdo_noise.md) | Polynomial-cost noisy circuit simulation via the v0.3 matrix-product density-operator engine. |
| 3 | [topological_band_structure.md](topological_band_structure.md) | Chern and Z_2 invariants on the SSH, Qi-Wu-Zhang, Haldane, Kane-Mele, Bernevig-Hughes-Zhang, and Hofstadter models with the v0.3 quantum-geometric-tensor module. |
| 4 | [adaptive_bond_tdvp.md](adaptive_bond_tdvp.md) | Real- and imaginary-time evolution of MPS states with the v0.4 entropy-feedback PID bond-dimension controller; C, Python, and Rust worked examples. |

## Conventions

- All examples target the C API; complete, compilable programs are
  available under `examples/` in the source tree.  The Python and
  Rust bindings track the C API one minor release behind unless
  otherwise noted.
- Build commands assume an out-of-tree build directory `build/` at
  the repository root; adjust paths if a different layout is used.
- Tutorials that give an analytical result also identify the unit
  test or benchmark that verifies the corresponding numerical
  output to a stated tolerance.  Local deviations beyond that
  tolerance should be reported as issues.

## See also

- `docs/reference/`: function-level API reference, organised one
  file per module (gates, error codes, configuration, QGT, MPDO,
  and others).
- `docs/research/`: design notes and method derivations covering
  CA-MPS variational-D, the quantum geometric tensor, and the Z_2
  lattice gauge theory implementation.
- `documents/index.md`: top-level documentation entry point with
  the complete module map.
