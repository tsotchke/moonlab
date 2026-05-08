# Moonlab tutorials

Hands-on walkthroughs that take you from a fresh checkout to running
real simulations.  Each tutorial is self-contained and assumes the
prerequisites listed at its top.

## Index

| # | Tutorial | What you'll do |
|---|---|---|
| 1 | [getting_started.md](getting_started.md) | Build the library, run your first 4-qubit circuit, dump amplitudes. |
| 2 | [mpdo_noise.md](mpdo_noise.md) | Simulate noisy circuits with the v0.3 matrix-product density-operator engine. |
| 3 | [topological_band_structure.md](topological_band_structure.md) | Compute Chern numbers and Z_2 invariants on QWZ, Haldane, Kane-Mele, BHZ, Kitaev, and Hofstadter models with the v0.3 QGT module. |

## Conventions

- All examples target the C API and live under `examples/` in the
  source tree.  Python and Rust binding parity tracks the C API one
  point release behind unless noted.
- Build commands assume an out-of-tree build directory `build/` at the
  project root.  Adjust paths if you use a different layout.
- Where a tutorial gives an analytical answer, every numerical result
  in the corresponding test or benchmark agrees to within the stated
  tolerance.  If your local run disagrees by more than that, please
  open an issue.

## See also

- `docs/reference/` — function-level API reference, one file per
  module (gates, error codes, configuration, qgt, mpdo, ...).
- `docs/research/` — design notes and method derivations: CA-MPS
  variational-D, quantum geometric tensor, Z_2 lattice gauge theory.
- `documents/index.md` — top-level documentation entry-point with the
  full module map.
