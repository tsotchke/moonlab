# Moonlab for Rust

Safe, idiomatic Rust bindings for the Moonlab quantum simulator.

The native Moonlab SDK must be installed. A Homebrew installation is detected
through pkg-config; custom SDK locations can be selected with
`MOONLAB_LIB_DIR` and `MOONLAB_INCLUDE_DIR`.

```rust
use moonlab::QuantumState;

let mut state = QuantumState::new(2).unwrap();
state.h(0).cnot(0, 1);
```

Quantum annealing exposes the same complete Ising/QUBO configuration and
result contract as C and Python:

```rust
use moonlab::annealing::{anneal_qubo, AnnealConfig};

let q = [-1.0, 1.0, 1.0, -1.0];
let result = anneal_qubo(&q, 1.0, &AnnealConfig::default()).unwrap();
assert_eq!(result.best_energy, 0.0);
```
