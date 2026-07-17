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
