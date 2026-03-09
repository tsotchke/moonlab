# Terminal User Interface

Interactive terminal application for quantum simulation.

**Crate**: `moonlab-tui`

## Overview

`moonlab-tui` provides a beautiful, interactive terminal user interface for exploring quantum computing concepts. Built with Ratatui, it offers:

- **Algorithm Browser**: Run and visualize quantum algorithms
- **Step-Through Mode**: Execute circuits gate by gate
- **Free Exploration**: Build custom circuits interactively
- **Feynman Diagrams**: Visualize particle physics processes
- **Real-Time Visualization**: Animated state evolution

## Installation

### From Source

```bash
cd bindings/rust/moonlab-tui
cargo build --release
```

The binary will be at `target/release/moonlab-tui`.

### Running

```bash
./moonlab-tui
```

Or with cargo:

```bash
cargo run --release
```

## Modes

### Algorithm Browser

The default mode for selecting and running quantum algorithms.

**Available Algorithms**:

| Algorithm | Description |
|-----------|-------------|
| Grover's Search | Quantum search with O(√N) speedup |
| Bell State | Maximally entangled 2-qubit state |
| GHZ State | Greenberger-Horne-Zeilinger entanglement |
| VQE | Variational Quantum Eigensolver ansatz |
| QAOA | Quantum Approximate Optimization |
| QFT | Quantum Fourier Transform |

**Controls**:

| Key | Action |
|-----|--------|
| `↑`/`k` | Select previous algorithm |
| `↓`/`j` | Select next algorithm |
| `Tab` | Cycle focus between panels |
| `Enter` | Run selected algorithm |
| `n` | Enter number of qubits |
| `s` | Step-through mode |
| `f` | Free exploration mode |
| `d` | Feynman diagram browser |
| `?` | Show help |
| `q`/`Esc` | Quit |

### Algorithm Running

Displays animated algorithm execution.

**Controls**:

| Key | Action |
|-----|--------|
| `Space` | Pause/resume animation |
| `r` | Restart algorithm |
| `b`/`Backspace` | Return to browser |
| `?` | Show help |

### Step-Through Mode

Execute circuits one gate at a time for detailed analysis.

**Controls**:

| Key | Action |
|-----|--------|
| `n`/`→` | Step forward |
| `p`/`←` | Step backward |
| `r` | Reset to initial state |
| `b`/`Backspace` | Return to browser |
| `?` | Show help |

### Free Exploration

Build custom quantum circuits interactively.

**Qubit Selection**:

| Key | Action |
|-----|--------|
| `0`-`9` | Select qubit by number |
| `↑`/`↓` | Cycle selected qubit |
| `Tab` | Cycle target qubit (for 2-qubit gates) |

**Single-Qubit Gates**:

| Key | Gate |
|-----|------|
| `h` | Hadamard |
| `x` | Pauli-X |
| `y` | Pauli-Y |
| `z` | Pauli-Z |
| `s` | S gate |
| `t` | T gate |

**Two-Qubit Gates**:

| Key | Gate |
|-----|------|
| `c` | CNOT (selected → target) |

**Other**:

| Key | Action |
|-----|--------|
| `r` | Reset state to \|0...0⟩ |
| `Esc`/`Backspace` | Exit to running mode |
| `?` | Show help |

### Feynman Diagram Browser

Explore QED Feynman diagrams.

**Available Diagrams**:

| Diagram | Process |
|---------|---------|
| QED Vertex | e⁻ → e⁻ + γ |
| e⁺e⁻ → μ⁺μ⁻ | Muon pair production |
| Compton Scattering | e⁻ + γ → e⁻ + γ |
| Pair Annihilation | e⁺ e⁻ → γ γ |
| Møller Scattering | e⁻ e⁻ → e⁻ e⁻ |
| Bhabha Scattering | e⁺ e⁻ → e⁺ e⁻ |
| Electron Self-Energy | 1-loop correction |
| Vacuum Polarization | 1-loop photon correction |

**Controls**:

| Key | Action |
|-----|--------|
| `↑`/`k` | Previous diagram |
| `↓`/`j` | Next diagram |
| `Enter` | Show diagram details |
| `b`/`Backspace` | Return to browser |
| `?` | Show help |

## User Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                      Moonlab Quantum Simulator                  │
├─────────────────┬───────────────────────────────────────────────┤
│   Algorithms    │                   Circuit                     │
│  ─────────────  │  q0: ─H──●──────                              │
│  > Bell State   │         │                                     │
│    GHZ State    │  q1: ────X──●───                              │
│    Grover's     │            │                                  │
│    VQE          │  q2: ───────X───                              │
│    QAOA         │                                               │
│    QFT          │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│   Parameters    │              State Probabilities              │
│  ─────────────  │                                               │
│  Qubits: 3      │  |000⟩ ████████████████████░░░░ 50.0%         │
│                 │  |111⟩ ████████████████████░░░░ 50.0%         │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│     Metrics     │              Entanglement                     │
│  ─────────────  │                                               │
│  Entropy: 0.693 │  S(0|12) = 0.693 bits (maximal)               │
│  Purity: 1.000  │                                               │
└─────────────────┴───────────────────────────────────────────────┘
 Status: Bell state: (|00⟩ + |11⟩)/√2
```

## Architecture

### Module Structure

```
moonlab-tui/
├── src/
│   ├── main.rs        # Entry point, terminal setup
│   ├── app.rs         # Application state and logic
│   ├── events.rs      # Event handling
│   └── ui/
│       ├── mod.rs     # UI rendering coordinator
│       ├── dashboard.rs    # Main dashboard layout
│       ├── circuit.rs      # Circuit diagram rendering
│       ├── amplitudes.rs   # State probability bars
│       ├── bloch.rs        # Bloch sphere visualization
│       ├── entropy.rs      # Entanglement display
│       └── feynman.rs      # Feynman diagram rendering
```

### App State

```rust
pub struct App {
    /// Current mode
    pub mode: AppMode,
    /// Current focus area
    pub focus: Focus,
    /// Selected algorithm
    pub selected_algorithm: usize,
    /// Number of qubits
    pub num_qubits: usize,
    /// Quantum state
    pub state: Option<QuantumState>,
    /// Animation progress (0.0 - 1.0)
    pub animation_progress: f64,
    /// Whether animation is running
    pub animating: bool,
    /// Circuit steps executed
    pub circuit_step: usize,
    /// Feynman diagram
    pub feynman_diagram: Option<FeynmanDiagram>,
    // ...
}
```

### App Modes

```rust
pub enum AppMode {
    AlgorithmBrowser,   // Browse and select algorithms
    AlgorithmRunning,   // Animated algorithm execution
    StepThrough,        // Step-by-step execution
    FreeExploration,    // Custom circuit building
    FeynmanBrowser,     // QED diagram browser
    Help,               // Help overlay
}
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `moonlab` | Quantum simulation engine |
| `ratatui` | Terminal UI framework |
| `crossterm` | Cross-platform terminal control |
| `anyhow` | Error handling |
| `tracing` | Logging |
| `num-complex` | Complex number support |

## Building from Source

### Prerequisites

- Rust 1.70+
- C compiler (for moonlab-sys)
- libclang (for bindgen)

### Build Steps

```bash
# Clone repository
git clone https://github.com/tsotchke/moonlab.git
cd moonlab

# Build TUI
cd bindings/rust/moonlab-tui
cargo build --release

# Run
./target/release/moonlab-tui
```

### Development Build

```bash
# With debug logging
RUST_LOG=debug cargo run
```

## Customization

### Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Set logging level (debug, info, warn, error) |
| `NO_COLOR` | Disable color output |

### Terminal Requirements

- Minimum 80x24 terminal size recommended
- True color support for best visuals
- UTF-8 encoding

## Examples

### Creating a Bell State

1. Launch `moonlab-tui`
2. Select "Bell State" with arrow keys
3. Press `Enter` to run
4. Observe the animation showing:
   - Hadamard gate on qubit 0
   - CNOT gate with control=0, target=1
   - Final state with 50% |00⟩ and 50% |11⟩

### Building a Custom GHZ State

1. Launch `moonlab-tui`
2. Press `n` and enter `4` for 4 qubits
3. Press `f` for free exploration
4. Press `h` to apply Hadamard on qubit 0
5. Press `Tab` to set target to qubit 1
6. Press `c` to apply CNOT(0→1)
7. Press `Tab` twice to set target to qubit 2
8. Press `c` to apply CNOT(0→2)
9. Press `Tab` to set target to qubit 3
10. Press `c` to apply CNOT(0→3)
11. Observe 4-qubit GHZ state: (|0000⟩ + |1111⟩)/√2

### Exploring Feynman Diagrams

1. Launch `moonlab-tui`
2. Press `d` for Feynman diagram browser
3. Use arrow keys to browse diagrams
4. Observe particle lines and vertices
5. Press `Enter` for process details
6. Press `b` to return to main browser

## Known Issues

- Large qubit counts (>12) may cause slow rendering
- Some terminal emulators may not display all Unicode characters
- Windows support requires Windows Terminal for best results

## See Also

- [moonlab](moonlab.md) - Rust quantum simulation library
- [moonlab-sys](moonlab-sys.md) - Low-level FFI bindings
- [Tutorials](../../tutorials/index.md) - Learning resources

