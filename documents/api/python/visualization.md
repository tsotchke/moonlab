# Archived Moonlab Documentation: Visualization API

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Visualization API

Complete reference for quantum circuit visualization in the Python API.

**Module**: `moonlab.visualization`

## Overview

The visualization module provides tools for rendering quantum circuits in multiple formats:

- **ASCII**: Terminal-friendly text diagrams
- **Matplotlib**: Publication-quality figures
- **LaTeX**: Quantikz-based circuit diagrams
- **SVG**: Scalable vector graphics for web
- **Interactive**: Jupyter notebook widgets

## Installation

[archived fence delimiter: ```python]
from moonlab.visualization import CircuitDrawer, draw_circuit
[archived fence delimiter: ```]

Optional dependencies:
[archived fence delimiter: ```bash]
pip install matplotlib  # For figure output
pip install pylatex     # For LaTeX generation
[archived fence delimiter: ```]

## Quick Start

[archived fence delimiter: ```python]
from moonlab import QuantumState
from moonlab.visualization import draw_circuit

# Create a circuit
state = QuantumState(3)
state.h(0).cnot(0, 1).cnot(1, 2)

# Draw in ASCII
print(draw_circuit(state))
[archived fence delimiter: ```]

Output:
[archived fence delimiter: ```]
q0: ─H──●──────
        │
q1: ────X──●───
           │
q2: ───────X───
[archived fence delimiter: ```]

## CircuitDrawer

Main class for circuit visualization.

### Constructor

[archived fence delimiter: ```python]
CircuitDrawer(
    num_qubits: int,
    style: str = 'default',
    scale: float = 1.0
)
[archived fence delimiter: ```]

**Parameters**:
- `num_qubits`: Number of qubits to display
- `style`: Visual style preset
  - `'default'`: Standard circuit style
  - `'ibm'`: IBM Qiskit-like style
  - `'google'`: Cirq-like style
  - `'minimal'`: Clean, minimal style
- `scale`: Scaling factor for output

### Methods

#### add_gate

[archived fence delimiter: ```python]
add_gate(
    gate_name: str,
    qubits: Union[int, List[int]],
    params: Optional[List[float]] = None,
    label: Optional[str] = None
) -> CircuitDrawer
[archived fence delimiter: ```]

Add a gate to the circuit diagram.

**Parameters**:
- `gate_name`: Gate identifier ('H', 'X', 'CNOT', 'RX', etc.)
- `qubits`: Target qubit(s)
- `params`: Gate parameters (for rotation gates)
- `label`: Custom label override

**Returns**: Self for method chaining

**Example**:
[archived fence delimiter: ```python]
from moonlab.visualization import CircuitDrawer

drawer = CircuitDrawer(3)
drawer.add_gate('H', 0)
drawer.add_gate('CNOT', [0, 1])
drawer.add_gate('RZ', 2, params=[3.14159])
drawer.add_gate('TOFFOLI', [0, 1, 2])
[archived fence delimiter: ```]

#### add_measurement

[archived fence delimiter: ```python]
add_measurement(
    qubit: int,
    classical_bit: Optional[int] = None
) -> CircuitDrawer
[archived fence delimiter: ```]

Add measurement symbol.

[archived fence delimiter: ```python]
drawer.add_measurement(0)
drawer.add_measurement(1, classical_bit=0)
[archived fence delimiter: ```]

#### add_barrier

[archived fence delimiter: ```python]
add_barrier(qubits: Optional[List[int]] = None) -> CircuitDrawer
[archived fence delimiter: ```]

Add visual barrier (dashed line).

[archived fence delimiter: ```python]
drawer.add_barrier()  # All qubits
drawer.add_barrier([0, 1])  # Specific qubits
[archived fence delimiter: ```]

#### add_label

[archived fence delimiter: ```python]
add_label(text: str, qubit: int) -> CircuitDrawer
[archived fence delimiter: ```]

Add text label at current position.

### Rendering Methods

#### to_ascii

[archived fence delimiter: ```python]
to_ascii(
    wire_char: str = '─',
    show_labels: bool = True
) -> str
[archived fence delimiter: ```]

Render as ASCII art.

**Parameters**:
- `wire_char`: Character for wire lines
- `show_labels`: Show qubit labels

**Example**:
[archived fence delimiter: ```python]
drawer = CircuitDrawer(2)
drawer.add_gate('H', 0)
drawer.add_gate('CNOT', [0, 1])
print(drawer.to_ascii())
[archived fence delimiter: ```]

Output:
[archived fence delimiter: ```]
q0: ─H──●──
        │
q1: ────X──
[archived fence delimiter: ```]

#### to_matplotlib

[archived fence delimiter: ```python]
to_matplotlib(
    figsize: Tuple[float, float] = None,
    dpi: int = 100,
    show_grid: bool = False
) -> matplotlib.figure.Figure
[archived fence delimiter: ```]

Render as matplotlib figure.

**Parameters**:
- `figsize`: Figure size in inches (auto-calculated if None)
- `dpi`: Resolution for raster output
- `show_grid`: Show background grid

**Example**:
[archived fence delimiter: ```python]
import matplotlib.pyplot as plt

drawer = CircuitDrawer(3, style='ibm')
drawer.add_gate('H', 0)
drawer.add_gate('CNOT', [0, 1])
drawer.add_gate('CNOT', [1, 2])
drawer.add_measurement(2)

fig = drawer.to_matplotlib(figsize=(8, 4))
fig.savefig('circuit.png', dpi=150)
plt.show()
[archived fence delimiter: ```]

#### to_latex

[archived fence delimiter: ```python]
to_latex(
    standalone: bool = True,
    package: str = 'quantikz'
) -> str
[archived fence delimiter: ```]

Generate LaTeX source code.

**Parameters**:
- `standalone`: Include document preamble
- `package`: LaTeX package ('quantikz' or 'qcircuit')

**Example**:
[archived fence delimiter: ```python]
drawer = CircuitDrawer(2)
drawer.add_gate('H', 0)
drawer.add_gate('CNOT', [0, 1])

latex_code = drawer.to_latex()
print(latex_code)
[archived fence delimiter: ```]

Output:
[archived fence delimiter: ```latex]
\documentclass{standalone}
\usepackage{quantikz}
\begin{document}
\begin{quantikz}
\lstick{$q_0$} & \gate{H} & \ctrl{1} & \qw \\
\lstick{$q_1$} & \qw & \targ{} & \qw
\end{quantikz}
\end{document}
[archived fence delimiter: ```]

#### to_svg

[archived fence delimiter: ```python]
to_svg(
    width: int = 800,
    height: Optional[int] = None
) -> str
[archived fence delimiter: ```]

Generate SVG markup.

**Example**:
[archived fence delimiter: ```python]
svg_code = drawer.to_svg(width=600)
with open('circuit.svg', 'w') as f:
    f.write(svg_code)
[archived fence delimiter: ```]

#### save

[archived fence delimiter: ```python]
save(
    filename: str,
    format: Optional[str] = None,
    **kwargs
) -> None
[archived fence delimiter: ```]

Save circuit to file.

**Parameters**:
- `filename`: Output file path
- `format`: Format override (inferred from extension if None)

**Supported formats**: png, pdf, svg, tex, txt

[archived fence delimiter: ```python]
drawer.save('circuit.png', dpi=300)
drawer.save('circuit.pdf')
drawer.save('circuit.tex', standalone=False)
drawer.save('circuit.svg')
drawer.save('circuit.txt')  # ASCII
[archived fence delimiter: ```]

## Convenience Functions

### draw_circuit

[archived fence delimiter: ```python]
draw_circuit(
    state: QuantumState,
    format: str = 'ascii',
    **kwargs
) -> Union[str, Figure]
[archived fence delimiter: ```]

Draw circuit from QuantumState history.

**Parameters**:
- `state`: QuantumState with gate history
- `format`: Output format ('ascii', 'matplotlib', 'latex', 'svg')

**Example**:
[archived fence delimiter: ```python]
from moonlab import QuantumState
from moonlab.visualization import draw_circuit

state = QuantumState(3)
state.h(0).cnot(0, 1).cnot(1, 2).t(2)

# ASCII output
print(draw_circuit(state))

# Matplotlib figure
fig = draw_circuit(state, format='matplotlib')
fig.savefig('ghz.png')
[archived fence delimiter: ```]

### draw_state

[archived fence delimiter: ```python]
draw_state(
    state: QuantumState,
    basis: str = 'computational',
    format: str = 'bar'
) -> matplotlib.figure.Figure
[archived fence delimiter: ```]

Visualize quantum state amplitudes/probabilities.

**Parameters**:
- `state`: QuantumState to visualize
- `basis`: Representation basis ('computational', 'bloch')
- `format`: Plot type ('bar', 'hinton', 'city', 'bloch')

**Example**:
[archived fence delimiter: ```python]
from moonlab import QuantumState
from moonlab.visualization import draw_state

state = QuantumState(3)
state.h(0).cnot(0, 1).cnot(1, 2)

# Probability bar chart
fig = draw_state(state, format='bar')
fig.savefig('ghz_probs.png')

# Bloch sphere (single qubit)
state1 = QuantumState(1)
state1.h(0).t(0)
fig = draw_state(state1, format='bloch')
[archived fence delimiter: ```]

### draw_bloch

[archived fence delimiter: ```python]
draw_bloch(
    theta: float,
    phi: float,
    r: float = 1.0,
    **kwargs
) -> matplotlib.figure.Figure
[archived fence delimiter: ```]

Draw point on Bloch sphere.

[archived fence delimiter: ```python]
from moonlab.visualization import draw_bloch
import numpy as np

# |+⟩ state
fig = draw_bloch(theta=np.pi/2, phi=0)
fig.savefig('plus_state.png')
[archived fence delimiter: ```]

## Gate Symbols

Standard gate symbols used in diagrams:

| Gate | ASCII | Symbol |
|------|-------|--------|
| Hadamard | `H` | $H$ |
| Pauli-X | `X` | $X$ |
| Pauli-Y | `Y` | $Y$ |
| Pauli-Z | `Z` | $Z$ |
| S | `S` | $S$ |
| T | `T` | $T$ |
| CNOT | `●─X` | Control + Target |
| CZ | `●─●` | Two controls |
| SWAP | `×─×` | Crossed lines |
| Toffoli | `●─●─X` | Two controls + target |
| RX(θ) | `RX(θ)` | $R_X(\theta)$ |
| RY(θ) | `RY(θ)` | $R_Y(\theta)$ |
| RZ(θ) | `RZ(θ)` | $R_Z(\theta)$ |
| Measure | `M` | Meter symbol |

## Styles

### Built-in Styles

[archived fence delimiter: ```python]
# IBM Qiskit style
drawer = CircuitDrawer(4, style='ibm')

# Google Cirq style
drawer = CircuitDrawer(4, style='google')

# Minimal clean style
drawer = CircuitDrawer(4, style='minimal')
[archived fence delimiter: ```]

### Custom Styles

[archived fence delimiter: ```python]
from moonlab.visualization import CircuitStyle

custom_style = CircuitStyle(
    wire_color='#333333',
    gate_fill='#4a90d9',
    gate_stroke='#2c5aa0',
    text_color='white',
    font_family='Arial',
    font_size=12,
    gate_width=40,
    gate_height=30,
    wire_spacing=50,
    padding=20
)

drawer = CircuitDrawer(3, style=custom_style)
[archived fence delimiter: ```]

### Style Properties

| Property | Type | Description |
|----------|------|-------------|
| `wire_color` | str | Wire line color |
| `gate_fill` | str | Gate background color |
| `gate_stroke` | str | Gate border color |
| `text_color` | str | Label text color |
| `control_color` | str | Control dot color |
| `measure_color` | str | Measurement symbol color |
| `font_family` | str | Font for labels |
| `font_size` | int | Label font size |
| `gate_width` | int | Gate box width (pixels) |
| `gate_height` | int | Gate box height (pixels) |
| `wire_spacing` | int | Vertical space between wires |
| `gate_spacing` | int | Horizontal space between gates |

## Jupyter Integration

### Interactive Display

[archived fence delimiter: ```python]
from moonlab.visualization import CircuitDrawer

drawer = CircuitDrawer(3)
drawer.add_gate('H', 0)
drawer.add_gate('CNOT', [0, 1])
drawer.add_gate('CNOT', [1, 2])

# In Jupyter, displays interactively
drawer  # Auto-renders in notebook
[archived fence delimiter: ```]

### Widget Controls

[archived fence delimiter: ```python]
from moonlab.visualization import InteractiveCircuit

# Create interactive circuit builder
circuit = InteractiveCircuit(num_qubits=4)
circuit.display()  # Shows drag-and-drop interface
[archived fence delimiter: ```]

## Algorithm Visualization

### Grover's Algorithm

[archived fence delimiter: ```python]
from moonlab.visualization import draw_grover_circuit

fig = draw_grover_circuit(
    num_qubits=4,
    marked_state=5,
    num_iterations=3
)
fig.savefig('grover.png')
[archived fence delimiter: ```]

### VQE Ansatz

[archived fence delimiter: ```python]
from moonlab.visualization import draw_vqe_ansatz

fig = draw_vqe_ansatz(
    num_qubits=4,
    num_layers=2,
    entanglement='linear'
)
fig.savefig('vqe_ansatz.png')
[archived fence delimiter: ```]

### QAOA Circuit

[archived fence delimiter: ```python]
from moonlab.visualization import draw_qaoa_circuit

fig = draw_qaoa_circuit(
    num_qubits=5,
    num_layers=3,
    show_params=True
)
fig.savefig('qaoa.png')
[archived fence delimiter: ```]

## Animation

### Gate-by-Gate Animation

[archived fence delimiter: ```python]
from moonlab.visualization import CircuitAnimator

state = QuantumState(3)
state.h(0).cnot(0, 1).cnot(1, 2)

animator = CircuitAnimator(state)
animator.save('circuit_animation.gif', fps=2)
[archived fence delimiter: ```]

### State Evolution

[archived fence delimiter: ```python]
from moonlab.visualization import animate_state_evolution

state = QuantumState(2)
history = []

state.h(0)
history.append(state.get_statevector().copy())

state.cnot(0, 1)
history.append(state.get_statevector().copy())

animate_state_evolution(history, 'evolution.gif')
[archived fence delimiter: ```]

## Complete Example

[archived fence delimiter: ```python]
from moonlab import QuantumState
from moonlab.visualization import CircuitDrawer, draw_state
import matplotlib.pyplot as plt

# Create quantum circuit
state = QuantumState(4)

# Build circuit
state.h(0)
state.h(1)
state.cnot(0, 2)
state.cnot(1, 3)
state.cz(2, 3)
state.h(2)
state.h(3)

# Visualize circuit
drawer = CircuitDrawer(4, style='ibm')
drawer.add_gate('H', 0)
drawer.add_gate('H', 1)
drawer.add_gate('CNOT', [0, 2])
drawer.add_gate('CNOT', [1, 3])
drawer.add_gate('CZ', [2, 3])
drawer.add_gate('H', 2)
drawer.add_gate('H', 3)
drawer.add_measurement(0)
drawer.add_measurement(1)
drawer.add_measurement(2)
drawer.add_measurement(3)

# Create figure with circuit and state
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Circuit diagram
circuit_fig = drawer.to_matplotlib()
# Copy to subplot...

# State probabilities
probs = state.probabilities()
axes[1].bar(range(16), probs)
axes[1].set_xlabel('Basis State')
axes[1].set_ylabel('Probability')
axes[1].set_title('Output Distribution')

plt.tight_layout()
plt.savefig('full_visualization.png', dpi=150)
plt.show()

# Export in multiple formats
drawer.save('circuit.pdf')
drawer.save('circuit.svg')
print(drawer.to_latex(standalone=False))
print(drawer.to_ascii())
[archived fence delimiter: ```]

## See Also

- [Core API](core.md) - QuantumState, Gates
- [Algorithms API](algorithms.md) - VQE, QAOA, Grover
- [PyTorch Integration](torch-layer.md) - Neural network layers

```
