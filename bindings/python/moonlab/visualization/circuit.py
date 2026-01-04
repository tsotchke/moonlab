"""
Quantum Circuit Diagram Visualization

Publication-quality quantum circuit diagram rendering in multiple formats:
- ASCII (terminal output)
- Matplotlib figure (interactive/saveable)
- SVG (web/publication)
- LaTeX/quantikz (papers)

Example:
--------
    >>> from moonlab.visualization import CircuitDiagram
    >>>
    >>> circuit = CircuitDiagram(2, title="Bell State")
    >>> circuit.h(0).cx(0, 1).measure_all()
    >>> circuit.render()  # Returns matplotlib figure
    >>> circuit.to_ascii()  # Returns ASCII string
    >>> circuit.to_latex()  # Returns LaTeX/quantikz code
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GateType(Enum):
    """Circuit element types"""
    SINGLE = "single"           # Single-qubit gate
    CONTROLLED = "controlled"   # Controlled gate
    MULTI = "multi"            # Multi-qubit gate (SWAP, Toffoli)
    MEASUREMENT = "measurement"
    BARRIER = "barrier"
    RESET = "reset"


@dataclass
class CircuitElement:
    """Single circuit element"""
    type: GateType
    qubit: int
    time_slot: int
    name: str = ""
    target: int = -1          # For controlled gates
    control2: int = -1        # For Toffoli
    angle: float = 0.0        # For rotation gates
    classical_bit: int = -1   # For measurement
    barrier_qubits: List[int] = field(default_factory=list)


class CircuitDiagram:
    """
    Quantum circuit diagram builder and renderer.

    Example:
    --------
        circuit = CircuitDiagram(3, title="GHZ State")
        circuit.h(0)
        for i in range(2):
            circuit.cx(i, i+1)
        circuit.measure_all()

        # Render
        fig = circuit.render()
        plt.show()

        # Export
        circuit.save("ghz_circuit.svg")
        print(circuit.to_latex())
    """

    def __init__(self, num_qubits: int, num_classical: int = None, title: str = ""):
        """
        Initialize circuit diagram.

        Parameters:
        -----------
        num_qubits : int
            Number of qubits in the circuit
        num_classical : int, optional
            Number of classical bits (defaults to num_qubits)
        title : str, optional
            Circuit title for display
        """
        self.num_qubits = num_qubits
        self.num_classical = num_classical if num_classical is not None else num_qubits
        self.title = title
        self.elements: List[CircuitElement] = []
        self.qubit_labels = [f"q{i}" for i in range(num_qubits)]
        self.current_time = [0] * num_qubits

    def set_qubit_label(self, qubit: int, label: str) -> 'CircuitDiagram':
        """Set custom label for a qubit wire."""
        if 0 <= qubit < self.num_qubits:
            self.qubit_labels[qubit] = label
        return self

    def _get_time(self, *qubits) -> int:
        """Get current time slot for qubits."""
        return max(self.current_time[q] for q in qubits if 0 <= q < self.num_qubits)

    def _advance_time(self, *qubits):
        """Advance time for qubits."""
        t = self._get_time(*qubits) + 1
        for q in qubits:
            if 0 <= q < self.num_qubits:
                self.current_time[q] = t

    # Single-qubit gates
    def h(self, qubit: int) -> 'CircuitDiagram':
        """Add Hadamard gate."""
        return self._add_single_gate(qubit, "H")

    def x(self, qubit: int) -> 'CircuitDiagram':
        """Add Pauli-X gate."""
        return self._add_single_gate(qubit, "X")

    def y(self, qubit: int) -> 'CircuitDiagram':
        """Add Pauli-Y gate."""
        return self._add_single_gate(qubit, "Y")

    def z(self, qubit: int) -> 'CircuitDiagram':
        """Add Pauli-Z gate."""
        return self._add_single_gate(qubit, "Z")

    def s(self, qubit: int) -> 'CircuitDiagram':
        """Add S (phase) gate."""
        return self._add_single_gate(qubit, "S")

    def t(self, qubit: int) -> 'CircuitDiagram':
        """Add T (pi/8) gate."""
        return self._add_single_gate(qubit, "T")

    def rx(self, qubit: int, angle: float) -> 'CircuitDiagram':
        """Add RX rotation gate."""
        return self._add_rotation_gate(qubit, "Rx", angle)

    def ry(self, qubit: int, angle: float) -> 'CircuitDiagram':
        """Add RY rotation gate."""
        return self._add_rotation_gate(qubit, "Ry", angle)

    def rz(self, qubit: int, angle: float) -> 'CircuitDiagram':
        """Add RZ rotation gate."""
        return self._add_rotation_gate(qubit, "Rz", angle)

    def _add_single_gate(self, qubit: int, name: str) -> 'CircuitDiagram':
        """Internal: add single-qubit gate."""
        t = self.current_time[qubit]
        self.elements.append(CircuitElement(
            type=GateType.SINGLE,
            qubit=qubit,
            time_slot=t,
            name=name
        ))
        self.current_time[qubit] = t + 1
        return self

    def _add_rotation_gate(self, qubit: int, name: str, angle: float) -> 'CircuitDiagram':
        """Internal: add rotation gate."""
        t = self.current_time[qubit]
        angle_str = self._format_angle(angle)
        self.elements.append(CircuitElement(
            type=GateType.SINGLE,
            qubit=qubit,
            time_slot=t,
            name=f"{name}({angle_str})",
            angle=angle
        ))
        self.current_time[qubit] = t + 1
        return self

    # Two-qubit gates
    def cx(self, control: int, target: int) -> 'CircuitDiagram':
        """Add CNOT (controlled-X) gate."""
        return self._add_controlled_gate(control, target, "X")

    def cz(self, control: int, target: int) -> 'CircuitDiagram':
        """Add CZ (controlled-Z) gate."""
        return self._add_controlled_gate(control, target, "Z")

    def cy(self, control: int, target: int) -> 'CircuitDiagram':
        """Add CY (controlled-Y) gate."""
        return self._add_controlled_gate(control, target, "Y")

    def cp(self, control: int, target: int, angle: float) -> 'CircuitDiagram':
        """Add controlled phase gate."""
        angle_str = self._format_angle(angle)
        return self._add_controlled_gate(control, target, f"P({angle_str})")

    def crx(self, control: int, target: int, angle: float) -> 'CircuitDiagram':
        """Add controlled RX gate."""
        angle_str = self._format_angle(angle)
        return self._add_controlled_gate(control, target, f"Rx({angle_str})")

    def cry(self, control: int, target: int, angle: float) -> 'CircuitDiagram':
        """Add controlled RY gate."""
        angle_str = self._format_angle(angle)
        return self._add_controlled_gate(control, target, f"Ry({angle_str})")

    def crz(self, control: int, target: int, angle: float) -> 'CircuitDiagram':
        """Add controlled RZ gate."""
        angle_str = self._format_angle(angle)
        return self._add_controlled_gate(control, target, f"Rz({angle_str})")

    def _add_controlled_gate(self, control: int, target: int, name: str) -> 'CircuitDiagram':
        """Internal: add controlled gate."""
        t = self._get_time(control, target)
        self.elements.append(CircuitElement(
            type=GateType.CONTROLLED,
            qubit=control,
            target=target,
            time_slot=t,
            name=name
        ))
        self._advance_time(control, target)
        return self

    # Multi-qubit gates
    def swap(self, qubit1: int, qubit2: int) -> 'CircuitDiagram':
        """Add SWAP gate."""
        t = self._get_time(qubit1, qubit2)
        self.elements.append(CircuitElement(
            type=GateType.MULTI,
            qubit=qubit1,
            target=qubit2,
            time_slot=t,
            name="SWAP"
        ))
        self._advance_time(qubit1, qubit2)
        return self

    def ccx(self, ctrl1: int, ctrl2: int, target: int) -> 'CircuitDiagram':
        """Add Toffoli (CCX) gate."""
        t = self._get_time(ctrl1, ctrl2, target)
        self.elements.append(CircuitElement(
            type=GateType.MULTI,
            qubit=ctrl1,
            target=target,
            control2=ctrl2,
            time_slot=t,
            name="CCX"
        ))
        self._advance_time(ctrl1, ctrl2, target)
        return self

    # Toffoli alias
    def toffoli(self, ctrl1: int, ctrl2: int, target: int) -> 'CircuitDiagram':
        """Add Toffoli gate (alias for ccx)."""
        return self.ccx(ctrl1, ctrl2, target)

    # Measurement
    def measure(self, qubit: int, classical_bit: int = None) -> 'CircuitDiagram':
        """Add measurement on single qubit."""
        if classical_bit is None:
            classical_bit = qubit
        t = self.current_time[qubit]
        self.elements.append(CircuitElement(
            type=GateType.MEASUREMENT,
            qubit=qubit,
            time_slot=t,
            name=f"M{classical_bit}",
            classical_bit=classical_bit
        ))
        self.current_time[qubit] = t + 1
        return self

    def measure_all(self) -> 'CircuitDiagram':
        """Add measurement on all qubits."""
        for q in range(self.num_qubits):
            self.measure(q, q)
        return self

    # Barrier
    def barrier(self, qubits: List[int] = None) -> 'CircuitDiagram':
        """Add barrier across specified qubits (or all if None)."""
        if qubits is None:
            qubits = list(range(self.num_qubits))
        t = self._get_time(*qubits)
        self.elements.append(CircuitElement(
            type=GateType.BARRIER,
            qubit=qubits[0],
            time_slot=t,
            barrier_qubits=qubits
        ))
        self._advance_time(*qubits)
        return self

    # Reset
    def reset(self, qubit: int) -> 'CircuitDiagram':
        """Add reset to |0>."""
        t = self.current_time[qubit]
        self.elements.append(CircuitElement(
            type=GateType.RESET,
            qubit=qubit,
            time_slot=t,
            name="|0>"
        ))
        self.current_time[qubit] = t + 1
        return self

    @property
    def depth(self) -> int:
        """Get circuit depth (number of time steps)."""
        return max(self.current_time) if self.current_time else 0

    @staticmethod
    def _format_angle(angle: float) -> str:
        """Format angle as pi fraction when possible."""
        pi_factor = angle / math.pi

        fractions = [
            (0.0, "0"), (0.25, "pi/4"), (0.5, "pi/2"), (0.75, "3pi/4"),
            (1.0, "pi"), (-0.25, "-pi/4"), (-0.5, "-pi/2"), (-0.75, "-3pi/4"),
            (-1.0, "-pi"), (1.0/3.0, "pi/3"), (2.0/3.0, "2pi/3"),
            (1.0/6.0, "pi/6"), (5.0/6.0, "5pi/6"), (2.0, "2pi"),
        ]

        for factor, name in fractions:
            if abs(pi_factor - factor) < 1e-10:
                return name

        return f"{angle:.3f}"

    # Rendering methods
    def to_ascii(self, gate_width: int = 5) -> str:
        """
        Render circuit as ASCII string.

        Parameters:
        -----------
        gate_width : int
            Width of gate boxes in characters

        Returns:
        --------
        str : ASCII representation of circuit
        """
        depth = self.depth
        if depth == 0:
            depth = 1

        label_width = max(len(l) for l in self.qubit_labels) + 2
        wire_spacing = 3
        grid_width = label_width + depth * (gate_width + 1) + 10
        grid_height = self.num_qubits * wire_spacing + 4

        # Create grid
        grid = [[' '] * grid_width for _ in range(grid_height)]

        # Title
        start_row = 0
        if self.title:
            for i, c in enumerate(self.title):
                if i < grid_width:
                    grid[0][i] = c
            start_row = 2

        # Draw wires and labels
        for q in range(self.num_qubits):
            y = start_row + 1 + q * wire_spacing
            label = f"{self.qubit_labels[q]}:"
            for i, c in enumerate(label):
                if i < label_width - 1:
                    grid[y][i] = c
            for x in range(label_width, grid_width - 1):
                grid[y][x] = '-'

        # Draw elements
        for elem in self.elements:
            x = label_width + elem.time_slot * (gate_width + 1) + 1
            y = start_row + 1 + elem.qubit * wire_spacing

            if elem.type == GateType.SINGLE:
                self._draw_ascii_gate(grid, x, y, elem.name, gate_width)

            elif elem.type == GateType.CONTROLLED:
                target_y = start_row + 1 + elem.target * wire_spacing
                x_center = x + gate_width // 2

                # Control dot
                grid[y][x_center] = '*'

                # Vertical line
                y_min, y_max = min(y, target_y), max(y, target_y)
                for vy in range(y_min, y_max + 1):
                    if grid[vy][x_center] == ' ' or grid[vy][x_center] == '-':
                        grid[vy][x_center] = '|'
                grid[y][x_center] = '*'

                # Target
                if elem.name == "X":
                    grid[target_y][x_center] = 'X'
                elif elem.name == "Z":
                    grid[target_y][x_center] = '*'
                else:
                    self._draw_ascii_gate(grid, x, target_y, elem.name, gate_width)

            elif elem.type == GateType.MULTI:
                target_y = start_row + 1 + elem.target * wire_spacing
                x_center = x + gate_width // 2

                if elem.name == "SWAP":
                    grid[y][x_center] = 'X'
                    grid[target_y][x_center] = 'X'
                    y_min, y_max = min(y, target_y), max(y, target_y)
                    for vy in range(y_min, y_max + 1):
                        if grid[vy][x_center] == ' ' or grid[vy][x_center] == '-':
                            grid[vy][x_center] = '|'

                elif elem.name == "CCX":
                    ctrl2_y = start_row + 1 + elem.control2 * wire_spacing
                    grid[y][x_center] = '*'
                    grid[ctrl2_y][x_center] = '*'
                    grid[target_y][x_center] = 'X'
                    y_min = min(y, ctrl2_y, target_y)
                    y_max = max(y, ctrl2_y, target_y)
                    for vy in range(y_min, y_max + 1):
                        if grid[vy][x_center] == ' ' or grid[vy][x_center] == '-':
                            grid[vy][x_center] = '|'

            elif elem.type == GateType.MEASUREMENT:
                self._draw_ascii_gate(grid, x, y, "M", 3)

            elif elem.type == GateType.BARRIER:
                for bq in elem.barrier_qubits:
                    by = start_row + 1 + bq * wire_spacing
                    x_center = x + gate_width // 2
                    grid[by - 1][x_center] = ':'
                    grid[by][x_center] = ':'
                    grid[by + 1][x_center] = ':'

            elif elem.type == GateType.RESET:
                self._draw_ascii_gate(grid, x, y, "|0>", gate_width)

        # Convert grid to string
        lines = []
        for row in grid:
            line = ''.join(row).rstrip()
            if line:
                lines.append(line)

        return '\n'.join(lines)

    def _draw_ascii_gate(self, grid, x, y, name, width):
        """Draw gate box in ASCII grid."""
        if y < 1 or y >= len(grid) - 1:
            return

        # Top border
        grid[y - 1][x] = '+'
        for i in range(1, width - 1):
            if x + i < len(grid[0]):
                grid[y - 1][x + i] = '-'
        if x + width - 1 < len(grid[0]):
            grid[y - 1][x + width - 1] = '+'

        # Middle with name
        grid[y][x] = '|'
        name_start = x + (width - len(name)) // 2
        for i, c in enumerate(name):
            if name_start + i < len(grid[0]) - 1:
                grid[y][name_start + i] = c
        if x + width - 1 < len(grid[0]):
            grid[y][x + width - 1] = '|'

        # Bottom border
        grid[y + 1][x] = '+'
        for i in range(1, width - 1):
            if x + i < len(grid[0]):
                grid[y + 1][x + i] = '-'
        if x + width - 1 < len(grid[0]):
            grid[y + 1][x + width - 1] = '+'

    def render(self, figsize: Tuple[float, float] = None,
               style: str = 'publication') -> 'plt.Figure':
        """
        Render circuit to matplotlib figure.

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        style : str
            Style preset: 'publication', 'colorful', 'minimal'

        Returns:
        --------
        matplotlib.Figure : Rendered circuit
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for render(). Install with: pip install matplotlib")

        depth = max(self.depth, 1)

        if figsize is None:
            figsize = (3 + depth * 0.8, 1 + self.num_qubits * 0.6)

        styles = {
            'publication': {
                'wire_color': 'black',
                'gate_fill': 'white',
                'gate_edge': 'black',
                'font': 'serif',
                'fontsize': 12
            },
            'colorful': {
                'wire_color': '#444444',
                'gate_fill': '#e3f2fd',
                'gate_edge': '#1976d2',
                'font': 'sans-serif',
                'fontsize': 11
            },
            'minimal': {
                'wire_color': '#666666',
                'gate_fill': '#f5f5f5',
                'gate_edge': '#333333',
                'font': 'monospace',
                'fontsize': 10
            }
        }
        s = styles.get(style, styles['publication'])

        fig, ax = plt.subplots(figsize=figsize)

        gate_width = 0.6
        wire_spacing = 1.0

        # Draw wires
        for q in range(self.num_qubits):
            y = (self.num_qubits - 1 - q) * wire_spacing
            ax.plot([-0.5, depth + 0.5], [y, y], color=s['wire_color'], linewidth=1)
            ax.text(-0.8, y, self.qubit_labels[q], fontsize=s['fontsize'],
                    fontfamily=s['font'], ha='right', va='center')

        # Draw elements
        for elem in self.elements:
            t = elem.time_slot
            y = (self.num_qubits - 1 - elem.qubit) * wire_spacing

            if elem.type == GateType.SINGLE:
                rect = FancyBboxPatch((t - gate_width/2, y - 0.3), gate_width, 0.6,
                                      boxstyle="round,pad=0.05",
                                      facecolor=s['gate_fill'],
                                      edgecolor=s['gate_edge'], linewidth=1.5)
                ax.add_patch(rect)
                ax.text(t, y, elem.name, fontsize=s['fontsize'], fontfamily=s['font'],
                        ha='center', va='center')

            elif elem.type == GateType.CONTROLLED:
                tgt_y = (self.num_qubits - 1 - elem.target) * wire_spacing

                # Control dot
                ax.add_patch(Circle((t, y), 0.08, facecolor='black'))

                # Vertical line
                ax.plot([t, t], [y, tgt_y], color='black', linewidth=1)

                # Target
                if elem.name == "X":
                    ax.add_patch(Circle((t, tgt_y), 0.2, facecolor='white',
                                        edgecolor='black', linewidth=1.5))
                    ax.plot([t - 0.2, t + 0.2], [tgt_y, tgt_y], color='black', linewidth=1)
                    ax.plot([t, t], [tgt_y - 0.2, tgt_y + 0.2], color='black', linewidth=1)
                elif elem.name == "Z":
                    ax.add_patch(Circle((t, tgt_y), 0.08, facecolor='black'))
                else:
                    rect = FancyBboxPatch((t - gate_width/2, tgt_y - 0.3), gate_width, 0.6,
                                          boxstyle="round,pad=0.05",
                                          facecolor=s['gate_fill'],
                                          edgecolor=s['gate_edge'], linewidth=1.5)
                    ax.add_patch(rect)
                    ax.text(t, tgt_y, elem.name, fontsize=s['fontsize'], fontfamily=s['font'],
                            ha='center', va='center')

            elif elem.type == GateType.MULTI:
                tgt_y = (self.num_qubits - 1 - elem.target) * wire_spacing

                if elem.name == "SWAP":
                    ax.plot([t, t], [y, tgt_y], color='black', linewidth=1)
                    for qy in [y, tgt_y]:
                        ax.plot([t - 0.15, t + 0.15], [qy - 0.15, qy + 0.15], 'k-', linewidth=1.5)
                        ax.plot([t + 0.15, t - 0.15], [qy - 0.15, qy + 0.15], 'k-', linewidth=1.5)

                elif elem.name == "CCX":
                    ctrl2_y = (self.num_qubits - 1 - elem.control2) * wire_spacing
                    y_min = min(y, ctrl2_y, tgt_y)
                    y_max = max(y, ctrl2_y, tgt_y)
                    ax.plot([t, t], [y_min, y_max], color='black', linewidth=1)
                    ax.add_patch(Circle((t, y), 0.08, facecolor='black'))
                    ax.add_patch(Circle((t, ctrl2_y), 0.08, facecolor='black'))
                    ax.add_patch(Circle((t, tgt_y), 0.2, facecolor='white',
                                        edgecolor='black', linewidth=1.5))
                    ax.plot([t - 0.2, t + 0.2], [tgt_y, tgt_y], color='black', linewidth=1)
                    ax.plot([t, t], [tgt_y - 0.2, tgt_y + 0.2], color='black', linewidth=1)

            elif elem.type == GateType.MEASUREMENT:
                rect = FancyBboxPatch((t - gate_width/2, y - 0.35), gate_width, 0.7,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#f5f5f5', edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                # Meter arc and arrow
                theta = np.linspace(np.pi, 0, 50)
                arc_x = t + 0.15 * np.cos(theta)
                arc_y = y - 0.1 + 0.15 * np.sin(theta)
                ax.plot(arc_x, arc_y, color='black', linewidth=1)
                ax.arrow(t, y - 0.1, 0.1, 0.15, head_width=0.05, head_length=0.03,
                         fc='black', ec='black')

            elif elem.type == GateType.BARRIER:
                for bq in elem.barrier_qubits:
                    by = (self.num_qubits - 1 - bq) * wire_spacing
                    ax.plot([t - 0.1, t - 0.1], [by - 0.4, by + 0.4],
                            color='gray', linewidth=2, linestyle='--')

            elif elem.type == GateType.RESET:
                rect = FancyBboxPatch((t - gate_width/2, y - 0.3), gate_width, 0.6,
                                      boxstyle="round,pad=0.05",
                                      facecolor=s['gate_fill'],
                                      edgecolor=s['gate_edge'], linewidth=1.5)
                ax.add_patch(rect)
                ax.text(t, y, "|0>", fontsize=s['fontsize'], fontfamily=s['font'],
                        ha='center', va='center')

        ax.set_xlim(-1.5, depth + 1)
        ax.set_ylim(-0.8, (self.num_qubits - 1) * wire_spacing + 0.8)
        ax.set_aspect('equal')
        ax.axis('off')

        if self.title:
            ax.set_title(self.title, fontsize=s['fontsize'] + 2, fontfamily=s['font'])

        return fig

    def to_latex(self) -> str:
        """
        Generate LaTeX/quantikz representation.

        Returns:
        --------
        str : LaTeX code using quantikz package
        """
        lines = ["% Requires: \\usepackage{quantikz}", "\\begin{quantikz}"]

        depth = self.depth

        for q in range(self.num_qubits):
            row = [f"\\lstick{{${self.qubit_labels[q]}$}}"]

            for t in range(depth + 1):
                elem = self._get_element_at(q, t)

                if elem is None:
                    row.append("\\qw")
                elif elem.type == GateType.BARRIER:
                    row.append("\\qw")
                elif elem.qubit == q:
                    if elem.type == GateType.SINGLE:
                        row.append(f"\\gate{{{elem.name}}}")
                    elif elem.type == GateType.CONTROLLED:
                        delta = elem.target - elem.qubit
                        row.append(f"\\ctrl{{{delta}}}")
                    elif elem.type == GateType.MULTI:
                        if elem.name == "SWAP":
                            delta = elem.target - elem.qubit
                            row.append(f"\\swap{{{delta}}}")
                        elif elem.name == "CCX":
                            delta = elem.target - elem.qubit
                            row.append(f"\\ctrl{{{delta}}}")
                        else:
                            row.append("\\qw")
                    elif elem.type == GateType.MEASUREMENT:
                        row.append("\\meter{}")
                    elif elem.type == GateType.RESET:
                        row.append("\\gate{|0\\rangle}")
                    else:
                        row.append("\\qw")
                elif elem.target == q:
                    if elem.type == GateType.CONTROLLED:
                        if elem.name == "X":
                            row.append("\\targ{}")
                        elif elem.name == "Z":
                            row.append("\\control{}")
                        else:
                            row.append(f"\\gate{{{elem.name}}}")
                    elif elem.type == GateType.MULTI:
                        if elem.name == "SWAP":
                            row.append("\\targX{}")
                        elif elem.name == "CCX":
                            row.append("\\targ{}")
                        else:
                            row.append("\\qw")
                    else:
                        row.append("\\qw")
                elif elem.control2 == q:
                    if elem.name == "CCX":
                        delta = elem.target - q
                        row.append(f"\\ctrl{{{delta}}}")
                    else:
                        row.append("\\qw")
                else:
                    row.append("\\qw")

            sep = " \\\\" if q < self.num_qubits - 1 else ""
            lines.append(" & ".join(row) + sep)

        lines.append("\\end{quantikz}")
        return "\n".join(lines)

    def _get_element_at(self, qubit: int, time: int) -> Optional[CircuitElement]:
        """Get element affecting qubit at time."""
        for elem in self.elements:
            if elem.time_slot == time:
                if elem.qubit == qubit:
                    return elem
                if elem.target == qubit:
                    return elem
                if elem.control2 == qubit:
                    return elem
                if elem.type == GateType.BARRIER and qubit in elem.barrier_qubits:
                    return elem
        return None

    def save(self, filename: str, format: str = None, dpi: int = 300):
        """
        Save circuit to file.

        Parameters:
        -----------
        filename : str
            Output filename
        format : str, optional
            File format ('svg', 'png', 'pdf'). Auto-detected from extension.
        dpi : int
            Resolution for raster formats
        """
        if format is None:
            format = filename.split('.')[-1].lower()

        if format == 'tex' or format == 'latex':
            with open(filename, 'w') as f:
                f.write(self.to_latex())
        elif format == 'txt' or format == 'ascii':
            with open(filename, 'w') as f:
                f.write(self.to_ascii())
        else:
            if not HAS_MATPLOTLIB:
                raise ImportError("matplotlib is required for graphical export")
            fig = self.render()
            fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

    def __repr__(self) -> str:
        return f"CircuitDiagram({self.num_qubits} qubits, depth={self.depth})"


def draw_circuit(circuit: Union[CircuitDiagram, 'QuantumState'] = None,
                 **kwargs) -> 'plt.Figure':
    """
    Quick function to draw a circuit.

    Parameters:
    -----------
    circuit : CircuitDiagram or QuantumState
        Circuit to draw (or QuantumState with gate history)
    **kwargs : dict
        Additional arguments passed to render()

    Returns:
    --------
    matplotlib.Figure : Rendered circuit
    """
    if isinstance(circuit, CircuitDiagram):
        return circuit.render(**kwargs)
    else:
        raise TypeError("Expected CircuitDiagram or QuantumState with gate history")
