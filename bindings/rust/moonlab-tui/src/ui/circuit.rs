//! Quantum circuit diagram visualization.
//!
//! ASCII art representation of quantum circuits with gate symbols.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Widget},
};

/// A quantum gate in the circuit.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Gate {
    /// Hadamard gate
    H(usize),
    /// Pauli-X gate
    X(usize),
    /// Pauli-Y gate
    Y(usize),
    /// Pauli-Z gate
    Z(usize),
    /// S gate
    S(usize),
    /// T gate
    T(usize),
    /// Rotation X
    Rx(usize, f64),
    /// Rotation Y
    Ry(usize, f64),
    /// Rotation Z
    Rz(usize, f64),
    /// CNOT gate
    CNOT(usize, usize),
    /// CZ gate
    CZ(usize, usize),
    /// Toffoli gate
    Toffoli(usize, usize, usize),
    /// Measurement
    Measure(usize),
}

impl Gate {
    fn symbol(&self) -> &'static str {
        match self {
            Gate::H(_) => "H",
            Gate::X(_) => "X",
            Gate::Y(_) => "Y",
            Gate::Z(_) => "Z",
            Gate::S(_) => "S",
            Gate::T(_) => "T",
            Gate::Rx(_, _) => "Rx",
            Gate::Ry(_, _) => "Ry",
            Gate::Rz(_, _) => "Rz",
            Gate::CNOT(_, _) => "⊕",
            Gate::CZ(_, _) => "Z",
            Gate::Toffoli(_, _, _) => "⊕",
            Gate::Measure(_) => "M",
        }
    }

    fn qubits(&self) -> Vec<usize> {
        match self {
            Gate::H(q) | Gate::X(q) | Gate::Y(q) | Gate::Z(q)
            | Gate::S(q) | Gate::T(q)
            | Gate::Rx(q, _) | Gate::Ry(q, _) | Gate::Rz(q, _)
            | Gate::Measure(q) => vec![*q],
            Gate::CNOT(c, t) | Gate::CZ(c, t) => vec![*c, *t],
            Gate::Toffoli(c1, c2, t) => vec![*c1, *c2, *t],
        }
    }

    fn is_controlled(&self) -> bool {
        matches!(self, Gate::CNOT(_, _) | Gate::CZ(_, _) | Gate::Toffoli(_, _, _))
    }

    fn control_qubits(&self) -> Vec<usize> {
        match self {
            Gate::CNOT(c, _) | Gate::CZ(c, _) => vec![*c],
            Gate::Toffoli(c1, c2, _) => vec![*c1, *c2],
            _ => vec![],
        }
    }

    fn target_qubit(&self) -> Option<usize> {
        match self {
            Gate::CNOT(_, t) | Gate::CZ(_, t) | Gate::Toffoli(_, _, t) => Some(*t),
            _ => None,
        }
    }
}

/// Circuit diagram widget.
pub struct CircuitDiagram {
    num_qubits: usize,
    gates: Vec<Gate>,
    current_step: usize,
    title: String,
}

#[allow(dead_code)]
impl CircuitDiagram {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            current_step: 0,
            title: String::from("Circuit"),
        }
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn gates(mut self, gates: Vec<Gate>) -> Self {
        self.gates = gates;
        self
    }

    pub fn current_step(mut self, step: usize) -> Self {
        self.current_step = step;
        self
    }

    /// Add gates for common algorithms.
    pub fn bell_circuit() -> Self {
        Self::new(2)
            .gates(vec![
                Gate::H(0),
                Gate::CNOT(0, 1),
            ])
            .title("Bell State Circuit")
    }

    pub fn ghz_circuit(n: usize) -> Self {
        let mut gates = vec![Gate::H(0)];
        for i in 1..n {
            gates.push(Gate::CNOT(0, i));
        }
        Self::new(n).gates(gates).title("GHZ State Circuit")
    }
}

impl Widget for CircuitDiagram {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(self.title.as_str())
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 2 || inner.width < 10 {
            return;
        }

        let rows_per_qubit = 2;
        let max_qubits = (inner.height as usize / rows_per_qubit).min(self.num_qubits);
        let gate_width = 5;
        let max_gates = (inner.width as usize - 5) / gate_width;

        // Draw qubit lines
        for q in 0..max_qubits {
            let y = inner.y + (q * rows_per_qubit) as u16;

            // Qubit label
            let label = format!("q{}", q);
            buf.set_string(inner.x, y, &label, Style::default().fg(Color::Cyan));

            // Wire
            let wire_start = inner.x + 3;
            let wire_end = inner.x + inner.width - 1;
            for x in wire_start..wire_end {
                buf.set_string(x, y, "─", Style::default().fg(Color::DarkGray));
            }
        }

        // Draw gates
        let mut gate_positions: Vec<usize> = vec![4; self.num_qubits]; // Next available x position for each qubit

        for (gate_idx, gate) in self.gates.iter().enumerate() {
            if gate_idx >= max_gates {
                break;
            }

            let qubits = gate.qubits();
            let min_q = *qubits.iter().min().unwrap_or(&0);
            let max_q = *qubits.iter().max().unwrap_or(&0);

            if max_q >= max_qubits {
                continue;
            }

            // Find the next available column for all involved qubits
            let mut col = 0;
            for &q in &qubits {
                col = col.max(gate_positions[q]);
            }

            let x = inner.x + col as u16;

            if x + 3 > inner.x + inner.width {
                break;
            }

            // Determine if this gate is current
            let is_current = gate_idx == self.current_step.saturating_sub(1);
            let gate_color = if is_current {
                Color::Yellow
            } else if gate_idx < self.current_step {
                Color::Green
            } else {
                Color::White
            };

            // Draw vertical line for multi-qubit gates
            if gate.is_controlled() {
                for q in min_q..=max_q {
                    let y = inner.y + (q * rows_per_qubit) as u16;
                    buf.set_string(x + 1, y, "│", Style::default().fg(gate_color));
                }

                // Draw control dots
                for &ctrl in &gate.control_qubits() {
                    if ctrl < max_qubits {
                        let y = inner.y + (ctrl * rows_per_qubit) as u16;
                        buf.set_string(x + 1, y, "●", Style::default().fg(gate_color));
                    }
                }
            }

            // Draw gate symbol
            for &q in &qubits {
                if q >= max_qubits {
                    continue;
                }

                let y = inner.y + (q * rows_per_qubit) as u16;

                if gate.is_controlled() {
                    if Some(q) == gate.target_qubit() {
                        // Target gate
                        buf.set_string(x, y, "[", Style::default().fg(gate_color));
                        buf.set_string(x + 1, y, gate.symbol(), Style::default().fg(gate_color));
                        buf.set_string(x + 2, y, "]", Style::default().fg(gate_color));
                    }
                    // Controls already drawn as dots
                } else {
                    // Single-qubit gate
                    buf.set_string(x, y, "[", Style::default().fg(gate_color));
                    buf.set_string(x + 1, y, gate.symbol(), Style::default().fg(gate_color));
                    buf.set_string(x + 2, y, "]", Style::default().fg(gate_color));
                }
            }

            // Update gate positions
            for &q in &qubits {
                if q < gate_positions.len() {
                    gate_positions[q] = col + gate_width;
                }
            }
        }

        // Show progress if we have steps
        if self.current_step > 0 && !self.gates.is_empty() {
            let progress = format!("Step {}/{}", self.current_step, self.gates.len());
            let x = inner.x + inner.width - progress.len() as u16 - 1;
            let y = inner.y + inner.height - 1;
            buf.set_string(x, y, &progress, Style::default().fg(Color::Magenta));
        }
    }
}

/// Simple gate legend widget.
pub struct GateLegend;

impl Widget for GateLegend {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Gate Legend")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        let legends = [
            ("H", "Hadamard", Color::White),
            ("X", "Pauli-X (NOT)", Color::White),
            ("●─⊕", "CNOT", Color::White),
            ("Rx", "Rotation-X", Color::White),
            ("M", "Measure", Color::Yellow),
        ];

        for (i, (sym, name, color)) in legends.iter().enumerate() {
            if i as u16 >= inner.height {
                break;
            }
            let y = inner.y + i as u16; 
            let text = format!("{:4} {}", sym, name);
            buf.set_string(inner.x + 1, y, &text, Style::default().fg(*color));
        }
    }
}
