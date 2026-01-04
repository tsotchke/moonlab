//! Entanglement entropy visualization.
//!
//! Displays a heatmap of pairwise entanglement between qubits.

use moonlab::QuantumState;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Widget},
};

/// Entanglement heatmap widget.
pub struct EntanglementHeatmap<'a> {
    state: Option<&'a QuantumState>,
    title: &'a str,
}

impl<'a> EntanglementHeatmap<'a> {
    pub fn new(state: Option<&'a QuantumState>) -> Self {
        Self {
            state,
            title: "Entanglement",
        }
    }

    pub fn title(mut self, title: &'a str) -> Self {
        self.title = title;
        self
    }

    /// Get color for entanglement value (0 to 1).
    fn entropy_color(entropy: f64) -> Color {
        // Gradient from black (no entanglement) to bright (max entanglement)
        if entropy < 0.05 {
            Color::DarkGray
        } else if entropy < 0.2 {
            Color::Blue
        } else if entropy < 0.4 {
            Color::Cyan
        } else if entropy < 0.6 {
            Color::Green
        } else if entropy < 0.8 {
            Color::Yellow
        } else {
            Color::Red
        }
    }

    /// Get block character for intensity.
    fn intensity_char(value: f64) -> char {
        if value < 0.1 {
            ' '
        } else if value < 0.25 {
            '░'
        } else if value < 0.5 {
            '▒'
        } else if value < 0.75 {
            '▓'
        } else {
            '█'
        }
    }
}

impl<'a> Widget for EntanglementHeatmap<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(self.title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta));

        let inner = block.inner(area);
        block.render(area, buf);

        let Some(state) = self.state else {
            let msg = "No state";
            buf.set_string(inner.x + 1, inner.y, msg, Style::default().fg(Color::DarkGray));
            return;
        };

        let n = state.num_qubits();
        if n < 2 {
            let msg = "Need ≥2 qubits";
            buf.set_string(inner.x + 1, inner.y, msg, Style::default().fg(Color::DarkGray));
            return;
        }

        let cell_width = 3;
        let max_qubits = ((inner.width as usize - 3) / cell_width).min(n).min(8);

        // Column headers
        for j in 0..max_qubits {
            let x = inner.x + 3 + (j * cell_width) as u16;
            let label = format!("Q{}", j);
            buf.set_string(x, inner.y, &label, Style::default().fg(Color::Cyan));
        }

        // Rows
        for i in 0..max_qubits.min(inner.height as usize - 1) {
            let y = inner.y + 1 + i as u16;
            if y >= inner.y + inner.height {
                break;
            }

            // Row label
            let label = format!("Q{}", i);
            buf.set_string(inner.x, y, &label, Style::default().fg(Color::Cyan));

            // Cells
            for j in 0..max_qubits {
                let x = inner.x + 3 + (j * cell_width) as u16;

                if i == j {
                    // Diagonal - purity of single qubit
                    let z = state.expectation_z(i).unwrap_or(0.0);
                    let x_exp = state.expectation_x(i).unwrap_or(0.0);
                    let y_exp = state.expectation_y(i).unwrap_or(0.0);
                    let purity = z * z + x_exp * x_exp + y_exp * y_exp;

                    let color = if purity > 0.9 {
                        Color::Green
                    } else if purity > 0.5 {
                        Color::Yellow
                    } else {
                        Color::Red
                    };
                    buf.set_string(x, y, " ● ", Style::default().fg(color));
                } else {
                    // Off-diagonal - ZZ correlation as proxy for entanglement
                    let zz = state.correlation_zz(i, j).unwrap_or(0.0).abs();
                    let zi = state.expectation_z(i).unwrap_or(0.0);
                    let zj = state.expectation_z(j).unwrap_or(0.0);

                    // Mutual information proxy: |ZZ - Zi*Zj|
                    let correlation = (zz - zi * zj).abs();

                    let char_val = Self::intensity_char(correlation);
                    let color = Self::entropy_color(correlation);

                    let cell = format!(" {} ", char_val);
                    buf.set_string(x, y, &cell, Style::default().fg(color));
                }
            }
        }

        // Legend
        if inner.height > max_qubits as u16 + 3 {
            let y = inner.y + inner.height - 1;
            let legend = "░▒▓█ = Correlation";
            buf.set_string(inner.x + 1, y, legend, Style::default().fg(Color::DarkGray));
        }
    }
}

/// Metrics panel showing various quantum state properties.
pub struct MetricsPanel<'a> {
    state: Option<&'a QuantumState>,
    elapsed_ms: f64,
    gate_count: usize,
}

#[allow(dead_code)]
impl<'a> MetricsPanel<'a> {
    pub fn new(state: Option<&'a QuantumState>) -> Self {
        Self {
            state,
            elapsed_ms: 0.0,
            gate_count: 0,
        }
    }

    pub fn elapsed(mut self, ms: f64) -> Self {
        self.elapsed_ms = ms;
        self
    }

    pub fn gate_count(mut self, count: usize) -> Self {
        self.gate_count = count;
        self
    }
}

impl<'a> Widget for MetricsPanel<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Metrics")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue));

        let inner = block.inner(area);
        block.render(area, buf);

        let metrics: Vec<(String, String, Color)> = if let Some(state) = self.state {
            let purity = state.purity();
            let entropy = state.entropy();

            vec![
                ("Qubits".to_string(), format!("{}", state.num_qubits()), Color::Cyan),
                ("Dim".to_string(), format!("2^{} = {}", state.num_qubits(), state.state_dim()), Color::White),
                ("Purity".to_string(), format!("{:.4}", purity), if purity > 0.99 { Color::Green } else { Color::Yellow }),
                ("Entropy".to_string(), format!("{:.3} bits", entropy), Color::White),
                ("Gates".to_string(), format!("{}", self.gate_count), Color::Magenta),
                ("Time".to_string(), format!("{:.2}ms", self.elapsed_ms), Color::DarkGray),
            ]
        } else {
            vec![
                ("Status".to_string(), "No state".to_string(), Color::DarkGray),
            ]
        };

        for (i, (label, value, color)) in metrics.iter().enumerate() {
            if i as u16 >= inner.height {
                break;
            }
            let y = inner.y + i as u16;
            let text = format!("{:>8}: {}", label, value);
            buf.set_string(inner.x + 1, y, &text, Style::default().fg(*color));
        }
    }
}
