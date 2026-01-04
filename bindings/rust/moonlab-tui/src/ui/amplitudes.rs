//! Amplitude bar visualization widget.
//!
//! Displays quantum state amplitudes as horizontal bars with
//! color encoding for probability and phase.

use moonlab::QuantumState;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Widget},
};

/// Widget for displaying quantum state amplitudes as bars.
#[allow(dead_code)]
pub struct AmplitudeBars<'a> {
    state: Option<&'a QuantumState>,
    title: &'a str,
    max_display: usize,
    show_phases: bool,
}

#[allow(dead_code)]
impl<'a> AmplitudeBars<'a> {
    pub fn new(state: Option<&'a QuantumState>) -> Self {
        Self {
            state,
            title: "State Amplitudes",
            max_display: 16,
            show_phases: true,
        }
    }

    pub fn title(mut self, title: &'a str) -> Self {
        self.title = title;
        self
    }

    pub fn max_display(mut self, max: usize) -> Self {
        self.max_display = max;
        self
    }

    /// Get color for probability value (gradient from blue to red).
    fn prob_color(prob: f64) -> Color {
        if prob < 0.001 {
            Color::DarkGray
        } else if prob < 0.1 {
            Color::Blue
        } else if prob < 0.3 {
            Color::Cyan
        } else if prob < 0.5 {
            Color::Yellow
        } else if prob < 0.7 {
            Color::Rgb(255, 165, 0) // Orange
        } else {
            Color::Red
        }
    }

    /// Get color for phase (HSV-like mapping).
    fn phase_color(phase: f64) -> Color {
        // Map phase from [-π, π] to hue [0, 360]
        let hue = ((phase + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * 360.0) as u16;

        // Simple HSV to RGB (saturation=1, value=1)
        let h = hue % 360;
        let x = 255 - ((h % 60) as i32 * 255 / 60).abs() as u8;

        match h / 60 {
            0 => Color::Rgb(255, x, 0),
            1 => Color::Rgb(x, 255, 0),
            2 => Color::Rgb(0, 255, x),
            3 => Color::Rgb(0, x, 255),
            4 => Color::Rgb(x, 0, 255),
            _ => Color::Rgb(255, 0, x),
        }
    }

    /// Format basis state label.
    fn format_basis_state(index: usize, num_qubits: usize) -> String {
        let binary = format!("{:0width$b}", index, width = num_qubits);
        format!("|{}⟩", binary)
    }

    /// Create a bar string with given width and fill percentage.
    fn bar_string(width: usize, fill: f64) -> String {
        let filled = ((width as f64) * fill.clamp(0.0, 1.0)) as usize;
        let full_blocks = filled;
        let partial = ((fill * width as f64) - full_blocks as f64) * 8.0;

        let mut bar = String::new();

        // Full blocks
        for _ in 0..full_blocks {
            bar.push('█');
        }

        // Partial block
        if full_blocks < width {
            let partial_char = match partial as usize {
                0 => ' ',
                1 => '▏',
                2 => '▎',
                3 => '▍',
                4 => '▌',
                5 => '▋',
                6 => '▊',
                7 => '▉',
                _ => '█',
            };
            bar.push(partial_char);

            // Empty space
            for _ in (full_blocks + 1)..width {
                bar.push(' ');
            }
        }

        bar
    }
}

impl<'a> Widget for AmplitudeBars<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(self.title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        let Some(state) = self.state else {
            let msg = "No quantum state";
            if inner.width as usize > msg.len() {
                buf.set_string(inner.x + 1, inner.y + inner.height / 2, msg, Style::default().fg(Color::DarkGray));
            }
            return;
        };

        let probs = state.probabilities();
        let amps = state.amplitudes();
        let num_qubits = state.num_qubits();
        let dim = probs.len();

        // Calculate display range
        let display_count = (inner.height as usize).min(dim).min(self.max_display);

        // Sort by probability to show most significant states
        let mut indices: Vec<usize> = (0..dim).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate max probability for scaling
        let max_prob = probs.iter().cloned().fold(0.0f64, f64::max).max(0.001);

        // Label width
        let label_width = num_qubits + 3; // |xxx⟩
        let bar_width = (inner.width as usize).saturating_sub(label_width + 10);

        for (row, &idx) in indices.iter().take(display_count).enumerate() {
            if row >= inner.height as usize {
                break;
            }

            let y = inner.y + row as u16;
            let prob = probs[idx];
            let amp = amps[idx];
            let _phase = amp.arg(); // Reserved for future phase visualization

            // Basis state label
            let label = Self::format_basis_state(idx, num_qubits);
            buf.set_string(inner.x, y, &label, Style::default().fg(Color::White));

            // Bar
            let bar_fill = prob / max_prob;
            let bar = Self::bar_string(bar_width, bar_fill);
            let bar_color = Self::prob_color(prob);
            buf.set_string(inner.x + label_width as u16, y, &bar, Style::default().fg(bar_color));

            // Probability value
            let prob_str = format!("{:5.1}%", prob * 100.0);
            let prob_x = inner.x + label_width as u16 + bar_width as u16 + 1;
            if prob_x + 6 <= inner.x + inner.width {
                buf.set_string(prob_x, y, &prob_str, Style::default().fg(Color::Yellow));
            }
        }

        // Show indicator if there are more states
        if dim > display_count {
            let more = format!("... {} more", dim - display_count);
            let y = inner.y + inner.height - 1;
            buf.set_string(inner.x, y, &more, Style::default().fg(Color::DarkGray));
        }
    }
}

/// Compact state view for smaller areas.
#[allow(dead_code)]
pub struct CompactStateView<'a> {
    state: Option<&'a QuantumState>,
}

#[allow(dead_code)]
impl<'a> CompactStateView<'a> {
    pub fn new(state: Option<&'a QuantumState>) -> Self {
        Self { state }
    }
}

impl<'a> Widget for CompactStateView<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("State |ψ⟩")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta));

        let inner = block.inner(area);
        block.render(area, buf);

        let Some(state) = self.state else {
            return;
        };

        let probs = state.probabilities();
        let num_qubits = state.num_qubits();

        // Find non-zero states and format as ket notation
        let mut terms: Vec<(usize, f64)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.01)
            .map(|(i, &p)| (i, p))
            .collect();

        terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut y = inner.y;
        for (idx, prob) in terms.iter().take(inner.height as usize) {
            let binary = format!("{:0width$b}", idx, width = num_qubits);
            let coeff = prob.sqrt();
            let term = format!("{:.2}|{}⟩", coeff, binary);
            buf.set_string(inner.x + 1, y, &term, Style::default().fg(Color::White));
            y += 1;
        }
    }
}
