//! Bloch sphere visualization widget.
//!
//! Displays a single-qubit state on the Bloch sphere using ASCII art.

use moonlab::QuantumState;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Widget},
};

/// ASCII Bloch sphere visualization.
pub struct BlochSphere<'a> {
    state: Option<&'a QuantumState>,
    qubit: usize,
    title: &'a str,
    frame: u64,
}

impl<'a> BlochSphere<'a> {
    pub fn new(state: Option<&'a QuantumState>, qubit: usize) -> Self {
        Self {
            state,
            qubit,
            title: "Bloch Sphere",
            frame: 0,
        }
    }

    pub fn title(mut self, title: &'a str) -> Self {
        self.title = title;
        self
    }

    pub fn frame(mut self, frame: u64) -> Self {
        self.frame = frame;
        self
    }

    /// Compute Bloch sphere coordinates (θ, φ, r) from single-qubit state.
    /// Returns (theta, phi, r, label) where theta is polar angle, phi is azimuthal, r is purity.
    fn compute_bloch_coords(&self) -> Option<(f64, f64, f64, String)> {
        let state = self.state?;

        if state.num_qubits() < self.qubit + 1 {
            return None;
        }

        // For a multi-qubit state, we trace out other qubits
        // For simplicity, we'll use expectation values
        let x = state.expectation_x(self.qubit).unwrap_or(0.0);
        let y = state.expectation_y(self.qubit).unwrap_or(0.0);
        let z = state.expectation_z(self.qubit).unwrap_or(1.0);

        // Convert to spherical coordinates
        // r is the Bloch vector magnitude (1.0 = pure state, 0.0 = maximally mixed)
        let r = (x * x + y * y + z * z).sqrt();

        if r < 0.001 {
            // Maximally mixed state - point at center
            return Some((std::f64::consts::FRAC_PI_2, 0.0, 0.0, "mixed".to_string()));
        }

        let theta = (z / r).acos(); // polar angle from +z
        let phi = y.atan2(x);        // azimuthal angle

        // Create label based on position
        let label = if r > 0.95 && z > 0.9 {
            "|0⟩".to_string()
        } else if r > 0.95 && z < -0.9 {
            "|1⟩".to_string()
        } else if r > 0.95 && x.abs() > 0.9 {
            if x > 0.0 { "|+⟩" } else { "|-⟩" }.to_string()
        } else if r > 0.95 && y.abs() > 0.9 {
            if y > 0.0 { "|+i⟩" } else { "|-i⟩" }.to_string()
        } else if r < 0.3 {
            "mixed".to_string()
        } else {
            format!("r={:.1}", r)
        };

        Some((theta, phi, r, label))
    }
}

impl<'a> Widget for BlochSphere<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(self.title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.width < 15 || inner.height < 8 {
            return; // Too small to render
        }

        // Sphere center
        let cx = inner.x + inner.width / 2;
        let cy = inner.y + inner.height / 2;
        let radius_x = (inner.width as f64 / 2.5) as i16;
        let radius_y = (inner.height as f64 / 2.5) as i16;

        // Alternative sphere outline (ellipse) - reserved for future use
        let _sphere_chars = [
            // Top arc
            (-3, -2, "╭───╮"),
            // Middle
            (-4, -1, "│"),
            (4, -1, "│"),
            (-4, 0, "│"),
            (4, 0, "│"),
            (-4, 1, "│"),
            (4, 1, "│"),
            // Bottom arc
            (-3, 2, "╰───╯"),
        ];

        // Draw a simple ASCII Bloch sphere
        let sphere_art = [
            "      z      ",
            "      │      ",
            "    ╭─┼─╮    ",
            "   ╱  │  ╲   ",
            "  │───●───│──y",
            "   ╲  │  ╱   ",
            "    ╰─┼─╯    ",
            "      │      ",
            "     ╱       ",
            "    x        ",
        ];

        let start_y = cy.saturating_sub(5);
        let start_x = cx.saturating_sub(6);

        // Draw sphere structure
        for (i, line) in sphere_art.iter().enumerate() {
            let y = start_y + i as u16;
            if y < inner.y + inner.height {
                for (j, ch) in line.chars().enumerate() {
                    let x = start_x + j as u16;
                    if x >= inner.x && x < inner.x + inner.width && ch != ' ' {
                        let color = match ch {
                            'x' | 'y' | 'z' => Color::Cyan,
                            '│' | '─' | '╭' | '╮' | '╰' | '╯' | '╱' | '╲' => Color::DarkGray,
                            '●' => Color::Yellow,
                            _ => Color::White,
                        };
                        buf.set_string(x, y, &ch.to_string(), Style::default().fg(color));
                    }
                }
            }
        }

        // Draw state point if we have one
        if let Some((theta, phi, r, label)) = self.compute_bloch_coords() {
            // Project 3D point to 2D
            // Scale by r (Bloch vector magnitude) - mixed states are near center
            let x_3d = r * theta.sin() * phi.cos();
            let _y_3d = r * theta.sin() * phi.sin();
            let z_3d = r * theta.cos();

            // Simple orthographic projection (looking from y-axis)
            let px = cx as i16 + (x_3d * radius_x as f64) as i16;
            let py = cy as i16 - (z_3d * radius_y as f64) as i16;

            // Draw state vector
            if px >= inner.x as i16 && (px as u16) < inner.x + inner.width
                && py >= inner.y as i16 && (py as u16) < inner.y + inner.height
            {
                // Animate the state point - different symbol for mixed states
                let blink = (self.frame / 8) % 2 == 0;
                let (state_char, color) = if r < 0.3 {
                    // Mixed state - show at center with different color
                    (if blink { "●" } else { "○" }, Color::Yellow)
                } else {
                    (if blink { "◆" } else { "◇" }, Color::Red)
                };
                buf.set_string(
                    px as u16,
                    py as u16,
                    state_char,
                    Style::default().fg(color),
                );
            }

            // Draw label
            let label_x = inner.x + inner.width - label.len() as u16 - 1;
            let label_y = inner.y + inner.height - 1;
            buf.set_string(label_x, label_y, &label, Style::default().fg(Color::Green));

            // Draw coordinates - show r for mixed states
            let coords = if r < 0.95 {
                format!("r={:.2}", r)
            } else {
                format!("θ={:.0}° φ={:.0}°", theta.to_degrees(), phi.to_degrees())
            };
            if inner.width > coords.len() as u16 + 2 {
                buf.set_string(inner.x + 1, inner.y, &coords, Style::default().fg(Color::DarkGray));
            }
        }
    }
}

/// Multi-qubit Bloch visualization (shows each qubit's reduced state).
#[allow(dead_code)]
pub struct MultiBloch<'a> {
    state: Option<&'a QuantumState>,
    frame: u64,
}

#[allow(dead_code)]
impl<'a> MultiBloch<'a> {
    pub fn new(state: Option<&'a QuantumState>) -> Self {
        Self { state, frame: 0 }
    }

    pub fn frame(mut self, frame: u64) -> Self {
        self.frame = frame;
        self
    }
}

impl<'a> Widget for MultiBloch<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Reduced States")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow));

        let inner = block.inner(area);
        block.render(area, buf);

        let Some(state) = self.state else {
            return;
        };

        let n = state.num_qubits();
        let per_qubit_width = (inner.width / n.max(1) as u16).max(8);

        for q in 0..n.min(inner.width as usize / 8) {
            let x = state.expectation_x(q).unwrap_or(0.0);
            let y = state.expectation_y(q).unwrap_or(0.0);
            let z = state.expectation_z(q).unwrap_or(1.0);

            let qx = inner.x + (q as u16) * per_qubit_width;

            // Qubit label
            let label = format!("Q{}", q);
            buf.set_string(qx, inner.y, &label, Style::default().fg(Color::Cyan));

            // Expectation values
            let info = format!("Z:{:+.1}", z);
            if inner.height > 1 {
                buf.set_string(qx, inner.y + 1, &info, Style::default().fg(Color::White));
            }

            // Simple purity indicator
            let purity = x * x + y * y + z * z;
            let purity_bar = if purity > 0.9 {
                "█"
            } else if purity > 0.5 {
                "▓"
            } else {
                "░"
            };
            if inner.height > 2 {
                buf.set_string(
                    qx,
                    inner.y + 2,
                    purity_bar,
                    Style::default().fg(if purity > 0.9 { Color::Green } else { Color::Yellow }),
                );
            }
        }
    }
}
