//! Feynman diagram visualization widget.
//!
//! Displays QFT Feynman diagrams using ASCII art in the terminal.

use moonlab::feynman::{FeynmanDiagram, ParticleType};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Widget},
};

/// Standard Feynman diagram types for the browser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardDiagram {
    /// QED vertex (e- → e- + γ)
    QedVertex,
    /// e+ e- → μ+ μ-
    EeToMuMu,
    /// Compton scattering
    Compton,
    /// Pair annihilation
    PairAnnihilation,
    /// Møller scattering
    Moller,
    /// Bhabha scattering
    Bhabha,
    /// Electron self-energy
    ElectronSelfEnergy,
    /// Vacuum polarization
    VacuumPolarization,
}

#[allow(dead_code)]
impl StandardDiagram {
    /// Get all standard diagrams.
    pub fn all() -> &'static [StandardDiagram] {
        &[
            StandardDiagram::QedVertex,
            StandardDiagram::EeToMuMu,
            StandardDiagram::Compton,
            StandardDiagram::PairAnnihilation,
            StandardDiagram::Moller,
            StandardDiagram::Bhabha,
            StandardDiagram::ElectronSelfEnergy,
            StandardDiagram::VacuumPolarization,
        ]
    }

    /// Get the display name.
    pub fn name(&self) -> &'static str {
        match self {
            StandardDiagram::QedVertex => "QED Vertex",
            StandardDiagram::EeToMuMu => "e⁺e⁻ → μ⁺μ⁻",
            StandardDiagram::Compton => "Compton Scattering",
            StandardDiagram::PairAnnihilation => "Pair Annihilation",
            StandardDiagram::Moller => "Møller Scattering",
            StandardDiagram::Bhabha => "Bhabha Scattering",
            StandardDiagram::ElectronSelfEnergy => "Electron Self-Energy",
            StandardDiagram::VacuumPolarization => "Vacuum Polarization",
        }
    }

    /// Get the process notation.
    pub fn process(&self) -> &'static str {
        match self {
            StandardDiagram::QedVertex => "e⁻ → e⁻ + γ",
            StandardDiagram::EeToMuMu => "e⁺ e⁻ → μ⁺ μ⁻",
            StandardDiagram::Compton => "e⁻ + γ → e⁻ + γ",
            StandardDiagram::PairAnnihilation => "e⁺ e⁻ → γ γ",
            StandardDiagram::Moller => "e⁻ e⁻ → e⁻ e⁻",
            StandardDiagram::Bhabha => "e⁺ e⁻ → e⁺ e⁻",
            StandardDiagram::ElectronSelfEnergy => "e⁻ → e⁻ (1-loop)",
            StandardDiagram::VacuumPolarization => "γ → γ (1-loop)",
        }
    }

    /// Get a description.
    pub fn description(&self) -> &'static str {
        match self {
            StandardDiagram::QedVertex => "Fundamental QED interaction vertex",
            StandardDiagram::EeToMuMu => "Muon pair production via virtual photon",
            StandardDiagram::Compton => "Photon-electron elastic scattering",
            StandardDiagram::PairAnnihilation => "Electron-positron annihilation to photons",
            StandardDiagram::Moller => "Electron-electron scattering",
            StandardDiagram::Bhabha => "Electron-positron scattering",
            StandardDiagram::ElectronSelfEnergy => "One-loop correction to electron propagator",
            StandardDiagram::VacuumPolarization => "One-loop correction to photon propagator",
        }
    }

    /// Create the corresponding Feynman diagram.
    pub fn create_diagram(&self) -> Option<FeynmanDiagram> {
        match self {
            StandardDiagram::QedVertex => FeynmanDiagram::qed_vertex().ok(),
            StandardDiagram::EeToMuMu => FeynmanDiagram::ee_to_mumu().ok(),
            StandardDiagram::Compton => FeynmanDiagram::compton_scattering().ok(),
            StandardDiagram::PairAnnihilation => FeynmanDiagram::pair_annihilation().ok(),
            StandardDiagram::Moller => FeynmanDiagram::moller_scattering().ok(),
            StandardDiagram::Bhabha => FeynmanDiagram::bhabha_scattering().ok(),
            StandardDiagram::ElectronSelfEnergy => FeynmanDiagram::electron_self_energy().ok(),
            StandardDiagram::VacuumPolarization => FeynmanDiagram::vacuum_polarization().ok(),
        }
    }

    /// Is this a loop diagram?
    pub fn is_loop(&self) -> bool {
        matches!(
            self,
            StandardDiagram::ElectronSelfEnergy | StandardDiagram::VacuumPolarization
        )
    }
}

/// Feynman diagram visualization widget.
#[allow(dead_code)]
pub struct FeynmanWidget<'a> {
    diagram: Option<&'a FeynmanDiagram>,
    diagram_type: Option<StandardDiagram>,
    title: &'a str,
    show_labels: bool,
    animate_frame: u64,
}

#[allow(dead_code)]
impl<'a> FeynmanWidget<'a> {
    /// Create a new Feynman widget.
    pub fn new(diagram: Option<&'a FeynmanDiagram>) -> Self {
        Self {
            diagram,
            diagram_type: None,
            title: "Feynman Diagram",
            show_labels: true,
            animate_frame: 0,
        }
    }

    /// Create from a standard diagram type.
    pub fn from_standard(diagram_type: StandardDiagram, diagram: Option<&'a FeynmanDiagram>) -> Self {
        Self {
            diagram,
            diagram_type: Some(diagram_type),
            title: diagram_type.name(),
            show_labels: true,
            animate_frame: 0,
        }
    }

    /// Set the title.
    pub fn title(mut self, title: &'a str) -> Self {
        self.title = title;
        self
    }

    /// Set whether to show labels.
    pub fn show_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Set animation frame for effects.
    pub fn frame(mut self, frame: u64) -> Self {
        self.animate_frame = frame;
        self
    }

    /// Get the line character for a particle type.
    fn particle_line(&self, particle_type: ParticleType, horizontal: bool) -> &'static str {
        match particle_type {
            ParticleType::Fermion | ParticleType::Antifermion => {
                if horizontal { "─" } else { "│" }
            }
            ParticleType::Photon => "~",
            ParticleType::Gluon => "@",
            ParticleType::WBoson | ParticleType::ZBoson | ParticleType::Scalar => "-",
            ParticleType::Higgs => ".",
            ParticleType::Ghost => ":",
            ParticleType::Graviton => "=",
        }
    }

    /// Get the color for a particle type.
    fn particle_color(&self, particle_type: ParticleType) -> Color {
        match particle_type {
            ParticleType::Fermion => Color::White,
            ParticleType::Antifermion => Color::LightCyan,
            ParticleType::Photon => Color::Yellow,
            ParticleType::Gluon => Color::Red,
            ParticleType::WBoson | ParticleType::ZBoson => Color::Blue,
            ParticleType::Higgs => Color::Magenta,
            ParticleType::Scalar => Color::Green,
            ParticleType::Ghost => Color::DarkGray,
            ParticleType::Graviton => Color::LightMagenta,
        }
    }

    /// Get the arrow character for fermions.
    fn arrow_char(&self, particle_type: ParticleType, right: bool) -> &'static str {
        match particle_type {
            ParticleType::Fermion => if right { "→" } else { "←" },
            ParticleType::Antifermion => if right { "←" } else { "→" },
            _ => "",
        }
    }

    /// Render a simple ASCII representation.
    fn render_simple_ascii(&self, buf: &mut Buffer, area: Rect) {
        let Some(diagram_type) = self.diagram_type else {
            return;
        };

        // Center position
        let cx = area.x + area.width / 2;
        let cy = area.y + area.height / 2;

        match diagram_type {
            StandardDiagram::QedVertex => {
                // Simple vertex diagram
                //     e-
                //      \
                //       ●~~~~ γ
                //      /
                //     e-
                let style_e = Style::default().fg(Color::White);
                let style_g = Style::default().fg(Color::Yellow);
                let style_v = Style::default().fg(Color::Red);

                buf.set_string(cx - 4, cy - 2, "e⁻", style_e);
                buf.set_string(cx - 2, cy - 1, "\\", style_e);
                buf.set_string(cx, cy, "●", style_v);
                buf.set_string(cx + 1, cy, "~~~~", style_g);
                buf.set_string(cx + 5, cy, "γ", style_g);
                buf.set_string(cx - 2, cy + 1, "/", style_e);
                buf.set_string(cx - 4, cy + 2, "e⁻", style_e);
            }
            StandardDiagram::EeToMuMu => {
                // e+ e- -> mu+ mu- via photon
                let style_e = Style::default().fg(Color::White);
                let style_m = Style::default().fg(Color::Cyan);
                let style_g = Style::default().fg(Color::Yellow);
                let style_v = Style::default().fg(Color::Red);

                buf.set_string(cx - 6, cy - 2, "e⁺ →", style_e);
                buf.set_string(cx - 2, cy - 1, "\\", style_e);
                buf.set_string(cx, cy, "●", style_v);
                buf.set_string(cx + 1, cy, "~~", style_g);
                buf.set_string(cx + 3, cy, "●", style_v);
                buf.set_string(cx + 4, cy - 1, "/", style_m);
                buf.set_string(cx + 5, cy - 2, "→ μ⁺", style_m);
                buf.set_string(cx - 6, cy + 2, "e⁻ →", style_e);
                buf.set_string(cx - 2, cy + 1, "/", style_e);
                buf.set_string(cx + 4, cy + 1, "\\", style_m);
                buf.set_string(cx + 5, cy + 2, "→ μ⁻", style_m);
            }
            StandardDiagram::Compton => {
                // Compton: e- + γ -> e- + γ
                let style_e = Style::default().fg(Color::White);
                let style_g = Style::default().fg(Color::Yellow);
                let style_v = Style::default().fg(Color::Red);

                buf.set_string(cx - 8, cy, "e⁻ →─", style_e);
                buf.set_string(cx - 3, cy, "●", style_v);
                buf.set_string(cx - 2, cy - 2, "γ", style_g);
                buf.set_string(cx - 2, cy - 1, "|", style_g);
                buf.set_string(cx - 2, cy, "~", style_g);
                buf.set_string(cx - 1, cy, "─", style_e);
                buf.set_string(cx, cy, "●", style_v);
                buf.set_string(cx + 1, cy, "─→ e⁻", style_e);
                buf.set_string(cx + 1, cy - 1, "|", style_g);
                buf.set_string(cx + 1, cy - 2, "γ", style_g);
            }
            StandardDiagram::PairAnnihilation => {
                // e+ e- -> γ γ
                let style_e = Style::default().fg(Color::White);
                let style_g = Style::default().fg(Color::Yellow);
                let style_v = Style::default().fg(Color::Red);

                buf.set_string(cx - 6, cy - 1, "e⁺ →", style_e);
                buf.set_string(cx - 2, cy - 1, "\\", style_e);
                buf.set_string(cx, cy, "●", style_v);
                buf.set_string(cx + 1, cy - 1, "~~~~", style_g);
                buf.set_string(cx + 5, cy - 1, "γ", style_g);
                buf.set_string(cx - 6, cy + 1, "e⁻ →", style_e);
                buf.set_string(cx - 2, cy + 1, "/", style_e);
                buf.set_string(cx + 1, cy + 1, "~~~~", style_g);
                buf.set_string(cx + 5, cy + 1, "γ", style_g);
            }
            StandardDiagram::Moller | StandardDiagram::Bhabha => {
                // t-channel scattering
                let (p1, p2) = if diagram_type == StandardDiagram::Moller {
                    ("e⁻", "e⁻")
                } else {
                    ("e⁺", "e⁻")
                };
                let style_e = Style::default().fg(Color::White);
                let style_g = Style::default().fg(Color::Yellow);
                let style_v = Style::default().fg(Color::Red);

                buf.set_string(cx - 6, cy - 2, &format!("{} →─", p1), style_e);
                buf.set_string(cx - 1, cy - 2, "●", style_v);
                buf.set_string(cx, cy - 2, "─→", style_e);
                buf.set_string(cx + 2, cy - 2, p1, style_e);
                buf.set_string(cx - 1, cy - 1, "|", style_g);
                buf.set_string(cx - 1, cy, "~", style_g);
                buf.set_string(cx - 1, cy + 1, "|", style_g);
                buf.set_string(cx - 6, cy + 2, &format!("{} →─", p2), style_e);
                buf.set_string(cx - 1, cy + 2, "●", style_v);
                buf.set_string(cx, cy + 2, "─→", style_e);
                buf.set_string(cx + 2, cy + 2, p2, style_e);
            }
            StandardDiagram::ElectronSelfEnergy => {
                // One-loop self-energy
                let style_e = Style::default().fg(Color::White);
                let style_g = Style::default().fg(Color::Yellow);
                let style_v = Style::default().fg(Color::Red);

                buf.set_string(cx - 8, cy, "e⁻ →─", style_e);
                buf.set_string(cx - 3, cy, "●", style_v);
                buf.set_string(cx - 2, cy - 1, "╭~~╮", style_g);
                buf.set_string(cx - 2, cy, "─", style_e);
                buf.set_string(cx + 1, cy, "─", style_e);
                buf.set_string(cx - 2, cy + 1, "╰──╯", style_e);
                buf.set_string(cx + 2, cy, "●", style_v);
                buf.set_string(cx + 3, cy, "─→ e⁻", style_e);
            }
            StandardDiagram::VacuumPolarization => {
                // Vacuum polarization loop
                let style_e = Style::default().fg(Color::White);
                let style_g = Style::default().fg(Color::Yellow);
                let style_v = Style::default().fg(Color::Red);

                buf.set_string(cx - 8, cy, "γ ~~~", style_g);
                buf.set_string(cx - 3, cy, "●", style_v);
                buf.set_string(cx - 2, cy - 1, "╭──╮", style_e);
                buf.set_string(cx - 2, cy + 1, "╰──╯", style_e);
                buf.set_string(cx + 2, cy, "●", style_v);
                buf.set_string(cx + 3, cy, "~~~ γ", style_g);
            }
        }
    }
}

impl<'a> Widget for FeynmanWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(self.title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 5 || inner.width < 15 {
            return; // Too small to render
        }

        // If we have a diagram type, render the simple ASCII version
        if self.diagram_type.is_some() {
            self.render_simple_ascii(buf, inner);

            // Show process notation at bottom
            if let Some(dt) = self.diagram_type {
                let process = dt.process();
                if inner.width as usize > process.len() + 2 {
                    let x = inner.x + (inner.width - process.len() as u16) / 2;
                    let y = inner.y + inner.height - 1;
                    buf.set_string(x, y, process, Style::default().fg(Color::Green));
                }
            }
            return;
        }

        // Fallback: If we have a diagram, try to get ASCII from C library
        if let Some(diagram) = self.diagram {
            let ascii = diagram.render_ascii();
            if !ascii.is_empty() {
                // Render ASCII output line by line
                for (i, line) in ascii.lines().enumerate() {
                    if i as u16 >= inner.height {
                        break;
                    }
                    let truncated: String = line.chars().take(inner.width as usize).collect();
                    buf.set_string(inner.x, inner.y + i as u16, &truncated, Style::default());
                }
            }
        } else {
            // No diagram - show placeholder
            let msg = "No diagram loaded";
            let x = inner.x + (inner.width.saturating_sub(msg.len() as u16)) / 2;
            let y = inner.y + inner.height / 2;
            buf.set_string(x, y, msg, Style::default().fg(Color::DarkGray));
        }
    }
}

/// Metrics panel for Feynman diagram.
pub struct FeynmanMetrics<'a> {
    diagram: Option<&'a FeynmanDiagram>,
    diagram_type: Option<StandardDiagram>,
}

impl<'a> FeynmanMetrics<'a> {
    pub fn new(diagram: Option<&'a FeynmanDiagram>, diagram_type: Option<StandardDiagram>) -> Self {
        Self { diagram, diagram_type }
    }
}

impl<'a> Widget for FeynmanMetrics<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Diagram Info")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        let mut y = inner.y;

        if let Some(dt) = self.diagram_type {
            // Type
            buf.set_string(inner.x + 1, y, "Type:", Style::default().fg(Color::Cyan));
            buf.set_string(inner.x + 7, y, dt.name(), Style::default().fg(Color::White));
            y += 1;

            // Loop order
            if y < inner.y + inner.height {
                let loop_text = if dt.is_loop() { "1-loop" } else { "Tree" };
                buf.set_string(inner.x + 1, y, "Order:", Style::default().fg(Color::Cyan));
                buf.set_string(inner.x + 8, y, loop_text, Style::default().fg(Color::Yellow));
                y += 1;
            }
        }

        if let Some(diagram) = self.diagram {
            // Vertices
            if y < inner.y + inner.height {
                buf.set_string(inner.x + 1, y, "Vertices:", Style::default().fg(Color::Cyan));
                buf.set_string(
                    inner.x + 11,
                    y,
                    &format!("{}", diagram.num_vertices()),
                    Style::default().fg(Color::White),
                );
                y += 1;
            }

            // Propagators
            if y < inner.y + inner.height {
                buf.set_string(inner.x + 1, y, "Lines:", Style::default().fg(Color::Cyan));
                buf.set_string(
                    inner.x + 8,
                    y,
                    &format!("{}", diagram.num_propagators()),
                    Style::default().fg(Color::White),
                );
                y += 1;
            }

            // Loop order from diagram
            if y < inner.y + inner.height {
                buf.set_string(inner.x + 1, y, "Loops:", Style::default().fg(Color::Cyan));
                buf.set_string(
                    inner.x + 8,
                    y,
                    &format!("{}", diagram.loop_order()),
                    Style::default().fg(Color::Yellow),
                );
            }
        }
    }
}

/// Particle legend widget.
pub struct ParticleLegend;

impl Widget for ParticleLegend {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Particle Legend")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        let legends = [
            ("───→", "Fermion (e⁻, μ⁻)", Color::White),
            ("───←", "Antifermion (e⁺)", Color::LightCyan),
            ("~~~~", "Photon (γ)", Color::Yellow),
            ("@@@@", "Gluon (g)", Color::Red),
            ("----", "W/Z Boson", Color::Blue),
            ("....", "Higgs (H)", Color::Magenta),
            ("  ● ", "Vertex", Color::Red),
        ];

        for (i, (sym, name, color)) in legends.iter().enumerate() {
            if i as u16 >= inner.height {
                break;
            }
            let y = inner.y + i as u16;
            buf.set_string(inner.x + 1, y, sym, Style::default().fg(*color));
            buf.set_string(
                inner.x + 6,
                y,
                name,
                Style::default().fg(Color::White),
            );
        }
    }
}
