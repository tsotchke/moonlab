//! Main dashboard layout.
//!
//! Organizes all widgets into a cohesive display.

use super::{AmplitudeBars, BlochSphere, CircuitDiagram, EntanglementHeatmap, FeynmanMetrics, FeynmanWidget, Gate, MetricsPanel, ParticleLegend, StandardDiagram};
use crate::app::{Algorithm, App, AppMode, FeynmanDiagramType, Focus, FreeGate};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame,
};

/// Render the full dashboard.
pub fn render_dashboard(frame: &mut Frame, app: &App) {
    // Clear with dark background
    frame.render_widget(Clear, frame.size());

    // Main layout: header, content, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Content
            Constraint::Length(3), // Footer/status
        ])
        .split(frame.size());

    render_header(frame, chunks[0], app);
    render_content(frame, chunks[1], app);
    render_footer(frame, chunks[2], app);

    // Render help overlay if active
    if app.mode == AppMode::Help {
        render_help_overlay(frame, app);
    }
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let title = vec![
        Span::styled("◆ ", Style::default().fg(Color::Cyan)),
        Span::styled("MOONLAB", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::styled(" QUANTUM SIMULATOR", Style::default().fg(Color::Cyan)),
        Span::raw("  │  "),
        Span::styled(
            app.current_algorithm().name(),
            Style::default().fg(Color::Yellow),
        ),
        Span::raw("  │  "),
        Span::styled(
            format!("{} qubits", app.num_qubits),
            Style::default().fg(Color::Green),
        ),
    ];

    let mode_text = match app.mode {
        AppMode::AlgorithmBrowser => "BROWSE",
        AppMode::AlgorithmRunning => if app.animating { "RUNNING" } else { "PAUSED" },
        AppMode::StepThrough => "STEP",
        AppMode::FreeExploration => "FREE",
        AppMode::FeynmanBrowser => "FEYNMAN",
        AppMode::Help => "HELP",
    };

    let header = Paragraph::new(Line::from(title))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(
                    format!(" {} ", mode_text),
                    Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
                )),
        );

    frame.render_widget(header, area);
}

fn render_content(frame: &mut Frame, area: Rect, app: &App) {
    if app.mode == AppMode::FeynmanBrowser {
        render_feynman_content(frame, area, app);
        return;
    }

    if app.mode == AppMode::FreeExploration {
        render_free_mode_content(frame, area, app);
        return;
    }

    // Split content into left panel (algorithms/params) and right panel (visualization)
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(28), // Left panel
            Constraint::Min(40),    // Main visualization
        ])
        .split(area);

    render_left_panel(frame, main_chunks[0], app);
    render_main_panel(frame, main_chunks[1], app);
}

fn render_feynman_content(frame: &mut Frame, area: Rect, app: &App) {
    // Split into left panel (diagram list) and right panel (visualization)
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(28), // Left panel
            Constraint::Min(40),    // Main visualization
        ])
        .split(area);

    // Left panel: Diagram list
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(FeynmanDiagramType::all().len() as u16 + 2), // Diagram list
            Constraint::Min(5), // Legend
        ])
        .split(main_chunks[0]);

    // Diagram list
    let diagrams: Vec<ListItem> = FeynmanDiagramType::all()
        .iter()
        .enumerate()
        .map(|(i, dt)| {
            let style = if i == app.selected_feynman {
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            let prefix = if i == app.selected_feynman { "▸ " } else { "  " };
            ListItem::new(format!("{}{}", prefix, dt.name())).style(style)
        })
        .collect();

    let diagram_list = List::new(diagrams).block(
        Block::default()
            .title("Feynman Diagrams")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta)),
    );

    frame.render_widget(diagram_list, left_chunks[0]);

    // Particle legend
    frame.render_widget(ParticleLegend, left_chunks[1]);

    // Right panel: Diagram visualization
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(70), // Diagram
            Constraint::Percentage(30), // Info
        ])
        .split(main_chunks[1]);

    // Convert app FeynmanDiagramType to UI StandardDiagram
    let diagram_type = match app.current_feynman_type() {
        FeynmanDiagramType::QedVertex => StandardDiagram::QedVertex,
        FeynmanDiagramType::EeToMuMu => StandardDiagram::EeToMuMu,
        FeynmanDiagramType::Compton => StandardDiagram::Compton,
        FeynmanDiagramType::PairAnnihilation => StandardDiagram::PairAnnihilation,
        FeynmanDiagramType::Moller => StandardDiagram::Moller,
        FeynmanDiagramType::Bhabha => StandardDiagram::Bhabha,
        FeynmanDiagramType::ElectronSelfEnergy => StandardDiagram::ElectronSelfEnergy,
        FeynmanDiagramType::VacuumPolarization => StandardDiagram::VacuumPolarization,
    };

    // Render Feynman diagram widget
    frame.render_widget(
        FeynmanWidget::from_standard(diagram_type, app.feynman_diagram.as_ref())
            .frame(app.frame()),
        right_chunks[0],
    );

    // Render diagram info
    let info_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(right_chunks[1]);

    frame.render_widget(
        FeynmanMetrics::new(app.feynman_diagram.as_ref(), Some(diagram_type)),
        info_chunks[0],
    );

    // Description panel
    let dt = app.current_feynman_type();
    let desc_text = vec![
        Line::from(vec![
            Span::styled("Process: ", Style::default().fg(Color::Cyan)),
            Span::styled(dt.process(), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            dt.description(),
            Style::default().fg(Color::White),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("Loop order: ", Style::default().fg(Color::Cyan)),
            Span::styled(
                if dt.is_loop() { "One-loop" } else { "Tree-level" },
                Style::default().fg(if dt.is_loop() { Color::Yellow } else { Color::Green }),
            ),
        ]),
    ];

    let desc = Paragraph::new(desc_text)
        .block(
            Block::default()
                .title("Description")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(desc, info_chunks[1]);
}

fn render_free_mode_content(frame: &mut Frame, area: Rect, app: &App) {
    // Split into left panel (controls) and right panel (visualization)
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(32), // Left panel - wider for controls
            Constraint::Min(40),    // Main visualization
        ])
        .split(area);

    // Left panel: Qubit selector, gate palette, applied gates
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(app.num_qubits as u16 + 4), // Qubit selector
            Constraint::Length(10),                         // Gate palette
            Constraint::Min(5),                             // Applied gates
        ])
        .split(main_chunks[0]);

    // Qubit selector panel
    render_qubit_selector(frame, left_chunks[0], app);

    // Gate palette
    render_gate_palette(frame, left_chunks[1], app);

    // Applied gates list
    render_applied_gates(frame, left_chunks[2], app);

    // Right panel: State visualization
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50), // Bloch + amplitudes
            Constraint::Percentage(50), // Entanglement + metrics
        ])
        .split(main_chunks[1]);

    // Top row: Bloch sphere and amplitude bars
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Bloch sphere
            Constraint::Percentage(60), // Amplitudes
        ])
        .split(right_chunks[0]);

    frame.render_widget(
        BlochSphere::new(app.state.as_ref(), app.selected_qubit)
            .title(format!("Bloch Sphere (Q{})", app.selected_qubit).as_str())
            .frame(app.frame()),
        top_chunks[0],
    );

    frame.render_widget(
        AmplitudeBars::new(app.state.as_ref())
            .title("State Amplitudes |ψ⟩"),
        top_chunks[1],
    );

    // Bottom row: Entanglement and metrics
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(right_chunks[1]);

    frame.render_widget(
        EntanglementHeatmap::new(app.state.as_ref())
            .title("Entanglement"),
        bottom_chunks[0],
    );

    frame.render_widget(
        MetricsPanel::new(app.state.as_ref())
            .gate_count(app.circuit_step),
        bottom_chunks[1],
    );
}

fn render_qubit_selector(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![
        Line::from(Span::styled(
            "Select qubit (↑↓ or 0-9):",
            Style::default().fg(Color::Cyan),
        )),
        Line::from(""),
    ];

    for i in 0..app.num_qubits {
        let is_selected = i == app.selected_qubit;
        let is_target = i == app.selected_target;

        let prefix = if is_selected { "▸ " } else { "  " };
        let suffix = if is_target { " ← target" } else { "" };

        let style = if is_selected {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else if is_target {
            Style::default().fg(Color::Magenta)
        } else {
            Style::default().fg(Color::White)
        };

        // Show qubit with visual state indicator
        let qubit_char = if is_selected { "●" } else { "○" };
        lines.push(Line::from(vec![
            Span::styled(prefix, style),
            Span::styled(format!("Q{} ", i), style),
            Span::styled(qubit_char, style),
            Span::styled(suffix, Style::default().fg(Color::Magenta)),
        ]));
    }

    let selector = Paragraph::new(lines).block(
        Block::default()
            .title(" Qubits ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow)),
    );

    frame.render_widget(selector, area);
}

fn render_gate_palette(frame: &mut Frame, area: Rect, app: &App) {
    let lines = vec![
        Line::from(vec![
            Span::styled("Single-qubit:", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled(" h", Style::default().fg(Color::Green)),
            Span::raw("=H  "),
            Span::styled("x", Style::default().fg(Color::Green)),
            Span::raw("=X  "),
            Span::styled("y", Style::default().fg(Color::Green)),
            Span::raw("=Y  "),
            Span::styled("z", Style::default().fg(Color::Green)),
            Span::raw("=Z"),
        ]),
        Line::from(vec![
            Span::styled(" s", Style::default().fg(Color::Green)),
            Span::raw("=S  "),
            Span::styled("t", Style::default().fg(Color::Green)),
            Span::raw("=T"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Two-qubit:", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled(" c", Style::default().fg(Color::Magenta)),
            Span::raw(format!("=CNOT (Q{}→Q{})", app.selected_qubit, app.selected_target)),
        ]),
        Line::from(vec![
            Span::styled(" Tab", Style::default().fg(Color::DarkGray)),
            Span::raw(" to change target"),
        ]),
    ];

    let palette = Paragraph::new(lines).block(
        Block::default()
            .title(" Gates ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green)),
    );

    frame.render_widget(palette, area);
}

fn render_applied_gates(frame: &mut Frame, area: Rect, app: &App) {
    let max_display = (area.height as usize).saturating_sub(3);

    let gates: Vec<ListItem> = app.free_mode_gates
        .iter()
        .rev()
        .take(max_display)
        .enumerate()
        .map(|(i, gate)| {
            let step = app.free_mode_gates.len() - i;
            let color = match gate {
                FreeGate::H(_) => Color::Cyan,
                FreeGate::X(_) | FreeGate::Y(_) | FreeGate::Z(_) => Color::Yellow,
                FreeGate::S(_) | FreeGate::T(_) => Color::Blue,
                FreeGate::CNOT(_, _) => Color::Magenta,
            };

            ListItem::new(format!("{:2}. {} {}", step, gate.name(), gate.qubits()))
                .style(Style::default().fg(color))
        })
        .collect();

    let title = if app.free_mode_gates.is_empty() {
        " Applied Gates (empty) "
    } else {
        " Applied Gates "
    };

    let list = List::new(gates).block(
        Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue)),
    );

    frame.render_widget(list, area);
}

fn render_left_panel(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(Algorithm::all().len() as u16 + 2), // Algorithms
            Constraint::Length(6), // Parameters
            Constraint::Min(5),    // Metrics
        ])
        .split(area);

    // Algorithm list
    let algorithms: Vec<ListItem> = Algorithm::all()
        .iter()
        .enumerate()
        .map(|(i, alg)| {
            let style = if i == app.selected_algorithm {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            let prefix = if i == app.selected_algorithm { "▸ " } else { "  " };
            ListItem::new(format!("{}{}", prefix, alg.name())).style(style)
        })
        .collect();

    let alg_border_color = if app.focus == Focus::Algorithms {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let algorithm_list = List::new(algorithms)
        .block(
            Block::default()
                .title("Algorithms")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(alg_border_color)),
        );

    frame.render_widget(algorithm_list, chunks[0]);

    // Parameters
    let param_border_color = if app.focus == Focus::Parameters {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let params_text = vec![
        Line::from(vec![
            Span::raw("  Qubits: "),
            Span::styled(format!("{}", app.num_qubits), Style::default().fg(Color::Cyan)),
            Span::styled(" ▲▼", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::raw("  Dim: "),
            Span::styled(format!("2^{}", app.num_qubits), Style::default().fg(Color::White)),
        ]),
    ];

    let params = Paragraph::new(params_text).block(
        Block::default()
            .title("Parameters")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(param_border_color)),
    );

    frame.render_widget(params, chunks[1]);

    // Metrics
    frame.render_widget(
        MetricsPanel::new(app.state.as_ref())
            .gate_count(app.circuit_step),
        chunks[2],
    );
}

fn render_main_panel(frame: &mut Frame, area: Rect, app: &App) {
    // Split main panel: top (circuit + bloch) and bottom (state + entropy)
    let vert_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(area);

    // Top row: Circuit diagram and Bloch sphere
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60), // Circuit
            Constraint::Percentage(40), // Bloch sphere
        ])
        .split(vert_chunks[0]);

    // Render circuit diagram
    let circuit = build_circuit_for_algorithm(app);
    let _circuit_border = if app.focus == Focus::Circuit {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let circuit_widget = circuit.current_step(app.circuit_step);
    // TODO: Apply circuit_border style when CircuitDiagram supports it
    frame.render_widget(circuit_widget, top_chunks[0]);

    // Render Bloch sphere
    frame.render_widget(
        BlochSphere::new(app.state.as_ref(), 0)
            .title("Bloch Sphere (Q0)")
            .frame(app.frame()),
        top_chunks[1],
    );

    // Bottom row: State amplitudes and entanglement
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60), // Amplitudes
            Constraint::Percentage(40), // Entanglement
        ])
        .split(vert_chunks[1]);

    let _state_border = if app.focus == Focus::StateView {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    // TODO: Apply state_border style when AmplitudeBars supports it
    frame.render_widget(
        AmplitudeBars::new(app.state.as_ref())
            .title("State Amplitudes |ψ⟩"),
        bottom_chunks[0],
    );

    frame.render_widget(
        EntanglementHeatmap::new(app.state.as_ref())
            .title("Entanglement"),
        bottom_chunks[1],
    );
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    // Keybindings based on mode
    let keys = match app.mode {
        AppMode::AlgorithmBrowser => {
            "↑↓ Select  │  Enter Run  │  s Step  │  f Free  │  d Diagrams  │  ? Help  │  q Quit"
        }
        AppMode::AlgorithmRunning => {
            "Space Pause  │  r Reset  │  b Back  │  ? Help  │  q Quit"
        }
        AppMode::StepThrough => {
            "← → Step  │  r Reset  │  b Back  │  ? Help  │  q Quit"
        }
        AppMode::FreeExploration => {
            "0-9/↑↓ Qubit  │  Tab Target  │  h/x/y/z/s/t Gates  │  c CNOT  │  ? Help  │  Esc Back"
        }
        AppMode::FeynmanBrowser => {
            "↑↓ Select diagram  │  Enter Details  │  b Back  │  ? Help  │  q Quit"
        }
        AppMode::Help => {
            "Press any key to close help"
        }
    };

    let footer = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(&app.status, Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled(keys, Style::default().fg(Color::DarkGray)),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );

    frame.render_widget(footer, area);
}

fn render_help_overlay(frame: &mut Frame, app: &App) {
    let area = centered_rect(70, 85, frame.size());

    frame.render_widget(Clear, area);

    let (title, help_text) = match app.help_from_mode {
        AppMode::AlgorithmBrowser => help_algorithm_browser(),
        AppMode::AlgorithmRunning => help_algorithm_running(),
        AppMode::StepThrough => help_step_through(),
        AppMode::FreeExploration => help_free_exploration(),
        AppMode::FeynmanBrowser => help_feynman_browser(),
        AppMode::Help => help_algorithm_browser(), // Fallback
    };

    let help = Paragraph::new(help_text)
        .block(
            Block::default()
                .title(format!(" {} ", title))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow)),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(help, area);
}

fn help_algorithm_browser() -> (&'static str, Vec<Line<'static>>) {
    ("Algorithm Browser Help", vec![
        Line::from(Span::styled(
            "◆ MOONLAB QUANTUM SIMULATOR",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled("Welcome! ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))),
        Line::from("This is the algorithm browser - your starting point for exploring"),
        Line::from("quantum algorithms. Select an algorithm and watch it run."),
        Line::from(""),
        Line::from(Span::styled("─── Navigation ───", Style::default().fg(Color::Yellow))),
        Line::from("  ↑/↓ or j/k   Select algorithm from the list"),
        Line::from("  Tab          Cycle focus between panels"),
        Line::from("  Enter        Run the selected algorithm"),
        Line::from("  q            Quit the application"),
        Line::from(""),
        Line::from(Span::styled("─── Modes ───", Style::default().fg(Color::Yellow))),
        Line::from("  s            Step-through: Execute algorithm one gate at a time"),
        Line::from("  f            Free exploration: Build your own quantum circuits"),
        Line::from("  d            Feynman diagrams: Explore particle physics diagrams"),
        Line::from(""),
        Line::from(Span::styled("─── Parameters ───", Style::default().fg(Color::Yellow))),
        Line::from("  Tab to Parameters panel, then use ↑/↓ to adjust qubit count."),
        Line::from("  More qubits = larger Hilbert space (2^n dimensional)."),
        Line::from(""),
        Line::from(Span::styled("─── Available Algorithms ───", Style::default().fg(Color::Yellow))),
        Line::from(vec![
            Span::styled("  Bell       ", Style::default().fg(Color::Magenta)),
            Span::raw("Creates maximally entangled pair |00⟩+|11⟩"),
        ]),
        Line::from(vec![
            Span::styled("  GHZ        ", Style::default().fg(Color::Magenta)),
            Span::raw("Multi-qubit entanglement (Greenberger-Horne-Zeilinger)"),
        ]),
        Line::from(vec![
            Span::styled("  Grover     ", Style::default().fg(Color::Magenta)),
            Span::raw("Quantum search - finds marked items in O(√N)"),
        ]),
        Line::from(vec![
            Span::styled("  QFT        ", Style::default().fg(Color::Magenta)),
            Span::raw("Quantum Fourier Transform - basis for Shor's algorithm"),
        ]),
        Line::from(vec![
            Span::styled("  VQE        ", Style::default().fg(Color::Magenta)),
            Span::raw("Variational Quantum Eigensolver - finds ground states"),
        ]),
        Line::from(vec![
            Span::styled("  QAOA       ", Style::default().fg(Color::Magenta)),
            Span::raw("Quantum Approximate Optimization Algorithm"),
        ]),
        Line::from(""),
        Line::from(Span::styled("Press any key to close", Style::default().fg(Color::DarkGray))),
    ])
}

fn help_algorithm_running() -> (&'static str, Vec<Line<'static>>) {
    ("Algorithm Running Help", vec![
        Line::from(Span::styled(
            "◆ ALGORITHM EXECUTION",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("You're watching a quantum algorithm execute! The visualization shows"),
        Line::from("how quantum states evolve as gates are applied."),
        Line::from(""),
        Line::from(Span::styled("─── Controls ───", Style::default().fg(Color::Yellow))),
        Line::from("  Space        Pause/resume animation"),
        Line::from("  r            Reset to initial state and restart"),
        Line::from("  b            Go back to algorithm browser"),
        Line::from("  q            Quit"),
        Line::from(""),
        Line::from(Span::styled("─── Understanding the Display ───", Style::default().fg(Color::Yellow))),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Circuit Diagram: ", Style::default().fg(Color::Green)),
            Span::raw("Shows quantum gates being applied"),
        ]),
        Line::from("    • Highlighted gate = currently executing"),
        Line::from("    • Lines = qubit wires, gates sit on wires"),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Bloch Sphere: ", Style::default().fg(Color::Green)),
            Span::raw("3D representation of qubit 0"),
        ]),
        Line::from("    • |0⟩ = north pole, |1⟩ = south pole"),
        Line::from("    • Superposition states are on the equator"),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Amplitude Bars: ", Style::default().fg(Color::Green)),
            Span::raw("Probability of each basis state"),
        ]),
        Line::from("    • Bar length = probability (|amplitude|²)"),
        Line::from("    • Color indicates complex phase"),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Entanglement Map: ", Style::default().fg(Color::Green)),
            Span::raw("Correlations between qubits"),
        ]),
        Line::from("    • Bright = strongly entangled"),
        Line::from("    • Dark = independent/separable"),
        Line::from(""),
        Line::from(Span::styled("Press any key to close", Style::default().fg(Color::DarkGray))),
    ])
}

fn help_step_through() -> (&'static str, Vec<Line<'static>>) {
    ("Step-Through Mode Help", vec![
        Line::from(Span::styled(
            "◆ STEP-THROUGH MODE",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Step through quantum algorithms one gate at a time. Perfect for"),
        Line::from("understanding exactly how quantum states transform."),
        Line::from(""),
        Line::from(Span::styled("─── Controls ───", Style::default().fg(Color::Yellow))),
        Line::from("  → or n       Step forward (apply next gate)"),
        Line::from("  ← or p       Step backward (undo last gate)"),
        Line::from("  r            Reset to initial |0...0⟩ state"),
        Line::from("  b            Back to algorithm browser"),
        Line::from("  q            Quit"),
        Line::from(""),
        Line::from(Span::styled("─── What to Watch For ───", Style::default().fg(Color::Yellow))),
        Line::from(""),
        Line::from(Span::styled("  Hadamard (H):", Style::default().fg(Color::Green))),
        Line::from("    Creates superposition: |0⟩ → (|0⟩+|1⟩)/√2"),
        Line::from("    Watch the amplitude bars split 50/50"),
        Line::from(""),
        Line::from(Span::styled("  CNOT:", Style::default().fg(Color::Green))),
        Line::from("    Entangles two qubits (flips target if control is |1⟩)"),
        Line::from("    Watch the entanglement heatmap light up"),
        Line::from(""),
        Line::from(Span::styled("  Phase gates (S, T):", Style::default().fg(Color::Green))),
        Line::from("    Add complex phases - affects Bloch sphere longitude"),
        Line::from(""),
        Line::from(Span::styled("─── Tips ───", Style::default().fg(Color::Yellow))),
        Line::from("  • After H on |0⟩, Bloch sphere points to +X axis"),
        Line::from("  • Entanglement appears after CNOT on superposed control"),
        Line::from("  • Try comparing step-by-step with running mode"),
        Line::from(""),
        Line::from(Span::styled("Press any key to close", Style::default().fg(Color::DarkGray))),
    ])
}

fn help_free_exploration() -> (&'static str, Vec<Line<'static>>) {
    ("Free Exploration Help", vec![
        Line::from(Span::styled(
            "◆ FREE EXPLORATION MODE",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Build your own quantum circuits! Apply gates to any qubit"),
        Line::from("and watch the quantum state evolve in real-time."),
        Line::from(""),
        Line::from(Span::styled("─── Qubit Selection ───", Style::default().fg(Color::Yellow))),
        Line::from("  0-9          Select qubit directly (if available)"),
        Line::from("  ↑/↓          Cycle through qubits"),
        Line::from("  Tab          Cycle target qubit (for 2-qubit gates)"),
        Line::from(""),
        Line::from(Span::styled("─── Single-Qubit Gates ───", Style::default().fg(Color::Yellow))),
        Line::from(vec![
            Span::styled("  h  Hadamard  ", Style::default().fg(Color::Green)),
            Span::raw("Creates superposition (|0⟩+|1⟩)/√2"),
        ]),
        Line::from(vec![
            Span::styled("  x  Pauli-X   ", Style::default().fg(Color::Green)),
            Span::raw("Bit flip: |0⟩↔|1⟩ (quantum NOT)"),
        ]),
        Line::from(vec![
            Span::styled("  y  Pauli-Y   ", Style::default().fg(Color::Green)),
            Span::raw("Bit+phase flip (combines X and Z)"),
        ]),
        Line::from(vec![
            Span::styled("  z  Pauli-Z   ", Style::default().fg(Color::Green)),
            Span::raw("Phase flip: |1⟩ → -|1⟩"),
        ]),
        Line::from(vec![
            Span::styled("  s  S-gate    ", Style::default().fg(Color::Green)),
            Span::raw("√Z gate (90° phase rotation)"),
        ]),
        Line::from(vec![
            Span::styled("  t  T-gate    ", Style::default().fg(Color::Green)),
            Span::raw("√S gate (45° phase rotation)"),
        ]),
        Line::from(""),
        Line::from(Span::styled("─── Two-Qubit Gates ───", Style::default().fg(Color::Yellow))),
        Line::from(vec![
            Span::styled("  c  CNOT      ", Style::default().fg(Color::Magenta)),
            Span::raw("Control=selected, Target=Tab selection"),
        ]),
        Line::from("                 Flips target qubit if control is |1⟩"),
        Line::from("                 Creates entanglement from superposition"),
        Line::from(""),
        Line::from(Span::styled("─── Other Controls ───", Style::default().fg(Color::Yellow))),
        Line::from("  r            Reset to |0...0⟩ state"),
        Line::from("  Esc          Return to view completed circuit"),
        Line::from(""),
        Line::from(Span::styled("─── Try These! ───", Style::default().fg(Color::Yellow))),
        Line::from("  Bell state:   h on Q0, then c with Q0→Q1"),
        Line::from("  Superposition: h on all qubits"),
        Line::from("  Phase kickback: x on target, h on control, then c"),
        Line::from(""),
        Line::from(Span::styled("Press any key to close", Style::default().fg(Color::DarkGray))),
    ])
}

fn help_feynman_browser() -> (&'static str, Vec<Line<'static>>) {
    ("Feynman Diagrams Help", vec![
        Line::from(Span::styled(
            "◆ FEYNMAN DIAGRAM BROWSER",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Explore Quantum Electrodynamics (QED) through Feynman diagrams."),
        Line::from("These diagrams represent particle interactions visually."),
        Line::from(""),
        Line::from(Span::styled("─── Controls ───", Style::default().fg(Color::Yellow))),
        Line::from("  ↑/↓          Select diagram"),
        Line::from("  Enter        View diagram details"),
        Line::from("  b            Back to algorithm browser"),
        Line::from("  q            Quit"),
        Line::from(""),
        Line::from(Span::styled("─── Reading Feynman Diagrams ───", Style::default().fg(Color::Yellow))),
        Line::from("  • Time flows left → right"),
        Line::from("  • Solid lines = fermions (electrons, positrons, muons)"),
        Line::from("  • Wavy lines = photons (γ)"),
        Line::from("  • Vertices = interactions (coupling constant α)"),
        Line::from(""),
        Line::from(Span::styled("─── Available Diagrams ───", Style::default().fg(Color::Yellow))),
        Line::from(vec![
            Span::styled("  QED Vertex     ", Style::default().fg(Color::Magenta)),
            Span::raw("Basic electron-photon interaction"),
        ]),
        Line::from(vec![
            Span::styled("  e⁺e⁻ → μ⁺μ⁻   ", Style::default().fg(Color::Magenta)),
            Span::raw("Electron-positron annihilation to muon pair"),
        ]),
        Line::from(vec![
            Span::styled("  Compton       ", Style::default().fg(Color::Magenta)),
            Span::raw("Photon scattering off electron"),
        ]),
        Line::from(vec![
            Span::styled("  Pair Annihil. ", Style::default().fg(Color::Magenta)),
            Span::raw("e⁺e⁻ → γγ (matter-antimatter → light)"),
        ]),
        Line::from(vec![
            Span::styled("  Møller        ", Style::default().fg(Color::Magenta)),
            Span::raw("Electron-electron scattering"),
        ]),
        Line::from(vec![
            Span::styled("  Bhabha        ", Style::default().fg(Color::Magenta)),
            Span::raw("Electron-positron scattering"),
        ]),
        Line::from(vec![
            Span::styled("  Self-Energy   ", Style::default().fg(Color::Yellow)),
            Span::raw("Electron loop correction (1-loop)"),
        ]),
        Line::from(vec![
            Span::styled("  Vacuum Pol.   ", Style::default().fg(Color::Yellow)),
            Span::raw("Photon loop correction (1-loop)"),
        ]),
        Line::from(""),
        Line::from(Span::styled("─── The Physics ───", Style::default().fg(Color::Yellow))),
        Line::from("  Tree-level diagrams = leading order approximation"),
        Line::from("  Loop diagrams = quantum corrections (renormalization)"),
        Line::from("  Coupling α ≈ 1/137 (fine structure constant)"),
        Line::from(""),
        Line::from(Span::styled("Press any key to close", Style::default().fg(Color::DarkGray))),
    ])
}

/// Helper to create a centered rectangle.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Build circuit diagram for the current algorithm.
fn build_circuit_for_algorithm(app: &App) -> CircuitDiagram {
    let n = app.num_qubits;

    match app.current_algorithm() {
        Algorithm::Bell => {
            CircuitDiagram::new(2.max(n))
                .title("Bell State")
                .gates(vec![Gate::H(0), Gate::CNOT(0, 1)])
        }
        Algorithm::GHZ => {
            let mut gates = vec![Gate::H(0)];
            for i in 1..n {
                gates.push(Gate::CNOT(0, i));
            }
            CircuitDiagram::new(n).title("GHZ State").gates(gates)
        }
        Algorithm::Grover => {
            let mut gates = Vec::new();
            // Hadamards
            for i in 0..n {
                gates.push(Gate::H(i));
            }
            // Oracle (simplified)
            if n >= 2 {
                gates.push(Gate::CZ(0, 1));
            }
            // Diffusion
            for i in 0..n {
                gates.push(Gate::H(i));
            }
            CircuitDiagram::new(n).title("Grover's Search").gates(gates)
        }
        Algorithm::QFT => {
            let mut gates = Vec::new();
            for i in 0..n {
                gates.push(Gate::H(i));
                // Controlled rotations (simplified)
            }
            CircuitDiagram::new(n).title("QFT").gates(gates)
        }
        Algorithm::VQE => {
            let mut gates = Vec::new();
            for i in 0..n {
                gates.push(Gate::Ry(i, 0.5));
            }
            for i in 0..n.saturating_sub(1) {
                gates.push(Gate::CNOT(i, i + 1));
            }
            CircuitDiagram::new(n).title("VQE Ansatz").gates(gates)
        }
        Algorithm::QAOA => {
            let mut gates = Vec::new();
            for i in 0..n {
                gates.push(Gate::H(i));
            }
            for i in 0..n.saturating_sub(1) {
                gates.push(Gate::CZ(i, i + 1));
            }
            for i in 0..n {
                gates.push(Gate::Rx(i, 0.5));
            }
            CircuitDiagram::new(n).title("QAOA p=1").gates(gates)
        }
    }
}
