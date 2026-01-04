//! Application state and logic.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use moonlab::{FeynmanDiagram, QuantumState};
use std::time::Instant;

/// Application execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    /// Interactive algorithm browser
    AlgorithmBrowser,
    /// Running an algorithm with visualization
    AlgorithmRunning,
    /// Step-by-step exploration
    StepThrough,
    /// Free circuit building
    FreeExploration,
    /// Feynman diagram browser
    FeynmanBrowser,
    /// Help overlay
    Help,
}

/// Algorithm types available in the simulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    Grover,
    Bell,
    GHZ,
    VQE,
    QAOA,
    QFT,
}

impl Algorithm {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Grover => "Grover's Search",
            Self::Bell => "Bell State",
            Self::GHZ => "GHZ State",
            Self::VQE => "VQE (H₂ Molecule)",
            Self::QAOA => "QAOA (MaxCut)",
            Self::QFT => "Quantum Fourier Transform",
        }
    }

    #[allow(dead_code)]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Grover => "Quantum search with O(√N) speedup",
            Self::Bell => "Maximally entangled 2-qubit state",
            Self::GHZ => "Greenberger-Horne-Zeilinger entanglement",
            Self::VQE => "Variational Quantum Eigensolver for chemistry",
            Self::QAOA => "Quantum Approximate Optimization",
            Self::QFT => "Quantum analog of discrete Fourier transform",
        }
    }

    pub fn all() -> &'static [Algorithm] {
        &[
            Self::Grover,
            Self::Bell,
            Self::GHZ,
            Self::VQE,
            Self::QAOA,
            Self::QFT,
        ]
    }
}

/// Focus area in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Algorithms,
    Parameters,
    Circuit,
    StateView,
    Metrics,
}

impl Focus {
    pub fn next(&self) -> Self {
        match self {
            Self::Algorithms => Self::Parameters,
            Self::Parameters => Self::Circuit,
            Self::Circuit => Self::StateView,
            Self::StateView => Self::Metrics,
            Self::Metrics => Self::Algorithms,
        }
    }

    pub fn prev(&self) -> Self {
        match self {
            Self::Algorithms => Self::Metrics,
            Self::Parameters => Self::Algorithms,
            Self::Circuit => Self::Parameters,
            Self::StateView => Self::Circuit,
            Self::Metrics => Self::StateView,
        }
    }
}

/// Standard Feynman diagram types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeynmanDiagramType {
    QedVertex,
    EeToMuMu,
    Compton,
    PairAnnihilation,
    Moller,
    Bhabha,
    ElectronSelfEnergy,
    VacuumPolarization,
}

impl FeynmanDiagramType {
    pub fn all() -> &'static [FeynmanDiagramType] {
        &[
            FeynmanDiagramType::QedVertex,
            FeynmanDiagramType::EeToMuMu,
            FeynmanDiagramType::Compton,
            FeynmanDiagramType::PairAnnihilation,
            FeynmanDiagramType::Moller,
            FeynmanDiagramType::Bhabha,
            FeynmanDiagramType::ElectronSelfEnergy,
            FeynmanDiagramType::VacuumPolarization,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::QedVertex => "QED Vertex",
            Self::EeToMuMu => "e⁺e⁻ → μ⁺μ⁻",
            Self::Compton => "Compton Scattering",
            Self::PairAnnihilation => "Pair Annihilation",
            Self::Moller => "Møller Scattering",
            Self::Bhabha => "Bhabha Scattering",
            Self::ElectronSelfEnergy => "Electron Self-Energy",
            Self::VacuumPolarization => "Vacuum Polarization",
        }
    }

    pub fn process(&self) -> &'static str {
        match self {
            Self::QedVertex => "e⁻ → e⁻ + γ",
            Self::EeToMuMu => "e⁺ e⁻ → μ⁺ μ⁻",
            Self::Compton => "e⁻ + γ → e⁻ + γ",
            Self::PairAnnihilation => "e⁺ e⁻ → γ γ",
            Self::Moller => "e⁻ e⁻ → e⁻ e⁻",
            Self::Bhabha => "e⁺ e⁻ → e⁺ e⁻",
            Self::ElectronSelfEnergy => "e⁻ → e⁻ (1-loop)",
            Self::VacuumPolarization => "γ → γ (1-loop)",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::QedVertex => "Fundamental QED interaction vertex",
            Self::EeToMuMu => "Muon pair production via virtual photon",
            Self::Compton => "Photon-electron elastic scattering",
            Self::PairAnnihilation => "Electron-positron annihilation to photons",
            Self::Moller => "Electron-electron scattering",
            Self::Bhabha => "Electron-positron scattering",
            Self::ElectronSelfEnergy => "One-loop correction to electron propagator",
            Self::VacuumPolarization => "One-loop correction to photon propagator",
        }
    }

    pub fn is_loop(&self) -> bool {
        matches!(
            self,
            Self::ElectronSelfEnergy | Self::VacuumPolarization
        )
    }

    pub fn create_diagram(&self) -> Option<FeynmanDiagram> {
        match self {
            Self::QedVertex => FeynmanDiagram::qed_vertex().ok(),
            Self::EeToMuMu => FeynmanDiagram::ee_to_mumu().ok(),
            Self::Compton => FeynmanDiagram::compton_scattering().ok(),
            Self::PairAnnihilation => FeynmanDiagram::pair_annihilation().ok(),
            Self::Moller => FeynmanDiagram::moller_scattering().ok(),
            Self::Bhabha => FeynmanDiagram::bhabha_scattering().ok(),
            Self::ElectronSelfEnergy => FeynmanDiagram::electron_self_energy().ok(),
            Self::VacuumPolarization => FeynmanDiagram::vacuum_polarization().ok(),
        }
    }
}

/// A gate applied in free mode (for tracking/display).
#[derive(Debug, Clone)]
pub enum FreeGate {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    T(usize),
    CNOT(usize, usize),
}

impl FreeGate {
    /// Get gate name for display.
    pub fn name(&self) -> &'static str {
        match self {
            Self::H(_) => "H",
            Self::X(_) => "X",
            Self::Y(_) => "Y",
            Self::Z(_) => "Z",
            Self::S(_) => "S",
            Self::T(_) => "T",
            Self::CNOT(_, _) => "CNOT",
        }
    }

    /// Get target qubit(s) for display.
    pub fn qubits(&self) -> String {
        match self {
            Self::H(q) | Self::X(q) | Self::Y(q) | Self::Z(q) | Self::S(q) | Self::T(q) => {
                format!("Q{}", q)
            }
            Self::CNOT(c, t) => format!("Q{}→Q{}", c, t),
        }
    }
}

/// Main application state.
#[allow(dead_code)]
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
    /// Total circuit steps
    pub total_steps: usize,
    /// Algorithm-specific parameter
    pub parameter: f64,
    /// Status message
    pub status: String,
    /// Whether in input mode
    input_mode: bool,
    /// Input buffer
    input_buffer: String,
    /// Last tick time
    last_tick: Instant,
    /// Frame counter for animations
    frame_count: u64,
    /// Selected Feynman diagram type
    pub selected_feynman: usize,
    /// Current Feynman diagram
    pub feynman_diagram: Option<FeynmanDiagram>,
    /// Selected qubit for free mode
    pub selected_qubit: usize,
    /// Selected target qubit for 2-qubit gates
    pub selected_target: usize,
    /// Mode we were in before entering Help (for context-sensitive help)
    pub help_from_mode: AppMode,
    /// Gates applied in free mode (for visualization)
    pub free_mode_gates: Vec<FreeGate>,
}

impl App {
    pub fn new() -> Self {
        Self {
            mode: AppMode::AlgorithmBrowser,
            focus: Focus::Algorithms,
            selected_algorithm: 0,
            num_qubits: 3,
            state: None,
            animation_progress: 0.0,
            animating: false,
            circuit_step: 0,
            total_steps: 0,
            parameter: 0.0,
            status: String::from("Press Enter to run algorithm, Tab to switch focus, d for diagrams"),
            input_mode: false,
            input_buffer: String::new(),
            last_tick: Instant::now(),
            frame_count: 0,
            selected_feynman: 0,
            feynman_diagram: None,
            selected_qubit: 0,
            selected_target: 1,
            help_from_mode: AppMode::AlgorithmBrowser,
            free_mode_gates: Vec::new(),
        }
    }

    /// Get the currently selected Feynman diagram type.
    pub fn current_feynman_type(&self) -> FeynmanDiagramType {
        FeynmanDiagramType::all()[self.selected_feynman]
    }

    /// Get the currently selected algorithm.
    pub fn current_algorithm(&self) -> Algorithm {
        Algorithm::all()[self.selected_algorithm]
    }

    /// Check if we're in input mode.
    pub fn is_input_mode(&self) -> bool {
        self.input_mode
    }

    /// Exit input mode.
    pub fn exit_input_mode(&mut self) {
        self.input_mode = false;
        self.input_buffer.clear();
    }

    /// Check if Esc should exit the application (only in browser mode).
    pub fn should_esc_exit(&self) -> bool {
        matches!(self.mode, AppMode::AlgorithmBrowser)
    }

    /// Handle a key event.
    pub fn handle_key(&mut self, key: KeyEvent) {
        if self.input_mode {
            self.handle_input_key(key);
            return;
        }

        match self.mode {
            AppMode::Help => {
                // Any key exits help - return to previous mode
                self.mode = self.help_from_mode;
            }
            AppMode::AlgorithmBrowser => self.handle_browser_key(key),
            AppMode::AlgorithmRunning => self.handle_running_key(key),
            AppMode::StepThrough => self.handle_step_key(key),
            AppMode::FreeExploration => self.handle_free_key(key),
            AppMode::FeynmanBrowser => self.handle_feynman_key(key),
        }
    }

    fn handle_input_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char(c) => {
                self.input_buffer.push(c);
            }
            KeyCode::Backspace => {
                self.input_buffer.pop();
            }
            KeyCode::Enter => {
                self.apply_input();
                self.exit_input_mode();
            }
            _ => {}
        }
    }

    fn apply_input(&mut self) {
        if let Ok(n) = self.input_buffer.parse::<usize>() {
            if n >= 1 && n <= 12 {
                self.num_qubits = n;
                self.status = format!("Qubits set to {}", n);
            }
        }
    }

    fn handle_browser_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Tab => {
                self.focus = if key.modifiers.contains(KeyModifiers::SHIFT) {
                    self.focus.prev()
                } else {
                    self.focus.next()
                };
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.focus == Focus::Algorithms {
                    self.selected_algorithm = self.selected_algorithm.saturating_sub(1);
                } else if self.focus == Focus::Parameters {
                    self.num_qubits = (self.num_qubits + 1).min(12);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.focus == Focus::Algorithms {
                    self.selected_algorithm =
                        (self.selected_algorithm + 1).min(Algorithm::all().len() - 1);
                } else if self.focus == Focus::Parameters {
                    self.num_qubits = self.num_qubits.saturating_sub(1).max(1);
                }
            }
            KeyCode::Enter => {
                self.run_algorithm();
            }
            KeyCode::Char('?') => {
                self.help_from_mode = AppMode::AlgorithmBrowser;
                self.mode = AppMode::Help;
            }
            KeyCode::Char('s') => {
                self.mode = AppMode::StepThrough;
                self.initialize_state();
                self.status = String::from("Step-through mode: n=next, p=prev, r=reset");
            }
            KeyCode::Char('f') => {
                self.mode = AppMode::FreeExploration;
                self.initialize_state();
                self.free_mode_gates.clear();
                self.selected_qubit = 0;
                self.selected_target = 1.min(self.num_qubits.saturating_sub(1));
                self.status = format!("Free mode: Select Q{} (target: Q{}) | Apply gates with h/x/y/z/s/t/c", self.selected_qubit, self.selected_target);
            }
            KeyCode::Char('n') => {
                self.input_mode = true;
                self.input_buffer.clear();
                self.status = String::from("Enter number of qubits (1-12):");
            }
            KeyCode::Char('d') => {
                self.mode = AppMode::FeynmanBrowser;
                self.load_feynman_diagram();
                self.status = String::from("Feynman Diagrams: ↑↓ to browse, b to go back");
            }
            _ => {}
        }
    }

    fn handle_feynman_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected_feynman = self.selected_feynman.saturating_sub(1);
                self.load_feynman_diagram();
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected_feynman =
                    (self.selected_feynman + 1).min(FeynmanDiagramType::all().len() - 1);
                self.load_feynman_diagram();
            }
            KeyCode::Enter => {
                // Could add additional diagram interaction here
                let diagram_type = self.current_feynman_type();
                self.status = format!("Selected: {} - {}", diagram_type.name(), diagram_type.description());
            }
            KeyCode::Char('b') | KeyCode::Backspace => {
                self.mode = AppMode::AlgorithmBrowser;
                self.status = String::from("Press Enter to run algorithm, Tab to switch focus, d for diagrams");
            }
            KeyCode::Char('h') | KeyCode::Char('?') => {
                self.help_from_mode = AppMode::FeynmanBrowser;
                self.mode = AppMode::Help;
            }
            _ => {}
        }
    }

    fn load_feynman_diagram(&mut self) {
        let diagram_type = self.current_feynman_type();
        self.feynman_diagram = diagram_type.create_diagram();
        self.status = format!("{}: {}", diagram_type.name(), diagram_type.process());
    }

    fn handle_running_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char(' ') => {
                if self.circuit_step >= self.total_steps {
                    // Replay from beginning
                    self.circuit_step = 0;
                    self.animation_progress = 0.0;
                    self.animating = true;
                    self.status = String::from("Replaying...");
                } else {
                    // Pause/resume
                    self.animating = !self.animating;
                    self.status = if self.animating {
                        String::from("Running...")
                    } else {
                        String::from("Paused - Space to resume")
                    };
                }
            }
            KeyCode::Char('r') => {
                self.run_algorithm();
            }
            KeyCode::Char('b') | KeyCode::Backspace => {
                self.mode = AppMode::AlgorithmBrowser;
                self.status = String::from("Press Enter to run algorithm");
            }
            KeyCode::Char('?') => {
                self.help_from_mode = AppMode::AlgorithmRunning;
                self.mode = AppMode::Help;
            }
            _ => {}
        }
    }

    fn handle_step_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('n') | KeyCode::Right => {
                self.step_forward();
            }
            KeyCode::Char('p') | KeyCode::Left => {
                self.step_backward();
            }
            KeyCode::Char('r') => {
                self.initialize_state();
                self.circuit_step = 0;
            }
            KeyCode::Char('b') | KeyCode::Backspace => {
                self.mode = AppMode::AlgorithmBrowser;
            }
            KeyCode::Char('?') => {
                self.help_from_mode = AppMode::StepThrough;
                self.mode = AppMode::Help;
            }
            _ => {}
        }
    }

    fn handle_free_key(&mut self, key: KeyEvent) {
        let max_qubit = self.num_qubits.saturating_sub(1);

        match key.code {
            // Number keys 0-9 select qubit
            KeyCode::Char(c @ '0'..='9') => {
                let q = c.to_digit(10).unwrap() as usize;
                if q < self.num_qubits {
                    self.selected_qubit = q;
                    self.status = format!("▸ Q{} selected (target: Q{})", q, self.selected_target);
                }
            }
            // Up/Down to cycle selected qubit
            KeyCode::Up => {
                self.selected_qubit = self.selected_qubit.saturating_sub(1);
                self.status = format!("▸ Q{} selected (target: Q{})", self.selected_qubit, self.selected_target);
            }
            KeyCode::Down => {
                self.selected_qubit = (self.selected_qubit + 1).min(max_qubit);
                self.status = format!("▸ Q{} selected (target: Q{})", self.selected_qubit, self.selected_target);
            }
            // Tab to cycle target qubit (for 2-qubit gates)
            KeyCode::Tab => {
                self.selected_target = (self.selected_target + 1) % self.num_qubits;
                if self.selected_target == self.selected_qubit {
                    self.selected_target = (self.selected_target + 1) % self.num_qubits;
                }
                self.status = format!("▸ Q{} selected (target: Q{})", self.selected_qubit, self.selected_target);
            }
            // Single-qubit gates (lowercase)
            KeyCode::Char('h') => {
                if let Some(ref mut state) = self.state {
                    state.h(self.selected_qubit);
                    self.free_mode_gates.push(FreeGate::H(self.selected_qubit));
                    self.circuit_step += 1;
                    self.status = format!("✓ H(Q{}) applied", self.selected_qubit);
                }
            }
            KeyCode::Char('x') => {
                if let Some(ref mut state) = self.state {
                    state.x(self.selected_qubit);
                    self.free_mode_gates.push(FreeGate::X(self.selected_qubit));
                    self.circuit_step += 1;
                    self.status = format!("✓ X(Q{}) applied", self.selected_qubit);
                }
            }
            KeyCode::Char('y') => {
                if let Some(ref mut state) = self.state {
                    state.y(self.selected_qubit);
                    self.free_mode_gates.push(FreeGate::Y(self.selected_qubit));
                    self.circuit_step += 1;
                    self.status = format!("✓ Y(Q{}) applied", self.selected_qubit);
                }
            }
            KeyCode::Char('z') => {
                if let Some(ref mut state) = self.state {
                    state.z(self.selected_qubit);
                    self.free_mode_gates.push(FreeGate::Z(self.selected_qubit));
                    self.circuit_step += 1;
                    self.status = format!("✓ Z(Q{}) applied", self.selected_qubit);
                }
            }
            KeyCode::Char('s') => {
                if let Some(ref mut state) = self.state {
                    state.s(self.selected_qubit);
                    self.free_mode_gates.push(FreeGate::S(self.selected_qubit));
                    self.circuit_step += 1;
                    self.status = format!("✓ S(Q{}) applied", self.selected_qubit);
                }
            }
            KeyCode::Char('t') => {
                if let Some(ref mut state) = self.state {
                    state.t(self.selected_qubit);
                    self.free_mode_gates.push(FreeGate::T(self.selected_qubit));
                    self.circuit_step += 1;
                    self.status = format!("✓ T(Q{}) applied", self.selected_qubit);
                }
            }
            // Two-qubit gates
            KeyCode::Char('c') => {
                if let Some(ref mut state) = self.state {
                    if self.selected_qubit != self.selected_target {
                        state.cnot(self.selected_qubit, self.selected_target);
                        self.free_mode_gates.push(FreeGate::CNOT(self.selected_qubit, self.selected_target));
                        self.circuit_step += 1;
                        self.status = format!("✓ CNOT(Q{}→Q{}) applied", self.selected_qubit, self.selected_target);
                    } else {
                        self.status = String::from("✗ Control and target must differ!");
                    }
                }
            }
            // Reset
            KeyCode::Char('r') => {
                self.initialize_state();
                self.free_mode_gates.clear();
                self.selected_qubit = 0;
                self.selected_target = 1.min(max_qubit);
                self.status = String::from("✓ State reset to |0...0⟩");
            }
            // Back - go to running mode to show the state we built
            KeyCode::Backspace | KeyCode::Esc => {
                self.mode = AppMode::AlgorithmRunning;
                self.animating = false;
                self.total_steps = self.circuit_step; // Mark as complete
                self.status = String::from("Custom circuit complete | Space: replay  b: browser  r: reset");
            }
            // Help
            KeyCode::Char('?') => {
                self.help_from_mode = AppMode::FreeExploration;
                self.mode = AppMode::Help;
            }
            _ => {}
        }
    }

    fn initialize_state(&mut self) {
        self.state = QuantumState::new(self.num_qubits).ok();
        self.circuit_step = 0;
        self.animation_progress = 0.0;
    }

    fn run_algorithm(&mut self) {
        // Capture values before mutable borrow
        let algorithm = self.current_algorithm();
        let n = self.num_qubits;

        self.initialize_state();

        let Some(ref mut state) = self.state else {
            self.status = String::from("Failed to initialize quantum state");
            return;
        };

        match algorithm {
            Algorithm::Bell => {
                state.h(0).cnot(0, 1);
                self.total_steps = 2;
                self.status = String::from("Bell state: (|00⟩ + |11⟩)/√2");
            }
            Algorithm::GHZ => {
                state.h(0);
                for i in 1..n {
                    state.cnot(0, i);
                }
                self.total_steps = n;
                self.status = format!("{}-qubit GHZ state created", n);
            }
            Algorithm::Grover => {
                // Simple Grover's for demonstration (marks state |11...1⟩)
                // Initial superposition
                for i in 0..n {
                    state.h(i);
                }
                // One iteration of Grover
                // Oracle (mark all-ones state)
                if n >= 2 {
                    // Multi-controlled Z is approximated
                    for i in 0..n {
                        state.z(i);
                    }
                }
                // Diffusion
                for i in 0..n {
                    state.h(i);
                }
                for i in 0..n {
                    state.x(i);
                }
                if n >= 2 {
                    state.cz(0, 1);
                }
                for i in 0..n {
                    state.x(i);
                }
                for i in 0..n {
                    state.h(i);
                }
                self.total_steps = 4 * n + 2;
                self.status = String::from("Grover's search (1 iteration)");
            }
            Algorithm::QFT => {
                let qubits: Vec<usize> = (0..n).collect();
                state.qft(&qubits);
                self.total_steps = n * (n + 1) / 2;
                self.status = format!("{}-qubit QFT applied", n);
            }
            Algorithm::VQE => {
                // Simplified VQE ansatz demonstration
                for i in 0..n {
                    state.ry(i, std::f64::consts::PI / 4.0);
                }
                for i in 0..n.saturating_sub(1) {
                    state.cnot(i, i + 1);
                }
                self.total_steps = 2 * n;
                self.status = String::from("VQE hardware-efficient ansatz");
            }
            Algorithm::QAOA => {
                // Simplified QAOA demonstration
                for i in 0..n {
                    state.h(i);
                }
                // Cost layer
                for i in 0..n.saturating_sub(1) {
                    state.rz(i, std::f64::consts::PI / 3.0);
                    state.cz(i, i + 1);
                }
                // Mixer layer
                for i in 0..n {
                    state.rx(i, std::f64::consts::PI / 4.0);
                }
                self.total_steps = 4 * n;
                self.status = String::from("QAOA p=1 circuit");
            }
        }

        self.mode = AppMode::AlgorithmRunning;
        self.circuit_step = 0;
        self.animation_progress = 0.0;
        self.animating = true;
    }

    fn step_forward(&mut self) {
        if self.circuit_step < self.total_steps {
            self.circuit_step += 1;
            // In a full implementation, we'd replay the circuit up to this step
        }
    }

    fn step_backward(&mut self) {
        self.circuit_step = self.circuit_step.saturating_sub(1);
        // In a full implementation, we'd replay the circuit up to this step
    }

    /// Update animation state.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick);
        self.last_tick = now;
        self.frame_count += 1;

        if self.animating && self.total_steps > 0 {
            let step_duration = 0.5; // seconds per step
            self.animation_progress += dt.as_secs_f64() / step_duration;

            // Calculate which step we should be on
            let target_step = ((self.animation_progress * self.total_steps as f64) as usize)
                .min(self.total_steps);

            if target_step > self.circuit_step {
                self.circuit_step = target_step;
            }

            if self.circuit_step >= self.total_steps {
                self.circuit_step = self.total_steps;
                self.animating = false;
                self.status = String::from("Algorithm complete. Press Space to replay, b to go back");
            }
        }
    }

    /// Get frame count for animations.
    pub fn frame(&self) -> u64 {
        self.frame_count
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
