//! Feynman Diagram Rendering
//!
//! Generate publication-quality Feynman diagrams in multiple formats:
//! - ASCII (terminal output)
//! - SVG (web/publication)
//! - LaTeX/TikZ-Feynman (papers)
//!
//! # Example
//!
//! ```no_run
//! use moonlab::feynman::{FeynmanDiagram, ParticleType};
//!
//! // Create a custom diagram
//! let mut diagram = FeynmanDiagram::new("e+ e- -> mu+ mu-").unwrap();
//!
//! // Add vertices
//! let v1 = diagram.add_vertex(-2.0, 1.0);
//! let v2 = diagram.add_vertex(-2.0, -1.0);
//! let v3 = diagram.add_vertex(0.0, 0.0);
//! let v4 = diagram.add_vertex(2.0, 1.0);
//! let v5 = diagram.add_vertex(2.0, -1.0);
//!
//! // Add propagators
//! diagram.add_fermion(v1, v3, "e-");
//! diagram.add_antifermion(v2, v3, "e+");
//! diagram.add_photon(v3, v3, "γ");
//! diagram.add_fermion(v3, v4, "μ-");
//! diagram.add_antifermion(v3, v5, "μ+");
//!
//! // Render
//! let ascii = diagram.render_ascii();
//! println!("{}", ascii);
//! ```
//!
//! # Standard Diagrams
//!
//! Use the predefined diagrams for common QFT processes:
//!
//! ```no_run
//! use moonlab::feynman::FeynmanDiagram;
//!
//! let qed = FeynmanDiagram::qed_vertex();
//! let compton = FeynmanDiagram::compton_scattering();
//! let bhabha = FeynmanDiagram::bhabha_scattering();
//! ```

use std::ffi::{CStr, CString};
use std::ptr::NonNull;

use moonlab_sys as ffi;
use crate::error::{QuantumError, Result};

/// Particle types for Feynman diagram propagators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticleType {
    /// Electron, muon, quark, neutrino (solid line with forward arrow)
    Fermion,
    /// Positron, antimuon, antiquark (solid line with backward arrow)
    Antifermion,
    /// Electromagnetic force carrier γ (wavy line)
    Photon,
    /// Strong force carrier g (curly line)
    Gluon,
    /// Weak force carrier W± (dashed line)
    WBoson,
    /// Weak force carrier Z⁰ (dashed line)
    ZBoson,
    /// Higgs boson H (dotted line)
    Higgs,
    /// Generic scalar field (dashed line)
    Scalar,
    /// Faddeev-Popov ghost
    Ghost,
    /// Graviton (theoretical)
    Graviton,
}

impl ParticleType {
    /// Convert to FFI particle type.
    fn to_ffi(self) -> ffi::particle_type_t {
        match self {
            ParticleType::Fermion => ffi::particle_type_t_PARTICLE_FERMION,
            ParticleType::Antifermion => ffi::particle_type_t_PARTICLE_ANTIFERMION,
            ParticleType::Photon => ffi::particle_type_t_PARTICLE_PHOTON,
            ParticleType::Gluon => ffi::particle_type_t_PARTICLE_GLUON,
            ParticleType::WBoson => ffi::particle_type_t_PARTICLE_W_BOSON,
            ParticleType::ZBoson => ffi::particle_type_t_PARTICLE_Z_BOSON,
            ParticleType::Higgs => ffi::particle_type_t_PARTICLE_HIGGS,
            ParticleType::Scalar => ffi::particle_type_t_PARTICLE_SCALAR,
            ParticleType::Ghost => ffi::particle_type_t_PARTICLE_GHOST,
            ParticleType::Graviton => ffi::particle_type_t_PARTICLE_GRAVITON,
        }
    }

    /// Convert from FFI particle type.
    fn from_ffi(ffi_type: ffi::particle_type_t) -> Self {
        match ffi_type {
            ffi::particle_type_t_PARTICLE_FERMION => ParticleType::Fermion,
            ffi::particle_type_t_PARTICLE_ANTIFERMION => ParticleType::Antifermion,
            ffi::particle_type_t_PARTICLE_PHOTON => ParticleType::Photon,
            ffi::particle_type_t_PARTICLE_GLUON => ParticleType::Gluon,
            ffi::particle_type_t_PARTICLE_W_BOSON => ParticleType::WBoson,
            ffi::particle_type_t_PARTICLE_Z_BOSON => ParticleType::ZBoson,
            ffi::particle_type_t_PARTICLE_HIGGS => ParticleType::Higgs,
            ffi::particle_type_t_PARTICLE_SCALAR => ParticleType::Scalar,
            ffi::particle_type_t_PARTICLE_GHOST => ParticleType::Ghost,
            ffi::particle_type_t_PARTICLE_GRAVITON => ParticleType::Graviton,
            _ => ParticleType::Scalar,
        }
    }

    /// Get the display name for this particle type.
    pub fn name(&self) -> &'static str {
        match self {
            ParticleType::Fermion => "fermion",
            ParticleType::Antifermion => "antifermion",
            ParticleType::Photon => "photon",
            ParticleType::Gluon => "gluon",
            ParticleType::WBoson => "W boson",
            ParticleType::ZBoson => "Z boson",
            ParticleType::Higgs => "Higgs",
            ParticleType::Scalar => "scalar",
            ParticleType::Ghost => "ghost",
            ParticleType::Graviton => "graviton",
        }
    }

    /// Get the line style character for ASCII rendering.
    pub fn ascii_char(&self) -> char {
        match self {
            ParticleType::Fermion | ParticleType::Antifermion => '─',
            ParticleType::Photon => '~',
            ParticleType::Gluon => '@',
            ParticleType::WBoson | ParticleType::ZBoson | ParticleType::Scalar => '-',
            ParticleType::Higgs => '.',
            ParticleType::Ghost => ':',
            ParticleType::Graviton => '=',
        }
    }
}

/// A vertex (interaction point) in a Feynman diagram.
#[derive(Debug, Clone)]
pub struct Vertex {
    /// Unique vertex ID.
    pub id: i32,
    /// X coordinate in diagram space.
    pub x: f64,
    /// Y coordinate in diagram space.
    pub y: f64,
    /// Optional label (e.g., coupling constant).
    pub label: Option<String>,
    /// Whether this is an external vertex (incoming/outgoing particle).
    pub is_external: bool,
}

/// A propagator (line between vertices) in a Feynman diagram.
#[derive(Debug, Clone)]
pub struct Propagator {
    /// Starting vertex ID.
    pub from_vertex: i32,
    /// Ending vertex ID.
    pub to_vertex: i32,
    /// Particle type (determines line style).
    pub particle_type: ParticleType,
    /// Particle label (e.g., "e-", "γ").
    pub label: String,
    /// Whether this is an external line.
    pub is_external: bool,
}

/// A Feynman diagram for quantum field theory visualization.
///
/// Feynman diagrams are graphical representations of particle interactions
/// in quantum field theory. This type provides a builder API for constructing
/// diagrams and rendering them in various formats.
#[derive(Debug)]
pub struct FeynmanDiagram {
    inner: NonNull<ffi::feynman_diagram_t>,
}

// Safety: The underlying C structure is not accessed concurrently.
unsafe impl Send for FeynmanDiagram {}

impl FeynmanDiagram {
    /// Create a new Feynman diagram with the given process notation.
    ///
    /// # Arguments
    ///
    /// * `process` - Process notation (e.g., "e+ e- -> mu+ mu-")
    ///
    /// # Example
    ///
    /// ```no_run
    /// use moonlab::feynman::FeynmanDiagram;
    ///
    /// let diagram = FeynmanDiagram::new("e+ e- -> gamma gamma").unwrap();
    /// ```
    pub fn new(process: &str) -> Result<Self> {
        let c_process = CString::new(process)
            .map_err(|_| QuantumError::Ffi("Invalid process string".to_string()))?;

        unsafe {
            let ptr = ffi::feynman_create(c_process.as_ptr());
            NonNull::new(ptr)
                .map(|inner| Self { inner })
                .ok_or(QuantumError::AllocationFailed(0))
        }
    }

    /// Create a standard QED vertex diagram (e- → e- + γ).
    pub fn qed_vertex() -> Result<Self> {
        unsafe {
            let ptr = ffi::feynman_create_qed_vertex();
            NonNull::new(ptr)
                .map(|inner| Self { inner })
                .ok_or(QuantumError::AllocationFailed(0))
        }
    }

    /// Create e+ e- → μ+ μ- diagram (s-channel).
    pub fn ee_to_mumu() -> Result<Self> {
        unsafe {
            let ptr = ffi::feynman_create_ee_to_mumu();
            NonNull::new(ptr)
                .map(|inner| Self { inner })
                .ok_or(QuantumError::AllocationFailed(0))
        }
    }

    /// Create Compton scattering diagram (e- + γ → e- + γ).
    pub fn compton_scattering() -> Result<Self> {
        unsafe {
            let ptr = ffi::feynman_create_compton();
            NonNull::new(ptr)
                .map(|inner| Self { inner })
                .ok_or(QuantumError::AllocationFailed(0))
        }
    }

    /// Create pair annihilation diagram (e+ e- → γ γ).
    pub fn pair_annihilation() -> Result<Self> {
        unsafe {
            let ptr = ffi::feynman_create_pair_annihilation();
            NonNull::new(ptr)
                .map(|inner| Self { inner })
                .ok_or(QuantumError::AllocationFailed(0))
        }
    }

    /// Create electron self-energy diagram (one-loop).
    pub fn electron_self_energy() -> Result<Self> {
        unsafe {
            let ptr = ffi::feynman_create_electron_self_energy();
            NonNull::new(ptr)
                .map(|inner| Self { inner })
                .ok_or(QuantumError::AllocationFailed(0))
        }
    }

    /// Create vacuum polarization diagram (one-loop photon).
    pub fn vacuum_polarization() -> Result<Self> {
        unsafe {
            let ptr = ffi::feynman_create_vacuum_polarization();
            NonNull::new(ptr)
                .map(|inner| Self { inner })
                .ok_or(QuantumError::AllocationFailed(0))
        }
    }

    /// Create Møller scattering diagram (e- e- → e- e-).
    pub fn moller_scattering() -> Result<Self> {
        unsafe {
            let ptr = ffi::feynman_create_moller_scattering();
            NonNull::new(ptr)
                .map(|inner| Self { inner })
                .ok_or(QuantumError::AllocationFailed(0))
        }
    }

    /// Create Bhabha scattering diagram (e+ e- → e+ e-).
    pub fn bhabha_scattering() -> Result<Self> {
        unsafe {
            let ptr = ffi::feynman_create_bhabha_scattering();
            NonNull::new(ptr)
                .map(|inner| Self { inner })
                .ok_or(QuantumError::AllocationFailed(0))
        }
    }

    /// Get pointer to underlying C structure.
    #[inline]
    fn as_ptr(&self) -> *mut ffi::feynman_diagram_t {
        self.inner.as_ptr()
    }

    /// Set the diagram title.
    pub fn set_title(&mut self, title: &str) -> &mut Self {
        if let Ok(c_title) = CString::new(title) {
            unsafe {
                ffi::feynman_set_title(self.as_ptr(), c_title.as_ptr());
            }
        }
        self
    }

    /// Set the loop order (0 = tree level, 1 = one-loop, etc.).
    pub fn set_loop_order(&mut self, order: i32) -> &mut Self {
        unsafe {
            ffi::feynman_set_loop_order(self.as_ptr(), order);
        }
        self
    }

    /// Get the process notation string.
    pub fn process(&self) -> String {
        unsafe {
            let diagram = self.inner.as_ref();
            let c_str = CStr::from_ptr(diagram.process.as_ptr());
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Get the diagram title.
    pub fn title(&self) -> String {
        unsafe {
            let diagram = self.inner.as_ref();
            let c_str = CStr::from_ptr(diagram.title.as_ptr());
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Get the number of vertices.
    pub fn num_vertices(&self) -> usize {
        unsafe { self.inner.as_ref().num_vertices as usize }
    }

    /// Get the number of propagators.
    pub fn num_propagators(&self) -> usize {
        unsafe { self.inner.as_ref().num_propagators as usize }
    }

    /// Get the loop order.
    pub fn loop_order(&self) -> i32 {
        unsafe { self.inner.as_ref().loop_order }
    }

    /// Add a vertex at the specified position.
    ///
    /// Returns the vertex ID for use in adding propagators.
    pub fn add_vertex(&mut self, x: f64, y: f64) -> i32 {
        unsafe { ffi::feynman_add_vertex(self.as_ptr(), x, y) }
    }

    /// Add a labeled vertex.
    pub fn add_vertex_labeled(&mut self, x: f64, y: f64, label: &str) -> i32 {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe { ffi::feynman_add_vertex_labeled(self.as_ptr(), x, y, c_label.as_ptr()) }
    }

    /// Add an external vertex (for incoming/outgoing particles).
    pub fn add_external_vertex(&mut self, x: f64, y: f64, label: &str) -> i32 {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe { ffi::feynman_add_external_vertex(self.as_ptr(), x, y, c_label.as_ptr()) }
    }

    /// Add a fermion propagator (solid line with forward arrow).
    pub fn add_fermion(&mut self, from: i32, to: i32, label: &str) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_fermion(self.as_ptr(), from, to, c_label.as_ptr());
        }
        self
    }

    /// Add an antifermion propagator (solid line with backward arrow).
    pub fn add_antifermion(&mut self, from: i32, to: i32, label: &str) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_antifermion(self.as_ptr(), from, to, c_label.as_ptr());
        }
        self
    }

    /// Add a photon propagator (wavy line).
    pub fn add_photon(&mut self, from: i32, to: i32, label: &str) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_photon(self.as_ptr(), from, to, c_label.as_ptr());
        }
        self
    }

    /// Add a gluon propagator (curly line).
    pub fn add_gluon(&mut self, from: i32, to: i32, label: &str) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_gluon(self.as_ptr(), from, to, c_label.as_ptr());
        }
        self
    }

    /// Add a W boson propagator (dashed line).
    pub fn add_w_boson(&mut self, from: i32, to: i32, label: &str) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_w_boson(self.as_ptr(), from, to, c_label.as_ptr());
        }
        self
    }

    /// Add a Z boson propagator (dashed line).
    pub fn add_z_boson(&mut self, from: i32, to: i32, label: &str) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_z_boson(self.as_ptr(), from, to, c_label.as_ptr());
        }
        self
    }

    /// Add a Higgs propagator (dotted line).
    pub fn add_higgs(&mut self, from: i32, to: i32, label: &str) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_higgs(self.as_ptr(), from, to, c_label.as_ptr());
        }
        self
    }

    /// Add a scalar propagator (dashed line).
    pub fn add_scalar(&mut self, from: i32, to: i32, label: &str) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_scalar(self.as_ptr(), from, to, c_label.as_ptr());
        }
        self
    }

    /// Add a generic propagator with specified type.
    pub fn add_propagator(
        &mut self,
        from: i32,
        to: i32,
        particle_type: ParticleType,
        label: &str,
    ) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_propagator(
                self.as_ptr(),
                from,
                to,
                particle_type.to_ffi(),
                c_label.as_ptr(),
            );
        }
        self
    }

    /// Add an incoming external line to a vertex.
    ///
    /// # Arguments
    ///
    /// * `vertex` - Target vertex ID
    /// * `particle_type` - Type of particle
    /// * `label` - Particle label
    /// * `direction` - Direction angle in degrees (0=right, 90=up, 180=left, 270=down)
    pub fn add_incoming(
        &mut self,
        vertex: i32,
        particle_type: ParticleType,
        label: &str,
        direction: f64,
    ) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_incoming(
                self.as_ptr(),
                vertex,
                particle_type.to_ffi(),
                c_label.as_ptr(),
                direction,
            );
        }
        self
    }

    /// Add an outgoing external line from a vertex.
    pub fn add_outgoing(
        &mut self,
        vertex: i32,
        particle_type: ParticleType,
        label: &str,
        direction: f64,
    ) -> &mut Self {
        let c_label = CString::new(label).unwrap_or_default();
        unsafe {
            ffi::feynman_add_outgoing(
                self.as_ptr(),
                vertex,
                particle_type.to_ffi(),
                c_label.as_ptr(),
                direction,
            );
        }
        self
    }

    /// Render the diagram to ASCII art.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use moonlab::feynman::FeynmanDiagram;
    ///
    /// let diagram = FeynmanDiagram::qed_vertex().unwrap();
    /// let ascii = diagram.render_ascii();
    /// println!("{}", ascii);
    /// ```
    pub fn render_ascii(&self) -> String {
        unsafe {
            let c_str = ffi::feynman_render_ascii(self.as_ptr(), std::ptr::null());
            if c_str.is_null() {
                return String::new();
            }
            let result = CStr::from_ptr(c_str).to_string_lossy().into_owned();
            libc::free(c_str as *mut libc::c_void);
            result
        }
    }

    /// Render the diagram to SVG format.
    ///
    /// # Arguments
    ///
    /// * `width` - SVG width in pixels
    /// * `height` - SVG height in pixels
    pub fn render_svg(&self, width: i32, height: i32) -> String {
        unsafe {
            let c_str = ffi::feynman_render_svg(self.as_ptr(), width, height);
            if c_str.is_null() {
                return String::new();
            }
            let result = CStr::from_ptr(c_str).to_string_lossy().into_owned();
            libc::free(c_str as *mut libc::c_void);
            result
        }
    }

    /// Render the diagram to LaTeX/TikZ-Feynman format.
    pub fn render_latex(&self) -> String {
        unsafe {
            let c_str = ffi::feynman_render_latex(self.as_ptr());
            if c_str.is_null() {
                return String::new();
            }
            let result = CStr::from_ptr(c_str).to_string_lossy().into_owned();
            libc::free(c_str as *mut libc::c_void);
            result
        }
    }

    /// Save the diagram to an SVG file.
    pub fn save_svg(&self, filename: &str, width: i32, height: i32) -> Result<()> {
        let c_filename = CString::new(filename)
            .map_err(|_| QuantumError::Ffi("Invalid filename".to_string()))?;

        unsafe {
            let result = ffi::feynman_save_svg(self.as_ptr(), c_filename.as_ptr(), width, height);
            if result == 0 {
                Ok(())
            } else {
                Err(QuantumError::Ffi(format!("Failed to save SVG to {}", filename)))
            }
        }
    }

    /// Save the diagram to a LaTeX file.
    pub fn save_latex(&self, filename: &str) -> Result<()> {
        let c_filename = CString::new(filename)
            .map_err(|_| QuantumError::Ffi("Invalid filename".to_string()))?;

        unsafe {
            let result = ffi::feynman_save_latex(self.as_ptr(), c_filename.as_ptr());
            if result == 0 {
                Ok(())
            } else {
                Err(QuantumError::Ffi(format!("Failed to save LaTeX to {}", filename)))
            }
        }
    }

    /// Get the vertices in this diagram.
    pub fn vertices(&self) -> Vec<Vertex> {
        unsafe {
            let diagram = self.inner.as_ref();
            let mut vertices = Vec::with_capacity(diagram.num_vertices as usize);

            for i in 0..diagram.num_vertices as usize {
                let v = &diagram.vertices[i];
                let label = CStr::from_ptr(v.label.as_ptr())
                    .to_string_lossy()
                    .into_owned();

                vertices.push(Vertex {
                    id: v.id,
                    x: v.x,
                    y: v.y,
                    label: if label.is_empty() { None } else { Some(label) },
                    is_external: v.is_external,
                });
            }

            vertices
        }
    }

    /// Get the propagators in this diagram.
    pub fn propagators(&self) -> Vec<Propagator> {
        unsafe {
            let diagram = self.inner.as_ref();
            let mut propagators = Vec::with_capacity(diagram.num_propagators as usize);

            for i in 0..diagram.num_propagators as usize {
                let p = &diagram.propagators[i];
                let label = CStr::from_ptr(p.label.as_ptr())
                    .to_string_lossy()
                    .into_owned();

                propagators.push(Propagator {
                    from_vertex: p.from_vertex,
                    to_vertex: p.to_vertex,
                    particle_type: ParticleType::from_ffi(p.type_),
                    label,
                    is_external: p.is_external,
                });
            }

            propagators
        }
    }

    /// Get the bounding box (min_x, min_y, max_x, max_y).
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        unsafe {
            let diagram = self.inner.as_ref();
            (diagram.min_x, diagram.min_y, diagram.max_x, diagram.max_y)
        }
    }

    /// Update the bounding box based on current vertices.
    pub fn update_bounds(&mut self) {
        unsafe {
            ffi::feynman_update_bounds(self.as_ptr());
        }
    }
}

impl Drop for FeynmanDiagram {
    fn drop(&mut self) {
        unsafe {
            ffi::feynman_free(self.as_ptr());
        }
    }
}

/// Standard QFT process diagrams.
pub mod standard {
    use super::*;

    /// Common QED (Quantum Electrodynamics) processes.
    pub mod qed {
        use super::*;

        /// QED vertex: e- → e- + γ
        pub fn vertex() -> Result<FeynmanDiagram> {
            FeynmanDiagram::qed_vertex()
        }

        /// Compton scattering: e- + γ → e- + γ
        pub fn compton() -> Result<FeynmanDiagram> {
            FeynmanDiagram::compton_scattering()
        }

        /// Pair annihilation: e+ e- → γ γ
        pub fn pair_annihilation() -> Result<FeynmanDiagram> {
            FeynmanDiagram::pair_annihilation()
        }

        /// Møller scattering: e- e- → e- e-
        pub fn moller() -> Result<FeynmanDiagram> {
            FeynmanDiagram::moller_scattering()
        }

        /// Bhabha scattering: e+ e- → e+ e-
        pub fn bhabha() -> Result<FeynmanDiagram> {
            FeynmanDiagram::bhabha_scattering()
        }

        /// e+ e- → μ+ μ- (muon pair production)
        pub fn ee_to_mumu() -> Result<FeynmanDiagram> {
            FeynmanDiagram::ee_to_mumu()
        }
    }

    /// Loop diagrams.
    pub mod loops {
        use super::*;

        /// Electron self-energy (one-loop)
        pub fn electron_self_energy() -> Result<FeynmanDiagram> {
            FeynmanDiagram::electron_self_energy()
        }

        /// Vacuum polarization (one-loop photon)
        pub fn vacuum_polarization() -> Result<FeynmanDiagram> {
            FeynmanDiagram::vacuum_polarization()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_diagram() {
        let diagram = FeynmanDiagram::new("e+ e- -> gamma gamma").unwrap();
        assert_eq!(diagram.process(), "e+ e- -> gamma gamma");
    }

    #[test]
    fn test_add_vertices() {
        let mut diagram = FeynmanDiagram::new("test").unwrap();

        let v1 = diagram.add_vertex(0.0, 0.0);
        let v2 = diagram.add_vertex(1.0, 0.0);

        assert!(v1 >= 0);
        assert!(v2 >= 0);
        assert_ne!(v1, v2);
        assert_eq!(diagram.num_vertices(), 2);
    }

    #[test]
    fn test_add_propagators() {
        let mut diagram = FeynmanDiagram::new("test").unwrap();

        let v1 = diagram.add_vertex(0.0, 0.0);
        let v2 = diagram.add_vertex(1.0, 0.0);

        diagram
            .add_fermion(v1, v2, "e-")
            .add_photon(v1, v2, "gamma");

        assert_eq!(diagram.num_propagators(), 2);
    }

    #[test]
    fn test_standard_diagrams() {
        let qed = FeynmanDiagram::qed_vertex().unwrap();
        assert!(qed.num_vertices() > 0);

        let compton = FeynmanDiagram::compton_scattering().unwrap();
        assert!(compton.num_vertices() > 0);
    }

    #[test]
    fn test_render_ascii() {
        let diagram = FeynmanDiagram::qed_vertex().unwrap();
        let ascii = diagram.render_ascii();
        assert!(!ascii.is_empty());
    }
}
