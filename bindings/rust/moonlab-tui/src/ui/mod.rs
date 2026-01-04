//! UI rendering module.
//!
//! Contains all visualization widgets and layout logic.

mod amplitudes;
mod bloch;
mod circuit;
mod dashboard;
mod entropy;
mod feynman;

use crate::app::App;
use ratatui::Frame;

// Re-export widgets for use by dashboard
pub use amplitudes::*;
pub use bloch::*;
pub use circuit::*;
#[allow(unused_imports)]
pub use dashboard::*;
pub use entropy::*;
pub use feynman::*;

/// Render the entire application.
pub fn render(frame: &mut Frame, app: &App) {
    dashboard::render_dashboard(frame, app);
}
