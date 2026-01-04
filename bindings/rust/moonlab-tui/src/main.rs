//! Moonlab TUI - Interactive Quantum Simulator
//!
//! A beautiful terminal user interface for exploring quantum computing.

mod app;
mod events;
mod ui;

use anyhow::Result;
use app::App;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::stdout;
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run
    let mut app = App::new();
    let result = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Handle result
    if let Err(err) = result {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        // Draw UI
        terminal.draw(|f| ui::render(f, app))?;

        // Handle input with timeout for animations
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match key.code {
                    KeyCode::Char('q') if !app.is_input_mode() => {
                        return Ok(());
                    }
                    KeyCode::Esc => {
                        if app.is_input_mode() {
                            app.exit_input_mode();
                        } else if app.should_esc_exit() {
                            return Ok(());
                        } else {
                            // Let app handle Esc for going back in other modes
                            app.handle_key(key);
                        }
                    }
                    _ => app.handle_key(key),
                }
            }
        }

        // Update animations/state
        app.tick();
    }
}
