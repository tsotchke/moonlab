//! Event handling for the TUI.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Input event types.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    Quit,
    Help,
    Tab,
    ShiftTab,
    Up,
    Down,
    Left,
    Right,
    Enter,
    Space,
    Escape,
    Reset,
    Back,
    Char(char),
}

impl From<KeyEvent> for InputEvent {
    fn from(key: KeyEvent) -> Self {
        match key.code {
            KeyCode::Char('q') => InputEvent::Quit,
            KeyCode::Char('h') if !key.modifiers.contains(KeyModifiers::SHIFT) => InputEvent::Help,
            KeyCode::Tab if key.modifiers.contains(KeyModifiers::SHIFT) => InputEvent::ShiftTab,
            KeyCode::Tab => InputEvent::Tab,
            KeyCode::Up | KeyCode::Char('k') => InputEvent::Up,
            KeyCode::Down | KeyCode::Char('j') => InputEvent::Down,
            KeyCode::Left => InputEvent::Left,
            KeyCode::Right => InputEvent::Right,
            KeyCode::Enter => InputEvent::Enter,
            KeyCode::Char(' ') => InputEvent::Space,
            KeyCode::Esc => InputEvent::Escape,
            KeyCode::Char('r') => InputEvent::Reset,
            KeyCode::Backspace | KeyCode::Char('b') => InputEvent::Back,
            KeyCode::Char(c) => InputEvent::Char(c),
            _ => InputEvent::Char('\0'),
        }
    }
}
