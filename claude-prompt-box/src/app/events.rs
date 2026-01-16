use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use crate::app::{AppState, Mode};
use crate::parsing::token::find_active_at_token;
use crate::fuzzy::matcher::FuzzyMatcher;

pub fn handle_key_event(
    app: &mut AppState,
    key: KeyEvent,
    fuzzy_matcher: &mut FuzzyMatcher,
) {
    match app.mode {
        Mode::Input => handle_input_mode(app, key, fuzzy_matcher),
        Mode::Picker => handle_picker_mode(app, key, fuzzy_matcher),
    }
}

fn handle_input_mode(
    app: &mut AppState,
    key: KeyEvent,
    fuzzy_matcher: &mut FuzzyMatcher,
) {
    match key.code {
        KeyCode::Char(c) => {
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                match c {
                    'c' => app.copy_to_clipboard(),
                    'q' => app.should_quit = true,
                    _ => {}
                }
            } else {
                app.insert_char(c);
                app.status_message = None;
                
                if c == '@' || is_in_at_token(app) {
                    trigger_picker_if_needed(app, fuzzy_matcher);
                }
            }
        }
        KeyCode::Backspace => {
            app.delete_char();
            app.status_message = None;
            
            if is_in_at_token(app) {
                trigger_picker_if_needed(app, fuzzy_matcher);
            }
        }
        KeyCode::Delete => {
            app.delete_char_forward();
            app.status_message = None;
        }
        KeyCode::Left => {
            app.move_cursor_left();
        }
        KeyCode::Right => {
            app.move_cursor_right();
        }
        KeyCode::Home => {
            app.move_cursor_home();
        }
        KeyCode::End => {
            app.move_cursor_end();
        }
        KeyCode::Up => {
            app.history_prev();
        }
        KeyCode::Down => {
            app.history_next();
        }
        KeyCode::Enter => {
            app.submit_input();
        }
        KeyCode::Tab => {
            if is_in_at_token(app) {
                trigger_picker_if_needed(app, fuzzy_matcher);
            }
        }
        _ => {}
    }
}

fn handle_picker_mode(
    app: &mut AppState,
    key: KeyEvent,
    fuzzy_matcher: &mut FuzzyMatcher,
) {
    match key.code {
        KeyCode::Esc => {
            app.exit_picker_mode();
        }
        KeyCode::Enter => {
            if !app.picker_state.results.is_empty() && 
               app.picker_state.selected_idx < app.picker_state.results.len() {
                let selected_path = &app.picker_state.results[app.picker_state.selected_idx].0;
                let path_str = format!("@{}", selected_path.to_string_lossy());
                app.replace_token(&path_str);
                app.exit_picker_mode();
                app.status_message = Some(format!("Added: {}", path_str));
            }
        }
        KeyCode::Up => {
            if app.picker_state.selected_idx > 0 {
                app.picker_state.selected_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.picker_state.selected_idx + 1 < app.picker_state.results.len() {
                app.picker_state.selected_idx += 1;
            }
        }
        KeyCode::Char(c) => {
            // update the query in the input and search
            app.insert_char(c);
            update_picker_search(app, fuzzy_matcher);
        }
        KeyCode::Backspace => {
            app.delete_char();
            update_picker_search(app, fuzzy_matcher);
        }
        _ => {}
    }
}

fn is_in_at_token(app: &AppState) -> bool {
    find_active_at_token(&app.input, app.cursor).is_some()
}

fn trigger_picker_if_needed(app: &mut AppState, fuzzy_matcher: &mut FuzzyMatcher) {
    if let Some(token) = find_active_at_token(&app.input, app.cursor) {
        app.enter_picker_mode(token.query.clone(), Some(token.range));
        update_picker_search(app, fuzzy_matcher);
    }
}

fn update_picker_search(app: &mut AppState, fuzzy_matcher: &mut FuzzyMatcher) {
    if let Some(token) = find_active_at_token(&app.input, app.cursor) {
        app.picker_state.query = token.query.clone();
        app.picker_state.token_range_to_replace = Some(token.range);
        
        fuzzy_matcher.update_candidates(app.index.clone());
        app.picker_state.results = fuzzy_matcher.search(&app.picker_state.query, 100);
        app.picker_state.selected_idx = 0;
    }
}
