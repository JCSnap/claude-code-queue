use std::path::PathBuf;
use std::time::SystemTime;
use clipboard::{ClipboardContext, ClipboardProvider};

#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    Input,
    Picker,
}

#[derive(Debug, Clone)]
pub struct PickerState {
    pub query: String,
    pub results: Vec<(PathBuf, f64)>, // (path, score)
    pub selected_idx: usize,
    pub scroll_offset: usize,
    pub token_range_to_replace: Option<(usize, usize)>, // (start, end) in input string
}

impl Default for PickerState {
    fn default() -> Self {
        Self {
            query: String::new(),
            results: Vec::new(),
            selected_idx: 0,
            scroll_offset: 0,
            token_range_to_replace: None,
        }
    }
}

#[derive(Debug)]
pub struct AppState {
    pub mode: Mode,
    pub input: String,
    pub cursor: usize,
    pub history: Vec<String>,
    pub history_index: Option<usize>,
    pub index: Vec<PathBuf>,
    pub picker_state: PickerState,
    pub last_index_build_time: Option<SystemTime>,
    pub should_quit: bool,
    pub status_message: Option<String>,
    pub is_indexing: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            mode: Mode::Input,
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: None,
            index: Vec::new(),
            picker_state: PickerState::default(),
            last_index_build_time: None,
            should_quit: false,
            status_message: None,
            is_indexing: false,
        }
    }
}

impl AppState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter_picker_mode(&mut self, query: String, token_range: Option<(usize, usize)>) {
        self.mode = Mode::Picker;
        self.picker_state = PickerState {
            query,
            results: Vec::new(),
            selected_idx: 0,
            scroll_offset: 0,
            token_range_to_replace: token_range,
        };
    }

    pub fn exit_picker_mode(&mut self) {
        self.mode = Mode::Input;
        self.picker_state = PickerState::default();
    }

    pub fn insert_char(&mut self, c: char) {
        if self.cursor <= self.input.len() {
            self.input.insert(self.cursor, c);
            self.cursor += c.len_utf8();
        }
    }

    pub fn delete_char(&mut self) {
        if self.cursor > 0 && !self.input.is_empty() {
            let prev_char_boundary = self.find_prev_char_boundary(self.cursor);
            self.input.remove(prev_char_boundary);
            self.cursor = prev_char_boundary;
        }
    }

    pub fn delete_char_forward(&mut self) {
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.find_prev_char_boundary(self.cursor);
        }
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor < self.input.len() {
            self.cursor = self.find_next_char_boundary(self.cursor);
        }
    }

    pub fn move_cursor_home(&mut self) {
        self.cursor = 0;
    }

    pub fn move_cursor_end(&mut self) {
        self.cursor = self.input.len();
    }

    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        
        let new_index = match self.history_index {
            None => Some(self.history.len() - 1),
            Some(0) => Some(0), // Stay at first item
            Some(idx) => Some(idx - 1),
        };
        
        if let Some(idx) = new_index {
            self.history_index = Some(idx);
            self.input = self.history[idx].clone();
            self.cursor = self.input.len();
        }
    }

    pub fn history_next(&mut self) {
        if let Some(idx) = self.history_index {
            if idx >= self.history.len() - 1 {
                self.history_index = None;
                self.input.clear();
                self.cursor = 0;
            } else {
                self.history_index = Some(idx + 1);
                self.input = self.history[idx + 1].clone();
                self.cursor = self.input.len();
            }
        }
    }

    pub fn submit_input(&mut self) {
        if !self.input.trim().is_empty() {
            if self.input.trim() == "exit" {
                self.should_quit = true;
                return;
            }
            
            if self.input.trim() == ":refresh" {
                self.status_message = Some("Refreshing file index...".to_string());
                return;
            }
            
            if self.history.last() != Some(&self.input) {
                self.history.push(self.input.clone());
            }
            
            self.status_message = Some(format!("Executed: {}", self.input));
            
            self.input.clear();
            self.cursor = 0;
            self.history_index = None;
        }
    }

    pub fn replace_token(&mut self, replacement: &str) {
        if let Some((start, end)) = self.picker_state.token_range_to_replace {
            self.input.replace_range(start..end, replacement);
            self.cursor = start + replacement.len();
        } else {
            self.insert_string(replacement);
        }
    }

    pub fn copy_to_clipboard(&mut self) {
        let result = clipboard::ClipboardContext::new()
            .and_then(|mut ctx: clipboard::ClipboardContext| {
                ctx.set_contents(self.input.clone())
            });
            
        match result {
            Ok(_) => {
                self.status_message = Some("Copied to clipboard!".to_string());
            }
            Err(_) => {
                self.status_message = Some("Failed to copy to clipboard".to_string());
            }
        }
    }

    fn insert_string(&mut self, s: &str) {
        if self.cursor <= self.input.len() {
            self.input.insert_str(self.cursor, s);
            self.cursor += s.len();
        }
    }

    fn find_prev_char_boundary(&self, pos: usize) -> usize {
        self.input.char_indices()
            .rev()
            .find(|(idx, _)| *idx < pos)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn find_next_char_boundary(&self, pos: usize) -> usize {
        self.input.char_indices()
            .find(|(idx, _)| *idx > pos)
            .map(|(idx, _)| idx)
            .unwrap_or(self.input.len())
    }
}
