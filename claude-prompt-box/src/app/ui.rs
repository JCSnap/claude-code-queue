use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use crate::app::{AppState, Mode};

pub fn render(f: &mut Frame, app: &mut AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),     // Main area
            Constraint::Length(1),  // Status line
        ])
        .split(f.area());

    match app.mode {
        Mode::Input => {
            render_input_mode(f, app, chunks[0]);
        }
        Mode::Picker => {
            render_input_mode(f, app, chunks[0]);
            render_picker_overlay(f, app, chunks[0]);
        }
    }

    render_status_line(f, app, chunks[1]);
}

fn render_input_mode(f: &mut Frame, app: &AppState, area: Rect) {
    let block = Block::default()
        .title("Interactive CLI")
        .borders(Borders::ALL);
    
    let input_area = block.inner(area);
    f.render_widget(block, area);

    let prompt = "> ";
    let full_text = format!("{}{}", prompt, app.input);
    let cursor_pos = prompt.len() + app.cursor;
    
    let mut spans = Vec::new();
    
    if app.mode == Mode::Input && cursor_pos <= full_text.len() {
        if cursor_pos > 0 {
            spans.push(Span::raw(&full_text[..cursor_pos]));
        }
        
        let cursor_char = if cursor_pos < full_text.len() {
            full_text.chars().nth(cursor_pos).unwrap_or(' ').to_string()
        } else {
            " ".to_string()
        };
        spans.push(Span::styled(
            cursor_char,
            Style::default().bg(Color::White).fg(Color::Black)
        ));
        
        // after cursor
        if cursor_pos + 1 < full_text.len() {
            spans.push(Span::raw(&full_text[cursor_pos + 1..]));
        }
    } else {
        // no cursor highlighting
        spans.push(Span::raw(&full_text));
        if app.mode == Mode::Input {
            spans.push(Span::styled(
                " ",
                Style::default().bg(Color::White).fg(Color::Black)
            ));
        }
    }

    let line = Line::from(spans);
    let text = Text::from(line);
    let paragraph = Paragraph::new(text)
        .wrap(Wrap { trim: false });
    
    f.render_widget(paragraph, input_area);
}

fn render_picker_overlay(f: &mut Frame, app: &AppState, area: Rect) {
    // calculate overlay size (80% width, 60% height, centered)
    let overlay_width = (area.width as f32 * 0.8) as u16;
    let overlay_height = (area.height as f32 * 0.6) as u16;
    
    let x = (area.width.saturating_sub(overlay_width)) / 2;
    let y = (area.height.saturating_sub(overlay_height)) / 2;
    
    let overlay_area = Rect {
        x: area.x + x,
        y: area.y + y,
        width: overlay_width,
        height: overlay_height,
    };

    f.render_widget(Clear, overlay_area);

    // picker block
    let block = Block::default()
        .title(format!("Add context: @{}", app.picker_state.query))
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::Black));

    let picker_area = block.inner(overlay_area);
    f.render_widget(block, overlay_area);

    // file list
    if !app.picker_state.results.is_empty() {
        let items: Vec<ListItem> = app.picker_state.results
            .iter()
            .enumerate()
            .map(|(i, (path, _score))| {
                let style = if i == app.picker_state.selected_idx {
                    Style::default()
                        .bg(Color::Blue)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                
                ListItem::new(Line::from(Span::styled(
                    path.to_string_lossy().to_string(),
                    style,
                )))
            })
            .collect();

        let mut list_state = ListState::default();
        list_state.select(Some(app.picker_state.selected_idx));

        let list = List::new(items)
            .highlight_style(
                Style::default()
                    .bg(Color::Blue)
                    .add_modifier(Modifier::BOLD)
            );

        f.render_stateful_widget(list, picker_area, &mut list_state);
    } else {
        let no_results = Paragraph::new("No files found")
            .style(Style::default().fg(Color::Gray));
        f.render_widget(no_results, picker_area);
    }
}

fn render_status_line(f: &mut Frame, app: &AppState, area: Rect) {
    let status_text = if let Some(ref message) = app.status_message {
        message.clone()
    } else if app.is_indexing {
        format!("Indexing… ({} files)", app.index.len())
    } else {
        match app.mode {
            Mode::Input => "Ready - Type @ to search files | Ctrl+C to copy | Ctrl+Q to quit".to_string(),
            Mode::Picker => "Select file: ↑↓ to navigate, Enter to select, Esc to cancel".to_string(),
        }
    };

    let status = Paragraph::new(status_text)
        .style(Style::default().fg(Color::Gray));
    
    f.render_widget(status, area);
}
