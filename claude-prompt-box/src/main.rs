use std::io;
use std::env;
use std::fs;
use clap::{Arg, Command};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use prompt_box::app::{AppState, events::handle_key_event, ui::render};
use prompt_box::index::scanner::{scan_files, ScanOptions};
use prompt_box::fuzzy::matcher::FuzzyMatcher;

fn main() -> anyhow::Result<()> {
    let matches = Command::new("prompt-box")
        .version("0.1.0")
        .about("Interactive CLI with fuzzy file search")
        .arg(
            Arg::new("example-files")
                .long("example-files")
                .help("Create example files for testing")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("example-prompt")
                .long("example-prompt")
                .help("Start with an example prompt containing @ mentions")
                .value_name("PROMPT")
                .action(clap::ArgAction::Set),
        )
        .get_matches();

    // Create example files if requested
    if matches.get_flag("example-files") {
        create_example_files()?;
        println!("Created example files for testing");
        return Ok(());
    }

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create cleanup guard to ensure terminal is restored on panic
    let _cleanup = CleanupGuard;

    // Initialize application state
    let mut app = AppState::new();
    let mut fuzzy_matcher = FuzzyMatcher::new();

    // Set example prompt if provided
    if let Some(prompt) = matches.get_one::<String>("example-prompt") {
        app.input = prompt.clone();
        app.cursor = prompt.len();
    }

    // Build file index
    let cwd = env::current_dir()?;
    app.is_indexing = true;
    let scan_options = ScanOptions::default();
    let files = scan_files(&cwd, &scan_options)?;
    app.index = files;
    app.is_indexing = false;

    // Main loop
    loop {
        terminal.draw(|f| render(f, &mut app))?;

        if app.should_quit {
            break;
        }

        if let Event::Key(key) = event::read()? {
            handle_key_event(&mut app, key, &mut fuzzy_matcher);
        }
    }

    // Cleanup
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

fn create_example_files() -> anyhow::Result<()> {
    let dirs = vec![
        "example_project/src",
        "example_project/tests", 
        "example_project/docs",
        "example_project/config",
    ];

    for dir in dirs {
        fs::create_dir_all(dir)?;
    }

    let files = vec![
        ("example_project/src/main.rs", "fn main() {\n    println!(\"Hello, world!\");\n}"),
        ("example_project/src/lib.rs", "pub mod utils;\npub mod parser;"),
        ("example_project/src/utils.rs", "pub fn helper() -> String {\n    \"helper\".to_string()\n}"),
        ("example_project/src/parser.rs", "pub fn parse(input: &str) -> Vec<String> {\n    input.split_whitespace().map(|s| s.to_string()).collect()\n}"),
        ("example_project/tests/integration_test.rs", "#[test]\nfn test_integration() {\n    assert_eq!(1 + 1, 2);\n}"),
        ("example_project/docs/README.md", "# Example Project\n\nThis is an example project for testing the CLI."),
        ("example_project/config/settings.toml", "[app]\nname = \"example\"\nversion = \"0.1.0\""),
        ("example_project/Cargo.toml", "[package]\nname = \"example_project\"\nversion = \"0.1.0\"\nedition = \"2021\""),
    ];

    for (path, content) in files {
        fs::write(path, content)?;
    }

    Ok(())
}

struct CleanupGuard;

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        // Ensure terminal is cleaned up even on panic
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
    }
}
