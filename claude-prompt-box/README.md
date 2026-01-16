# MyCLI - Interactive CLI with Fuzzy File Search

A Rust-based interactive CLI application (TUI-like) that provides a single-line input box (REPL) with fuzzy file search functionality. When you type `@`, it triggers an inline fuzzy file picker that allows you to quickly insert file paths into your input.

## Features

- **Interactive Input**: Single-line editable input with cursor navigation and command history
- **Fuzzy File Search**: Type `@` to trigger an inline file picker with fuzzy search
- **Smart Token Detection**: Distinguishes between file mentions (`@src/main.rs`) and email addresses (`user@example.com`)
- **Fast File Indexing**: Efficiently scans and indexes files in the current directory and subdirectories
- **Customizable Ignore Rules**: Built-in ignore patterns for common directories and file types
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd claude-prompt-box
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Run the application:
   ```bash
   ./target/release/mycli
   ```

## Usage

### Basic Operation

Start the interactive CLI:
```bash
mycli
```

You'll see a prompt where you can type commands:
```
> hello world
```

### File Search with @ Mentions

Type `@` to trigger the fuzzy file picker:
```
> please check @
```

The picker will appear showing files in your project. You can:
- Type to filter files (e.g., `@src` to find files containing "src")
- Use ↑/↓ arrow keys to navigate
- Press Enter to select a file
- Press Esc to cancel

After selection, the file path is inserted:
```
> please check @src/main.rs
```

### Command Line Options

- `--example-files`: Create example files for testing
- `--example-prompt "text"`: Start with a pre-filled prompt

Example:
```bash
# Create example project structure for testing
mycli --example-files

# Start with a prompt containing @ mentions
mycli --example-prompt "Review @src and @tests"
```

## Keybindings

### Input Mode
- **Left/Right arrows**: Move cursor
- **Home/End**: Jump to start/end of line
- **Up/Down arrows**: Navigate command history
- **Backspace/Delete**: Delete characters
- **Enter**: Submit command
- **Ctrl+C**: Copy textbox content to clipboard
- **Ctrl+Q**: Quit application
- **Tab**: Alternative trigger for file picker (when in @ token)

### File Picker Mode
- **Up/Down arrows**: Navigate file list
- **Enter**: Select highlighted file
- **Esc**: Cancel and return to input
- **Type**: Filter files by name

## Special Commands

- `exit`: Quit the application
- `:refresh`: Rebuild the file index

## How @ Token Detection Works

The application intelligently detects `@` tokens:

✅ **Will trigger file picker:**
- `@` (at start of input)
- `hello @src` (after whitespace)
- `check @main.rs` (at word boundary)

❌ **Will NOT trigger file picker:**
- `user@example.com` (email addresses)
- `variable@domain` (when @ is not at word start)

## Ignore Rules

By default, the following directories and files are excluded from the file index:

### Directories
- `.git/`, `target/`, `node_modules/`
- `dist/`, `build/`, `.next/`, `.turbo/`
- `.cache/`, `.venv/`, `venv/`
- `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`
- `.idea/`, `.vscode/`, `coverage/`

### File Extensions
- **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.svg`
- **Videos**: `.mp4`, `.mov`, `.mkv`
- **Audio**: `.mp3`, `.wav`, `.flac`
- **Archives**: `.zip`, `.tar`, `.gz`, `.7z`

### Files
- `.DS_Store`

**Note**: Package lock files like `package-lock.json`, `pnpm-lock.yaml`, and `yarn.lock` are included by default.

### Customizing Ignore Rules

To modify the ignore rules, edit the constants in `src/index/scanner.rs`:

```rust
pub const DEFAULT_IGNORE_DIRS: &[&str] = &[
    ".git", "target", "node_modules", 
    // Add your custom directories here
];

pub const DEFAULT_IGNORE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", 
    // Add your custom extensions here
];

pub const DEFAULT_IGNORE_FILES: &[&str] = &[
    ".DS_Store",
    // Add your custom files here
];
```

## Architecture

The project is organized into modular components:

- **`src/app/`**: Application state, UI rendering, and event handling
- **`src/parsing/`**: Token parsing for @ mentions
- **`src/index/`**: File scanning and indexing
- **`src/fuzzy/`**: Fuzzy matching with substring search
- **`src/main.rs`**: Application entry point and terminal setup

## Dependencies

- **ratatui**: Terminal UI framework
- **crossterm**: Cross-platform terminal manipulation
- **walkdir**: Recursive directory traversal
- **anyhow**: Error handling
- **clap**: Command line argument parsing

## Testing

Run the test suite:
```bash
cargo test
```

The tests cover:
- Token parsing for various @ mention scenarios
- File scanning and ignore rule application
- Fuzzy matching functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).
