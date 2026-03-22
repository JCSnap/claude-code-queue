use std::path::{Path, PathBuf};
use ignore::WalkBuilder;

/// Default directories to ignore during scanning (applied on top of .gitignore)
pub const DEFAULT_IGNORE_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".turbo",
    ".cache",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "coverage",
];

/// Default file extensions to ignore (binary/media files)
pub const DEFAULT_IGNORE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "svg",
    "mp4", "mov", "mkv",
    "mp3", "wav", "flac",
    "zip", "tar", "gz", "7z",
];

/// Default files to ignore
pub const DEFAULT_IGNORE_FILES: &[&str] = &[
    ".DS_Store",
];

#[derive(Debug, Clone)]
pub struct ScanOptions {
    pub ignore_dirs: Vec<String>,
    pub ignore_extensions: Vec<String>,
    pub ignore_files: Vec<String>,
    /// Whether to respect .gitignore files (default: true)
    pub respect_gitignore: bool,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            ignore_dirs: DEFAULT_IGNORE_DIRS.iter().map(|s| s.to_string()).collect(),
            ignore_extensions: DEFAULT_IGNORE_EXTENSIONS.iter().map(|s| s.to_string()).collect(),
            ignore_files: DEFAULT_IGNORE_FILES.iter().map(|s| s.to_string()).collect(),
            respect_gitignore: true,
        }
    }
}

/// Scan files in the given directory recursively, respecting .gitignore and custom ignore rules
pub fn scan_files(cwd: &Path, options: &ScanOptions) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    let walker = WalkBuilder::new(cwd)
        .git_ignore(options.respect_gitignore)
        .git_global(options.respect_gitignore)
        .git_exclude(options.respect_gitignore)
        .hidden(false) // don't skip all hidden files — let .gitignore decide
        .follow_links(false)
        .build();

    for entry in walker {
        let entry = entry?;
        let path = entry.path();

        // Skip the root directory itself
        if path == cwd {
            continue;
        }

        // Skip ignored directories by name (additional hardcoded rules)
        if path.is_dir() {
            continue;
        }

        // Check hardcoded directory ignore list against all path components
        if has_ignored_ancestor(path, cwd, options) {
            continue;
        }

        // Only include files
        if !path.is_file() {
            continue;
        }

        // Check if file should be ignored by extension/name
        if should_ignore_file(path, options) {
            continue;
        }

        // Convert to relative path with forward slashes
        let relative_path = normalize_path(path.strip_prefix(cwd).unwrap_or(path));
        files.push(relative_path);
    }

    Ok(files)
}

/// Check if any ancestor directory of the path is in the ignore list
fn has_ignored_ancestor(path: &Path, cwd: &Path, options: &ScanOptions) -> bool {
    let relative = path.strip_prefix(cwd).unwrap_or(path);
    for component in relative.components() {
        if let std::path::Component::Normal(name) = component {
            if let Some(name_str) = name.to_str() {
                if options.ignore_dirs.iter().any(|ignore| ignore == name_str) {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if a file should be ignored based on the options
fn should_ignore_file(path: &Path, options: &ScanOptions) -> bool {
    // Check filename
    if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
        if options.ignore_files.iter().any(|ignore| ignore == filename) {
            return true;
        }
    }

    // Check extension
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        if options.ignore_extensions.iter().any(|ignore| ignore == ext) {
            return true;
        }
    }

    false
}

/// Normalize path to use forward slashes and be relative
fn normalize_path(path: &Path) -> PathBuf {
    let path_str = path.to_string_lossy();

    // Replace backslashes with forward slashes for cross-platform consistency
    let normalized = path_str.replace('\\', "/");

    PathBuf::from(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_should_ignore_file() {
        let options = ScanOptions::default();

        // Should ignore binary files
        assert!(should_ignore_file(Path::new("image.png"), &options));
        assert!(should_ignore_file(Path::new("video.mp4"), &options));
        assert!(should_ignore_file(Path::new("archive.zip"), &options));

        // Should not ignore source files
        assert!(!should_ignore_file(Path::new("main.rs"), &options));
        assert!(!should_ignore_file(Path::new("package.json"), &options));
        assert!(!should_ignore_file(Path::new("package-lock.json"), &options));

        // Should ignore specific files
        assert!(should_ignore_file(Path::new(".DS_Store"), &options));
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path(Path::new("src/main.rs")), PathBuf::from("src/main.rs"));
        assert_eq!(normalize_path(Path::new("src\\main.rs")), PathBuf::from("src/main.rs"));
        assert_eq!(normalize_path(Path::new("foo")), PathBuf::from("foo"));
    }

    #[test]
    fn test_scan_files() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();

        // Create test file structure
        fs::create_dir_all(temp_path.join("src"))?;
        fs::create_dir_all(temp_path.join("node_modules/pkg"))?;
        fs::create_dir_all(temp_path.join(".git"))?;

        fs::write(temp_path.join("src/main.rs"), "fn main() {}")?;
        fs::write(temp_path.join("src/lib.rs"), "pub mod test;")?;
        fs::write(temp_path.join("package.json"), "{}")?;
        fs::write(temp_path.join("image.png"), "fake image")?;
        fs::write(temp_path.join(".DS_Store"), "metadata")?;
        fs::write(temp_path.join("node_modules/pkg/index.js"), "module.exports = {};")?;
        fs::write(temp_path.join(".git/config"), "[core]")?;

        let options = ScanOptions::default();
        let files = scan_files(temp_path, &options)?;

        // Should include source files but exclude ignored ones
        let file_names: Vec<String> = files.iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        assert!(file_names.contains(&"src/main.rs".to_string()));
        assert!(file_names.contains(&"src/lib.rs".to_string()));
        assert!(file_names.contains(&"package.json".to_string()));

        // Should exclude ignored files and directories
        assert!(!file_names.iter().any(|f| f.contains("node_modules")));
        assert!(!file_names.iter().any(|f| f.contains(".git")));
        assert!(!file_names.iter().any(|f| f.contains("image.png")));
        assert!(!file_names.iter().any(|f| f.contains(".DS_Store")));

        Ok(())
    }

    #[test]
    fn test_scan_respects_gitignore() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();

        // Init a git repo so .gitignore is respected
        fs::create_dir_all(temp_path.join(".git"))?;

        // Create .gitignore
        fs::write(temp_path.join(".gitignore"), ".worktrees/\nignored_dir/\n")?;

        // Create files
        fs::create_dir_all(temp_path.join("src"))?;
        fs::create_dir_all(temp_path.join(".worktrees/feature-branch/src"))?;
        fs::create_dir_all(temp_path.join("ignored_dir"))?;

        fs::write(temp_path.join("src/main.rs"), "fn main() {}")?;
        fs::write(temp_path.join(".worktrees/feature-branch/src/main.rs"), "fn main() {}")?;
        fs::write(temp_path.join("ignored_dir/file.txt"), "ignored")?;

        let options = ScanOptions::default();
        let files = scan_files(temp_path, &options)?;

        let file_names: Vec<String> = files.iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        // Should include non-ignored files
        assert!(file_names.contains(&"src/main.rs".to_string()));

        // Should exclude gitignored directories
        assert!(!file_names.iter().any(|f| f.contains(".worktrees")));
        assert!(!file_names.iter().any(|f| f.contains("ignored_dir")));

        Ok(())
    }
}
