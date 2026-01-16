use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Default directories to ignore during scanning
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
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            ignore_dirs: DEFAULT_IGNORE_DIRS.iter().map(|s| s.to_string()).collect(),
            ignore_extensions: DEFAULT_IGNORE_EXTENSIONS.iter().map(|s| s.to_string()).collect(),
            ignore_files: DEFAULT_IGNORE_FILES.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Scan files in the given directory recursively, applying ignore rules
pub fn scan_files(cwd: &Path, options: &ScanOptions) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    
    let walker = WalkDir::new(cwd)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            let path = entry.path();
            
            // Skip ignored directories
            if path.is_dir() {
                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    if options.ignore_dirs.iter().any(|ignore| ignore == dir_name) {
                        return false;
                    }
                }
            }
            
            true
        });
    
    for entry in walker {
        let entry = entry?;
        let path = entry.path();
        
        // Only include files, not directories
        if !path.is_file() {
            continue;
        }
        
        // Check if file should be ignored
        if should_ignore_file(path, options) {
            continue;
        }
        
        // Convert to relative path with forward slashes
        let relative_path = normalize_path(path.strip_prefix(cwd).unwrap_or(path));
        files.push(relative_path);
    }
    
    Ok(files)
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
}