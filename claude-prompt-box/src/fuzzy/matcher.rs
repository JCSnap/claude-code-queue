use std::path::PathBuf;

pub struct FuzzyMatcher {
    candidates: Vec<PathBuf>,
}

impl FuzzyMatcher {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    /// Update the candidates list
    pub fn update_candidates(&mut self, candidates: Vec<PathBuf>) {
        self.candidates = candidates;
    }

    /// Perform fuzzy search and return top K results with scores
    pub fn search(&mut self, query: &str, limit: usize) -> Vec<(PathBuf, f64)> {
        if query.is_empty() {
            // Return all candidates if no query
            return self.candidates
                .iter()
                .take(limit)
                .map(|p| (p.clone(), 1.0))
                .collect();
        }

        // Simple substring matching for now (can be improved with proper fuzzy matching later)
        let query_lower = query.to_lowercase();
        let mut matches: Vec<(PathBuf, f64)> = self.candidates
            .iter()
            .filter_map(|path| {
                let path_str = path.to_string_lossy().to_lowercase();
                if path_str.contains(&query_lower) {
                    // Simple scoring: prefer exact matches and shorter paths
                    let score = if path_str == query_lower {
                        1.0
                    } else if path_str.starts_with(&query_lower) {
                        0.9
                    } else {
                        0.5
                    };
                    Some((path.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score (highest first)
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top K results
        matches.into_iter().take(limit).collect()
    }
}

impl Default for FuzzyMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_fuzzy_matcher_empty_query() {
        let mut matcher = FuzzyMatcher::new();
        let candidates = vec![
            PathBuf::from("src/main.rs"),
            PathBuf::from("src/lib.rs"),
            PathBuf::from("tests/test.rs"),
        ];
        
        matcher.update_candidates(candidates.clone());
        let results = matcher.search("", 10);
        
        // Should return all candidates when query is empty
        assert_eq!(results.len(), candidates.len());
    }

    #[test]
    fn test_fuzzy_matcher_with_query() {
        let mut matcher = FuzzyMatcher::new();
        let candidates = vec![
            PathBuf::from("src/main.rs"),
            PathBuf::from("src/lib.rs"),
            PathBuf::from("tests/test.rs"),
            PathBuf::from("docs/readme.md"),
        ];
        
        matcher.update_candidates(candidates);
        let results = matcher.search("src", 10);
        
        // Should find files containing "src"
        assert!(!results.is_empty());
        assert!(results.iter().any(|(path, _)| path.to_string_lossy().contains("src")));
    }

    #[test]
    fn test_fuzzy_matcher_limit() {
        let mut matcher = FuzzyMatcher::new();
        let candidates = vec![
            PathBuf::from("file1.rs"),
            PathBuf::from("file2.rs"),
            PathBuf::from("file3.rs"),
            PathBuf::from("file4.rs"),
        ];
        
        matcher.update_candidates(candidates);
        let results = matcher.search("file", 2);
        
        // Should respect the limit
        assert!(results.len() <= 2);
    }
}