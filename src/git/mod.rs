mod diff;
mod error;
mod parser;

pub use error::GitError;
pub use parser::RepositoryParser;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitInfo {
    pub hash: String,
    pub author: String,
    pub date: DateTime<Utc>,
    pub message: String,
    pub diff_summary: String,
}

/// Prefix of the line in `diff_summary` that lists the commit's changed paths.
pub const FILES_PREFIX: &str = "Files: ";

impl CommitInfo {
    pub fn to_text(&self, include_diff: bool) -> String {
        let mut text = format!("{}\n{}", self.message, self.author);

        if include_diff && !self.diff_summary.is_empty() {
            text.push('\n');
            text.push_str(&self.diff_summary);
        }

        text
    }

    /// Paths this commit touched, as recorded by the diff extractor.
    ///
    /// Returns `None` for indexes built before paths were recorded, letting
    /// callers fall back rather than silently treating every commit as touching
    /// nothing.
    pub fn changed_files(&self) -> Option<Vec<&str>> {
        let line = self
            .diff_summary
            .lines()
            .next()?
            .strip_prefix(FILES_PREFIX)?;

        Some(
            line.split(", ")
                .map(str::trim)
                .filter(|p| !p.is_empty() && *p != "...")
                .collect(),
        )
    }

    /// Whether this commit touched a path containing `needle`.
    ///
    /// Prefers the recorded path list. On a legacy index with no path list it
    /// degrades to the old behaviour — substring search over the whole diff —
    /// which is imprecise but better than matching nothing.
    pub fn touches_path(&self, needle: &str) -> bool {
        match self.changed_files() {
            Some(paths) => paths.iter().any(|path| path.contains(needle)),
            None => self.diff_summary.contains(needle),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_commit() -> CommitInfo {
        CommitInfo {
            hash: "abc1234def5678".to_string(),
            author: "Alice Chen".to_string(),
            date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            message: "fix: resolve race condition in auth".to_string(),
            diff_summary: "+mutex.lock()\n-unsafe_access()".to_string(),
        }
    }

    #[test]
    fn test_to_text_without_diff() {
        let commit = sample_commit();
        let text = commit.to_text(false);
        assert_eq!(text, "fix: resolve race condition in auth\nAlice Chen");
        assert!(!text.contains("mutex"));
    }

    #[test]
    fn test_to_text_with_diff() {
        let commit = sample_commit();
        let text = commit.to_text(true);
        assert!(text.contains("fix: resolve race condition in auth"));
        assert!(text.contains("Alice Chen"));
        assert!(text.contains("+mutex.lock()"));
        assert!(text.contains("-unsafe_access()"));
    }

    #[test]
    fn test_to_text_with_diff_flag_but_empty_diff() {
        let mut commit = sample_commit();
        commit.diff_summary = String::new();
        let text = commit.to_text(true);
        // Should not have trailing newline when diff is empty
        assert_eq!(text, "fix: resolve race condition in auth\nAlice Chen");
    }

    #[test]
    fn test_to_text_includes_message_and_author() {
        let commit = sample_commit();
        let text = commit.to_text(false);
        assert!(text.starts_with(&commit.message));
        assert!(text.ends_with(&commit.author));
    }

    fn commit_with_summary(diff_summary: &str) -> CommitInfo {
        let mut commit = sample_commit();
        commit.diff_summary = diff_summary.to_string();
        commit
    }

    #[test]
    fn test_changed_files_parses_the_files_line() {
        let commit = commit_with_summary("Files: Cargo.toml, src/auth.rs\n+token");
        assert_eq!(
            commit.changed_files(),
            Some(vec!["Cargo.toml", "src/auth.rs"])
        );
    }

    #[test]
    fn test_changed_files_handles_a_single_path() {
        let commit = commit_with_summary("Files: src/main.rs\n+fn main() {}");
        assert_eq!(commit.changed_files(), Some(vec!["src/main.rs"]));
    }

    #[test]
    fn test_changed_files_drops_the_truncation_marker() {
        let commit = commit_with_summary("Files: a.rs, b.rs, ...\n+x");
        assert_eq!(commit.changed_files(), Some(vec!["a.rs", "b.rs"]));
    }

    #[test]
    fn test_changed_files_is_none_without_the_prefix() {
        let commit = commit_with_summary("+modified src/auth.rs");
        assert_eq!(
            commit.changed_files(),
            None,
            "a legacy summary must be distinguishable from one with no paths"
        );
    }

    #[test]
    fn test_changed_files_is_none_for_empty_summary() {
        assert_eq!(commit_with_summary("").changed_files(), None);
    }

    #[test]
    fn test_touches_path_matches_recorded_paths() {
        let commit = commit_with_summary("Files: src/auth.rs, src/main.rs\n+x");
        assert!(commit.touches_path("src/auth.rs"));
        assert!(commit.touches_path("auth"));
        assert!(commit.touches_path("src/"));
        assert!(!commit.touches_path("src/index.rs"));
    }

    #[test]
    fn test_touches_path_ignores_diff_body_mentions() {
        let commit = commit_with_summary("Files: src/main.rs\n-use src/auth.rs::Token;");
        assert!(
            !commit.touches_path("src/auth.rs"),
            "a path mentioned in the diff body is not a changed file"
        );
    }

    #[test]
    fn test_touches_path_falls_back_for_legacy_summaries() {
        let commit = commit_with_summary("+modified src/auth.rs");
        assert!(
            commit.touches_path("src/auth.rs"),
            "legacy indexes keep the old substring behaviour"
        );
    }

    #[test]
    fn test_to_text_includes_changed_paths_when_diffs_are_indexed() {
        let commit = commit_with_summary("Files: src/auth.rs\n+token");
        assert!(
            commit.to_text(true).contains("src/auth.rs"),
            "paths should reach the embedded text too"
        );
    }
}
