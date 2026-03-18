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

impl CommitInfo {
    pub fn to_text(&self, include_diff: bool) -> String {
        let mut text = format!("{}\n{}", self.message, self.author);

        if include_diff && !self.diff_summary.is_empty() {
            text.push('\n');
            text.push_str(&self.diff_summary);
        }

        text
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
}

