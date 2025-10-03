mod diff;
mod parser;

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
            text.push_str("\n");
            text.push_str(&self.diff_summary);
        }

        text
    }
}

