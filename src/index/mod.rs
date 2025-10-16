mod builder;
mod storage;

pub use builder::IndexBuilder;
pub use storage::IndexStorage;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::git::CommitInfo;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub commit: CommitInfo,
    pub embedding: Vec<f32>, // Serializable version of ndarray
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub total_commits: usize,
    pub include_diffs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticIndex {
    pub entries: Vec<IndexEntry>,
    pub model_version: String,
    pub last_commit: String,
    pub metadata: IndexMetadata,
}

impl SemanticIndex {
    pub fn new(model_version: String, last_commit: String) -> Self {
        let now = Utc::now();
        Self {
            entries: Vec::new(),
            model_version,
            last_commit,
            metadata: IndexMetadata {
                created_at: now,
                updated_at: now,
                total_commits: 0,
                include_diffs: true,
            },
        }
    }
}

