mod ann;
mod builder;
mod error;
mod storage;

pub use ann::{AnnSidecar, EXACT_SCAN_THRESHOLD, build_graph};
pub use builder::IndexBuilder;
pub use error::IndexError;
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
    pub fn new(model_version: String, last_commit: String, include_diffs: bool) -> Self {
        let now = Utc::now();
        Self {
            entries: Vec::new(),
            model_version,
            last_commit,
            metadata: IndexMetadata {
                created_at: now,
                updated_at: now,
                total_commits: 0,
                include_diffs,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::CommitInfo;

    #[test]
    fn test_semantic_index_new() {
        let index =
            SemanticIndex::new("bge-small-en-v1.5".to_string(), "abc1234".to_string(), true);
        assert_eq!(index.model_version, "bge-small-en-v1.5");
        assert_eq!(index.last_commit, "abc1234");
        assert!(index.entries.is_empty());
        assert_eq!(index.metadata.total_commits, 0);
        assert!(index.metadata.include_diffs);
    }

    #[test]
    fn test_semantic_index_quick_mode() {
        let index = SemanticIndex::new(
            "bge-small-en-v1.5".to_string(),
            "abc1234".to_string(),
            false,
        );
        assert!(!index.metadata.include_diffs);
    }

    #[test]
    fn test_semantic_index_timestamps_are_set() {
        let before = Utc::now();
        let index = SemanticIndex::new("model".to_string(), "hash".to_string(), true);
        let after = Utc::now();
        assert!(index.metadata.created_at >= before && index.metadata.created_at <= after);
        assert!(index.metadata.updated_at >= before && index.metadata.updated_at <= after);
    }

    #[test]
    fn test_index_entry_serialization_roundtrip() {
        let entry = IndexEntry {
            commit: CommitInfo {
                hash: "abc1234".to_string(),
                author: "Alice".to_string(),
                date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
                message: "test commit".to_string(),
                diff_summary: "+added line".to_string(),
            },
            embedding: vec![0.1, 0.2, 0.3],
        };

        let mut index = SemanticIndex::new("model".to_string(), "abc1234".to_string(), true);
        index.entries.push(entry);
        index.metadata.total_commits = 1;

        let serialized = bincode::serialize(&index).unwrap();
        let deserialized: SemanticIndex = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.entries.len(), 1);
        assert_eq!(deserialized.entries[0].commit.hash, "abc1234");
        assert_eq!(deserialized.entries[0].commit.author, "Alice");
        assert_eq!(deserialized.entries[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(deserialized.model_version, "model");
        assert_eq!(deserialized.metadata.total_commits, 1);
        assert!(deserialized.metadata.include_diffs);
    }
}
