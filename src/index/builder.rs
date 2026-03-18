use chrono::Utc;
use tracing::debug;

use crate::embedding::ModelManager;
use crate::git::CommitInfo;

use super::{IndexEntry, IndexError, SemanticIndex};

pub struct IndexBuilder {
    entries: Vec<IndexEntry>,
    model_manager: ModelManager,
    model_version: String,
    last_commit: Option<String>,
    include_diffs: bool,
    created_at: Option<chrono::DateTime<Utc>>,
}

impl IndexBuilder {
    pub fn new(mut model_manager: ModelManager, include_diffs: bool) -> Result<Self, IndexError> {
        model_manager.init()?;
        let model_version = model_manager.model_version();

        Ok(Self {
            entries: Vec::new(),
            model_manager,
            model_version,
            last_commit: None,
            include_diffs,
            created_at: None,
        })
    }

    pub fn from_existing(index: SemanticIndex, mut model_manager: ModelManager) -> Result<Self, IndexError> {
        model_manager.init()?;
        let model_version = model_manager.model_version();
        let include_diffs = index.metadata.include_diffs;
        let created_at = Some(index.metadata.created_at);

        Ok(Self {
            entries: index.entries,
            model_manager,
            model_version,
            last_commit: Some(index.last_commit),
            include_diffs,
            created_at,
        })
    }

    /// Set the newest indexed commit hash (should be HEAD or newest new commit).
    pub fn set_last_commit(&mut self, hash: String) {
        self.last_commit = Some(hash);
    }

    pub fn add_commit(&mut self, commit: CommitInfo) -> Result<(), IndexError> {
        if self.entries.iter().any(|e| e.commit.hash == commit.hash) {
            debug!("Commit {} already indexed, skipping", &commit.hash[..7]);
            return Ok(());
        }

        debug!("Adding commit: {}", &commit.hash[..7]);

        let text = commit.to_text(self.include_diffs);
        let embedding_array = self.model_manager.encode_text(&text)?;
        let embedding = embedding_array.to_vec();

        let entry = IndexEntry { commit, embedding };
        self.entries.push(entry);

        Ok(())
    }

    pub fn build(self) -> SemanticIndex {
        let last_commit = self
            .last_commit
            .unwrap_or_else(|| "unknown".to_string());

        let mut index = SemanticIndex::new(self.model_version, last_commit, self.include_diffs);
        index.entries = self.entries;
        index.metadata.total_commits = index.entries.len();
        index.metadata.updated_at = Utc::now();
        if let Some(created_at) = self.created_at {
            index.metadata.created_at = created_at;
        }

        index
    }
}
