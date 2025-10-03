use anyhow::Result;
use chrono::Utc;
use tracing::debug;

use crate::embedding::ModelManager;
use crate::git::CommitInfo;

use super::{IndexEntry, SemanticIndex};

pub struct IndexBuilder {
    entries: Vec<IndexEntry>,
    model_manager: ModelManager,
    model_version: String,
    last_commit: Option<String>,
}

impl IndexBuilder {
    pub fn new(model_manager: ModelManager) -> Result<Self> {
        let model_version = model_manager.model_version();

        Ok(Self {
            entries: Vec::new(),
            model_manager,
            model_version,
            last_commit: None,
        })
    }

    pub fn from_existing(index: SemanticIndex, model_manager: ModelManager) -> Result<Self> {
        let model_version = model_manager.model_version();

        Ok(Self {
            entries: index.entries,
            model_manager,
            model_version,
            last_commit: Some(index.last_commit),
        })
    }

    pub fn add_commit(&mut self, commit: CommitInfo) -> Result<()> {
        debug!("Adding commit: {}", &commit.hash[..7]);

        // Generate text representation
        let text = commit.to_text(true);

        // Generate embedding
        let embedding_array = self.model_manager.encode_text(&text)?;
        let embedding = embedding_array.to_vec();

        // Store the commit hash as last_commit
        self.last_commit = Some(commit.hash.clone());

        // Create index entry
        let entry = IndexEntry { commit, embedding };

        self.entries.push(entry);

        Ok(())
    }

    pub fn build(self) -> Result<SemanticIndex> {
        let last_commit = self
            .last_commit
            .unwrap_or_else(|| "unknown".to_string());

        let mut index = SemanticIndex::new(self.model_version, last_commit);
        index.entries = self.entries;
        index.metadata.total_commits = index.entries.len();
        index.metadata.updated_at = Utc::now();

        Ok(index)
    }
}

