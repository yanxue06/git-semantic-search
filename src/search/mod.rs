mod engine;
mod filter;

pub use engine::SearchEngine;

use crate::git::CommitInfo;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub commit: CommitInfo,
    pub similarity: f32,
    pub rank: usize,
}

