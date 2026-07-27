mod engine;
mod error;
mod filter;
mod fusion;

pub use engine::{RetrievalMode, SearchEngine, SearchOptions, SearchOutcome, SearchStrategy};
pub use error::SearchError;
pub use fusion::{RRF_K, Ranking, reciprocal_rank_fusion};

use crate::git::CommitInfo;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub commit: CommitInfo,
    pub similarity: f32,
    pub rank: usize,
}
