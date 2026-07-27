//! On-disk cache for the BM25 inverted index.
//!
//! Same sidecar contract as [`crate::index::ann`]: stored beside
//! `semantic-index` rather than inside it, so the primary index format is
//! untouched and a stale or corrupt cache is a miss that rebuilds rather than an
//! error. Reuses the ANN fingerprint, since both caches are invalidated by
//! exactly the same events.

use serde::{Deserialize, Serialize};

use crate::text::{Bm25Index, Bm25Params};

use super::SemanticIndex;
use super::ann::fingerprint;

const SIDECAR_FORMAT: u32 = 1;

/// A persisted BM25 index plus the fingerprint of the index it came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalSidecar {
    format: u32,
    fingerprint: u64,
    index: Bm25Index,
}

impl LexicalSidecar {
    pub fn new(index: &SemanticIndex, lexical: Bm25Index) -> Self {
        Self {
            format: SIDECAR_FORMAT,
            fingerprint: fingerprint(index),
            index: lexical,
        }
    }

    pub fn matches(&self, index: &SemanticIndex) -> bool {
        self.format == SIDECAR_FORMAT && self.fingerprint == fingerprint(index)
    }

    pub fn index(&self) -> &Bm25Index {
        &self.index
    }

    pub fn into_index(self) -> Bm25Index {
        self.index
    }
}

/// Build a BM25 index over every commit, in entry order.
///
/// Document ids are index positions — the same convention as the ANN graph — so
/// a hit from either retriever resolves against `index.entries[id]` and the two
/// rankings can be fused without a translation table.
///
/// Indexes the author too, so `--author`-shaped queries typed as free text
/// ("commits by renovate") still land somewhere sensible.
pub fn build_lexical(index: &SemanticIndex, params: Bm25Params) -> Bm25Index {
    Bm25Index::build(
        params,
        index.entries.iter().map(|entry| {
            // `to_text(true)` is what was embedded: message, author, and — since
            // paths are recorded — the changed-file list. Indexing the same text
            // keeps the two retrievers looking at the same evidence.
            entry.commit.to_text(true)
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::CommitInfo;
    use crate::index::IndexEntry;

    fn entry(hash: &str, message: &str, files: &str) -> IndexEntry {
        IndexEntry {
            commit: CommitInfo {
                hash: hash.to_string(),
                author: "Alice".to_string(),
                date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                    .unwrap()
                    .with_timezone(&chrono::Utc),
                message: message.to_string(),
                diff_summary: format!("Files: {files}\n+something"),
            },
            embedding: vec![0.1; 32],
        }
    }

    fn sample() -> SemanticIndex {
        let mut index =
            SemanticIndex::new("bge-small-en-v1.5".to_string(), "head".to_string(), true);
        index
            .entries
            .push(entry("a1", "fix: race in login", "src/auth.rs"));
        index
            .entries
            .push(entry("b2", "chore: bump clap", "Cargo.toml"));
        index
            .entries
            .push(entry("c3", "docs: explain threshold", "README.md"));
        index.metadata.total_commits = 3;
        index
    }

    #[test]
    fn build_indexes_every_entry() {
        let lexical = build_lexical(&sample(), Bm25Params::default());
        assert_eq!(lexical.len(), 3);
    }

    #[test]
    fn document_ids_line_up_with_entry_positions() {
        let index = sample();
        let lexical = build_lexical(&index, Bm25Params::default());

        let hits = lexical.search("clap", 5);
        assert_eq!(hits[0].0, 1, "clap belongs to entry 1");
    }

    #[test]
    fn changed_paths_are_searchable() {
        let lexical = build_lexical(&sample(), Bm25Params::default());
        let hits = lexical.search("src/auth.rs", 5);
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn sidecar_matches_its_own_index() {
        let index = sample();
        let sidecar = LexicalSidecar::new(&index, build_lexical(&index, Bm25Params::default()));
        assert!(sidecar.matches(&index));
        assert_eq!(sidecar.index().len(), 3);
    }

    #[test]
    fn sidecar_invalidates_when_the_index_changes() {
        let index = sample();
        let sidecar = LexicalSidecar::new(&index, build_lexical(&index, Bm25Params::default()));

        let mut grown = sample();
        grown.entries.push(entry(
            "d4",
            "feat: add hybrid search",
            "src/search/fusion.rs",
        ));
        grown.metadata.total_commits = 4;

        assert!(!sidecar.matches(&grown));
    }

    #[test]
    fn sidecar_roundtrips_through_bincode() {
        let index = sample();
        let sidecar = LexicalSidecar::new(&index, build_lexical(&index, Bm25Params::default()));

        let bytes = bincode::serialize(&sidecar).unwrap();
        let restored: LexicalSidecar = bincode::deserialize(&bytes).unwrap();

        assert!(restored.matches(&index));
        assert_eq!(restored.into_index().search("clap", 5)[0].0, 1);
    }

    #[test]
    fn quick_mode_index_still_builds() {
        // No diff summaries at all — messages and authors only.
        let mut index =
            SemanticIndex::new("bge-small-en-v1.5".to_string(), "head".to_string(), false);
        for (i, message) in ["fix: one", "feat: two"].iter().enumerate() {
            let mut e = entry(&format!("h{i}"), message, "ignored");
            e.commit.diff_summary = String::new();
            index.entries.push(e);
        }
        index.metadata.total_commits = 2;

        let lexical = build_lexical(&index, Bm25Params::default());
        assert_eq!(lexical.len(), 2);
        assert_eq!(lexical.search("feat two", 5)[0].0, 1);
    }
}
