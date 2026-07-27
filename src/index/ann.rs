//! On-disk cache for the approximate nearest-neighbor graph.
//!
//! The graph is stored as a *sidecar* file next to `semantic-index` rather than
//! inside it. That keeps the primary index format byte-compatible with every
//! release before this one — an existing index keeps loading, and the graph is
//! built once on first search and reused after that. A stale or corrupt sidecar
//! is a cache miss, never an error.

use serde::{Deserialize, Serialize};

use crate::vector::{HnswIndex, HnswParams};

use super::SemanticIndex;

/// Bumped whenever the sidecar layout changes so old files are ignored rather
/// than misread.
const SIDECAR_FORMAT: u32 = 1;

/// Below this many commits an exact scan beats graph traversal outright: the
/// dot products stream through cache and there is no descent overhead.
/// Measured in `benches/search.rs`, where the two paths cross at ~1-2k vectors.
pub const EXACT_SCAN_THRESHOLD: usize = 2_048;

/// A persisted graph plus the fingerprint of the index it was built from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnSidecar {
    format: u32,
    fingerprint: u64,
    graph: HnswIndex,
}

impl AnnSidecar {
    pub fn new(index: &SemanticIndex, graph: HnswIndex) -> Self {
        Self {
            format: SIDECAR_FORMAT,
            fingerprint: fingerprint(index),
            graph,
        }
    }

    /// True when this sidecar was built from the index it is being loaded for.
    pub fn matches(&self, index: &SemanticIndex) -> bool {
        self.format == SIDECAR_FORMAT && self.fingerprint == fingerprint(index)
    }

    pub fn graph(&self) -> &HnswIndex {
        &self.graph
    }

    pub fn into_graph(self) -> HnswIndex {
        self.graph
    }
}

/// Build a graph over every embedding in `index`, in entry order.
///
/// Node ids are index positions, so a hit maps straight back to
/// `index.entries[id]` with no side table.
pub fn build_graph(index: &SemanticIndex, params: HnswParams) -> HnswIndex {
    let dim = index
        .entries
        .first()
        .map(|e| e.embedding.len())
        .unwrap_or(384);

    HnswIndex::build(
        dim,
        params,
        index.entries.iter().map(|e| e.embedding.as_slice()),
    )
}

/// Cheap staleness check over the fields that change when the index changes.
///
/// FNV-1a rather than `DefaultHasher` because SipHash keys are not stable
/// across Rust releases, which would silently invalidate every cache on a
/// toolchain bump.
pub(crate) fn fingerprint(index: &SemanticIndex) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET;
    let mut feed = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    };

    feed(&(index.entries.len() as u64).to_le_bytes());
    feed(index.model_version.as_bytes());
    feed(index.last_commit.as_bytes());
    feed(&[index.metadata.include_diffs as u8]);

    // Guard against same-count edits (a rebase that swaps one commit for
    // another) by folding in the first and last commit hashes.
    if let Some(first) = index.entries.first() {
        feed(first.commit.hash.as_bytes());
    }
    if let Some(last) = index.entries.last() {
        feed(last.commit.hash.as_bytes());
    }

    // Sample the embeddings themselves. Re-embedding the same commits — because
    // the text fed to the model changed, not the history — leaves count, HEAD,
    // and hashes identical, so nothing above would notice. Three fixed probes
    // keep this O(1) while catching that case.
    for position in sample_positions(index.entries.len()) {
        let embedding = &index.entries[position].embedding;
        feed(&(embedding.len() as u64).to_le_bytes());
        for value in embedding.iter().take(EMBEDDING_PROBE_FLOATS) {
            feed(&value.to_bits().to_le_bytes());
        }
    }

    hash
}

/// How many leading floats of a probed embedding to fold in.
const EMBEDDING_PROBE_FLOATS: usize = 8;

/// First, middle, and last entry — deduplicated for tiny indexes.
fn sample_positions(len: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }

    let mut positions = vec![0, len / 2, len - 1];
    positions.dedup();
    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::CommitInfo;
    use crate::index::IndexEntry;

    fn entry(hash: &str, seed: f32) -> IndexEntry {
        IndexEntry {
            commit: CommitInfo {
                hash: hash.to_string(),
                author: "Alice".to_string(),
                date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                    .unwrap()
                    .with_timezone(&chrono::Utc),
                message: format!("commit {hash}"),
                diff_summary: String::new(),
            },
            embedding: (0..32).map(|i| (i as f32 * seed).sin()).collect(),
        }
    }

    fn sample(hashes: &[&str]) -> SemanticIndex {
        let mut index =
            SemanticIndex::new("bge-small-en-v1.5".to_string(), "head".to_string(), true);
        for (i, h) in hashes.iter().enumerate() {
            index.entries.push(entry(h, 0.1 + i as f32 * 0.05));
        }
        index.metadata.total_commits = index.entries.len();
        index
    }

    #[test]
    fn sidecar_matches_its_own_index() {
        let index = sample(&["a1", "b2", "c3"]);
        let sidecar = AnnSidecar::new(&index, build_graph(&index, HnswParams::default()));
        assert!(sidecar.matches(&index));
    }

    #[test]
    fn sidecar_detects_added_commits() {
        let index = sample(&["a1", "b2", "c3"]);
        let sidecar = AnnSidecar::new(&index, build_graph(&index, HnswParams::default()));

        let grown = sample(&["a1", "b2", "c3", "d4"]);
        assert!(!sidecar.matches(&grown), "added commit must invalidate");
    }

    #[test]
    fn sidecar_detects_rewritten_history_at_equal_count() {
        let index = sample(&["a1", "b2", "c3"]);
        let sidecar = AnnSidecar::new(&index, build_graph(&index, HnswParams::default()));

        let rebased = sample(&["z9", "b2", "c3"]);
        assert!(
            !sidecar.matches(&rebased),
            "same commit count with different hashes must invalidate"
        );
    }

    #[test]
    fn sidecar_detects_model_change() {
        let index = sample(&["a1", "b2"]);
        let sidecar = AnnSidecar::new(&index, build_graph(&index, HnswParams::default()));

        let mut other = sample(&["a1", "b2"]);
        other.model_version = "some-other-model".to_string();
        assert!(!sidecar.matches(&other));
    }

    #[test]
    fn sidecar_detects_mode_change() {
        let index = sample(&["a1", "b2"]);
        let sidecar = AnnSidecar::new(&index, build_graph(&index, HnswParams::default()));

        let mut other = sample(&["a1", "b2"]);
        other.metadata.include_diffs = false;
        assert!(!sidecar.matches(&other));
    }

    #[test]
    fn build_graph_ids_line_up_with_entry_positions() {
        let index = sample(&["a1", "b2", "c3", "d4", "e5"]);
        let graph = build_graph(&index, HnswParams::default());
        assert_eq!(graph.len(), 5);

        for (position, entry) in index.entries.iter().enumerate() {
            let hits = graph.search(&entry.embedding, 1, None);
            assert_eq!(
                hits[0].0 as usize, position,
                "node id must equal entry position"
            );
        }
    }

    #[test]
    fn build_graph_on_empty_index_is_empty() {
        let index = sample(&[]);
        let graph = build_graph(&index, HnswParams::default());
        assert!(graph.is_empty());
        assert_eq!(graph.dim(), 384, "should fall back to the model dimension");
    }

    #[test]
    fn sidecar_roundtrips_through_bincode() {
        let index = sample(&["a1", "b2", "c3"]);
        let sidecar = AnnSidecar::new(&index, build_graph(&index, HnswParams::default()));

        let bytes = bincode::serialize(&sidecar).unwrap();
        let restored: AnnSidecar = bincode::deserialize(&bytes).unwrap();

        assert!(restored.matches(&index));
        assert_eq!(restored.graph().len(), 3);
    }

    #[test]
    fn fingerprint_is_stable_across_calls() {
        let index = sample(&["a1", "b2"]);
        assert_eq!(fingerprint(&index), fingerprint(&index));
    }

    #[test]
    fn sidecar_detects_reembedding_at_identical_history() {
        let index = sample(&["a1", "b2", "c3"]);
        let sidecar = AnnSidecar::new(&index, build_graph(&index, HnswParams::default()));

        // Same commits, same HEAD, same count — only the vectors changed, as
        // happens when the text fed to the model changes.
        let mut reembedded = sample(&["a1", "b2", "c3"]);
        for entry in &mut reembedded.entries {
            entry.embedding.iter_mut().for_each(|v| *v = -*v);
        }

        assert!(
            !sidecar.matches(&reembedded),
            "re-embedded vectors must invalidate the cached graph"
        );
    }

    #[test]
    fn sample_positions_dedupes_for_tiny_indexes() {
        assert!(sample_positions(0).is_empty());
        assert_eq!(sample_positions(1), vec![0]);
        assert_eq!(sample_positions(2), vec![0, 1]);
        assert_eq!(sample_positions(3), vec![0, 1, 2]);
        assert_eq!(sample_positions(100), vec![0, 50, 99]);
    }
}
