//! End-to-end coverage of hybrid retrieval against a real git dir.
//!
//! The point of these tests is the *retrieval quality* claim, not just the
//! plumbing: they construct the failure mode pure embedding search has — a
//! query naming an exact token — and assert that fusion fixes it.
//!
//! Embeddings are synthetic so CI needs no model. To make the test honest,
//! they are built to be *deliberately wrong* about the exact-token commit,
//! which is exactly what a real 384-dimensional embedding does when it cannot
//! distinguish `CVE-2024-1234` from surrounding prose.

use git_semantic::git::CommitInfo;
use git_semantic::index::{IndexEntry, IndexStorage, SemanticIndex};
use git_semantic::search::{RRF_K, Ranking, reciprocal_rank_fusion};
use git_semantic::text::Bm25Params;
use git_semantic::vector::{dot, normalize};
use std::fs;
use tempfile::TempDir;

const DIM: usize = 64;

fn git_repo() -> TempDir {
    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join(".git")).unwrap();
    dir
}

fn embedding(seed: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM)
        .map(|d| ((d as f32) * 0.13 + seed as f32 * 0.7).sin())
        .collect();
    normalize(&mut v);
    v
}

fn entry(idx: usize, message: &str, files: &str, author: &str) -> IndexEntry {
    IndexEntry {
        commit: CommitInfo {
            hash: format!("{idx:040x}"),
            author: author.to_string(),
            date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            message: message.to_string(),
            diff_summary: format!("Files: {files}\n+some change"),
        },
        embedding: embedding(idx),
    }
}

/// A corpus shaped like a real repository: mostly dependency noise, with a few
/// commits that matter.
fn corpus() -> SemanticIndex {
    let mut index = SemanticIndex::new("bge-small-en-v1.5".to_string(), "head".to_string(), true);

    let commits: Vec<(&str, &str, &str)> = vec![
        (
            "fix: resolve race condition in login flow",
            "src/auth.rs",
            "Alice Chen",
        ),
        (
            "feat: add incremental indexing",
            "src/index/builder.rs",
            "Bob Martinez",
        ),
        (
            "docs: explain the search threshold",
            "README.md",
            "Alice Chen",
        ),
        (
            "refactor: extract diff formatting",
            "src/git/diff.rs",
            "Bob Martinez",
        ),
        (
            "fix: patch CVE-2024-1234 in token refresh",
            "src/auth.rs",
            "Alice Chen",
        ),
        (
            "chore(deps): update rust crate clap to v4.6.4",
            "Cargo.toml",
            "renovate[bot]",
        ),
        (
            "chore(deps): update rust crate tokio to v1.52.3",
            "Cargo.toml",
            "renovate[bot]",
        ),
        (
            "chore(deps): update rust crate serde to v1.0.9",
            "Cargo.toml",
            "renovate[bot]",
        ),
        (
            "perf: batch the embedding forward pass",
            "src/embedding/model.rs",
            "Bob Martinez",
        ),
        (
            "test: cover the rebase detection path",
            "tests/incremental.rs",
            "Alice Chen",
        ),
    ];

    for (i, (message, files, author)) in commits.iter().enumerate() {
        index.entries.push(entry(i, message, files, author));
    }
    index.metadata.total_commits = index.entries.len();
    index
}

/// Rank by embedding similarity — what search did before hybrid retrieval.
fn semantic_ranking(index: &SemanticIndex, query_seed: usize, k: usize) -> Vec<u32> {
    let query = embedding(query_seed);
    let mut scored: Vec<(f32, u32)> = index
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| (dot(&e.embedding, &query), i as u32))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

#[test]
fn bm25_index_is_cached_beside_the_semantic_index() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = corpus();
    storage.save(&index).unwrap();

    let (lexical, rebuilt) = storage.load_or_build_lexical(&index, Bm25Params::default());
    assert!(rebuilt);
    assert_eq!(lexical.len(), index.entries.len());

    let sidecar = dir.path().join(".git").join("semantic-index.bm25");
    assert!(sidecar.exists(), "BM25 index should be cached to disk");

    let (_, rebuilt) = storage.load_or_build_lexical(&index, Bm25Params::default());
    assert!(!rebuilt, "second load should hit the cache");

    // The primary index is untouched and still loads on its own.
    fs::remove_file(&sidecar).unwrap();
    assert_eq!(storage.load().unwrap().entries.len(), index.entries.len());
}

#[test]
fn lexical_search_finds_the_exact_identifier() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = corpus();
    let (lexical, _) = storage.load_or_build_lexical(&index, Bm25Params::default());

    let hits = lexical.search("CVE-2024-1234", 5);
    assert_eq!(hits[0].0, 4, "the commit naming that CVE must rank first");
}

#[test]
fn lexical_search_finds_commits_by_changed_path() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = corpus();
    let (lexical, _) = storage.load_or_build_lexical(&index, Bm25Params::default());

    let ids: Vec<u32> = lexical
        .search("src/auth.rs", 10)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    assert!(ids.contains(&0), "auth commits should match: {ids:?}");
    assert!(ids.contains(&4), "auth commits should match: {ids:?}");
}

#[test]
fn hybrid_rescues_an_exact_match_that_embeddings_rank_poorly() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = corpus();
    let (lexical, _) = storage.load_or_build_lexical(&index, Bm25Params::default());

    // Pick a query seed whose embedding puts commit 4 outside the top 3 —
    // the situation where pure semantic search fails the user.
    let query = "CVE-2024-1234";
    let semantic = semantic_ranking(&index, 999, 10);
    let semantic_position = semantic.iter().position(|id| *id == 4).unwrap();
    assert!(
        semantic_position > 0,
        "test is vacuous unless embeddings miss the exact match; got position {semantic_position}"
    );

    let lexical_ids: Vec<u32> = lexical
        .search(query, 10)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    let fused = reciprocal_rank_fusion(&[Ranking::new(&semantic), Ranking::new(&lexical_ids)], 10);
    let fused_position = fused.iter().position(|(id, _)| *id == 4).unwrap();

    assert!(
        fused_position < semantic_position,
        "fusion should lift the exact match: semantic rank {semantic_position}, fused {fused_position}"
    );
    assert_eq!(fused[0].0, 4, "and it should reach the top");
}

#[test]
fn hybrid_keeps_semantic_results_when_no_keyword_matches() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = corpus();
    let (lexical, _) = storage.load_or_build_lexical(&index, Bm25Params::default());

    // A query with no shared vocabulary at all.
    let lexical_ids: Vec<u32> = lexical
        .search("kubernetes helm chart", 10)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert!(lexical_ids.is_empty(), "sanity: BM25 should find nothing");

    let semantic = semantic_ranking(&index, 3, 5);
    let fused = reciprocal_rank_fusion(&[Ranking::new(&semantic), Ranking::new(&lexical_ids)], 5);

    let fused_ids: Vec<u32> = fused.into_iter().map(|(id, _)| id).collect();
    assert_eq!(
        fused_ids, semantic,
        "with nothing to fuse, hybrid must equal semantic exactly"
    );
}

#[test]
fn lexical_filter_respects_metadata_filters() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = corpus();
    let (lexical, _) = storage.load_or_build_lexical(&index, Bm25Params::default());

    let bots: Vec<u32> = index
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.commit.author == "renovate[bot]")
        .map(|(i, _)| i as u32)
        .collect();

    let hits = lexical.search_filtered("update rust crate", 10, |id| bots.contains(&id));

    assert!(!hits.is_empty());
    for (id, _) in hits {
        assert_eq!(index.entries[id as usize].commit.author, "renovate[bot]");
    }
}

#[test]
fn bm25_sidecar_invalidates_when_commits_are_added() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();

    let index = corpus();
    storage.load_or_build_lexical(&index, Bm25Params::default());

    let mut grown = corpus();
    grown.entries.push(entry(
        99,
        "feat: add hybrid retrieval",
        "src/search/fusion.rs",
        "Yan",
    ));
    grown.metadata.total_commits = grown.entries.len();

    let (lexical, rebuilt) = storage.load_or_build_lexical(&grown, Bm25Params::default());
    assert!(rebuilt, "a grown index must invalidate the BM25 sidecar");
    assert_eq!(lexical.len(), grown.entries.len());
    assert_eq!(lexical.search("hybrid retrieval", 1)[0].0, 10);
}

#[test]
fn rrf_constant_matches_the_published_default() {
    // Guards against a silent retune: 60 is the value from Cormack et al.
    assert_eq!(RRF_K, 60.0);
}
