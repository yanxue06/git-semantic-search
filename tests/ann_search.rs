//! End-to-end coverage of the approximate search path against a real git dir.
//!
//! Everything the ANN change touches is exercised here — index persistence,
//! sidecar build/load/invalidate, graph traversal, filtered traversal, and
//! agreement with an exhaustive scan. The only thing stubbed is the ONNX
//! forward pass: embeddings are synthetic, which is exactly what lets these
//! tests run in CI without a 130 MB model download.

use git_semantic::git::CommitInfo;
use git_semantic::index::{EXACT_SCAN_THRESHOLD, IndexEntry, IndexStorage, SemanticIndex};
use git_semantic::vector::{HnswParams, dot, normalize};
use std::fs;
use tempfile::TempDir;

const DIM: usize = 96;

fn git_repo() -> TempDir {
    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join(".git")).unwrap();
    dir
}

/// Deterministic unit vectors that cluster, so nearest-neighbor structure is
/// meaningful rather than uniformly random noise.
fn embedding(seed: usize) -> Vec<f32> {
    let cluster = (seed % 12) as f32;
    let mut v: Vec<f32> = (0..DIM)
        .map(|d| {
            let base = ((d as f32) * 0.19 + cluster * 1.7).sin();
            let jitter = ((seed * 31 + d * 7) as f32 * 0.013).cos() * 0.25;
            base + jitter
        })
        .collect();
    normalize(&mut v);
    v
}

fn index_with(count: usize) -> SemanticIndex {
    let mut index = SemanticIndex::new(
        "bge-small-en-v1.5".to_string(),
        format!("head{count}"),
        true,
    );

    for i in 0..count {
        let author = match i % 3 {
            0 => "Alice Chen",
            1 => "Bob Martinez",
            _ => "renovate[bot]",
        };
        index.entries.push(IndexEntry {
            commit: CommitInfo {
                hash: format!("{i:040x}"),
                author: author.to_string(),
                date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                    .unwrap()
                    .with_timezone(&chrono::Utc),
                message: format!("commit {i}"),
                diff_summary: format!("+src/module{}.rs", i % 7),
            },
            embedding: embedding(i),
        });
    }

    index.metadata.total_commits = count;
    index
}

/// The exhaustive ranking the graph is expected to approximate.
fn exact_ranking(index: &SemanticIndex, query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = index
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| (dot(&e.embedding, query), i as u32))
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap()
            .then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

#[test]
fn graph_survives_a_save_load_cycle_and_matches_exact_top_hit() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = index_with(3_000);

    storage.save(&index).unwrap();
    let loaded = storage.load().unwrap();
    assert_eq!(loaded.entries.len(), 3_000);

    let (graph, rebuilt) = storage.load_or_build_ann(&loaded, HnswParams::default());
    assert!(rebuilt);
    assert_eq!(graph.len(), 3_000);
    assert_eq!(graph.dim(), DIM);

    for probe in [0usize, 977, 2_999] {
        let query = &loaded.entries[probe].embedding;
        let hits = graph.search(query, 5, None);
        assert_eq!(
            hits[0].0 as usize, probe,
            "graph should retrieve an indexed commit as its own best match"
        );
        assert_eq!(exact_ranking(&loaded, query, 1)[0] as usize, probe);
    }
}

#[test]
fn graph_recall_at_10_tracks_exhaustive_search() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = index_with(3_000);
    storage.save(&index).unwrap();

    let (graph, _) = storage.load_or_build_ann(&index, HnswParams::default());

    let mut matched = 0usize;
    let mut total = 0usize;

    // Queries that are not themselves indexed — the realistic case.
    for probe in 0..40 {
        let mut query = embedding(probe * 37 + 5);
        for (d, value) in query.iter_mut().enumerate() {
            *value += ((d + probe) as f32 * 0.05).sin() * 0.3;
        }
        normalize(&mut query);

        let exact = exact_ranking(&index, &query, 10);
        let approx: Vec<u32> = graph
            .search(&query, 10, None)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        total += exact.len();
        matched += exact.iter().filter(|id| approx.contains(id)).count();
    }

    let recall = matched as f64 / total as f64;
    assert!(
        recall >= 0.95,
        "recall@10 against the exhaustive scan dropped to {recall:.3}"
    );
}

#[test]
fn cached_sidecar_is_reused_and_invalidated_on_reindex() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();

    let index = index_with(2_500);
    storage.save(&index).unwrap();
    let (_, rebuilt) = storage.load_or_build_ann(&index, HnswParams::default());
    assert!(rebuilt, "cold start must build");

    let (_, rebuilt) = storage.load_or_build_ann(&index, HnswParams::default());
    assert!(!rebuilt, "warm start must reuse the sidecar");

    // Simulate new commits landing.
    let grown = index_with(2_600);
    storage.save(&grown).unwrap();
    let (graph, rebuilt) = storage.load_or_build_ann(&grown, HnswParams::default());
    assert!(rebuilt, "a grown index must invalidate the sidecar");
    assert_eq!(graph.len(), 2_600);
}

#[test]
fn sidecar_lives_beside_the_index_and_never_replaces_it() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = index_with(2_100);

    storage.save(&index).unwrap();
    storage.refresh_ann(&index, HnswParams::default()).unwrap();

    let index_file = dir.path().join(".git").join("semantic-index");
    let sidecar = dir.path().join(".git").join("semantic-index.hnsw");
    assert!(index_file.exists(), "primary index must remain");
    assert!(sidecar.exists(), "sidecar must be written alongside it");

    // The primary index is still readable by itself — that is what keeps this
    // change backward compatible with indexes built by earlier releases.
    fs::remove_file(&sidecar).unwrap();
    assert_eq!(storage.load().unwrap().entries.len(), 2_100);
}

#[test]
fn filtered_graph_search_returns_only_matching_commits_and_fills_k() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = index_with(3_000);
    let (graph, _) = storage.load_or_build_ann(&index, HnswParams::default());

    let query = &index.entries[42].embedding;
    let bots: Vec<u32> = (0..index.entries.len() as u32)
        .filter(|i| index.entries[*i as usize].commit.author == "renovate[bot]")
        .collect();
    assert!(bots.len() > 500, "sanity: the filter should match ~1/3");

    let hits = graph.search_filtered(query, 10, None, |id| bots.contains(&id));

    assert_eq!(hits.len(), 10, "a 1-in-3 filter must still fill k results");
    for (id, _) in hits {
        assert_eq!(
            index.entries[id as usize].commit.author,
            "renovate[bot]",
            "filter leaked a non-matching commit"
        );
    }
}

#[test]
fn small_repositories_stay_below_the_graph_threshold() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = index_with(100);
    storage.save(&index).unwrap();

    assert!(
        index.entries.len() <= EXACT_SCAN_THRESHOLD,
        "a 100-commit repo should be scanned exhaustively"
    );

    // `index` never calls refresh for a repo this size, so no sidecar appears.
    assert!(!storage.ann_path().exists());
    assert_eq!(storage.load().unwrap().entries.len(), 100);
}

#[test]
fn empty_index_produces_an_empty_graph() {
    let dir = git_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = index_with(0);
    storage.save(&index).unwrap();

    let (graph, _) = storage.load_or_build_ann(&index, HnswParams::default());
    assert!(graph.is_empty());
    assert!(graph.search(&embedding(1), 10, None).is_empty());
}
