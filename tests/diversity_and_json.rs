//! MMR diversification and JSON output, end to end.
//!
//! The diversification test uses a corpus shaped like this repository's actual
//! history — mostly near-identical dependency bumps — because that is the case
//! the feature exists for.

use git_semantic::cli::JsonOutput;
use git_semantic::git::CommitInfo;
use git_semantic::index::{IndexEntry, SemanticIndex};
use git_semantic::search::{
    Candidate, DEFAULT_LAMBDA, RetrievalMode, SearchOutcome, SearchResult, SearchStrategy, rerank,
};
use git_semantic::vector::normalize;

const DIM: usize = 32;

/// Vectors that cluster by `family`, with small within-family jitter — the
/// shape a batch of renovate commits actually produces.
///
/// Each family occupies its own slice of the space so families are genuinely
/// near-orthogonal. (A phase-shifted sinusoid is *not* enough: shifting by
/// `7 * 0.9 ≈ 2π` produces almost the same vector, which silently makes a
/// diversity test vacuous.)
fn embedding(family: usize, jitter: usize) -> Vec<f32> {
    const BLOCK: usize = 8;
    let mut v = vec![0.0f32; DIM];
    let base = (family * BLOCK) % DIM;

    for i in 0..BLOCK {
        v[(base + i) % DIM] = 1.0 - i as f32 * 0.05;
    }
    // Tiny within-family jitter: members are near-identical, never equal.
    v[(base + (jitter % BLOCK)) % DIM] += 0.02;

    normalize(&mut v);
    v
}

fn entry(idx: usize, message: &str, family: usize) -> IndexEntry {
    IndexEntry {
        commit: CommitInfo {
            hash: format!("{idx:040x}"),
            author: "renovate[bot]".to_string(),
            date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            message: message.to_string(),
            // Sorted, matching what the extractor's BTreeSet actually emits.
            diff_summary: "Files: Cargo.lock, Cargo.toml\n+version".to_string(),
        },
        embedding: embedding(family, idx),
    }
}

/// Eight dependency bumps (one family) and two substantive commits.
fn index_with_duplicate_heavy_history() -> SemanticIndex {
    let mut index = SemanticIndex::new("bge-small-en-v1.5".to_string(), "head".to_string(), true);

    for (i, crate_name) in [
        "clap", "tokio", "serde", "anyhow", "chrono", "rayon", "git2", "ort",
    ]
    .iter()
    .enumerate()
    {
        index.entries.push(entry(
            i,
            &format!("chore(deps): update rust crate {crate_name}"),
            0,
        ));
    }

    index
        .entries
        .push(entry(8, "feat: add incremental indexing", 1));
    index
        .entries
        .push(entry(9, "fix: resolve race in login", 2));

    index.metadata.total_commits = index.entries.len();
    index
}

fn candidates<'a>(
    index: &'a SemanticIndex,
    order: &[u32],
    relevance: &[f32],
) -> Vec<Candidate<'a>> {
    order
        .iter()
        .enumerate()
        .map(|(position, &id)| Candidate {
            id,
            relevance: relevance[position],
            embedding: &index.entries[id as usize].embedding,
        })
        .collect()
}

#[test]
fn diversification_breaks_up_a_wall_of_dependency_bumps() {
    let index = index_with_duplicate_heavy_history();

    // Relevance as a real query would produce it: the eight bumps score nearly
    // identically and edge out the two substantive commits.
    let order: Vec<u32> = (0..10).collect();
    let relevance = [0.88, 0.87, 0.87, 0.86, 0.86, 0.85, 0.85, 0.84, 0.72, 0.70];
    let cands = candidates(&index, &order, &relevance);

    let plain = rerank(&cands, 5, 1.0);
    assert_eq!(
        plain,
        vec![0, 1, 2, 3, 4],
        "sanity: pure relevance returns five near-identical bumps"
    );

    let diverse = rerank(&cands, 5, DEFAULT_LAMBDA);
    assert_eq!(diverse[0], 0, "the best answer must not move");

    let substantive = diverse.iter().filter(|id| **id >= 8).count();
    assert!(
        substantive > 0,
        "diversification should surface a non-dependency commit: {diverse:?}"
    );

    let bumps = diverse.iter().filter(|id| **id < 8).count();
    assert!(
        bumps < 5,
        "should not still be all dependency bumps: {diverse:?}"
    );
}

#[test]
fn diversification_never_drops_below_k_results() {
    let index = index_with_duplicate_heavy_history();
    let order: Vec<u32> = (0..10).collect();
    let relevance = vec![0.8f32; 10];
    let cands = candidates(&index, &order, &relevance);

    for k in [1usize, 3, 5, 10] {
        let out = rerank(&cands, k, DEFAULT_LAMBDA);
        assert_eq!(out.len(), k, "k={k} returned {} results", out.len());

        let mut unique = out.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), k, "k={k} produced duplicates: {out:?}");
    }
}

fn sample_outcome() -> SearchOutcome {
    let index = index_with_duplicate_heavy_history();
    let results = vec![
        SearchResult {
            commit: index.entries[0].commit.clone(),
            similarity: 0.88,
            rank: 1,
        },
        SearchResult {
            commit: index.entries[8].commit.clone(),
            // A keyword-only hit: no cosine to report.
            similarity: f32::NAN,
            rank: 2,
        },
    ];

    SearchOutcome {
        results,
        strategy: SearchStrategy::Approximate,
        candidate_count: 10,
        mode: RetrievalMode::Hybrid,
        diversified: true,
    }
}

#[test]
fn json_output_is_valid_and_round_trips() {
    let outcome = sample_outcome();
    let document = JsonOutput::new("dependency update", &outcome, true, 1.75);

    let text = serde_json::to_string_pretty(&document).unwrap();

    // Parse as untyped JSON first — this is what a consumer's `jq` would do.
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["query"], "dependency update");
    assert_eq!(value["mode"], "hybrid");
    assert_eq!(value["strategy"], "approximate");
    assert_eq!(value["candidates"], 10);
    assert_eq!(value["diversified"], true);
    assert_eq!(value["results"].as_array().unwrap().len(), 2);

    // And as the typed form.
    let parsed: JsonOutput = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed.results[0].hash.len(), 40, "full hash, not truncated");
}

#[test]
fn json_omits_similarity_for_keyword_only_hits() {
    let outcome = sample_outcome();
    let document = JsonOutput::new("q", &outcome, false, 1.0);
    let text = serde_json::to_string(&document).unwrap();

    // NaN has no JSON representation; emitting it produces a document that no
    // parser accepts. The field must simply be absent.
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let results = value["results"].as_array().unwrap();

    assert!(
        results[0].get("similarity").is_some(),
        "semantic hit keeps it"
    );
    assert!(
        results[1].get("similarity").is_none(),
        "keyword-only hit must omit it, got {}",
        results[1]
    );
    assert!(!text.contains("NaN"), "raw NaN leaked into JSON: {text}");
}

#[test]
fn json_exposes_changed_files_for_scripting() {
    let outcome = sample_outcome();
    let document = JsonOutput::new("q", &outcome, false, 1.0);
    let value: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&document).unwrap()).unwrap();

    let files = value["results"][0]["files"].as_array().unwrap();
    let names: Vec<&str> = files.iter().map(|f| f.as_str().unwrap()).collect();
    assert_eq!(names, vec!["Cargo.lock", "Cargo.toml"]);
}

#[test]
fn json_handles_an_empty_result_set() {
    let mut outcome = sample_outcome();
    outcome.results.clear();

    let document = JsonOutput::new("nothing matches", &outcome, false, 0.3);
    let text = serde_json::to_string(&document).unwrap();

    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["results"].as_array().unwrap().len(), 0);
    assert_eq!(value["query"], "nothing matches");
}
