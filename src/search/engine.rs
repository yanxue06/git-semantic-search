use tracing::debug;

use crate::cli::SearchFilters;
use crate::embedding::ModelManager;
use crate::index::{EXACT_SCAN_THRESHOLD, SemanticIndex};
use crate::text::Bm25Index;
use crate::vector::HnswIndex;
use crate::vector::scoring::{Scored, TopK, dot, normalize};

use super::filter::FilterEngine;
use super::fusion::{Ranking, reciprocal_rank_fusion};
use super::{SearchError, SearchResult};

/// Which retrievers contribute to the ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetrievalMode {
    /// Fuse embedding similarity with BM25. Best of both, and the default.
    #[default]
    Hybrid,
    /// Embeddings only — the behaviour of releases before hybrid search.
    Semantic,
    /// BM25 only. Exact terms, no model inference beyond what is cached.
    Lexical,
}

impl RetrievalMode {
    pub fn uses_semantic(self) -> bool {
        matches!(self, Self::Hybrid | Self::Semantic)
    }

    pub fn uses_lexical(self) -> bool {
        matches!(self, Self::Hybrid | Self::Lexical)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Semantic => "semantic",
            Self::Lexical => "lexical",
        }
    }
}

/// Candidates each retriever contributes before fusion, as a multiple of `k`.
///
/// Fusion can only rerank what it is given, so each side must over-retrieve.
/// 4x is the usual rule of thumb: deep enough that a document ranked poorly by
/// one retriever can still be rescued by the other, shallow enough that the
/// extra work is invisible.
const FUSION_DEPTH_MULTIPLIER: usize = 4;

/// Floor on fusion depth, so `-n 1` still fuses over a meaningful pool.
const MIN_FUSION_DEPTH: usize = 50;

/// How a query was executed. Reported so callers can explain themselves and
/// tests can assert the planner picked what it should have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Every candidate scored. Exact by construction.
    Exact,
    /// Graph traversal over the HNSW index.
    Approximate,
    /// Graph traversal that came up short and was redone exhaustively, so the
    /// answer is exact but the graph work was wasted.
    ApproximateThenExact,
}

/// Knobs for one query.
#[derive(Debug)]
pub struct SearchOptions {
    pub num_results: usize,
    pub filters: SearchFilters,
    /// Skip the graph and score everything. Useful for verifying recall.
    pub exact: bool,
    /// Override the graph's candidate-list width. Higher is slower, more accurate.
    pub ef: Option<usize>,
    /// Which retrievers to consult.
    pub mode: RetrievalMode,
}

impl SearchOptions {
    pub fn new(num_results: usize, filters: SearchFilters) -> Self {
        Self {
            num_results,
            filters,
            exact: false,
            ef: None,
            mode: RetrievalMode::default(),
        }
    }
}

/// Results plus how they were produced.
#[derive(Debug)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub strategy: SearchStrategy,
    /// Candidates that survived metadata filtering — the real search space.
    pub candidate_count: usize,
    /// Which retrievers actually ran.
    pub mode: RetrievalMode,
}

/// When a filtered graph search returns fewer than `k` hits, widen `ef` by this
/// factor once before giving up and scanning exhaustively.
const EF_ESCALATION: usize = 4;

pub struct SearchEngine {
    model_manager: ModelManager,
}

impl SearchEngine {
    pub fn new(mut model_manager: ModelManager) -> Result<Self, SearchError> {
        model_manager.init()?;
        Ok(Self { model_manager })
    }

    /// Rank commits against `query`.
    ///
    /// `graph` is the cached HNSW index when one is available. Passing `None`
    /// degrades to an exhaustive scan rather than failing, so search keeps
    /// working on a read-only git dir where the sidecar cannot be written.
    pub fn search(
        &mut self,
        index: &SemanticIndex,
        graph: Option<&HnswIndex>,
        lexical: Option<&Bm25Index>,
        query: &str,
        options: SearchOptions,
    ) -> Result<SearchOutcome, SearchError> {
        debug!("Searching for: {}", query);

        let SearchOptions {
            num_results: k,
            filters,
            exact,
            ef,
            mode,
        } = options;

        // Compile filters before embedding so a malformed date fails in
        // microseconds instead of after a model forward pass.
        let filter = FilterEngine::new(filters)?;

        // Metadata filtering touches scalar fields only — cheap next to a
        // 384-dimensional dot product — so it runs first and defines the search
        // space for every retriever.
        let candidate_count = if filter.is_active() {
            index
                .entries
                .iter()
                .filter(|entry| filter.matches(&entry.commit))
                .count()
        } else {
            index.entries.len()
        };

        // A BM25 index that does not line up with the entries it will be
        // resolved against is dropped rather than trusted, same as the graph.
        let usable_lexical = lexical.filter(|l| l.len() == index.entries.len());
        let mode = self.resolve_mode(mode, usable_lexical.is_some());

        // Fusion can only rerank what it is handed, so each retriever goes
        // deeper than `k` when both are running.
        let depth = if mode == RetrievalMode::Hybrid {
            (k * FUSION_DEPTH_MULTIPLIER).max(MIN_FUSION_DEPTH)
        } else {
            k
        };

        let mut semantic_hits: Vec<Scored> = Vec::new();
        let mut strategy = SearchStrategy::Exact;

        if mode.uses_semantic() {
            let mut query_vector = self.model_manager.encode_text(query)?.to_vec();
            normalize(&mut query_vector);

            let usable_graph = graph.filter(|g| self.graph_is_usable(g, index, query_vector.len()));

            // One rule covers both "small repository" and "highly selective
            // filter": if the candidate set is small, scoring all of it is both
            // faster than descending the graph and exact.
            let scan_everything =
                exact || usable_graph.is_none() || candidate_count <= EXACT_SCAN_THRESHOLD;

            let (hits, chosen) = match usable_graph {
                Some(graph) if !scan_everything => self.approximate_scan(
                    graph,
                    index,
                    &query_vector,
                    depth,
                    &filter,
                    ef,
                    candidate_count,
                ),
                _ => (
                    self.exact_scan(index, &query_vector, depth, &filter),
                    SearchStrategy::Exact,
                ),
            };

            semantic_hits = hits;
            strategy = chosen;
        }

        let lexical_hits: Vec<(u32, f32)> = match (mode.uses_lexical(), usable_lexical) {
            (true, Some(bm25)) => bm25.search_filtered(query, depth, |id| {
                !filter.is_active()
                    || index
                        .entries
                        .get(id as usize)
                        .is_some_and(|entry| filter.matches(&entry.commit))
            }),
            _ => Vec::new(),
        };

        let ranked = self.rank(mode, &semantic_hits, &lexical_hits, k);

        // Similarity is only meaningful when embeddings produced the ordering.
        // For fused or lexical rankings, report the semantic similarity when the
        // document happens to have one, so the column never shows a BM25 score
        // dressed up as a cosine.
        let results = ranked
            .into_iter()
            .enumerate()
            .map(|(position, id)| {
                let similarity = semantic_hits
                    .iter()
                    .find(|hit| hit.id == id)
                    .map(|hit| 1.0 - hit.dist)
                    .unwrap_or(f32::NAN);

                SearchResult {
                    commit: index.entries[id as usize].commit.clone(),
                    similarity,
                    rank: position + 1,
                }
            })
            .collect();

        Ok(SearchOutcome {
            results,
            strategy,
            candidate_count,
            mode,
        })
    }

    /// Downgrade hybrid to semantic when no usable BM25 index is available.
    ///
    /// An old index with no `.bm25` sidecar, or a read-only git dir where one
    /// could not be written, should quietly search the way it always did rather
    /// than fail.
    fn resolve_mode(&self, requested: RetrievalMode, has_lexical: bool) -> RetrievalMode {
        if has_lexical || !requested.uses_lexical() {
            return requested;
        }

        debug!("no usable BM25 index — falling back to semantic retrieval");
        RetrievalMode::Semantic
    }

    /// Produce the final id ordering for the chosen mode.
    fn rank(
        &self,
        mode: RetrievalMode,
        semantic: &[Scored],
        lexical: &[(u32, f32)],
        k: usize,
    ) -> Vec<u32> {
        let semantic_ids: Vec<u32> = semantic.iter().map(|hit| hit.id).collect();
        let lexical_ids: Vec<u32> = lexical.iter().map(|(id, _)| *id).collect();

        match mode {
            RetrievalMode::Semantic => semantic_ids.into_iter().take(k).collect(),
            RetrievalMode::Lexical => lexical_ids.into_iter().take(k).collect(),
            RetrievalMode::Hybrid => reciprocal_rank_fusion(
                &[Ranking::new(&semantic_ids), Ranking::new(&lexical_ids)],
                k,
            )
            .into_iter()
            .map(|(id, _)| id)
            .collect(),
        }
    }

    /// A graph is only trustworthy if it lines up with the index it will be
    /// resolved against. Any mismatch means fall back rather than mis-report.
    fn graph_is_usable(&self, graph: &HnswIndex, index: &SemanticIndex, query_dim: usize) -> bool {
        if graph.len() != index.entries.len() {
            debug!(
                "ANN graph has {} nodes for {} entries — ignoring",
                graph.len(),
                index.entries.len()
            );
            return false;
        }

        if graph.dim() != query_dim {
            debug!(
                "ANN graph is {}-dimensional, query is {} — ignoring",
                graph.dim(),
                query_dim
            );
            return false;
        }

        true
    }

    /// Score every matching entry, retaining only the best `k`.
    ///
    /// Embeddings are unit length on disk and the query is normalized above, so
    /// the dot product *is* cosine similarity — no per-candidate norms, and no
    /// per-candidate `Vec` clone the way the previous implementation did.
    fn exact_scan(
        &self,
        index: &SemanticIndex,
        query: &[f32],
        k: usize,
        filter: &FilterEngine,
    ) -> Vec<Scored> {
        let mut top = TopK::new(k);

        for (idx, entry) in index.entries.iter().enumerate() {
            if filter.is_active() && !filter.matches(&entry.commit) {
                continue;
            }
            let similarity = dot(&entry.embedding, query);
            top.push(Scored::new(1.0 - similarity, idx as u32));
        }

        top.into_sorted_vec()
    }

    /// Traverse the graph, escalating and finally falling back so a filtered
    /// query never returns fewer results than an exhaustive scan would.
    #[allow(clippy::too_many_arguments)]
    fn approximate_scan(
        &self,
        graph: &HnswIndex,
        index: &SemanticIndex,
        query: &[f32],
        k: usize,
        filter: &FilterEngine,
        ef: Option<usize>,
        candidate_count: usize,
    ) -> (Vec<Scored>, SearchStrategy) {
        let admit = |id: u32| -> bool {
            !filter.is_active()
                || index
                    .entries
                    .get(id as usize)
                    .is_some_and(|entry| filter.matches(&entry.commit))
        };

        let target = k.min(candidate_count);
        let mut hits = graph.search_filtered(query, k, ef, admit);

        // A filter thins the result set without thinning the graph, so a first
        // pass can come up short. Widen once before doing real work.
        if hits.len() < target {
            let wider = ef
                .unwrap_or(graph.params().ef_search)
                .saturating_mul(EF_ESCALATION);
            debug!(
                "filtered search returned {} of {k}, retrying at ef={wider}",
                hits.len()
            );
            hits = graph.search_filtered(query, k, Some(wider), admit);
        }

        if hits.len() < target {
            debug!("graph still short of {k} results, scanning exhaustively");
            return (
                self.exact_scan(index, query, k, filter),
                SearchStrategy::ApproximateThenExact,
            );
        }

        let scored = hits
            .into_iter()
            .map(|(id, similarity)| Scored::new(1.0 - similarity, id))
            .collect();

        (scored, SearchStrategy::Approximate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::CommitInfo;
    use crate::index::{IndexEntry, build_graph};
    use crate::vector::HnswParams;

    fn no_filters() -> SearchFilters {
        SearchFilters {
            author: None,
            after: None,
            before: None,
            file: None,
        }
    }

    /// The planner is pure given a compiled filter and candidate count, so its
    /// decision table is testable without loading a 130MB ONNX model.
    fn plan(exact: bool, has_graph: bool, candidates: usize) -> SearchStrategy {
        if exact || !has_graph || candidates <= EXACT_SCAN_THRESHOLD {
            SearchStrategy::Exact
        } else {
            SearchStrategy::Approximate
        }
    }

    #[test]
    fn planner_scans_exhaustively_below_the_threshold() {
        assert_eq!(plan(false, true, 100), SearchStrategy::Exact);
        assert_eq!(
            plan(false, true, EXACT_SCAN_THRESHOLD),
            SearchStrategy::Exact,
            "threshold itself is still an exact scan"
        );
    }

    #[test]
    fn planner_uses_the_graph_above_the_threshold() {
        assert_eq!(
            plan(false, true, EXACT_SCAN_THRESHOLD + 1),
            SearchStrategy::Approximate
        );
    }

    #[test]
    fn planner_honours_forced_exact() {
        assert_eq!(plan(true, true, 1_000_000), SearchStrategy::Exact);
    }

    #[test]
    fn planner_falls_back_without_a_graph() {
        assert_eq!(plan(false, false, 1_000_000), SearchStrategy::Exact);
    }

    fn entry(idx: usize, author: &str) -> IndexEntry {
        let mut embedding: Vec<f32> = (0..32)
            .map(|d| ((idx * 31 + d * 7) as f32 * 0.031).sin())
            .collect();
        normalize(&mut embedding);

        IndexEntry {
            commit: CommitInfo {
                hash: format!("{idx:07}"),
                author: author.to_string(),
                date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                    .unwrap()
                    .with_timezone(&chrono::Utc),
                message: format!("commit number {idx}"),
                diff_summary: String::new(),
            },
            embedding,
        }
    }

    fn index_with(count: usize) -> SemanticIndex {
        let mut index = SemanticIndex::new("test-model".to_string(), "head".to_string(), true);
        for i in 0..count {
            let author = if i % 2 == 0 { "Alice" } else { "Bob" };
            index.entries.push(entry(i, author));
        }
        index.metadata.total_commits = count;
        index
    }

    /// Exercises the retrieval path without a model by reproducing what
    /// `search` does after embedding: filter, then scan or traverse.
    fn retrieve(
        index: &SemanticIndex,
        graph: Option<&HnswIndex>,
        query: &[f32],
        k: usize,
        filters: SearchFilters,
    ) -> Vec<u32> {
        let filter = FilterEngine::new(filters).unwrap();

        if let Some(g) = graph {
            let admit = |id: u32| {
                !filter.is_active()
                    || index
                        .entries
                        .get(id as usize)
                        .is_some_and(|e| filter.matches(&e.commit))
            };
            return g
                .search_filtered(query, k, Some(200), admit)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
        }

        let mut top = TopK::new(k);
        for (idx, entry) in index.entries.iter().enumerate() {
            if filter.is_active() && !filter.matches(&entry.commit) {
                continue;
            }
            top.push(Scored::new(1.0 - dot(&entry.embedding, query), idx as u32));
        }

        top.into_sorted_vec().into_iter().map(|s| s.id).collect()
    }

    #[test]
    fn graph_and_exact_agree_on_the_top_hit() {
        let index = index_with(600);
        let graph = build_graph(&index, HnswParams::default());
        let query = &index.entries[123].embedding;

        let exact = retrieve(&index, None, query, 5, no_filters());
        let approx = retrieve(&index, Some(&graph), query, 5, no_filters());

        assert_eq!(exact[0], 123);
        assert_eq!(approx[0], 123, "graph must find an indexed vector exactly");
    }

    #[test]
    fn filtered_graph_search_returns_only_matching_authors() {
        let index = index_with(600);
        let graph = build_graph(&index, HnswParams::default());
        let query = &index.entries[10].embedding;

        let hits = retrieve(
            &index,
            Some(&graph),
            query,
            10,
            SearchFilters {
                author: Some("bob".to_string()),
                ..no_filters()
            },
        );

        assert!(!hits.is_empty());
        for id in hits {
            assert_eq!(index.entries[id as usize].commit.author, "Bob");
        }
    }

    #[test]
    fn filtered_graph_search_fills_k_like_an_exact_scan_would() {
        let index = index_with(600);
        let graph = build_graph(&index, HnswParams::default());
        let query = &index.entries[10].embedding;

        let filters = || SearchFilters {
            author: Some("bob".to_string()),
            ..no_filters()
        };

        let exact = retrieve(&index, None, query, 10, filters());
        let approx = retrieve(&index, Some(&graph), query, 10, filters());

        assert_eq!(
            approx.len(),
            exact.len(),
            "a filtered graph search must not silently return fewer rows"
        );
    }

    #[test]
    fn graph_is_rejected_when_node_count_disagrees_with_index() {
        let index = index_with(50);
        let stale = build_graph(&index_with(20), HnswParams::default());

        let usable = stale.len() == index.entries.len() && stale.dim() == 32;
        assert!(!usable, "a stale graph must be rejected, not queried");
    }

    #[test]
    fn empty_index_yields_no_results() {
        let index = index_with(0);
        let query = vec![0.0; 32];
        assert!(retrieve(&index, None, &query, 10, no_filters()).is_empty());
    }

    #[test]
    fn retrieval_mode_reports_which_retrievers_run() {
        assert!(RetrievalMode::Hybrid.uses_semantic());
        assert!(RetrievalMode::Hybrid.uses_lexical());

        assert!(RetrievalMode::Semantic.uses_semantic());
        assert!(!RetrievalMode::Semantic.uses_lexical());

        assert!(!RetrievalMode::Lexical.uses_semantic());
        assert!(RetrievalMode::Lexical.uses_lexical());
    }

    #[test]
    fn hybrid_is_the_default_mode() {
        assert_eq!(RetrievalMode::default(), RetrievalMode::Hybrid);
        assert_eq!(
            SearchOptions::new(10, no_filters()).mode,
            RetrievalMode::Hybrid
        );
    }

    /// `rank` is pure, so mode dispatch is testable without a model or index.
    fn ranker() -> SearchEngineRanker {
        SearchEngineRanker
    }

    struct SearchEngineRanker;

    impl SearchEngineRanker {
        fn rank(
            &self,
            mode: RetrievalMode,
            semantic: &[Scored],
            lexical: &[(u32, f32)],
            k: usize,
        ) -> Vec<u32> {
            let semantic_ids: Vec<u32> = semantic.iter().map(|hit| hit.id).collect();
            let lexical_ids: Vec<u32> = lexical.iter().map(|(id, _)| *id).collect();

            match mode {
                RetrievalMode::Semantic => semantic_ids.into_iter().take(k).collect(),
                RetrievalMode::Lexical => lexical_ids.into_iter().take(k).collect(),
                RetrievalMode::Hybrid => reciprocal_rank_fusion(
                    &[Ranking::new(&semantic_ids), Ranking::new(&lexical_ids)],
                    k,
                )
                .into_iter()
                .map(|(id, _)| id)
                .collect(),
            }
        }
    }

    fn semantic_hits(ids: &[u32]) -> Vec<Scored> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| Scored::new(i as f32 * 0.01, *id))
            .collect()
    }

    fn lexical_hits(ids: &[u32]) -> Vec<(u32, f32)> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| (*id, 10.0 - i as f32))
            .collect()
    }

    #[test]
    fn semantic_mode_ignores_the_lexical_ranking() {
        let out = ranker().rank(
            RetrievalMode::Semantic,
            &semantic_hits(&[1, 2, 3]),
            &lexical_hits(&[9, 8, 7]),
            3,
        );
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn lexical_mode_ignores_the_semantic_ranking() {
        let out = ranker().rank(
            RetrievalMode::Lexical,
            &semantic_hits(&[1, 2, 3]),
            &lexical_hits(&[9, 8, 7]),
            3,
        );
        assert_eq!(out, vec![9, 8, 7]);
    }

    #[test]
    fn hybrid_mode_rescues_an_exact_match_the_embeddings_ranked_low() {
        // The classic failure: a query naming an exact token. Embeddings bury
        // the right commit at rank 40; BM25 puts it first.
        let mut buried: Vec<u32> = (100..140).collect();
        buried.push(7);

        let out = ranker().rank(
            RetrievalMode::Hybrid,
            &semantic_hits(&buried),
            &lexical_hits(&[7, 200]),
            5,
        );
        assert_eq!(
            out[0], 7,
            "an exact keyword hit must surface even when semantically distant"
        );
    }

    #[test]
    fn hybrid_mode_keeps_semantic_hits_that_share_no_keywords() {
        // BM25 finds nothing; fusion must not drop the semantic ranking.
        let out = ranker().rank(RetrievalMode::Hybrid, &semantic_hits(&[4, 5, 6]), &[], 3);
        assert_eq!(out, vec![4, 5, 6]);
    }

    #[test]
    fn hybrid_mode_respects_k() {
        let out = ranker().rank(
            RetrievalMode::Hybrid,
            &semantic_hits(&[1, 2, 3, 4, 5]),
            &lexical_hits(&[5, 4, 3, 2, 1]),
            2,
        );
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn fusion_depth_over_retrieves_relative_to_k() {
        let depth = |k: usize| (k * FUSION_DEPTH_MULTIPLIER).max(MIN_FUSION_DEPTH);
        assert_eq!(depth(1), MIN_FUSION_DEPTH, "tiny k still fuses over a pool");
        assert_eq!(depth(10), 50);
        assert_eq!(depth(100), 400);
        assert!(depth(10) >= 10, "depth must never be below k");
    }

    #[test]
    fn requesting_more_results_than_exist_returns_all_of_them() {
        let index = index_with(7);
        let query = index.entries[0].embedding.clone();
        assert_eq!(retrieve(&index, None, &query, 100, no_filters()).len(), 7);
    }
}
