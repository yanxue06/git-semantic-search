//! BM25 over an inverted index of commit text.
//!
//! # Why lexical search at all
//!
//! Embeddings are good at meaning and bad at exact strings. A 384-dimensional
//! vector cannot reliably distinguish `CVE-2024-1234` from `CVE-2024-5678`, and
//! it has no special respect for `src/auth.rs`, an error code, or a commit hash
//! — those are precisely the queries where a user knows the exact token and
//! wants the commit that contains it.
//!
//! BM25 is the complement: exact-term matching with saturation on term
//! frequency and length normalization, so a 10 KB diff does not outrank a
//! one-line commit just for being longer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::tokenize::tokenize;

/// Standard BM25 tuning. `k1` controls how fast term frequency saturates, `b`
/// how strongly document length is normalized.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// One document's occurrence count for a term.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Posting {
    doc: u32,
    tf: u32,
}

/// An inverted index with BM25 scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Index {
    params: Bm25Params,
    /// term -> documents containing it, ascending by doc id.
    postings: HashMap<String, Vec<Posting>>,
    /// Token count per document, for length normalization.
    doc_lengths: Vec<u32>,
    total_tokens: u64,
}

impl Bm25Index {
    pub fn new(params: Bm25Params) -> Self {
        Self {
            params,
            postings: HashMap::new(),
            doc_lengths: Vec::new(),
            total_tokens: 0,
        }
    }

    /// Build an index over documents in order; ids are positions.
    pub fn build<I, S>(params: Bm25Params, documents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut index = Self::new(params);
        for doc in documents {
            index.add(doc.as_ref());
        }
        index
    }

    /// Append one document and return its id.
    pub fn add(&mut self, text: &str) -> u32 {
        let doc = self.doc_lengths.len() as u32;
        let tokens = tokenize(text);

        // Collapse to per-term counts first so each term gets one posting.
        let mut counts: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }

        for (term, tf) in counts {
            self.postings
                .entry(term)
                .or_default()
                .push(Posting { doc, tf });
        }

        self.doc_lengths.push(tokens.len() as u32);
        self.total_tokens += tokens.len() as u64;

        doc
    }

    pub fn len(&self) -> usize {
        self.doc_lengths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_lengths.is_empty()
    }

    pub fn vocabulary_size(&self) -> usize {
        self.postings.len()
    }

    fn average_length(&self) -> f32 {
        if self.doc_lengths.is_empty() {
            return 0.0;
        }
        self.total_tokens as f32 / self.doc_lengths.len() as f32
    }

    /// Probabilistic IDF with the +0.5 smoothing that keeps it positive even
    /// for a term appearing in every document.
    fn idf(&self, doc_frequency: usize) -> f32 {
        let n = self.doc_lengths.len() as f32;
        let df = doc_frequency as f32;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Top `k` documents for `query`, best first, as `(doc, score)`.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f32)> {
        self.search_filtered(query, k, |_| true)
    }

    /// Top `k` documents restricted to ids where `allow` returns true.
    ///
    /// Only documents that share a term with the query are touched, so cost
    /// scales with posting-list length rather than corpus size.
    pub fn search_filtered<P>(&self, query: &str, k: usize, allow: P) -> Vec<(u32, f32)>
    where
        P: Fn(u32) -> bool,
    {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let average = self.average_length();
        let mut scores: HashMap<u32, f32> = HashMap::new();

        // A repeated query term is scored once; BM25 has no query-side term
        // frequency component in this formulation.
        let mut seen_terms: Vec<&str> = Vec::with_capacity(query_terms.len());

        for term in &query_terms {
            if seen_terms.contains(&term.as_str()) {
                continue;
            }
            seen_terms.push(term);

            let Some(postings) = self.postings.get(term) else {
                continue;
            };

            let idf = self.idf(postings.len());

            for posting in postings {
                if !allow(posting.doc) {
                    continue;
                }

                let tf = posting.tf as f32;
                let length = self.doc_lengths[posting.doc as usize] as f32;
                let normalizer =
                    self.params.k1 * (1.0 - self.params.b + self.params.b * length / average);

                let contribution = idf * (tf * (self.params.k1 + 1.0)) / (tf + normalizer);
                *scores.entry(posting.doc).or_insert(0.0) += contribution;
            }
        }

        let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
        // Descending by score, ties broken by id so output is stable.
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(k);
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<&'static str> {
        vec![
            "Files: src/auth.rs\nfix: resolve race condition in login",
            "Files: src/index/storage.rs\nfeat: add incremental indexing",
            "Files: Cargo.toml, Cargo.lock\nchore(deps): update rust crate clap to v4.6.4",
            "Files: src/auth.rs\nfix: patch CVE-2024-1234 in token refresh",
            "Files: README.md\ndocs: explain the search threshold",
            "Files: Cargo.toml, Cargo.lock\nchore(deps): update rust crate tokio to v1.52.3",
        ]
    }

    fn index() -> Bm25Index {
        Bm25Index::build(Bm25Params::default(), corpus())
    }

    fn top_ids(hits: &[(u32, f32)]) -> Vec<u32> {
        hits.iter().map(|(id, _)| *id).collect()
    }

    #[test]
    fn empty_index_returns_nothing() {
        let index = Bm25Index::new(Bm25Params::default());
        assert!(index.is_empty());
        assert!(index.search("anything", 10).is_empty());
    }

    #[test]
    fn indexes_every_document() {
        let index = index();
        assert_eq!(index.len(), 6);
        assert!(index.vocabulary_size() > 20, "vocabulary looks too small");
    }

    #[test]
    fn finds_an_exact_identifier_embeddings_would_blur() {
        let index = index();
        let hits = index.search("CVE-2024-1234", 3);
        assert_eq!(hits[0].0, 3, "the commit naming that CVE must rank first");
    }

    #[test]
    fn finds_commits_by_changed_path() {
        let index = index();
        let hits = index.search("src/auth.rs", 10);
        let ids = top_ids(&hits);
        assert!(ids.contains(&0), "got {ids:?}");
        assert!(ids.contains(&3), "got {ids:?}");
    }

    #[test]
    fn ranks_a_rare_term_above_a_common_one() {
        let index = index();
        // "clap" appears once; "chore"/"deps"/"update" appear in two documents.
        let hits = index.search("clap", 5);
        assert_eq!(hits[0].0, 2);
        assert_eq!(hits.len(), 1, "only one document mentions clap");
    }

    #[test]
    fn unknown_terms_score_nothing() {
        let index = index();
        assert!(index.search("kubernetes helm chart", 10).is_empty());
    }

    #[test]
    fn empty_query_returns_nothing() {
        let index = index();
        assert!(index.search("", 10).is_empty());
        assert!(index.search("--- +++", 10).is_empty());
    }

    #[test]
    fn respects_k() {
        let index = index();
        assert_eq!(index.search("update rust crate", 1).len(), 1);
    }

    #[test]
    fn zero_k_returns_nothing() {
        assert!(index().search("fix", 0).is_empty());
    }

    #[test]
    fn scores_are_positive_and_descending() {
        let index = index();
        let hits = index.search("fix auth token", 10);
        assert!(!hits.is_empty());
        for (_, score) in &hits {
            assert!(*score > 0.0, "BM25 scores should be positive, got {score}");
        }
        for pair in hits.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "scores out of order: {hits:?}");
        }
    }

    #[test]
    fn filter_restricts_results_without_changing_order() {
        let index = index();
        let all = top_ids(&index.search("update rust crate", 10));
        assert!(all.len() >= 2);

        let filtered = top_ids(&index.search_filtered("update rust crate", 10, |id| id == 5));
        assert_eq!(filtered, vec![5]);
    }

    #[test]
    fn filter_matching_nothing_returns_empty() {
        assert!(index().search_filtered("fix", 10, |_| false).is_empty());
    }

    #[test]
    fn length_normalization_favours_the_focused_commit() {
        let mut index = Bm25Index::new(Bm25Params::default());
        index.add("fix authentication");
        index.add(&format!(
            "fix authentication {}",
            "unrelated words ".repeat(200)
        ));

        let hits = index.search("fix authentication", 2);
        assert_eq!(
            hits[0].0, 0,
            "the short, focused commit should win on equal term counts"
        );
    }

    #[test]
    fn term_frequency_saturates() {
        let mut index = Bm25Index::new(Bm25Params::default());
        index.add("leak");
        index.add("leak leak leak leak leak leak leak leak");

        let hits = index.search("leak", 2);
        let ratio = hits[0].1 / hits[1].1;
        assert!(
            ratio < 4.0,
            "8x the term count must not mean 8x the score; ratio was {ratio}"
        );
    }

    #[test]
    fn repeated_query_terms_are_not_double_counted() {
        let index = index();
        let once = index.search("clap", 5);
        let thrice = index.search("clap clap clap", 5);
        assert_eq!(once, thrice);
    }

    #[test]
    fn case_insensitive() {
        let index = index();
        assert_eq!(index.search("CLAP", 5), index.search("clap", 5));
    }

    #[test]
    fn serialization_roundtrip_preserves_ranking() {
        let index = index();
        let before = index.search("fix auth token", 10);

        let bytes = bincode::serialize(&index).unwrap();
        let restored: Bm25Index = bincode::deserialize(&bytes).unwrap();

        assert_eq!(restored.len(), index.len());
        assert_eq!(restored.search("fix auth token", 10), before);
    }

    #[test]
    fn idf_stays_positive_for_a_term_in_every_document() {
        let mut index = Bm25Index::new(Bm25Params::default());
        for _ in 0..10 {
            index.add("chore deps update");
        }

        let hits = index.search("chore", 10);
        assert_eq!(hits.len(), 10);
        for (_, score) in hits {
            assert!(
                score > 0.0,
                "smoothed IDF must not go negative, got {score}"
            );
        }
    }

    #[test]
    fn single_document_index_works() {
        let mut index = Bm25Index::new(Bm25Params::default());
        index.add("fix: the only commit");
        let hits = index.search("commit", 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].1 > 0.0);
    }

    #[test]
    fn documents_with_no_tokens_never_match() {
        let mut index = Bm25Index::new(Bm25Params::default());
        index.add("");
        index.add("real content here");

        let hits = index.search("content", 5);
        assert_eq!(top_ids(&hits), vec![1]);
    }
}
