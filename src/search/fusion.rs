//! Reciprocal Rank Fusion for combining semantic and lexical rankings.
//!
//! # Why ranks and not scores
//!
//! Cosine similarity and BM25 are not on the same scale and never will be.
//! Cosine lands in a narrow band — real bge-small commit similarities cluster
//! around 0.6–0.9 — while BM25 is unbounded and its magnitude depends on corpus
//! statistics, so the same query scores differently in a 500-commit repo than a
//! 50,000-commit one. Interpolating them (`α·cos + (1-α)·bm25`) requires
//! normalizing both per corpus, and any α that is tuned on one repository is
//! wrong on the next.
//!
//! RRF sidesteps that entirely by discarding the magnitudes and keeping only
//! the ranks:
//!
//! ```text
//! RRF(d) = Σ  1 / (k + rank_r(d))
//!          r
//! ```
//!
//! A document ranked first by either retriever gets a large contribution;
//! agreement between retrievers compounds. There is nothing to calibrate and
//! nothing to re-tune when the corpus changes.
//!
//! Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion outperforms Condorcet
//! and individual rank learning methods* (SIGIR 2009).

/// Rank-offset constant. 60 is the value from the original paper and the de
/// facto default since; it damps the difference between the top few ranks so a
/// single retriever's first place cannot dominate outright.
pub const RRF_K: f32 = 60.0;

/// One retriever's ranking: document ids, best first.
#[derive(Debug, Clone, Copy)]
pub struct Ranking<'a> {
    pub ids: &'a [u32],
}

impl<'a> Ranking<'a> {
    pub fn new(ids: &'a [u32]) -> Self {
        Self { ids }
    }
}

/// Fuse rankings and return the top `k` document ids with their RRF scores.
///
/// A document missing from a ranking simply contributes nothing for it, which
/// is what lets one retriever return 200 candidates and another 12 without any
/// padding or renormalization.
pub fn reciprocal_rank_fusion(rankings: &[Ranking<'_>], k: usize) -> Vec<(u32, f32)> {
    if k == 0 {
        return Vec::new();
    }

    // Small candidate sets (a few hundred), so a Vec of pairs beats a HashMap
    // on both allocation count and cache behaviour.
    let mut fused: Vec<(u32, f32)> = Vec::new();

    for ranking in rankings {
        for (position, &id) in ranking.ids.iter().enumerate() {
            let rank = position as f32 + 1.0; // ranks are 1-based
            let contribution = 1.0 / (RRF_K + rank);

            match fused.iter_mut().find(|(existing, _)| *existing == id) {
                Some((_, score)) => *score += contribution,
                None => fused.push((id, contribution)),
            }
        }
    }

    // Descending by score; ties break on id so output is stable across runs.
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    fused.truncate(k);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(hits: &[(u32, f32)]) -> Vec<u32> {
        hits.iter().map(|(id, _)| *id).collect()
    }

    #[test]
    fn single_ranking_passes_through_in_order() {
        let a = [7, 3, 9];
        let out = reciprocal_rank_fusion(&[Ranking::new(&a)], 10);
        assert_eq!(ids(&out), vec![7, 3, 9]);
    }

    #[test]
    fn agreement_between_retrievers_wins() {
        // 5 is second in both lists; 1 and 9 are first in one and absent in the other.
        let semantic = [1, 5, 2];
        let lexical = [9, 5, 8];

        let out = reciprocal_rank_fusion(&[Ranking::new(&semantic), Ranking::new(&lexical)], 5);
        assert_eq!(
            out[0].0, 5,
            "a document both retrievers like should outrank either one's favourite"
        );
    }

    #[test]
    fn a_unanimous_first_place_stays_first() {
        let semantic = [4, 1, 2];
        let lexical = [4, 8, 9];
        let out = reciprocal_rank_fusion(&[Ranking::new(&semantic), Ranking::new(&lexical)], 5);
        assert_eq!(out[0].0, 4);
    }

    #[test]
    fn documents_unique_to_one_retriever_are_kept() {
        let semantic = [1, 2];
        let lexical = [3, 4];
        let out = reciprocal_rank_fusion(&[Ranking::new(&semantic), Ranking::new(&lexical)], 10);

        let fused = ids(&out);
        for expected in [1, 2, 3, 4] {
            assert!(fused.contains(&expected), "lost {expected}: {fused:?}");
        }
    }

    #[test]
    fn rankings_of_different_lengths_need_no_padding() {
        // 199 is dead last semantically (rank 200) but first lexically; 5 is
        // 6th and 2nd. RRF prefers 5 — good in both beats great in one — while
        // still pulling 199 into the top 5 from rank 200.
        let semantic: Vec<u32> = (0..200).collect();
        let lexical = [199, 5];

        let out = reciprocal_rank_fusion(&[Ranking::new(&semantic), Ranking::new(&lexical)], 5);
        let fused = ids(&out);

        assert_eq!(fused[0], 5, "consistent across both retrievers should win");
        assert!(
            fused.contains(&199),
            "a lexical-only hit must still be rescued from rank 200: {fused:?}"
        );
    }

    #[test]
    fn a_lexical_only_hit_outranks_a_mediocre_semantic_one() {
        // Nothing rescues 150 (semantic rank 151, absent lexically); 199 has a
        // lexical first place. This is the exact-identifier case.
        let semantic: Vec<u32> = (0..200).collect();
        let lexical = [199];

        let out = reciprocal_rank_fusion(&[Ranking::new(&semantic), Ranking::new(&lexical)], 200);
        let fused = ids(&out);

        let position_of = |id: u32| fused.iter().position(|x| *x == id).unwrap();
        assert!(
            position_of(199) < position_of(150),
            "the keyword hit should climb above a mid-pack semantic result"
        );
    }

    #[test]
    fn empty_rankings_fuse_to_nothing() {
        assert!(reciprocal_rank_fusion(&[], 10).is_empty());

        let empty: [u32; 0] = [];
        assert!(reciprocal_rank_fusion(&[Ranking::new(&empty)], 10).is_empty());
    }

    #[test]
    fn zero_k_returns_nothing() {
        let a = [1, 2, 3];
        assert!(reciprocal_rank_fusion(&[Ranking::new(&a)], 0).is_empty());
    }

    #[test]
    fn respects_k() {
        let a = [1, 2, 3, 4, 5];
        let b = [5, 4, 3, 2, 1];
        let out = reciprocal_rank_fusion(&[Ranking::new(&a), Ranking::new(&b)], 3);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn scores_are_descending() {
        let a = [1, 2, 3, 4];
        let b = [3, 1, 5];
        let out = reciprocal_rank_fusion(&[Ranking::new(&a), Ranking::new(&b)], 10);
        for pair in out.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "out of order: {out:?}");
        }
    }

    #[test]
    fn ties_break_on_id_for_stable_output() {
        // 1 and 2 hold symmetric positions, so their scores are equal.
        let a = [1, 2];
        let b = [2, 1];
        let out = reciprocal_rank_fusion(&[Ranking::new(&a), Ranking::new(&b)], 10);
        assert_eq!(ids(&out), vec![1, 2]);
    }

    #[test]
    fn a_duplicate_within_one_ranking_only_helps_once_per_position() {
        // Defensive: a malformed ranking listing the same id twice should not
        // crash or produce a duplicate output row.
        let a = [1, 1, 2];
        let out = reciprocal_rank_fusion(&[Ranking::new(&a)], 10);
        assert_eq!(ids(&out), vec![1, 2], "no duplicate rows: {out:?}");
    }

    #[test]
    fn three_way_fusion_works() {
        let a = [1, 2];
        let b = [2, 3];
        let c = [2, 4];
        let out =
            reciprocal_rank_fusion(&[Ranking::new(&a), Ranking::new(&b), Ranking::new(&c)], 5);
        assert_eq!(out[0].0, 2, "present in all three");
    }
}
