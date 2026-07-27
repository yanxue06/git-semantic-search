//! Maximal Marginal Relevance reranking.
//!
//! # The problem it solves
//!
//! Relevance ranking has no opinion about redundancy. Ask this repository for
//! "dependency update" and the top ten are ten renovate commits that differ
//! only in a crate name — technically the ten best answers, practically one
//! answer repeated ten times. The user learns nothing from results 2 through 10.
//!
//! MMR (Carbonell & Goldstein, SIGIR 1998) picks results one at a time,
//! scoring each candidate on relevance *minus* its similarity to what has
//! already been selected:
//!
//! ```text
//! MMR = argmax [ λ · rel(d) − (1 − λ) · max sim(d, s) ]
//!        d∉S                        s∈S
//! ```
//!
//! λ = 1 is pure relevance (the original ranking); λ = 0 is pure novelty. The
//! first pick is always the top-ranked result, so the best answer never moves.

use crate::vector::scoring::dot;

/// Default relevance/novelty balance.
///
/// 0.7 keeps relevance clearly dominant — this reorders near-duplicates, it
/// does not go hunting for tangents.
pub const DEFAULT_LAMBDA: f32 = 0.7;

/// One candidate: its id, its relevance, and the vector used for redundancy.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
    pub id: u32,
    pub relevance: f32,
    pub embedding: &'a [f32],
}

/// Rerank `candidates` for diversity, returning at most `k` ids.
///
/// `candidates` must be ordered best-first; `relevance` is only used for
/// relative comparison, so any monotonic score works. Embeddings are assumed
/// unit length, as they are everywhere else in the index, so similarity is a
/// dot product.
pub fn rerank(candidates: &[Candidate<'_>], k: usize, lambda: f32) -> Vec<u32> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    let lambda = lambda.clamp(0.0, 1.0);
    let limit = k.min(candidates.len());

    // Relevance is used as-is, only clamped into [0, 1].
    //
    // Min-max normalizing it across the candidate pool is tempting and wrong:
    // real cosine similarities cluster in a narrow band, so rescaling stretches
    // a 0.02 relevance gap into a 0.5 one and the redundancy term can never
    // outweigh it — λ stops meaning anything. Both terms here are cosines in
    // the same embedding space, so they are already directly comparable.
    let relevance = clamped_relevance(candidates);

    let mut selected: Vec<usize> = Vec::with_capacity(limit);
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();

    // The top-ranked result is always kept first: diversification reorders what
    // comes after the best answer, it never displaces it.
    selected.push(remaining.remove(0));

    while selected.len() < limit && !remaining.is_empty() {
        let mut best_position = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (position, &candidate) in remaining.iter().enumerate() {
            let redundancy = selected
                .iter()
                .map(|&chosen| {
                    dot(
                        candidates[candidate].embedding,
                        candidates[chosen].embedding,
                    )
                })
                .fold(f32::NEG_INFINITY, f32::max);

            let score = lambda * relevance[candidate] - (1.0 - lambda) * redundancy;

            // Strict `>` keeps the earlier (more relevant) candidate on ties,
            // which makes the output deterministic.
            if score > best_score {
                best_score = score;
                best_position = position;
            }
        }

        selected.push(remaining.remove(best_position));
    }

    selected.into_iter().map(|i| candidates[i].id).collect()
}

/// Clamp relevance into `[0, 1]`, the range the redundancy term already lives in.
///
/// A negative cosine means the result is pointing away from the query, which is
/// no more useful than zero. A non-finite score marks a fused or keyword-only
/// hit that has no cosine at all; those sit mid-pack rather than letting NaN
/// poison every comparison.
fn clamped_relevance(candidates: &[Candidate<'_>]) -> Vec<f32> {
    candidates
        .iter()
        .map(|c| {
            if c.relevance.is_finite() {
                c.relevance.clamp(0.0, 1.0)
            } else {
                0.5
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::scoring::normalize;

    /// Build a unit vector in a given "direction"; same direction => similar.
    fn vector(direction: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 8];
        v[direction % 8] = 1.0;
        v[(direction + 1) % 8] = 0.15;
        normalize(&mut v);
        v
    }

    /// Three tight clusters, descending relevance within each.
    fn clustered() -> Vec<Vec<f32>> {
        vec![
            vector(0), // 0  cluster A
            vector(0), // 1  cluster A
            vector(0), // 2  cluster A
            vector(3), // 3  cluster B
            vector(3), // 4  cluster B
            vector(6), // 5  cluster C
        ]
    }

    fn candidates<'a>(vectors: &'a [Vec<f32>], relevance: &[f32]) -> Vec<Candidate<'a>> {
        vectors
            .iter()
            .enumerate()
            .map(|(i, v)| Candidate {
                id: i as u32,
                relevance: relevance[i],
                embedding: v,
            })
            .collect()
    }

    #[test]
    fn empty_input_returns_nothing() {
        assert!(rerank(&[], 5, DEFAULT_LAMBDA).is_empty());
    }

    #[test]
    fn zero_k_returns_nothing() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]);
        assert!(rerank(&cands, 0, DEFAULT_LAMBDA).is_empty());
    }

    #[test]
    fn the_top_result_is_never_displaced() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]);

        for lambda in [0.0, 0.3, 0.5, 0.7, 1.0] {
            let out = rerank(&cands, 4, lambda);
            assert_eq!(out[0], 0, "λ={lambda} moved the best answer");
        }
    }

    #[test]
    fn lambda_one_reproduces_the_original_ranking() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]);
        assert_eq!(rerank(&cands, 6, 1.0), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn diversification_breaks_up_a_dominant_cluster() {
        // Pure relevance would return 0, 1, 2 — three near-identical commits.
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, 0.98, 0.96, 0.7, 0.68, 0.5]);

        let relevance_only = rerank(&cands, 3, 1.0);
        assert_eq!(relevance_only, vec![0, 1, 2], "sanity: cluster A dominates");

        let diverse = rerank(&cands, 3, DEFAULT_LAMBDA);
        assert_eq!(diverse[0], 0, "best answer stays");
        assert!(
            diverse.contains(&3) || diverse.contains(&5),
            "a second cluster should appear: {diverse:?}"
        );
        assert!(
            !(diverse.contains(&1) && diverse.contains(&2)),
            "should not keep all of cluster A: {diverse:?}"
        );
    }

    #[test]
    fn lambda_zero_maximizes_novelty() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, 0.98, 0.96, 0.7, 0.68, 0.5]);

        let out = rerank(&cands, 3, 0.0);
        assert_eq!(out[0], 0);
        // With relevance ignored entirely, picks 2 and 3 must come from the
        // other clusters.
        assert!(out[1..].iter().all(|id| *id >= 3), "got {out:?}");
    }

    #[test]
    fn respects_k_and_never_repeats() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]);

        let out = rerank(&cands, 4, DEFAULT_LAMBDA);
        assert_eq!(out.len(), 4);

        let mut sorted = out.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "duplicate id in output: {out:?}");
    }

    #[test]
    fn k_larger_than_input_returns_everything_once() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]);

        let out = rerank(&cands, 100, DEFAULT_LAMBDA);
        assert_eq!(out.len(), 6);

        let mut sorted = out.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn single_candidate_passes_through() {
        let vectors = vec![vector(0)];
        let cands = candidates(&vectors, &[0.9]);
        assert_eq!(rerank(&cands, 5, DEFAULT_LAMBDA), vec![0]);
    }

    #[test]
    fn lambda_is_clamped_to_a_valid_range() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]);

        assert_eq!(rerank(&cands, 3, 5.0), rerank(&cands, 3, 1.0));
        assert_eq!(rerank(&cands, 3, -2.0), rerank(&cands, 3, 0.0));
    }

    #[test]
    fn flat_relevance_falls_back_to_novelty_ordering() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[0.8; 6]);

        // No relevance signal at all — redundancy alone should decide, and
        // nothing should divide by zero.
        let out = rerank(&cands, 3, DEFAULT_LAMBDA);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], 0);
        assert!(out[1..].iter().any(|id| *id >= 3), "got {out:?}");
    }

    #[test]
    fn a_narrow_relevance_band_still_diversifies() {
        // Regression guard. Real cosine similarities cluster tightly; an
        // earlier version min-max normalized relevance across the pool, which
        // stretched a 0.02 gap into a 0.5 one and made the redundancy term
        // unable to ever win. λ has to keep meaning what it says.
        let vectors = clustered();
        let cands = candidates(&vectors, &[0.84, 0.83, 0.82, 0.81, 0.80, 0.79]);

        let out = rerank(&cands, 3, DEFAULT_LAMBDA);
        assert_eq!(out[0], 0);
        assert!(
            out[1..].iter().any(|id| *id >= 3),
            "near-identical relevance must let redundancy decide: {out:?}"
        );
    }

    #[test]
    fn negative_relevance_is_clamped_not_amplified() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[0.9, -0.5, -0.6, 0.4, 0.3, 0.2]);

        let out = rerank(&cands, 3, DEFAULT_LAMBDA);
        assert_eq!(out[0], 0);
        assert!(
            !out.contains(&1) && !out.contains(&2),
            "results pointing away from the query should not be promoted: {out:?}"
        );
    }

    #[test]
    fn non_finite_relevance_does_not_poison_selection() {
        // Fused hits carry NaN similarity; MMR must still produce k results.
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, f32::NAN, 0.8, f32::NAN, 0.6, 0.5]);

        let out = rerank(&cands, 4, DEFAULT_LAMBDA);
        assert_eq!(out.len(), 4);

        let mut sorted = out.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "NaN caused a duplicate: {out:?}");
    }

    #[test]
    fn output_is_deterministic() {
        let vectors = clustered();
        let cands = candidates(&vectors, &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]);
        assert_eq!(
            rerank(&cands, 5, DEFAULT_LAMBDA),
            rerank(&cands, 5, DEFAULT_LAMBDA)
        );
    }

    #[test]
    fn identical_vectors_still_fill_k() {
        // Pathological: every candidate is the same commit shape. MMR has no
        // diversity to find but must not stall or return short.
        let vectors: Vec<Vec<f32>> = (0..5).map(|_| vector(0)).collect();
        let cands = candidates(&vectors, &[1.0, 0.9, 0.8, 0.7, 0.6]);

        let out = rerank(&cands, 5, DEFAULT_LAMBDA);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], 0);
    }
}
