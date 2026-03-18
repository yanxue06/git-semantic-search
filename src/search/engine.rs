use ndarray::Array1;
use tracing::debug;

use crate::cli::SearchFilters;
use crate::embedding::ModelManager;
use crate::index::SemanticIndex;

use super::filter::FilterEngine;
use super::{SearchError, SearchResult};

pub struct SearchEngine {
    model_manager: ModelManager,
}

impl SearchEngine {
    pub fn new(mut model_manager: ModelManager) -> Result<Self, SearchError> {
        model_manager.init()?;
        Ok(Self { model_manager })
    }

    pub fn search(
        &mut self,
        index: &SemanticIndex,
        query: &str,
        num_results: usize,
        filters: SearchFilters,
    ) -> Result<Vec<SearchResult>, SearchError> {
        debug!("Searching for: {}", query);

        let query_embedding = self.model_manager.encode_text(query)?;

        let mut results: Vec<SearchResult> = index
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let embedding = Array1::from_vec(entry.embedding.clone());
                let similarity = cosine_similarity(&query_embedding, &embedding);

                SearchResult {
                    commit: entry.commit.clone(),
                    similarity,
                    rank: idx + 1,
                }
            })
            .collect();

        let filter_engine = FilterEngine::new(filters);
        results = filter_engine.apply(results)?;

        results.sort_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(num_results);

        for (idx, result) in results.iter_mut().enumerate() {
            result.rank = idx + 1;
        }

        Ok(results)
    }
}

pub(crate) fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should have similarity ~1.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have similarity ~0.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![-1.0, -2.0, -3.0]);
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "opposite vectors should have similarity ~-1.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "zero vector should give similarity 0.0");
    }

    #[test]
    fn test_cosine_similarity_normalized_vectors() {
        // Pre-normalized vectors (unit length)
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let val = std::f32::consts::FRAC_1_SQRT_2;
        let b = Array1::from_vec(vec![val, val]); // 45 degrees
        let sim = cosine_similarity(&a, &b);
        assert!((sim - val).abs() < 0.01, "45-degree angle should give ~0.707, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_384_dimensions() {
        // Simulate real embedding dimension (384 for bge-small-en-v1.5)
        let a = Array1::from_vec((0..384).map(|i| (i as f32) / 384.0).collect());
        let b = Array1::from_vec((0..384).map(|i| ((383 - i) as f32) / 384.0).collect());
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.0 && sim < 1.0, "should be between 0 and 1, got {sim}");
    }
}
