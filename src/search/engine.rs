use anyhow::Result;
use ndarray::Array1;
use tracing::debug;

use crate::cli::SearchFilters;
use crate::embedding::ModelManager;
use crate::index::SemanticIndex;

use super::filter::FilterEngine;
use super::SearchResult;

pub struct SearchEngine {
    model_manager: ModelManager,
}

impl SearchEngine {
    pub fn new(mut model_manager: ModelManager) -> Result<Self> {
        // Initialize the model before use
        model_manager.init()?;
        Ok(Self { model_manager })
    }

    pub fn search(
        &mut self,
        index: &SemanticIndex,
        query: &str,
        num_results: usize,
        filters: SearchFilters,
    ) -> Result<Vec<SearchResult>> {
        debug!("Searching for: {}", query);

        // Generate query embedding
        let query_embedding = self.model_manager.encode_text(query)?;

        // Compute similarities
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

        // Apply filters
        let filter_engine = FilterEngine::new(filters);
        results = filter_engine.apply(results)?;

        // Sort by similarity
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        // Take top N results
        results.truncate(num_results);

        // Update ranks
        for (idx, result) in results.iter_mut().enumerate() {
            result.rank = idx + 1;
        }

        Ok(results)
    }
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

