mod error;
mod model;

pub use error::EmbeddingError;
pub use model::ModelManager;

use ndarray::Array1;
use serde::{Deserialize, Serialize};

pub type Embedding = Array1<f32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_name: String,
    pub dimension: usize,
    pub max_length: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "bge-small-en-v1.5".to_string(),
            dimension: 384,
            max_length: 512,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config_defaults() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model_name, "bge-small-en-v1.5");
        assert_eq!(config.dimension, 384);
        assert_eq!(config.max_length, 512);
    }

    #[test]
    fn test_embedding_type_is_384_dimensional() {
        let embedding: Embedding = Array1::zeros(384);
        assert_eq!(embedding.len(), 384);
    }
}
