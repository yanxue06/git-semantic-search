mod model;

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

