mod error;
mod model;
mod spec;

pub use error::EmbeddingError;
pub use model::{ModelManager, download_repo};
pub use spec::{
    DEFAULT_MODEL_NAME, DEFAULT_MODEL_REPO, DeclaredPooling, MAX_TOKENS, ModelChoice,
    ModelSettings, ModelSource, ModelSpec, Pooling, clear_active_model, downloaded_models,
    load_active_model, model_dir, model_files_present, parse_model_choice, read_model_settings,
    save_active_model,
};

use ndarray::Array1;

pub type Embedding = Array1<f32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_type_is_384_dimensional() {
        let embedding: Embedding = Array1::zeros(384);
        assert_eq!(embedding.len(), 384);
    }
}
