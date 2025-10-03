use anyhow::{Context, Result};
use directories::ProjectDirs;
use std::fs;
use std::path::PathBuf;
use tracing::info;

use super::{Embedding, EmbeddingConfig};

pub struct ModelManager {
    config: EmbeddingConfig,
    model_dir: PathBuf,
}

impl ModelManager {
    pub fn new() -> Result<Self> {
        let config = EmbeddingConfig::default();

        let project_dirs = ProjectDirs::from("com", "git-semantic", "git-semantic")
            .context("Failed to get project directories")?;

        let model_dir = project_dirs.data_dir().join("models");
        fs::create_dir_all(&model_dir)?;

        Ok(Self { config, model_dir })
    }

    pub fn is_model_downloaded(&self) -> Result<bool> {
        let model_path = self.model_path();
        Ok(model_path.exists())
    }

    pub fn download_model(&self) -> Result<()> {
        // For MVP, we'll implement a simple download
        // In production, this would download from HuggingFace or similar
        info!("Downloading model: {}", self.config.model_name);

        // TODO: Implement actual model download
        // For now, create a placeholder file to indicate model is "downloaded"
        let model_path = self.model_path();
        fs::write(
            model_path,
            "# Model placeholder - implement actual download in Phase 1",
        )?;

        info!("Model downloaded successfully");
        Ok(())
    }

    pub fn encode_text(&self, text: &str) -> Result<Embedding> {
        // TODO: Implement actual encoding with ONNX Runtime
        // For now, return a dummy embedding
        info!("Encoding text: {}", &text[..text.len().min(50)]);

        // Placeholder: return random-ish embedding based on text hash
        let hash = text.len() % self.config.dimension;
        let mut embedding = vec![0.0; self.config.dimension];
        embedding[hash] = 1.0;

        Ok(ndarray::Array1::from_vec(embedding))
    }

    pub fn encode_batch(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        // TODO: Implement efficient batch encoding
        texts
            .iter()
            .map(|text| self.encode_text(text))
            .collect()
    }

    pub fn model_version(&self) -> String {
        self.config.model_name.clone()
    }

    fn model_path(&self) -> PathBuf {
        self.model_dir.join(format!("{}.onnx", self.config.model_name))
    }
}

