use anyhow::{Context, Result};
use directories::ProjectDirs;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array1;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::fs;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tracing::{info, debug};

use super::{Embedding, EmbeddingConfig};

pub struct ModelManager {
    config: EmbeddingConfig,
    model_dir: PathBuf,
    session: Option<Session>,
    tokenizer: Option<Tokenizer>,
}

impl ModelManager {
    pub fn new() -> Result<Self> {
        let config = EmbeddingConfig::default();

        let project_dirs = ProjectDirs::from("com", "git-semantic", "git-semantic")
            .context("Failed to get project directories")?;

        let model_dir = project_dirs.data_dir().join("models");
        fs::create_dir_all(&model_dir)?;

        Ok(Self {
            config,
            model_dir,
            session: None,
            tokenizer: None,
        })
    }

    /// Initialize the model (load ONNX session and tokenizer)
    pub fn init(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }

        info!("Loading ONNX model...");
        let model_path = self.model_path();
        
        if !model_path.exists() {
            anyhow::bail!(
                "Model not found. Please run 'git-semantic init' first to download the model."
            );
        }

        // Create ONNX Runtime session
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&model_path)?;

        info!("Loading tokenizer...");
        let tokenizer_path = self.tokenizer_path();
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        self.session = Some(session);
        self.tokenizer = Some(tokenizer);

        info!("Model loaded successfully");
        Ok(())
    }

    pub fn is_model_downloaded(&self) -> Result<bool> {
        let model_path = self.model_path();
        let tokenizer_path = self.tokenizer_path();
        Ok(model_path.exists() && tokenizer_path.exists())
    }

    pub fn download_model(&self) -> Result<()> {
        info!("Downloading model: {}", self.config.model_name);

        // URLs for BGE-small-en-v1.5 ONNX model
        let base_url = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main";
        
        let files = vec![
            ("model.onnx", "onnx/model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ];

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;

        for (filename, remote_path) in files {
            let url = format!("{}/{}", base_url, remote_path);
            let target_path = self.model_dir.join(filename);

            info!("Downloading {} from {}", filename, url);

            // Download file
            let response = client.get(&url).send()?;
            
            if !response.status().is_success() {
                anyhow::bail!("Failed to download {}: HTTP {}", filename, response.status());
            }

            let total_size = response
                .content_length()
                .ok_or_else(|| anyhow::anyhow!("Failed to get content length"))?;

            // Create progress bar
            let pb = ProgressBar::new(total_size);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg}\n[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            pb.set_message(format!("Downloading {}", filename));

            // Stream download to file
            let mut file = fs::File::create(&target_path)?;
            let mut downloaded = 0u64;
            let mut content = response;

            use std::io::Write;
            let mut buffer = [0; 8192];
            
            loop {
                let bytes_read = std::io::Read::read(&mut content, &mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                file.write_all(&buffer[..bytes_read])?;
                downloaded += bytes_read as u64;
                pb.set_position(downloaded);
            }

            pb.finish_with_message(format!("✅ Downloaded {}", filename));
        }

        info!("All model files downloaded successfully");
        Ok(())
    }

    pub fn encode_text(&mut self, text: &str) -> Result<Embedding> {
        debug!("Encoding text: {}", &text[..text.len().min(50)]);

        let session = self.session.as_mut()
            .context("Model not initialized. Call init() first.")?;
        let tokenizer = self.tokenizer.as_ref()
            .context("Tokenizer not initialized. Call init() first.")?;

        // Tokenize the text
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Truncate to max length
        let max_len = self.config.max_length.min(input_ids.len());
        let input_ids = &input_ids[..max_len];
        let attention_mask = &attention_mask[..max_len];

        // Prepare inputs as 2D arrays (batch_size=1, sequence_length=max_len)
        let input_ids_array: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let attention_mask_array: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
        let token_type_ids_array: Vec<i64> = vec![0; max_len]; // BERT uses 0 for all tokens

        // Create ONNX input tensors
        use ort::value::Value;
        
        let input_ids_array_2d = ndarray::Array2::from_shape_vec(
            (1, max_len),
            input_ids_array,
        )?;
        let attention_mask_array_2d = ndarray::Array2::from_shape_vec(
            (1, max_len),
            attention_mask_array,
        )?;
        let token_type_ids_array_2d = ndarray::Array2::from_shape_vec(
            (1, max_len),
            token_type_ids_array,
        )?;
        
        let input_ids_tensor = Value::from_array((input_ids_array_2d.shape(), input_ids_array_2d.as_slice().unwrap().to_vec()))?;
        let attention_mask_tensor = Value::from_array((attention_mask_array_2d.shape(), attention_mask_array_2d.as_slice().unwrap().to_vec()))?;
        let token_type_ids_tensor = Value::from_array((token_type_ids_array_2d.shape(), token_type_ids_array_2d.as_slice().unwrap().to_vec()))?;

        // Run inference
        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];
        let outputs = session.run(inputs)?;

        // Extract the embedding from the output
        // BGE models output a tensor of shape [batch_size, sequence_length, hidden_size]
        let output_tensor = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()?;
        
        // Get the [CLS] token embedding (first token)
        let (shape, data) = output_tensor;
        let _batch_size = shape[0] as usize;
        let seq_len = shape[1] as usize;
        let hidden_size = shape[2] as usize;
        
        // Extract the [CLS] token (index 0) from the first batch
        let cls_start = 0 * seq_len * hidden_size + 0 * hidden_size;
        let cls_end = cls_start + hidden_size;
        let embedding: Vec<f32> = data[cls_start..cls_end].to_vec();

        // Normalize the embedding (L2 normalization)
        let embedding_array = Array1::from_vec(embedding);
        let norm = embedding_array.mapv(|x| x * x).sum().sqrt();
        let normalized = if norm > 0.0 {
            embedding_array / norm
        } else {
            embedding_array
        };

        Ok(normalized)
    }


    pub fn model_version(&self) -> String {
        self.config.model_name.clone()
    }

    fn model_path(&self) -> PathBuf {
        self.model_dir.join("model.onnx")
    }

    fn tokenizer_path(&self) -> PathBuf {
        self.model_dir.join("tokenizer.json")
    }
}

