use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array1;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use super::spec::{
    ModelSource, ModelSpec, Pooling, SETTINGS_FILES, load_active_model, model_dir,
    model_files_present, onnx_file,
};
use super::{Embedding, EmbeddingError};

const HUGGING_FACE: &str = "https://huggingface.co";

/// Where repos keep the ONNX export, most common first.
const ONNX_LOCATIONS: [&str; 2] = ["onnx/model.onnx", "model.onnx"];

/// The inputs a BERT-style encoder can declare. A model that asks for anything
/// else needs feeding this tool doesn't know how to do.
const KNOWN_INPUTS: [&str; 3] = ["input_ids", "attention_mask", "token_type_ids"];

pub struct ModelManager {
    spec: ModelSpec,
    model_dir: PathBuf,
    session: Option<Session>,
    tokenizer: Option<Tokenizer>,
    /// The inputs the loaded graph declares, in its own order.
    inputs: Vec<String>,
    /// The output vectors are read from.
    output: String,
    /// Whether that output is one vector per text already, so no pooling.
    output_is_pooled: bool,
}

impl ModelManager {
    /// The active model: whichever `git sem model` last chose.
    pub fn new() -> Result<Self, EmbeddingError> {
        Self::with_spec(load_active_model()?)
    }

    pub fn with_spec(spec: ModelSpec) -> Result<Self, EmbeddingError> {
        let model_dir = model_dir(&spec.source)?;
        Ok(Self {
            spec,
            model_dir,
            session: None,
            tokenizer: None,
            inputs: Vec::new(),
            output: String::new(),
            output_is_pooled: false,
        })
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Load the ONNX session and tokenizer, and work out how to feed the graph.
    pub fn init(&mut self) -> Result<(), EmbeddingError> {
        if self.session.is_some() {
            return Ok(());
        }

        if !self.is_model_downloaded() {
            return Err(self.missing_files_error());
        }

        info!("Loading ONNX model...");
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(onnx_file(&self.model_dir))?;

        let name = self.spec.name();
        self.inputs = graph_inputs(&session, &name)?;
        let (output, output_is_pooled) = graph_output(&session, &name)?;
        self.output = output;
        self.output_is_pooled = output_is_pooled;

        info!("Loading tokenizer...");
        let mut tokenizer = Tokenizer::from_file(self.model_dir.join("tokenizer.json"))
            .map_err(|e| EmbeddingError::Tokenization(format!("failed to load tokenizer: {e}")))?;
        // Texts are encoded one at a time, so padding only adds work. Some
        // repos ship a tokenizer that pads everything to a fixed length:
        // all-MiniLM-L6-v2 pads each commit to 128 tokens.
        tokenizer.with_padding(None);

        self.session = Some(session);
        self.tokenizer = Some(tokenizer);

        info!("Model loaded successfully");
        Ok(())
    }

    pub fn is_model_downloaded(&self) -> bool {
        model_files_present(&self.model_dir)
    }

    /// Fetch the files that running the model needs. A local model has nowhere
    /// to fetch from, so missing files there are an error.
    pub fn download_model(&self) -> Result<(), EmbeddingError> {
        match &self.spec.source {
            ModelSource::HuggingFace(repo) => {
                info!("Downloading model: {repo}");
                fs::create_dir_all(&self.model_dir)?;
                download_model_files(&http_client()?, repo, &self.model_dir)?;
                info!("All model files downloaded successfully");
                Ok(())
            }
            ModelSource::Local(_) => Err(self.missing_files_error()),
        }
    }

    fn missing_files_error(&self) -> EmbeddingError {
        match &self.spec.source {
            ModelSource::HuggingFace(_) => EmbeddingError::ModelNotDownloaded,
            ModelSource::Local(dir) => EmbeddingError::local_model_missing(dir),
        }
    }

    /// Embed a search query, with the prefix the model expects on queries.
    pub fn encode_query(&mut self, query: &str) -> Result<Embedding, EmbeddingError> {
        let text = format!("{}{}", self.spec.query_prefix, query);
        self.encode_text(&text)
    }

    /// Embed a commit's text, with the prefix the model expects on documents.
    pub fn encode_document(&mut self, document: &str) -> Result<Embedding, EmbeddingError> {
        let text = format!("{}{}", self.spec.document_prefix, document);
        self.encode_text(&text)
    }

    pub fn encode_text(&mut self, text: &str) -> Result<Embedding, EmbeddingError> {
        debug!("Encoding text: {}", &text[..text.len().min(50)]);

        let session = self
            .session
            .as_mut()
            .ok_or(EmbeddingError::ModelNotInitialized)?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(EmbeddingError::ModelNotInitialized)?;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| EmbeddingError::Tokenization(e.to_string()))?;

        let length = self.spec.max_tokens.min(encoding.get_ids().len());
        let mut input_ids = Vec::with_capacity(length);
        let mut attention_mask = Vec::with_capacity(length);
        for position in 0..length {
            input_ids.push(encoding.get_ids()[position] as i64);
            attention_mask.push(encoding.get_attention_mask()[position] as i64);
        }

        // Feed exactly the inputs the graph declares: mpnet and jina take no
        // token_type_ids, and BERT rejects a call that leaves one out.
        let mut inputs = Vec::with_capacity(self.inputs.len());
        for name in &self.inputs {
            let values = match name.as_str() {
                "input_ids" => input_ids.clone(),
                "attention_mask" => attention_mask.clone(),
                _ => vec![0; length],
            };
            inputs.push((name.clone(), Tensor::from_array(([1, length], values))?));
        }

        let outputs = session.run(inputs)?;
        let (shape, data) = outputs[self.output.as_str()].try_extract_tensor::<f32>()?;

        let vector = if self.output_is_pooled {
            let width = shape[1] as usize;
            data[..width].to_vec()
        } else {
            let tokens = shape[1] as usize;
            let width = shape[2] as usize;
            pool(data, tokens, width, &attention_mask, self.spec.pooling)
        };

        let embedding_array = Array1::from_vec(vector);
        let norm = embedding_array.mapv(|x| x * x).sum().sqrt();
        let normalized = if norm > 0.0 {
            embedding_array / norm
        } else {
            embedding_array
        };

        Ok(normalized)
    }

    /// What indexes built with this model record. See [`ModelSpec::index_key`].
    pub fn model_version(&self) -> String {
        self.spec.index_key()
    }
}

/// The graph's inputs, checked against the ones this tool can supply.
fn graph_inputs(session: &Session, model: &str) -> Result<Vec<String>, EmbeddingError> {
    let mut inputs = Vec::new();
    for input in &session.inputs {
        if !KNOWN_INPUTS.contains(&input.name.as_str()) {
            return Err(EmbeddingError::UnsupportedModel {
                model: model.to_string(),
                reason: format!("it takes an input named `{}`", input.name),
            });
        }
        inputs.push(input.name.clone());
    }

    if !inputs.iter().any(|name| name == "input_ids") {
        return Err(EmbeddingError::UnsupportedModel {
            model: model.to_string(),
            reason: "it has no `input_ids` input".to_string(),
        });
    }
    Ok(inputs)
}

/// Pick the output to read vectors from, and say whether it is already pooled.
///
/// `sentence_embedding` is the model's own finished vector, including any
/// projection after pooling, so it wins when present. Otherwise the per-token
/// output gets pooled here.
fn graph_output(session: &Session, model: &str) -> Result<(String, bool), EmbeddingError> {
    let mut names = Vec::new();
    for output in &session.outputs {
        names.push(output.name.clone());
    }

    let has = |wanted: &str| names.iter().any(|name| name == wanted);
    if has("sentence_embedding") {
        return Ok(("sentence_embedding".to_string(), true));
    }
    for per_token in ["last_hidden_state", "token_embeddings"] {
        if has(per_token) {
            return Ok((per_token.to_string(), false));
        }
    }

    Err(EmbeddingError::UnsupportedModel {
        model: model.to_string(),
        reason: format!(
            "it has no last_hidden_state or sentence_embedding output (it has: {})",
            names.join(", ")
        ),
    })
}

/// Turn `tokens` rows of `width` floats into one vector.
fn pool(data: &[f32], tokens: usize, width: usize, mask: &[i64], pooling: Pooling) -> Vec<f32> {
    match pooling {
        Pooling::Cls => data[..width].to_vec(),
        Pooling::Mean => {
            let mut sum = vec![0.0f32; width];
            let mut counted = 0.0f32;
            for token in 0..tokens {
                if mask[token] == 0 {
                    continue;
                }
                let row = &data[token * width..(token + 1) * width];
                for (total, value) in sum.iter_mut().zip(row) {
                    *total += value;
                }
                counted += 1.0;
            }

            if counted > 0.0 {
                for total in sum.iter_mut() {
                    *total /= counted;
                }
            }
            sum
        }
    }
}

fn http_client() -> Result<reqwest::blocking::Client, EmbeddingError> {
    // No overall deadline: a 1 GB model on a slow link can outlast any fixed
    // one. Connecting still has to happen promptly.
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(None)
        .build()?;
    Ok(client)
}

/// Fetch everything `git sem model` reads from a Hugging Face repo into `dir`.
pub fn download_repo(repo: &str, dir: &Path) -> Result<(), EmbeddingError> {
    fs::create_dir_all(dir)?;
    let client = http_client()?;

    // The settings files come first and are small. Many repos lack some or
    // all of them, which is fine.
    for remote in SETTINGS_FILES {
        fetch_small_file(&client, repo, remote, &dir.join(remote))?;
    }

    download_model_files(&client, repo, dir)
}

/// The ONNX graph and the tokenizer, which is all running a model needs.
///
/// The graph goes first: a repo without one is the usual reason a model
/// can't be used, and that check costs two small requests.
fn download_model_files(
    client: &reqwest::blocking::Client,
    repo: &str,
    dir: &Path,
) -> Result<(), EmbeddingError> {
    let graph = dir.join("model.onnx");
    let mut found_graph = false;
    for remote in ONNX_LOCATIONS {
        if download_with_progress(client, repo, remote, &graph)? {
            found_graph = true;
            break;
        }
    }
    if !found_graph {
        return Err(EmbeddingError::NoOnnxExport(repo.to_string()));
    }

    let tokenizer = dir.join("tokenizer.json");
    if !download_with_progress(client, repo, "tokenizer.json", &tokenizer)? {
        return Err(EmbeddingError::UnsupportedModel {
            model: repo.to_string(),
            reason: "it has no tokenizer.json".to_string(),
        });
    }
    Ok(())
}

fn file_url(repo: &str, remote: &str) -> String {
    format!("{HUGGING_FACE}/{repo}/resolve/main/{remote}")
}

/// Whether a response carries the file: true for success, false when the repo
/// exists but lacks it. Hugging Face answers 401 for a repo that doesn't exist
/// and for gated ones, so those mean the repo itself can't be reached.
fn file_exists(
    status: reqwest::StatusCode,
    repo: &str,
    remote: &str,
) -> Result<bool, EmbeddingError> {
    if status.is_success() {
        return Ok(true);
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        debug!("{repo} has no {remote}");
        return Ok(false);
    }
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(EmbeddingError::ModelNotFound(repo.to_string()));
    }
    Err(EmbeddingError::DownloadFailed {
        filename: remote.to_string(),
        reason: format!("HTTP {status}"),
    })
}

/// Fetch a small settings file. Returns false when the repo doesn't have it.
fn fetch_small_file(
    client: &reqwest::blocking::Client,
    repo: &str,
    remote: &str,
    target: &Path,
) -> Result<bool, EmbeddingError> {
    let response = client.get(file_url(repo, remote)).send()?;
    if !file_exists(response.status(), repo, remote)? {
        return Ok(false);
    }

    let bytes = response.bytes()?;
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(target, &bytes)?;
    Ok(true)
}

/// Download one file with a progress bar. Returns false when the repo doesn't
/// have it.
///
/// Writes to a `.part` file and renames it at the end, so an interrupted
/// download never leaves a truncated model that looks complete.
fn download_with_progress(
    client: &reqwest::blocking::Client,
    repo: &str,
    remote: &str,
    target: &Path,
) -> Result<bool, EmbeddingError> {
    let url = file_url(repo, remote);
    info!("Downloading {remote} from {url}");

    let mut response = client.get(&url).send()?;
    if !file_exists(response.status(), repo, remote)? {
        return Ok(false);
    }

    let total_size = response
        .content_length()
        .ok_or_else(|| EmbeddingError::MissingContentLength(remote.to_string()))?;

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg}\n[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message(format!("Downloading {remote}"));

    let mut partial_name = target.as_os_str().to_owned();
    partial_name.push(".part");
    let partial = PathBuf::from(partial_name);

    let mut file = fs::File::create(&partial)?;
    let mut downloaded = 0u64;
    let mut buffer = [0; 8192];

    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }

    file.flush()?;
    drop(file);
    fs::rename(&partial, target)?;

    pb.finish_with_message(format!("✅ Downloaded {remote}"));
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cls_pooling_takes_the_first_token() {
        let data = [1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
        assert_eq!(pool(&data, 3, 2, &[1, 1, 1], Pooling::Cls), vec![1.0, 2.0]);
    }

    #[test]
    fn mean_pooling_averages_the_tokens_the_mask_keeps() {
        let data = [1.0, 2.0, 3.0, 4.0, 1000.0, 1000.0];
        assert_eq!(pool(&data, 3, 2, &[1, 1, 0], Pooling::Mean), vec![2.0, 3.0]);
    }

    #[test]
    fn mean_pooling_with_nothing_kept_is_zero_not_nan() {
        let data = [5.0, 5.0];
        assert_eq!(pool(&data, 1, 2, &[0], Pooling::Mean), vec![0.0, 0.0]);
    }

    #[test]
    fn files_resolve_under_the_main_branch() {
        assert_eq!(
            file_url("BAAI/bge-small-en-v1.5", "onnx/model.onnx"),
            "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx"
        );
    }

    #[test]
    fn a_missing_file_is_not_a_missing_repo() {
        use reqwest::StatusCode;
        let check = |status| file_exists(status, "owner/name", "tokenizer.json");

        assert!(check(StatusCode::OK).unwrap());
        assert!(!check(StatusCode::NOT_FOUND).unwrap());
        assert!(matches!(
            check(StatusCode::UNAUTHORIZED),
            Err(EmbeddingError::ModelNotFound(_))
        ));
        assert!(matches!(
            check(StatusCode::INTERNAL_SERVER_ERROR),
            Err(EmbeddingError::DownloadFailed { .. })
        ));
    }
}
