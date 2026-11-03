//! Which embedding model is active, and what running it takes.
//!
//! BGE-small is the default. `git-semantic model <id>` swaps in any compatible
//! model: a Hugging Face repo or a local directory holding an ONNX encoder and
//! a `tokenizer.json`. Whatever the model's own settings files declare
//! (pooling, prompts, context length) is read from them; flags cover the rest.

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::EmbeddingError;

/// The name every index built before custom models recorded. The default
/// model keeps it so those indexes still match.
pub const DEFAULT_MODEL_NAME: &str = "bge-small-en-v1.5";

/// Where the default model is downloaded from.
pub const DEFAULT_MODEL_REPO: &str = "BAAI/bge-small-en-v1.5";

/// Longest input fed to a model, in tokens. Models with a longer context still
/// stop here: past this, a long diff costs time without changing the ranking.
pub const MAX_TOKENS: usize = 512;

/// Settings files a model may ship, at the paths sentence-transformers uses.
pub const SETTINGS_FILES: [&str; 3] = [
    "config.json",
    "1_Pooling/config.json",
    "config_sentence_transformers.json",
];

/// How a model's per-token vectors become one vector for the whole text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    /// The first token's vector. BGE, Arctic, and mxbai work this way.
    Cls,
    /// The average of every token's vector. MiniLM, mpnet, nomic, e5, and gte.
    Mean,
}

impl Pooling {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cls => "cls",
            Self::Mean => "mean",
        }
    }
}

/// Where a model's files come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    /// A Hugging Face repo id, like `BAAI/bge-base-en-v1.5`.
    HuggingFace(String),
    /// A directory holding `model.onnx` (or `onnx/model.onnx`) and `tokenizer.json`.
    Local(PathBuf),
}

impl ModelSource {
    /// What `git-semantic model` prints and accepts for this model.
    pub fn name(&self) -> String {
        match self {
            Self::HuggingFace(repo) => {
                if repo == DEFAULT_MODEL_REPO {
                    DEFAULT_MODEL_NAME.to_string()
                } else {
                    repo.clone()
                }
            }
            Self::Local(dir) => dir.display().to_string(),
        }
    }
}

/// Everything needed to embed text the same way every time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub source: ModelSource,
    pub pooling: Pooling,
    /// Put before every search query. Some models are trained to expect one.
    pub query_prefix: String,
    /// Put before every commit's text at indexing time.
    pub document_prefix: String,
    pub max_tokens: usize,
    /// Length of the vectors the model produces, measured when it was chosen.
    pub dimension: usize,
}

impl ModelSpec {
    pub fn default_model() -> Self {
        Self {
            source: ModelSource::HuggingFace(DEFAULT_MODEL_REPO.to_string()),
            pooling: Pooling::Cls,
            query_prefix: String::new(),
            document_prefix: String::new(),
            max_tokens: MAX_TOKENS,
            dimension: 384,
        }
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default_model()
    }

    pub fn name(&self) -> String {
        self.source.name()
    }

    /// What an index records about the model that built it.
    ///
    /// Covers every setting that changes a commit's vector, so an index built
    /// under different settings never passes for current. The query prefix is
    /// left out: it only touches queries, and the stored vectors stay valid.
    pub fn index_key(&self) -> String {
        let name = self.name();
        let is_default_model = self.source == Self::default_model().source;
        if is_default_model && self.pooling == Pooling::Cls && self.document_prefix.is_empty() {
            return name;
        }

        let mut settings = format!("{} pooling", self.pooling.as_str());
        if !self.document_prefix.is_empty() {
            settings.push_str(&format!(", document prefix {:?}", self.document_prefix));
        }
        format!("{name} ({settings})")
    }
}

/// What the user asked `git-semantic model` to switch to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelChoice {
    Default,
    HuggingFace(String),
    Local(PathBuf),
}

/// Read the argument to `git-semantic model`.
///
/// An existing directory wins over a Hugging Face id, so a local path that
/// happens to contain one slash still means the directory.
pub fn parse_model_choice(arg: &str) -> Result<ModelChoice, EmbeddingError> {
    if arg == "default" || arg == DEFAULT_MODEL_NAME || arg == DEFAULT_MODEL_REPO {
        return Ok(ModelChoice::Default);
    }

    let path = Path::new(arg);
    if path.is_dir() {
        return Ok(ModelChoice::Local(fs::canonicalize(path)?));
    }

    if is_repo_id(arg) {
        return Ok(ModelChoice::HuggingFace(arg.to_string()));
    }

    Err(EmbeddingError::InvalidModelName(arg.to_string()))
}

/// `owner/name`, using the characters Hugging Face allows in each half.
fn is_repo_id(arg: &str) -> bool {
    let parts: Vec<&str> = arg.split('/').collect();
    if parts.len() != 2 {
        return false;
    }

    for part in parts {
        if part.is_empty() || part == "." || part == ".." {
            return false;
        }
        for c in part.chars() {
            let allowed = c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.';
            if !allowed {
                return false;
            }
        }
    }
    true
}

/// Where a model's files live.
///
/// The default model keeps the flat `models/` layout that every release before
/// custom models used, so upgrading never downloads it again. Other Hugging
/// Face models get a folder each, and a local model is used where it is.
pub fn model_dir(source: &ModelSource) -> Result<PathBuf, EmbeddingError> {
    match source {
        ModelSource::Local(dir) => Ok(dir.clone()),
        ModelSource::HuggingFace(repo) => {
            let root = models_root()?;
            if repo == DEFAULT_MODEL_REPO {
                Ok(root)
            } else {
                Ok(root.join(repo.replace('/', "--")))
            }
        }
    }
}

/// The ONNX graph inside a model directory.
pub fn onnx_file(dir: &Path) -> PathBuf {
    let flat = dir.join("model.onnx");
    if flat.exists() {
        return flat;
    }
    // Where a cloned Hugging Face repo keeps it.
    dir.join("onnx").join("model.onnx")
}

pub fn model_files_present(dir: &Path) -> bool {
    onnx_file(dir).exists() && dir.join("tokenizer.json").exists()
}

/// Names of the Hugging Face models that have files on disk.
pub fn downloaded_models() -> Result<Vec<String>, EmbeddingError> {
    let root = models_root()?;
    let mut names = Vec::new();
    if model_files_present(&root) {
        names.push(DEFAULT_MODEL_NAME.to_string());
    }

    let entries = match fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(names),
        Err(err) => return Err(err.into()),
    };

    for entry in entries {
        let path = entry?.path();
        if !path.is_dir() || !model_files_present(&path) {
            continue;
        }
        if let Some(folder) = path.file_name() {
            // Hugging Face bans `--` inside ids, so it splits owner from name cleanly.
            names.push(folder.to_string_lossy().replacen("--", "/", 1));
        }
    }

    names.sort();
    Ok(names)
}

/// The model `git-semantic model` last chose, or the default.
pub fn load_active_model() -> Result<ModelSpec, EmbeddingError> {
    let path = active_model_file()?;
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(ModelSpec::default_model()),
        Err(err) => return Err(err.into()),
    };

    serde_json::from_str(&text).map_err(|err| EmbeddingError::InvalidModelConfig(err.to_string()))
}

pub fn save_active_model(spec: &ModelSpec) -> Result<(), EmbeddingError> {
    let path = active_model_file()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(spec).map_err(io::Error::other)?;
    fs::write(path, text + "\n")?;
    Ok(())
}

/// Forget the saved choice, so the built-in default applies again.
pub fn clear_active_model() -> Result<(), EmbeddingError> {
    match fs::remove_file(active_model_file()?) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

// Named for the crate, not the binary: changing these strings would strand
// every model already downloaded and the saved model choice.
fn project_dirs() -> Result<ProjectDirs, EmbeddingError> {
    ProjectDirs::from("com", "git-semantic", "git-semantic")
        .ok_or(EmbeddingError::ProjectDirsNotFound)
}

fn models_root() -> Result<PathBuf, EmbeddingError> {
    Ok(project_dirs()?.data_dir().join("models"))
}

fn active_model_file() -> Result<PathBuf, EmbeddingError> {
    Ok(project_dirs()?.config_dir().join("model.json"))
}

/// What `1_Pooling/config.json` asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclaredPooling {
    Supported(Pooling),
    /// A mode this tool can't do, like `max_tokens` or `lasttoken`.
    Unsupported(String),
    /// No pooling file, or one that sets no mode.
    Missing,
}

/// What a model's own settings files say about running it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSettings {
    pub pooling: DeclaredPooling,
    pub query_prefix: String,
    pub document_prefix: String,
    pub max_tokens: usize,
}

/// Read the sentence-transformers settings files from a model directory.
/// Any file can be missing; plenty of repos ship none of them.
pub fn read_model_settings(dir: &Path) -> ModelSettings {
    let mut settings = ModelSettings {
        pooling: DeclaredPooling::Missing,
        query_prefix: String::new(),
        document_prefix: String::new(),
        max_tokens: MAX_TOKENS,
    };

    if let Some(config) = read_json(&dir.join("1_Pooling").join("config.json")) {
        settings.pooling = declared_pooling(&config);
    }
    if let Some(config) = read_json(&dir.join("config_sentence_transformers.json")) {
        let (query, document) = declared_prompts(&config);
        settings.query_prefix = query;
        settings.document_prefix = document;
    }
    if let Some(config) = read_json(&dir.join("config.json")) {
        settings.max_tokens = declared_max_tokens(&config);
    }

    settings
}

fn read_json(path: &Path) -> Option<Value> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn declared_pooling(config: &Value) -> DeclaredPooling {
    if config["pooling_mode_cls_token"] == true {
        return DeclaredPooling::Supported(Pooling::Cls);
    }
    if config["pooling_mode_mean_tokens"] == true {
        return DeclaredPooling::Supported(Pooling::Mean);
    }

    if let Some(fields) = config.as_object() {
        for (key, value) in fields {
            if let Some(mode) = key.strip_prefix("pooling_mode_")
                && *value == true
            {
                return DeclaredPooling::Unsupported(mode.to_string());
            }
        }
    }
    DeclaredPooling::Missing
}

/// The query and document prompts, from `config_sentence_transformers.json`.
/// Repos name the document prompt either `document` or `passage`.
pub fn declared_prompts(config: &Value) -> (String, String) {
    let prompts = &config["prompts"];
    let query = prompts["query"].as_str().unwrap_or("");

    let mut document = prompts["document"].as_str().unwrap_or("");
    if document.is_empty() {
        document = prompts["passage"].as_str().unwrap_or("");
    }

    (query.to_string(), document.to_string())
}

/// The model's context length from `config.json`, capped at [`MAX_TOKENS`].
pub fn declared_max_tokens(config: &Value) -> usize {
    match config["max_position_embeddings"].as_u64() {
        Some(limit) if (limit as usize) < MAX_TOKENS => limit as usize,
        _ => MAX_TOKENS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn custom(repo: &str, pooling: Pooling) -> ModelSpec {
        ModelSpec {
            source: ModelSource::HuggingFace(repo.to_string()),
            pooling,
            query_prefix: String::new(),
            document_prefix: String::new(),
            max_tokens: MAX_TOKENS,
            dimension: 768,
        }
    }

    #[test]
    fn the_default_model_keeps_the_name_old_indexes_recorded() {
        let spec = ModelSpec::default_model();
        assert_eq!(spec.name(), "bge-small-en-v1.5");
        assert_eq!(spec.index_key(), "bge-small-en-v1.5");
        assert_eq!(spec.pooling, Pooling::Cls);
        assert_eq!(spec.max_tokens, 512);
        assert_eq!(spec.dimension, 384);
        assert!(spec.is_default());
    }

    #[test]
    fn a_custom_model_key_names_its_pooling() {
        let spec = custom("BAAI/bge-base-en-v1.5", Pooling::Cls);
        assert_eq!(spec.index_key(), "BAAI/bge-base-en-v1.5 (cls pooling)");
        assert!(!spec.is_default());
    }

    #[test]
    fn settings_that_change_stored_vectors_change_the_key() {
        let base = custom("nomic-ai/nomic-embed-text-v1.5", Pooling::Mean);

        let mut other_pooling = base.clone();
        other_pooling.pooling = Pooling::Cls;
        assert_ne!(base.index_key(), other_pooling.index_key());

        let mut with_document_prefix = base.clone();
        with_document_prefix.document_prefix = "search_document: ".to_string();
        assert_ne!(base.index_key(), with_document_prefix.index_key());
    }

    #[test]
    fn a_query_prefix_leaves_the_index_valid() {
        let mut spec = ModelSpec::default_model();
        spec.query_prefix = "Represent this sentence for searching relevant passages: ".to_string();
        assert_eq!(spec.index_key(), "bge-small-en-v1.5");
        assert!(!spec.is_default(), "but it is no longer the plain default");
    }

    #[test]
    fn the_default_model_with_other_pooling_is_a_different_index() {
        let mut spec = ModelSpec::default_model();
        spec.pooling = Pooling::Mean;
        assert_eq!(spec.index_key(), "bge-small-en-v1.5 (mean pooling)");
    }

    #[test]
    fn every_spelling_of_the_default_means_the_default() {
        for arg in ["default", "bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"] {
            assert_eq!(
                parse_model_choice(arg).unwrap(),
                ModelChoice::Default,
                "{arg}"
            );
        }
    }

    #[test]
    fn a_repo_id_is_a_hugging_face_choice() {
        assert_eq!(
            parse_model_choice("sentence-transformers/all-MiniLM-L6-v2").unwrap(),
            ModelChoice::HuggingFace("sentence-transformers/all-MiniLM-L6-v2".to_string())
        );
    }

    #[test]
    fn names_that_are_neither_a_repo_nor_a_directory_are_rejected() {
        for arg in [
            "nomic-embed-text",
            "a/b/c",
            "/bge",
            "BAAI/",
            "BAAI/bge small",
            "../x",
        ] {
            assert!(
                matches!(
                    parse_model_choice(arg),
                    Err(EmbeddingError::InvalidModelName(_))
                ),
                "{arg} should be rejected"
            );
        }
    }

    #[test]
    fn an_existing_directory_is_a_local_choice() {
        let dir = tempfile::TempDir::new().unwrap();
        let choice = parse_model_choice(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(
            choice,
            ModelChoice::Local(fs::canonicalize(dir.path()).unwrap())
        );
    }

    #[test]
    fn hugging_face_models_get_a_folder_each_and_the_default_stays_flat() {
        let default_dir = model_dir(&ModelSpec::default_model().source).unwrap();
        let other_dir = model_dir(&ModelSource::HuggingFace(
            "BAAI/bge-base-en-v1.5".to_string(),
        ))
        .unwrap();

        assert_eq!(default_dir.file_name().unwrap(), "models");
        assert_eq!(other_dir.parent().unwrap(), default_dir);
        assert_eq!(other_dir.file_name().unwrap(), "BAAI--bge-base-en-v1.5");
    }

    #[test]
    fn a_local_model_is_used_where_it_is() {
        let dir = PathBuf::from("/opt/models/mine");
        assert_eq!(model_dir(&ModelSource::Local(dir.clone())).unwrap(), dir);
    }

    #[test]
    fn the_onnx_file_may_sit_flat_or_in_an_onnx_folder() {
        let dir = tempfile::TempDir::new().unwrap();
        assert_eq!(
            onnx_file(dir.path()),
            dir.path().join("onnx").join("model.onnx")
        );

        fs::write(dir.path().join("model.onnx"), b"").unwrap();
        assert_eq!(onnx_file(dir.path()), dir.path().join("model.onnx"));
    }

    #[test]
    fn pooling_is_read_from_the_sentence_transformers_config() {
        let cls = json!({"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false});
        let mean = json!({"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true});
        let last = json!({"pooling_mode_cls_token": false, "pooling_mode_lasttoken": true});
        let none = json!({"word_embedding_dimension": 384});

        assert_eq!(
            declared_pooling(&cls),
            DeclaredPooling::Supported(Pooling::Cls)
        );
        assert_eq!(
            declared_pooling(&mean),
            DeclaredPooling::Supported(Pooling::Mean)
        );
        assert_eq!(
            declared_pooling(&last),
            DeclaredPooling::Unsupported("lasttoken".to_string())
        );
        assert_eq!(declared_pooling(&none), DeclaredPooling::Missing);
    }

    #[test]
    fn prompts_are_read_under_either_document_name() {
        let arctic = json!({"prompts": {"query": "Represent this sentence for searching relevant passages: "}});
        let mxbai = json!({"prompts": {"query": "q: ", "passage": "p: "}});
        let nomic_style =
            json!({"prompts": {"query": "search_query: ", "document": "search_document: "}});
        let none = json!({"__version__": {}});

        assert_eq!(
            declared_prompts(&arctic),
            (
                "Represent this sentence for searching relevant passages: ".to_string(),
                String::new()
            )
        );
        assert_eq!(
            declared_prompts(&mxbai),
            ("q: ".to_string(), "p: ".to_string())
        );
        assert_eq!(
            declared_prompts(&nomic_style),
            (
                "search_query: ".to_string(),
                "search_document: ".to_string()
            )
        );
        assert_eq!(declared_prompts(&none), (String::new(), String::new()));
    }

    #[test]
    fn the_context_limit_only_ever_lowers_the_cap() {
        assert_eq!(
            declared_max_tokens(&json!({"max_position_embeddings": 128})),
            128
        );
        assert_eq!(
            declared_max_tokens(&json!({"max_position_embeddings": 8192})),
            512
        );
        assert_eq!(declared_max_tokens(&json!({"hidden_size": 384})), 512);
    }

    #[test]
    fn settings_files_are_all_optional() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings = read_model_settings(dir.path());
        assert_eq!(settings.pooling, DeclaredPooling::Missing);
        assert_eq!(settings.query_prefix, "");
        assert_eq!(settings.document_prefix, "");
        assert_eq!(settings.max_tokens, MAX_TOKENS);
    }

    #[test]
    fn a_spec_survives_the_trip_through_its_saved_form() {
        let mut spec = custom("nomic-ai/nomic-embed-text-v1.5", Pooling::Mean);
        spec.query_prefix = "search_query: ".to_string();
        spec.document_prefix = "search_document: ".to_string();

        let text = serde_json::to_string_pretty(&spec).unwrap();
        let back: ModelSpec = serde_json::from_str(&text).unwrap();
        assert_eq!(back, spec);
        assert!(
            text.contains("\"pooling\": \"mean\""),
            "saved form is readable: {text}"
        );
    }
}
