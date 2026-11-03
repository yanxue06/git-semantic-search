use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("failed to determine application data directory")]
    ProjectDirsNotFound,

    #[error("model not found — run 'git sem init' first")]
    ModelNotDownloaded,

    #[error("model not initialized — call init() before encoding")]
    ModelNotInitialized,

    #[error("tokenization failed: {0}")]
    Tokenization(String),

    #[error("failed to download {filename}: {reason}")]
    DownloadFailed { filename: String, reason: String },

    #[error("missing content length for download of {0}")]
    MissingContentLength(String),

    #[error("ONNX runtime error: {}", sanitize_ort_error(.0))]
    Ort(#[from] ort::Error),

    #[error("file I/O error: {}", sanitize_io_error(.0))]
    Io(#[from] std::io::Error),

    #[error("network error: {}", sanitize_http_error(.0))]
    Http(#[from] reqwest::Error),

    #[error("tensor shape mismatch: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("'{0}' is not a Hugging Face model id or a directory")]
    InvalidModelName(String),

    #[error("{0} was not found on Hugging Face")]
    ModelNotFound(String),

    #[error("{0} has no ONNX export (looked for onnx/model.onnx and model.onnx)")]
    NoOnnxExport(String),

    #[error("{0} doesn't say how it pools token vectors")]
    UnknownPooling(String),

    #[error("{model} can't be used: {reason}")]
    UnsupportedModel { model: String, reason: String },

    #[error("no model.onnx and tokenizer.json in {0}")]
    LocalModelMissing(String),

    #[error("the saved model choice is unreadable: {0}")]
    InvalidModelConfig(String),
}

impl EmbeddingError {
    /// Returns a user-facing hint for how to resolve this error.
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Self::ProjectDirsNotFound => {
                Some("Ensure your system supports standard data directories (e.g. $HOME is set).")
            }
            Self::ModelNotDownloaded => Some("Run: git sem init"),
            Self::ModelNotInitialized => Some(
                "This is an internal error. Please report it at https://github.com/yanxue06/git-semantic-search/issues",
            ),
            Self::Tokenization(_) => {
                Some("The input text may contain unsupported characters. Try a simpler query.")
            }
            Self::DownloadFailed { .. } => Some(
                "Check your internet connection and try again. If behind a proxy, ensure HTTPS_PROXY is set.",
            ),
            Self::MissingContentLength(_) => {
                Some("The model server returned an unexpected response. Try again later.")
            }
            Self::Http(_) => Some("Check your internet connection and firewall settings."),
            Self::Ort(_) => Some("The ONNX model may be corrupted. Try: git sem init --force"),
            Self::Io(_) => Some("Check file permissions and available disk space."),
            Self::Shape(_) => {
                Some("The model produced unexpected output. Try: git sem init --force")
            }
            Self::InvalidModelName(_) => Some(
                "Use a Hugging Face id like BAAI/bge-base-en-v1.5, a directory holding model.onnx and tokenizer.json, or `default`.",
            ),
            Self::ModelNotFound(_) => {
                Some("Check the id on huggingface.co. Private and gated repos aren't supported.")
            }
            Self::NoOnnxExport(_) => Some(
                "Pick a repo that ships an ONNX file, or export one with `optimum-cli export onnx` and pass the directory.",
            ),
            Self::UnknownPooling(_) => {
                Some("Pass --pooling mean or --pooling cls. The model card says which.")
            }
            Self::UnsupportedModel { .. } => Some(
                "Supported models are BERT-style encoders: input_ids, attention_mask, and token_type_ids in; last_hidden_state or sentence_embedding out.",
            ),
            Self::LocalModelMissing(_) => {
                Some("Point at a directory holding both files. model.onnx may also sit in onnx/.")
            }
            Self::InvalidModelConfig(_) => Some("Reset it with: git sem model default"),
        }
    }

    /// A local model directory that lacks the files to run it. The path is
    /// shortened the same way I/O errors are.
    pub fn local_model_missing(dir: &std::path::Path) -> Self {
        Self::LocalModelMissing(sanitize_path_in_message(&dir.display().to_string()))
    }

    /// Returns an error code for programmatic identification.
    pub fn code(&self) -> &'static str {
        match self {
            Self::ProjectDirsNotFound => "E1001",
            Self::ModelNotDownloaded => "E1002",
            Self::ModelNotInitialized => "E1003",
            Self::Tokenization(_) => "E1004",
            Self::DownloadFailed { .. } => "E1005",
            Self::MissingContentLength(_) => "E1006",
            Self::Ort(_) => "E1007",
            Self::Io(_) => "E1008",
            Self::Http(_) => "E1009",
            Self::Shape(_) => "E1010",
            Self::InvalidModelName(_) => "E1011",
            Self::ModelNotFound(_) => "E1012",
            Self::NoOnnxExport(_) => "E1013",
            Self::UnknownPooling(_) => "E1014",
            Self::UnsupportedModel { .. } => "E1015",
            Self::LocalModelMissing(_) => "E1016",
            Self::InvalidModelConfig(_) => "E1017",
        }
    }
}

/// Sanitize ONNX runtime errors to avoid leaking internal file paths.
fn sanitize_ort_error(err: &ort::Error) -> String {
    sanitize_path_in_message(&err.to_string())
}

/// Sanitize I/O errors to avoid leaking full file system paths.
fn sanitize_io_error(err: &std::io::Error) -> String {
    sanitize_path_in_message(&err.to_string())
}

/// Sanitize HTTP errors to avoid leaking URLs with potential tokens.
fn sanitize_http_error(err: &reqwest::Error) -> String {
    let msg = err.to_string();
    // Strip query parameters which might contain tokens
    if let Some(idx) = msg.find('?') {
        format!("{}[query params redacted]", &msg[..idx])
    } else {
        msg
    }
}

/// Replace absolute paths in error messages with redacted versions
/// to avoid leaking usernames or directory structure.
fn sanitize_path_in_message(msg: &str) -> String {
    // Redact home directory paths (Unix and macOS)
    let sanitized = if let Some(home) = std::env::var_os("HOME") {
        msg.replace(&home.to_string_lossy().to_string(), "~")
    } else {
        msg.to_string()
    };

    // Redact Windows-style user paths
    regex_lite_replace_user_paths(&sanitized)
}

/// Simple replacement for Windows-style user paths (C:\Users\username\...)
fn regex_lite_replace_user_paths(msg: &str) -> String {
    // Look for C:\Users\<username>\ pattern and replace username
    if let Some(idx) = msg.find(":\\Users\\")
        && let Some(end_idx) = msg[idx + 8..].find('\\')
    {
        let before = &msg[..idx + 8];
        let after = &msg[idx + 8 + end_idx..];
        return format!("{}<user>{}", before, after);
    }
    msg.to_string()
}
