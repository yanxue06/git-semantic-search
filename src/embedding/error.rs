use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("failed to determine application data directory")]
    ProjectDirsNotFound,

    #[error("model not found — run 'git-semantic init' first")]
    ModelNotDownloaded,

    #[error("model not initialized — call init() before encoding")]
    ModelNotInitialized,

    #[error("tokenization failed: {0}")]
    Tokenization(String),

    #[error("failed to download {filename}: {reason}")]
    DownloadFailed { filename: String, reason: String },

    #[error("missing content length for download of {0}")]
    MissingContentLength(String),

    #[error(transparent)]
    Ort(#[from] ort::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    Shape(#[from] ndarray::ShapeError),
}
