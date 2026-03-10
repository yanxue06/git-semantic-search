use thiserror::Error;

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("not a git repository: .git directory not found")]
    NotAGitRepository,

    #[error("invalid .git file format — unable to resolve worktree path")]
    InvalidGitFile,

    #[error("index not found — run 'git-semantic index' first")]
    IndexNotFound,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("failed to serialize/deserialize index")]
    Bincode(#[from] bincode::Error),

    #[error(transparent)]
    Embedding(#[from] crate::embedding::EmbeddingError),
}
