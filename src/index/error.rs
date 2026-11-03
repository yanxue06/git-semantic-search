use thiserror::Error;

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("not a git repository: .git directory not found")]
    NotAGitRepository,

    #[error("invalid .git file format — unable to resolve worktree path")]
    InvalidGitFile,

    #[error("index not found — run 'git sem index' first")]
    IndexNotFound,

    #[error("index is corrupted or was created by an incompatible version")]
    CorruptedIndex(#[source] bincode::Error),

    #[error("failed to write index: disk may be full or permissions denied")]
    SaveFailed(#[source] std::io::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("failed to serialize/deserialize index")]
    Bincode(#[from] bincode::Error),

    #[error(transparent)]
    Embedding(#[from] crate::embedding::EmbeddingError),

    #[error("index was built with {index}, but the active model is {active}")]
    ModelMismatch { index: String, active: String },
}

impl IndexError {
    /// Returns a user-facing hint for how to resolve this error.
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Self::NotAGitRepository => {
                Some("Make sure you're inside a git repository, or use --path to specify one.")
            }
            Self::InvalidGitFile => {
                Some("The .git file may be corrupted. This can happen with broken worktree setups.")
            }
            Self::IndexNotFound => Some("Run: git sem index"),
            Self::CorruptedIndex(_) => {
                Some("The index file is unreadable. Rebuild it with: git sem index --force")
            }
            Self::SaveFailed(_) => {
                Some("Check available disk space and file permissions on the .git directory.")
            }
            Self::Io(_) => Some("Check file permissions and available disk space."),
            Self::Bincode(_) => Some(
                "The index may be from an incompatible version. Rebuild with: git sem index --force",
            ),
            Self::Embedding(_) => None, // Delegate to EmbeddingError's own hint
            Self::ModelMismatch { .. } => {
                Some("Rebuild it with the active model: git sem index --force")
            }
        }
    }

    /// Returns an error code for programmatic identification.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotAGitRepository => "E3001",
            Self::InvalidGitFile => "E3002",
            Self::IndexNotFound => "E3003",
            Self::CorruptedIndex(_) => "E3004",
            Self::SaveFailed(_) => "E3005",
            Self::Io(_) => "E3006",
            Self::Bincode(_) => "E3007",
            Self::Embedding(_) => "E3008",
            Self::ModelMismatch { .. } => "E3009",
        }
    }
}
