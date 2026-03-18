use thiserror::Error;

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("invalid date format '{value}' — expected YYYY-MM-DD")]
    InvalidDateFormat {
        value: String,
        #[source]
        source: chrono::ParseError,
    },

    #[error("no index found for this repository")]
    IndexNotLoaded,

    #[error(transparent)]
    Embedding(#[from] crate::embedding::EmbeddingError),
}

impl SearchError {
    /// Returns a user-facing hint for how to resolve this error.
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Self::InvalidDateFormat { .. } => Some(
                "Use the format YYYY-MM-DD, e.g. --after 2024-01-15"
            ),
            Self::IndexNotLoaded => Some(
                "Run: git-semantic index"
            ),
            Self::Embedding(_) => None, // Delegate to EmbeddingError's own hint
        }
    }

    /// Returns an error code for programmatic identification.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidDateFormat { .. } => "E4001",
            Self::IndexNotLoaded => "E4002",
            Self::Embedding(_) => "E4003",
        }
    }
}
