use thiserror::Error;

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("invalid date format '{value}' — expected YYYY-MM-DD")]
    InvalidDateFormat {
        value: String,
        #[source]
        source: chrono::ParseError,
    },

    #[error(transparent)]
    Embedding(#[from] crate::embedding::EmbeddingError),
}
