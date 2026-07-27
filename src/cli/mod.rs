pub mod commands;

pub use crate::search::RetrievalMode;

#[derive(Debug)]
pub struct SearchFilters {
    pub author: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
    pub file: Option<String>,
}
