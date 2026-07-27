pub mod commands;
pub mod output;

pub use crate::search::RetrievalMode;
pub use output::{JsonOutput, JsonResult};

/// Everything one `search` invocation needs.
///
/// Bundled rather than passed positionally — the flag list has grown past the
/// point where eight bare parameters are readable at the call site.
#[derive(Debug)]
pub struct SearchRequest {
    pub query: String,
    pub num_results: usize,
    pub filters: SearchFilters,
    /// Score every commit instead of using the approximate graph.
    pub exact: bool,
    /// Override the graph's candidate-list width.
    pub ef: Option<usize>,
    pub mode: RetrievalMode,
    /// MMR lambda when diversification is on.
    pub diversity: Option<f32>,
    /// Emit JSON instead of formatted text.
    pub json: bool,
}

#[derive(Debug)]
pub struct SearchFilters {
    pub author: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
    pub file: Option<String>,
}
