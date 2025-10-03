pub mod commands;

#[derive(Debug)]
pub struct SearchFilters {
    pub author: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
    pub file: Option<String>,
}

