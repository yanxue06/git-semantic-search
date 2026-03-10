use chrono::NaiveDate;

use crate::cli::SearchFilters;

use super::{SearchError, SearchResult};

pub struct FilterEngine {
    filters: SearchFilters,
}

impl FilterEngine {
    pub fn new(filters: SearchFilters) -> Self {
        Self { filters }
    }

    pub fn apply(&self, results: Vec<SearchResult>) -> Result<Vec<SearchResult>, SearchError> {
        let mut filtered = results;

        if let Some(ref author) = self.filters.author {
            filtered.retain(|r| r.commit.author.to_lowercase().contains(&author.to_lowercase()));
        }

        if let Some(ref after) = self.filters.after {
            let after_date = NaiveDate::parse_from_str(after, "%Y-%m-%d")
                .map_err(|e| SearchError::InvalidDateFormat {
                    value: after.clone(),
                    source: e,
                })?;
            let after_datetime = after_date.and_hms_opt(0, 0, 0).unwrap().and_utc();
            filtered.retain(|r| r.commit.date >= after_datetime);
        }

        if let Some(ref before) = self.filters.before {
            let before_date = NaiveDate::parse_from_str(before, "%Y-%m-%d")
                .map_err(|e| SearchError::InvalidDateFormat {
                    value: before.clone(),
                    source: e,
                })?;
            let before_datetime = before_date.and_hms_opt(23, 59, 59).unwrap().and_utc();
            filtered.retain(|r| r.commit.date <= before_datetime);
        }

        if let Some(ref file) = self.filters.file {
            filtered.retain(|r| r.commit.diff_summary.contains(file));
        }

        Ok(filtered)
    }
}
