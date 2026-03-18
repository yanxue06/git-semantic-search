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
            filtered.retain(|r| {
                r.commit
                    .author
                    .to_lowercase()
                    .contains(&author.to_lowercase())
            });
        }

        if let Some(ref after) = self.filters.after {
            let after_date = NaiveDate::parse_from_str(after, "%Y-%m-%d").map_err(|e| {
                SearchError::InvalidDateFormat {
                    value: after.clone(),
                    source: e,
                }
            })?;
            let after_datetime = after_date.and_hms_opt(0, 0, 0).unwrap().and_utc();
            filtered.retain(|r| r.commit.date >= after_datetime);
        }

        if let Some(ref before) = self.filters.before {
            let before_date = NaiveDate::parse_from_str(before, "%Y-%m-%d").map_err(|e| {
                SearchError::InvalidDateFormat {
                    value: before.clone(),
                    source: e,
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::CommitInfo;

    fn make_result(author: &str, date: &str, diff_summary: &str) -> SearchResult {
        SearchResult {
            commit: CommitInfo {
                hash: "abc1234".to_string(),
                author: author.to_string(),
                date: chrono::DateTime::parse_from_rfc3339(&format!("{date}T12:00:00Z"))
                    .unwrap()
                    .with_timezone(&chrono::Utc),
                message: "test commit".to_string(),
                diff_summary: diff_summary.to_string(),
            },
            similarity: 0.9,
            rank: 1,
        }
    }

    fn no_filters() -> SearchFilters {
        SearchFilters {
            author: None,
            after: None,
            before: None,
            file: None,
        }
    }

    #[test]
    fn test_no_filters_passes_all() {
        let results = vec![
            make_result("Alice", "2024-06-01", ""),
            make_result("Bob", "2024-07-01", ""),
        ];
        let engine = FilterEngine::new(no_filters());
        let filtered = engine.apply(results).unwrap();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_author_filter_case_insensitive() {
        let results = vec![
            make_result("Alice Chen", "2024-06-01", ""),
            make_result("Bob Martinez", "2024-07-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            author: Some("alice".to_string()),
            ..no_filters()
        });
        let filtered = engine.apply(results).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].commit.author, "Alice Chen");
    }

    #[test]
    fn test_author_filter_partial_match() {
        let results = vec![
            make_result("Alice Chen", "2024-06-01", ""),
            make_result("Bob Martinez", "2024-07-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            author: Some("chen".to_string()),
            ..no_filters()
        });
        let filtered = engine.apply(results).unwrap();
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_after_date_filter() {
        let results = vec![
            make_result("Alice", "2024-01-15", ""),
            make_result("Bob", "2024-06-15", ""),
            make_result("Carol", "2024-12-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            after: Some("2024-06-01".to_string()),
            ..no_filters()
        });
        let filtered = engine.apply(results).unwrap();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_before_date_filter() {
        let results = vec![
            make_result("Alice", "2024-01-15", ""),
            make_result("Bob", "2024-06-15", ""),
            make_result("Carol", "2024-12-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            before: Some("2024-06-30".to_string()),
            ..no_filters()
        });
        let filtered = engine.apply(results).unwrap();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_date_range_filter() {
        let results = vec![
            make_result("Alice", "2024-01-15", ""),
            make_result("Bob", "2024-06-15", ""),
            make_result("Carol", "2024-12-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            after: Some("2024-03-01".to_string()),
            before: Some("2024-09-01".to_string()),
            ..no_filters()
        });
        let filtered = engine.apply(results).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].commit.author, "Bob");
    }

    #[test]
    fn test_file_filter() {
        let results = vec![
            make_result("Alice", "2024-06-01", "+modified src/auth.rs\n-old line"),
            make_result("Bob", "2024-07-01", "+modified src/main.rs"),
        ];
        let engine = FilterEngine::new(SearchFilters {
            file: Some("src/auth.rs".to_string()),
            ..no_filters()
        });
        let filtered = engine.apply(results).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].commit.author, "Alice");
    }

    #[test]
    fn test_combined_filters() {
        let results = vec![
            make_result("Alice", "2024-06-01", "+src/auth.rs"),
            make_result("Alice", "2024-01-01", "+src/auth.rs"),
            make_result("Bob", "2024-06-01", "+src/auth.rs"),
        ];
        let engine = FilterEngine::new(SearchFilters {
            author: Some("alice".to_string()),
            after: Some("2024-03-01".to_string()),
            file: Some("src/auth.rs".to_string()),
            before: None,
        });
        let filtered = engine.apply(results).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].commit.author, "Alice");
        assert_eq!(
            filtered[0].commit.date.format("%Y-%m-%d").to_string(),
            "2024-06-01"
        );
    }

    #[test]
    fn test_invalid_date_format_returns_error() {
        let results = vec![make_result("Alice", "2024-06-01", "")];
        let engine = FilterEngine::new(SearchFilters {
            after: Some("not-a-date".to_string()),
            ..no_filters()
        });
        assert!(engine.apply(results).is_err());
    }

    #[test]
    fn test_filter_returns_empty_when_nothing_matches() {
        let results = vec![make_result("Alice", "2024-06-01", "")];
        let engine = FilterEngine::new(SearchFilters {
            author: Some("nonexistent".to_string()),
            ..no_filters()
        });
        let filtered = engine.apply(results).unwrap();
        assert!(filtered.is_empty());
    }
}
