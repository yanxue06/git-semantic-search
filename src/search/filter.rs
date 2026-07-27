use chrono::{DateTime, NaiveDate, Utc};

use crate::cli::SearchFilters;
use crate::git::CommitInfo;

use super::SearchError;

/// Metadata filters, compiled once per query.
///
/// The previous version re-parsed `--after` and `--before` on every call and
/// only knew how to transform a fully materialized result list. Search now
/// evaluates filters *before* scoring, so it needs a per-commit predicate that
/// is cheap enough to run N times — hence parsing up front and lowercasing the
/// author once.
pub struct FilterEngine {
    author: Option<String>,
    after: Option<DateTime<Utc>>,
    before: Option<DateTime<Utc>>,
    file: Option<String>,
}

impl FilterEngine {
    /// Compile raw CLI strings, surfacing date errors immediately.
    pub fn new(filters: SearchFilters) -> Result<Self, SearchError> {
        let after = filters
            .after
            .as_deref()
            .map(|raw| parse_day(raw, DayEdge::Start))
            .transpose()?;

        let before = filters
            .before
            .as_deref()
            .map(|raw| parse_day(raw, DayEdge::End))
            .transpose()?;

        Ok(Self {
            author: filters.author.map(|a| a.to_lowercase()),
            after,
            before,
            file: filters.file,
        })
    }

    /// True when at least one filter would exclude something.
    pub fn is_active(&self) -> bool {
        self.author.is_some()
            || self.after.is_some()
            || self.before.is_some()
            || self.file.is_some()
    }

    /// Whether `commit` passes every active filter.
    pub fn matches(&self, commit: &CommitInfo) -> bool {
        if let Some(author) = &self.author
            && !commit.author.to_lowercase().contains(author)
        {
            return false;
        }

        if let Some(after) = self.after
            && commit.date < after
        {
            return false;
        }

        if let Some(before) = self.before
            && commit.date > before
        {
            return false;
        }

        // Matches against the commit's recorded path list, not the raw diff
        // text, so `--file src/auth.rs` no longer depends on that string
        // happening to appear inside an added or removed line.
        if let Some(file) = &self.file
            && !commit.touches_path(file)
        {
            return false;
        }

        true
    }
}

enum DayEdge {
    Start,
    End,
}

/// Parse `YYYY-MM-DD` to an inclusive bound at the requested edge of the day.
fn parse_day(raw: &str, edge: DayEdge) -> Result<DateTime<Utc>, SearchError> {
    let date =
        NaiveDate::parse_from_str(raw, "%Y-%m-%d").map_err(|e| SearchError::InvalidDateFormat {
            value: raw.to_string(),
            source: e,
        })?;

    let time = match edge {
        DayEdge::Start => date.and_hms_opt(0, 0, 0),
        DayEdge::End => date.and_hms_opt(23, 59, 59),
    };

    // Both literals are valid for every representable date, so this cannot fail;
    // the fallback keeps the function total instead of panicking.
    Ok(time
        .unwrap_or_else(|| date.and_hms_opt(0, 0, 0).unwrap_or_default())
        .and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commit(author: &str, date: &str, diff_summary: &str) -> CommitInfo {
        CommitInfo {
            hash: "abc1234".to_string(),
            author: author.to_string(),
            date: chrono::DateTime::parse_from_rfc3339(&format!("{date}T12:00:00Z"))
                .unwrap()
                .with_timezone(&chrono::Utc),
            message: "test commit".to_string(),
            diff_summary: diff_summary.to_string(),
        }
    }

    /// Authors of the commits that survive the filter — the pre-scoring
    /// candidate set the engine will actually score.
    fn surviving(engine: &FilterEngine, commits: &[CommitInfo]) -> Vec<String> {
        commits
            .iter()
            .filter(|c| !engine.is_active() || engine.matches(c))
            .map(|c| c.author.clone())
            .collect()
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
        let commits = [
            commit("Alice", "2024-06-01", ""),
            commit("Bob", "2024-07-01", ""),
        ];
        let engine = FilterEngine::new(no_filters()).unwrap();
        assert!(!engine.is_active());
        assert_eq!(surviving(&engine, &commits).len(), 2);
    }

    #[test]
    fn test_author_filter_case_insensitive() {
        let commits = [
            commit("Alice Chen", "2024-06-01", ""),
            commit("Bob Martinez", "2024-07-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            author: Some("alice".to_string()),
            ..no_filters()
        })
        .unwrap();
        assert_eq!(surviving(&engine, &commits), vec!["Alice Chen"]);
    }

    #[test]
    fn test_author_filter_partial_match() {
        let commits = [
            commit("Alice Chen", "2024-06-01", ""),
            commit("Bob Martinez", "2024-07-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            author: Some("chen".to_string()),
            ..no_filters()
        })
        .unwrap();
        assert_eq!(surviving(&engine, &commits), vec!["Alice Chen"]);
    }

    #[test]
    fn test_after_date_filter() {
        let commits = [
            commit("Alice", "2024-01-15", ""),
            commit("Bob", "2024-06-15", ""),
            commit("Carol", "2024-12-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            after: Some("2024-06-01".to_string()),
            ..no_filters()
        })
        .unwrap();
        assert_eq!(surviving(&engine, &commits), vec!["Bob", "Carol"]);
    }

    #[test]
    fn test_before_date_filter() {
        let commits = [
            commit("Alice", "2024-01-15", ""),
            commit("Bob", "2024-06-15", ""),
            commit("Carol", "2024-12-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            before: Some("2024-06-30".to_string()),
            ..no_filters()
        })
        .unwrap();
        assert_eq!(surviving(&engine, &commits), vec!["Alice", "Bob"]);
    }

    #[test]
    fn test_date_range_filter() {
        let commits = [
            commit("Alice", "2024-01-15", ""),
            commit("Bob", "2024-06-15", ""),
            commit("Carol", "2024-12-01", ""),
        ];
        let engine = FilterEngine::new(SearchFilters {
            after: Some("2024-03-01".to_string()),
            before: Some("2024-09-01".to_string()),
            ..no_filters()
        })
        .unwrap();
        assert_eq!(surviving(&engine, &commits), vec!["Bob"]);
    }

    #[test]
    fn test_file_filter() {
        let commits = [
            commit(
                "Alice",
                "2024-06-01",
                "Files: src/auth.rs\n+let token = ...;",
            ),
            commit("Bob", "2024-07-01", "Files: src/main.rs\n+fn main() {}"),
        ];
        let engine = FilterEngine::new(SearchFilters {
            file: Some("src/auth.rs".to_string()),
            ..no_filters()
        })
        .unwrap();
        assert_eq!(surviving(&engine, &commits), vec!["Alice"]);
    }

    #[test]
    fn test_file_filter_matches_one_of_several_changed_paths() {
        let commits = [commit(
            "Alice",
            "2024-06-01",
            "Files: Cargo.toml, src/auth.rs, src/main.rs\n+something",
        )];
        for needle in ["Cargo.toml", "src/auth.rs", "src/main.rs", "src/"] {
            let engine = FilterEngine::new(SearchFilters {
                file: Some(needle.to_string()),
                ..no_filters()
            })
            .unwrap();
            assert_eq!(
                surviving(&engine, &commits).len(),
                1,
                "{needle} should match"
            );
        }
    }

    #[test]
    fn test_file_filter_ignores_paths_mentioned_only_in_diff_content() {
        // The commit edits main.rs and merely *mentions* auth.rs in a removed
        // line. Substring-searching the whole diff would wrongly match.
        let commits = [commit(
            "Alice",
            "2024-06-01",
            "Files: src/main.rs\n-use crate::src/auth.rs::Token;",
        )];
        let engine = FilterEngine::new(SearchFilters {
            file: Some("src/auth.rs".to_string()),
            ..no_filters()
        })
        .unwrap();
        assert!(
            surviving(&engine, &commits).is_empty(),
            "a path only referenced in diff text is not a changed file"
        );
    }

    #[test]
    fn test_file_filter_falls_back_on_a_legacy_index() {
        // No `Files:` line — an index built before paths were recorded.
        let commits = [commit("Alice", "2024-06-01", "+modified src/auth.rs")];
        let engine = FilterEngine::new(SearchFilters {
            file: Some("src/auth.rs".to_string()),
            ..no_filters()
        })
        .unwrap();
        assert_eq!(
            surviving(&engine, &commits).len(),
            1,
            "legacy indexes should keep the old substring behaviour"
        );
    }

    #[test]
    fn test_combined_filters() {
        let commits = [
            commit("Alice", "2024-06-01", "Files: src/auth.rs"),
            commit("Alice", "2024-01-01", "Files: src/auth.rs"),
            commit("Bob", "2024-06-01", "Files: src/auth.rs"),
        ];
        let engine = FilterEngine::new(SearchFilters {
            author: Some("alice".to_string()),
            after: Some("2024-03-01".to_string()),
            file: Some("src/auth.rs".to_string()),
            before: None,
        })
        .unwrap();

        let kept: Vec<&CommitInfo> = commits.iter().filter(|c| engine.matches(c)).collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].author, "Alice");
        assert_eq!(kept[0].date.format("%Y-%m-%d").to_string(), "2024-06-01");
    }

    #[test]
    fn test_invalid_date_format_fails_at_compile_time() {
        let result = FilterEngine::new(SearchFilters {
            after: Some("not-a-date".to_string()),
            ..no_filters()
        });
        assert!(
            result.is_err(),
            "a bad date should fail before any embedding work"
        );
    }

    #[test]
    fn test_invalid_before_date_is_also_rejected() {
        let result = FilterEngine::new(SearchFilters {
            before: Some("2024-13-45".to_string()),
            ..no_filters()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_returns_empty_when_nothing_matches() {
        let commits = [commit("Alice", "2024-06-01", "")];
        let engine = FilterEngine::new(SearchFilters {
            author: Some("nonexistent".to_string()),
            ..no_filters()
        })
        .unwrap();
        assert!(surviving(&engine, &commits).is_empty());
    }

    #[test]
    fn test_matches_is_case_insensitive_on_author() {
        let engine = FilterEngine::new(SearchFilters {
            author: Some("alice".to_string()),
            ..no_filters()
        })
        .unwrap();

        assert!(engine.matches(&commit("Alice Chen", "2024-06-01", "")));
        assert!(!engine.matches(&commit("Bob Martinez", "2024-06-01", "")));
    }

    #[test]
    fn test_is_active_reflects_each_filter() {
        assert!(!FilterEngine::new(no_filters()).unwrap().is_active());

        for filters in [
            SearchFilters {
                author: Some("a".into()),
                ..no_filters()
            },
            SearchFilters {
                after: Some("2024-01-01".into()),
                ..no_filters()
            },
            SearchFilters {
                before: Some("2024-01-01".into()),
                ..no_filters()
            },
            SearchFilters {
                file: Some("f".into()),
                ..no_filters()
            },
        ] {
            assert!(FilterEngine::new(filters).unwrap().is_active());
        }
    }

    #[test]
    fn test_date_bounds_are_inclusive() {
        let engine = FilterEngine::new(SearchFilters {
            after: Some("2024-06-15".to_string()),
            before: Some("2024-06-15".to_string()),
            ..no_filters()
        })
        .unwrap();

        assert!(
            engine.matches(&commit("Alice", "2024-06-15", "")),
            "a commit on the boundary day must be included"
        );
    }
}
