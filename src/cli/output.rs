//! Machine-readable search output.
//!
//! The human format is tuned for reading — emoji, blank lines, a truncated
//! hash, a two-line diff preview. None of that survives a pipe. `--json` emits
//! one stable object per query so results can feed `jq`, a script, or an LLM
//! without anyone parsing decorated text.

use serde::{Deserialize, Serialize};

use crate::search::{SearchOutcome, SearchStrategy};

/// A single ranked commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonResult {
    pub rank: usize,
    /// Full 40-character hash — the human view truncates, machines should not.
    pub hash: String,
    pub author: String,
    /// RFC 3339, so it parses everywhere without a format string.
    pub date: String,
    /// First line of the commit message.
    pub subject: String,
    /// Full commit message, trailing whitespace trimmed.
    pub message: String,
    /// Paths the commit touched, when the index recorded them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<String>>,
    /// Cosine similarity, omitted when the ranking did not come from embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f32>,
}

/// One query's full response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonOutput {
    pub query: String,
    /// `hybrid`, `semantic`, or `lexical`.
    pub mode: String,
    /// `exact`, `approximate`, or `approximate_then_exact`.
    pub strategy: String,
    /// Commits that survived metadata filtering — the real search space.
    pub candidates: usize,
    /// Whether MMR diversification reordered the results.
    pub diversified: bool,
    pub took_ms: f64,
    pub results: Vec<JsonResult>,
}

impl JsonOutput {
    pub fn new(query: &str, outcome: &SearchOutcome, diversified: bool, took_ms: f64) -> Self {
        Self {
            query: query.to_string(),
            mode: outcome.mode.as_str().to_string(),
            strategy: strategy_name(outcome.strategy).to_string(),
            candidates: outcome.candidate_count,
            diversified,
            took_ms,
            results: outcome.results.iter().map(JsonResult::from).collect(),
        }
    }
}

impl From<&crate::search::SearchResult> for JsonResult {
    fn from(result: &crate::search::SearchResult) -> Self {
        let commit = &result.commit;

        Self {
            rank: result.rank,
            hash: commit.hash.clone(),
            author: commit.author.clone(),
            date: commit.date.to_rfc3339(),
            subject: commit.message.lines().next().unwrap_or("").to_string(),
            message: commit.message.trim_end().to_string(),
            files: commit
                .changed_files()
                .map(|paths| paths.into_iter().map(str::to_string).collect()),
            // NaN marks "no embedding produced this ranking"; JSON has no NaN,
            // so the field is omitted rather than emitted as null or 0.
            similarity: result.similarity.is_finite().then_some(result.similarity),
        }
    }
}

fn strategy_name(strategy: SearchStrategy) -> &'static str {
    match strategy {
        SearchStrategy::Exact => "exact",
        SearchStrategy::Approximate => "approximate",
        SearchStrategy::ApproximateThenExact => "approximate_then_exact",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::CommitInfo;
    use crate::search::{RetrievalMode, SearchResult};

    fn commit(hash: &str, message: &str, diff_summary: &str) -> CommitInfo {
        CommitInfo {
            hash: hash.to_string(),
            author: "Alice Chen".to_string(),
            date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            message: message.to_string(),
            diff_summary: diff_summary.to_string(),
        }
    }

    fn result(similarity: f32) -> SearchResult {
        SearchResult {
            commit: commit(
                "abc1234def5678901234567890123456789012ab",
                "fix: resolve race condition\n\nLonger body here.\n",
                "Files: src/auth.rs, Cargo.toml\n+token",
            ),
            similarity,
            rank: 1,
        }
    }

    fn outcome(results: Vec<SearchResult>) -> SearchOutcome {
        SearchOutcome {
            results,
            strategy: SearchStrategy::Approximate,
            candidate_count: 4_211,
            mode: RetrievalMode::Hybrid,
            diversified: false,
        }
    }

    #[test]
    fn emits_the_full_hash_not_the_truncated_one() {
        let json = JsonOutput::new("race", &outcome(vec![result(0.83)]), false, 1.5);
        assert_eq!(json.results[0].hash.len(), 40);
    }

    #[test]
    fn splits_subject_from_full_message() {
        let json = JsonOutput::new("race", &outcome(vec![result(0.83)]), false, 1.5);
        assert_eq!(json.results[0].subject, "fix: resolve race condition");
        assert!(json.results[0].message.contains("Longer body here."));
        assert!(
            !json.results[0].message.ends_with('\n'),
            "trailing whitespace should be trimmed"
        );
    }

    #[test]
    fn dates_are_rfc3339() {
        let json = JsonOutput::new("race", &outcome(vec![result(0.83)]), false, 1.5);
        assert!(
            chrono::DateTime::parse_from_rfc3339(&json.results[0].date).is_ok(),
            "got {}",
            json.results[0].date
        );
    }

    #[test]
    fn changed_files_are_exposed_as_a_list() {
        let json = JsonOutput::new("race", &outcome(vec![result(0.83)]), false, 1.5);
        assert_eq!(
            json.results[0].files,
            Some(vec!["src/auth.rs".to_string(), "Cargo.toml".to_string()])
        );
    }

    #[test]
    fn files_is_omitted_for_a_legacy_index() {
        let mut r = result(0.83);
        r.commit.diff_summary = "+no paths recorded".to_string();
        let json = JsonOutput::new("race", &outcome(vec![r]), false, 1.5);

        assert!(json.results[0].files.is_none());
        let text = serde_json::to_string(&json).unwrap();
        assert!(
            !text.contains("\"files\""),
            "absent field should be omitted"
        );
    }

    #[test]
    fn similarity_is_omitted_for_a_keyword_only_hit() {
        let json = JsonOutput::new("race", &outcome(vec![result(f32::NAN)]), false, 1.5);
        assert!(json.results[0].similarity.is_none());

        let text = serde_json::to_string(&json).unwrap();
        assert!(
            !text.contains("similarity"),
            "NaN must not reach JSON, which cannot represent it: {text}"
        );
    }

    #[test]
    fn similarity_is_present_for_a_semantic_hit() {
        let json = JsonOutput::new("race", &outcome(vec![result(0.83)]), false, 1.5);
        assert_eq!(json.results[0].similarity, Some(0.83));
    }

    #[test]
    fn serializes_to_valid_parseable_json() {
        let json = JsonOutput::new("race condition", &outcome(vec![result(0.83)]), true, 2.25);
        let text = serde_json::to_string_pretty(&json).unwrap();

        let parsed: JsonOutput = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed.query, "race condition");
        assert_eq!(parsed.mode, "hybrid");
        assert_eq!(parsed.strategy, "approximate");
        assert_eq!(parsed.candidates, 4_211);
        assert!(parsed.diversified);
        assert_eq!(parsed.results.len(), 1);
    }

    #[test]
    fn empty_results_still_produce_a_valid_document() {
        let json = JsonOutput::new("nothing", &outcome(Vec::new()), false, 0.4);
        let text = serde_json::to_string(&json).unwrap();

        let parsed: JsonOutput = serde_json::from_str(&text).unwrap();
        assert!(parsed.results.is_empty());
        assert_eq!(parsed.query, "nothing");
    }

    #[test]
    fn strategy_names_are_stable_snake_case() {
        assert_eq!(strategy_name(SearchStrategy::Exact), "exact");
        assert_eq!(strategy_name(SearchStrategy::Approximate), "approximate");
        assert_eq!(
            strategy_name(SearchStrategy::ApproximateThenExact),
            "approximate_then_exact"
        );
    }

    #[test]
    fn quotes_and_newlines_in_commit_text_are_escaped() {
        let mut r = result(0.5);
        r.commit.message = "fix: handle \"quoted\" input\nand newlines".to_string();
        let json = JsonOutput::new("q", &outcome(vec![r]), false, 1.0);

        let text = serde_json::to_string(&json).unwrap();
        let parsed: JsonOutput = serde_json::from_str(&text).unwrap();
        assert!(parsed.results[0].message.contains("\"quoted\""));
        assert_eq!(parsed.results[0].subject, "fix: handle \"quoted\" input");
    }
}
