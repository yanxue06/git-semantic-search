use git2::{Commit, Diff, DiffOptions, Repository};
use std::collections::BTreeSet;

use super::GitError;

/// Total diff text kept per commit, in bytes.
const MAX_DIFF_SIZE: usize = 10_000;

/// Share of the budget reserved for the changed-file list, so a commit that
/// touches hundreds of files still leaves room for actual diff content.
const MAX_PATH_SECTION: usize = 2_000;

pub struct DiffExtractor;

impl DiffExtractor {
    pub fn extract_diff(repo: &Repository, commit: &Commit) -> Result<String, GitError> {
        let tree = commit.tree()?;

        let parent_tree = commit.parent(0).ok().and_then(|parent| parent.tree().ok());

        let mut diff_opts = DiffOptions::new();
        diff_opts.context_lines(0);

        let diff =
            repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), Some(&mut diff_opts))?;

        let paths = Self::changed_paths(&diff)?;
        let body = Self::format_diff(&diff)?;

        let mut summary = String::with_capacity(paths.len() + body.len() + 1);
        summary.push_str(&paths);
        if !paths.is_empty() && !body.is_empty() {
            summary.push('\n');
        }
        summary.push_str(&body);

        Ok(truncate_on_char_boundary(summary, MAX_DIFF_SIZE))
    }

    /// A `Files:` line naming every path the commit touched.
    ///
    /// libgit2 reports paths in the file-header line class, which
    /// [`Self::format_diff`] deliberately drops — it keeps only `+`/`-` content.
    /// So paths never reached the stored summary, which is what `--file`
    /// searches. Collecting them from the deltas directly puts them back.
    ///
    /// Both sides of a rename are recorded, so either name finds the commit.
    fn changed_paths(diff: &Diff) -> Result<String, GitError> {
        let mut paths: BTreeSet<String> = BTreeSet::new();

        diff.foreach(
            &mut |delta, _progress| {
                for file in [delta.old_file(), delta.new_file()] {
                    if let Some(path) = file.path().and_then(|p| p.to_str()) {
                        paths.insert(path.to_string());
                    }
                }
                true
            },
            None,
            None,
            None,
        )?;

        if paths.is_empty() {
            return Ok(String::new());
        }

        let mut line = String::from("Files: ");
        for (i, path) in paths.iter().enumerate() {
            let separator = if i == 0 { "" } else { ", " };
            if line.len() + separator.len() + path.len() > MAX_PATH_SECTION {
                line.push_str(", ...");
                break;
            }
            line.push_str(separator);
            line.push_str(path);
        }

        Ok(line)
    }

    fn format_diff(diff: &Diff) -> Result<String, GitError> {
        let mut result = String::new();

        diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
            match line.origin() {
                '+' | '-' => {
                    if let Ok(content) = std::str::from_utf8(line.content()) {
                        result.push(line.origin());
                        result.push_str(content);
                    }
                }
                _ => {}
            }
            true
        })?;

        Ok(result)
    }
}

/// Truncate to at most `limit` bytes without splitting a UTF-8 character.
fn truncate_on_char_boundary(text: String, limit: usize) -> String {
    if text.len() <= limit {
        return text;
    }

    // Find valid UTF-8 boundary to handle multi-byte chars (e.g. emojis)
    let truncate_at = text
        .char_indices()
        .take_while(|(i, _)| *i < limit)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(limit);

    text[..truncate_at].to_string() + "\n... (truncated)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_at_utf8_char_boundary() {
        let s = "1234567890🔍abc".to_string(); // emoji at byte 10-13
        let out = truncate_on_char_boundary(s, 10);
        assert_eq!(out, "1234567890\n... (truncated)");
    }

    #[test]
    fn test_truncate_leaves_short_text_alone() {
        let s = "short".to_string();
        assert_eq!(truncate_on_char_boundary(s.clone(), 100), s);
    }

    #[test]
    fn test_truncate_at_exact_limit_is_untouched() {
        let s = "1234567890".to_string();
        assert_eq!(truncate_on_char_boundary(s.clone(), 10), s);
    }
}
