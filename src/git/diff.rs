use anyhow::Result;
use git2::{Commit, Diff, DiffOptions, Repository};

pub struct DiffExtractor;

impl DiffExtractor {
    pub fn extract_diff(repo: &Repository, commit: &Commit) -> Result<String> {
        let tree = commit.tree()?;

        let parent_tree = commit
            .parent(0)
            .ok()
            .and_then(|parent| parent.tree().ok());

        let mut diff_opts = DiffOptions::new();
        diff_opts.context_lines(0); // We just want changes, not context

        let diff = repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), Some(&mut diff_opts))?;

        let summary = Self::format_diff(&diff)?;

        const MAX_DIFF_SIZE: usize = 10_000;
        if summary.len() > MAX_DIFF_SIZE {
            let truncate_at = summary
                .char_indices()
                .take_while(|(i, _)| *i < MAX_DIFF_SIZE)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(MAX_DIFF_SIZE); // Find valid UTF-8 boundary to handle multi-byte chars (e.g. emojis)
            Ok(summary[..truncate_at].to_string() + "\n... (truncated)")
        } else {
            Ok(summary)
        }
    }

    fn format_diff(diff: &Diff) -> Result<String> {
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_truncate_at_utf8_char_boundary() {
        const MAX_SIZE: usize = 10;
        let s = "1234567890🔍abc"; // emoji at byte 10-13

        let truncate_at = s
            .char_indices()
            .take_while(|(i, _)| *i < MAX_SIZE)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(MAX_SIZE);

        assert_eq!(&s[..truncate_at], "1234567890"); // Includes chars starting before limit, excludes emoji
    }
}

