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

        // Limit size to avoid huge diffs
        const MAX_DIFF_SIZE: usize = 10_000;
        if summary.len() > MAX_DIFF_SIZE {
            Ok(summary[..MAX_DIFF_SIZE].to_string() + "\n... (truncated)")
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

