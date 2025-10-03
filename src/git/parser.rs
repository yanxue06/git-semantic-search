use anyhow::{Context, Result};
use git2::Repository;
use std::path::Path;
use tracing::debug;

use super::diff::DiffExtractor;
use super::CommitInfo;

pub struct RepositoryParser {
    repo: Repository,
}

impl RepositoryParser {
    pub fn new(path: &Path) -> Result<Self> {
        let repo = Repository::discover(path)
            .context("Failed to open git repository. Make sure you're in a git repository.")?;

        Ok(Self { repo })
    }

    pub fn parse_commits(&self, include_diffs: bool) -> Result<Vec<CommitInfo>> {
        let mut revwalk = self.repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(git2::Sort::TIME)?;

        let mut commits = Vec::new();

        for oid in revwalk {
            let oid = oid?;
            let commit = self.repo.find_commit(oid)?;

            let hash = oid.to_string();
            let author = commit.author().name().unwrap_or("Unknown").to_string();
            let date = chrono::DateTime::from_timestamp(commit.time().seconds(), 0)
                .unwrap_or_default();
            let message = commit.message().unwrap_or("").to_string();

            let diff_summary = if include_diffs {
                DiffExtractor::extract_diff(&self.repo, &commit)?
            } else {
                String::new()
            };

            debug!(
                "Parsed commit: {} by {} at {}",
                &hash[..7],
                author,
                date
            );

            commits.push(CommitInfo {
                hash,
                author,
                date,
                message,
                diff_summary,
            });
        }

        Ok(commits)
    }

    pub fn parse_commits_since(&self, since_hash: &str) -> Result<Vec<CommitInfo>> {
        let mut revwalk = self.repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(git2::Sort::TIME)?;

        let since_oid = git2::Oid::from_str(since_hash)?;
        let mut commits = Vec::new();
        let mut found_since = false;

        for oid in revwalk {
            let oid = oid?;

            if oid == since_oid {
                found_since = true;
                break;
            }

            let commit = self.repo.find_commit(oid)?;

            let hash = oid.to_string();
            let author = commit.author().name().unwrap_or("Unknown").to_string();
            let date = chrono::DateTime::from_timestamp(commit.time().seconds(), 0)
                .unwrap_or_default();
            let message = commit.message().unwrap_or("").to_string();
            let diff_summary = DiffExtractor::extract_diff(&self.repo, &commit)?;

            commits.push(CommitInfo {
                hash,
                author,
                date,
                message,
                diff_summary,
            });
        }

        if !found_since {
            anyhow::bail!("Could not find commit {} in history", since_hash);
        }

        Ok(commits)
    }

    pub fn get_head_commit_hash(&self) -> Result<String> {
        let head = self.repo.head()?;
        let oid = head.target().context("HEAD has no target")?;
        Ok(oid.to_string())
    }
}

