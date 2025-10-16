use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

use super::SemanticIndex;

pub struct IndexStorage {
    index_path: PathBuf,
}

impl IndexStorage {
    pub fn new(repo_path: &Path) -> Result<Self> {
        let git_dir = repo_path.join(".git");

        // Handle both normal repos and worktrees
        let index_path = if git_dir.is_dir() {
            git_dir.join("semantic-index")
        } else if git_dir.is_file() {
            // It's a worktree - read the actual git dir
            let content = fs::read_to_string(&git_dir)?;
            let git_dir_path = content
                .strip_prefix("gitdir: ")
                .and_then(|s| s.trim().split('\n').next())
                .context("Invalid .git file format")?;
            PathBuf::from(git_dir_path).join("semantic-index")
        } else {
            anyhow::bail!("Not a git repository");
        };

        Ok(Self { index_path })
    }

    pub fn save(&self, index: &SemanticIndex) -> Result<()> {
        let encoded = bincode::serialize(index)?;
        fs::write(&self.index_path, encoded)?;
        Ok(())
    }

    pub fn load(&self) -> Result<SemanticIndex> {
        let data = fs::read(&self.index_path)
            .context("Index file not found. Run 'git-semantic index' first.")?;

        let index = bincode::deserialize(&data)?;
        Ok(index)
    }


    pub fn index_size_mb(&self) -> Result<f64> {
        let metadata = fs::metadata(&self.index_path)?;
        Ok(metadata.len() as f64 / 1_024_000.0)
    }
}

