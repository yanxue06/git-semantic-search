use std::fs;
use std::path::{Path, PathBuf};

use super::{IndexError, SemanticIndex};

pub struct IndexStorage {
    index_path: PathBuf,
}

impl IndexStorage {
    pub fn new(repo_path: &Path) -> Result<Self, IndexError> {
        let git_dir = repo_path.join(".git");

        let index_path = if git_dir.is_dir() {
            git_dir.join("semantic-index")
        } else if git_dir.is_file() {
            let content = fs::read_to_string(&git_dir)?;
            let git_dir_path = content
                .strip_prefix("gitdir: ")
                .and_then(|s| s.trim().split('\n').next())
                .ok_or(IndexError::InvalidGitFile)?;
            PathBuf::from(git_dir_path).join("semantic-index")
        } else {
            return Err(IndexError::NotAGitRepository);
        };

        Ok(Self { index_path })
    }

    pub fn save(&self, index: &SemanticIndex) -> Result<(), IndexError> {
        let encoded = bincode::serialize(index)?;
        fs::write(&self.index_path, encoded)?;
        Ok(())
    }

    pub fn load(&self) -> Result<SemanticIndex, IndexError> {
        let data = fs::read(&self.index_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                IndexError::IndexNotFound
            } else {
                IndexError::Io(e)
            }
        })?;

        let index = bincode::deserialize(&data)?;
        Ok(index)
    }

    pub fn index_size_mb(&self) -> Result<f64, IndexError> {
        let metadata = fs::metadata(&self.index_path)?;
        Ok(metadata.len() as f64 / 1_024_000.0)
    }
}
