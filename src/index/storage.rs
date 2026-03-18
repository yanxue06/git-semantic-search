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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::CommitInfo;
    use crate::index::{IndexEntry, SemanticIndex};
    use tempfile::TempDir;

    fn create_git_repo() -> TempDir {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join(".git")).unwrap();
        dir
    }

    fn sample_index() -> SemanticIndex {
        let mut index = SemanticIndex::new(
            "bge-small-en-v1.5".to_string(),
            "abc1234".to_string(),
            true,
        );
        index.entries.push(IndexEntry {
            commit: CommitInfo {
                hash: "abc1234".to_string(),
                author: "Alice".to_string(),
                date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                    .unwrap()
                    .with_timezone(&chrono::Utc),
                message: "test commit".to_string(),
                diff_summary: String::new(),
            },
            embedding: vec![0.1; 384],
        });
        index.metadata.total_commits = 1;
        index
    }

    #[test]
    fn test_storage_new_with_git_dir() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path());
        assert!(storage.is_ok());
    }

    #[test]
    fn test_storage_new_without_git_dir() {
        let dir = TempDir::new().unwrap();
        let storage = IndexStorage::new(dir.path());
        assert!(storage.is_err());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();

        let original = sample_index();
        storage.save(&original).unwrap();

        let loaded = storage.load().unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].commit.hash, "abc1234");
        assert_eq!(loaded.entries[0].embedding.len(), 384);
        assert_eq!(loaded.model_version, "bge-small-en-v1.5");
        assert_eq!(loaded.last_commit, "abc1234");
        assert!(loaded.metadata.include_diffs);
    }

    #[test]
    fn test_load_nonexistent_index() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();
        let result = storage.load();
        assert!(matches!(result, Err(IndexError::IndexNotFound)));
    }

    #[test]
    fn test_index_size_mb() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();
        let index = sample_index();
        storage.save(&index).unwrap();

        let size = storage.index_size_mb().unwrap();
        assert!(size > 0.0, "index size should be > 0, got {size}");
    }

    #[test]
    fn test_save_overwrites_existing() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();

        let mut index1 = sample_index();
        index1.last_commit = "first".to_string();
        storage.save(&index1).unwrap();

        let mut index2 = sample_index();
        index2.last_commit = "second".to_string();
        storage.save(&index2).unwrap();

        let loaded = storage.load().unwrap();
        assert_eq!(loaded.last_commit, "second");
    }

    #[test]
    fn test_index_stored_in_git_dir() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();
        let index = sample_index();
        storage.save(&index).unwrap();

        let index_file = dir.path().join(".git").join("semantic-index");
        assert!(index_file.exists(), "index should be stored in .git/semantic-index");
    }
}
