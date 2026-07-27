use std::fs;
use std::path::{Path, PathBuf};

use crate::vector::{HnswIndex, HnswParams};

use super::ann::{AnnSidecar, build_graph};
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

    /// Path of the ANN graph sidecar. Sits beside the index inside the git dir
    /// so deleting the repository takes both.
    pub fn ann_path(&self) -> PathBuf {
        let mut path = self.index_path.clone();
        let name = path
            .file_name()
            .map(|n| format!("{}.hnsw", n.to_string_lossy()))
            .unwrap_or_else(|| "semantic-index.hnsw".to_string());
        path.set_file_name(name);
        path
    }

    /// Load the cached graph, rebuilding it when absent, stale, or unreadable.
    ///
    /// Returns the graph and whether it had to be rebuilt, so callers can
    /// mention the one-off cost. Failing to *write* the cache is swallowed: a
    /// read-only git dir should slow searches down, not break them.
    pub fn load_or_build_ann(
        &self,
        index: &SemanticIndex,
        params: HnswParams,
    ) -> (HnswIndex, bool) {
        if let Some(sidecar) = self.read_ann_sidecar()
            && sidecar.matches(index)
        {
            return (sidecar.into_graph(), false);
        }

        let graph = build_graph(index, params);
        let sidecar = AnnSidecar::new(index, graph.clone());
        if let Err(err) = self.write_ann_sidecar(&sidecar) {
            tracing::debug!("could not cache ANN graph: {err}");
        }

        (graph, true)
    }

    /// Build and persist the graph unconditionally. Called after indexing so the
    /// first search does not pay for construction.
    pub fn refresh_ann(&self, index: &SemanticIndex, params: HnswParams) -> Result<(), IndexError> {
        let sidecar = AnnSidecar::new(index, build_graph(index, params));
        self.write_ann_sidecar(&sidecar)
    }

    /// A missing, truncated, or format-mismatched sidecar is a cache miss.
    fn read_ann_sidecar(&self) -> Option<AnnSidecar> {
        let bytes = fs::read(self.ann_path()).ok()?;
        bincode::deserialize(&bytes).ok()
    }

    fn write_ann_sidecar(&self, sidecar: &AnnSidecar) -> Result<(), IndexError> {
        let encoded = bincode::serialize(sidecar)?;
        fs::write(self.ann_path(), encoded)?;
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

    fn index_with(count: usize) -> SemanticIndex {
        let mut index =
            SemanticIndex::new("bge-small-en-v1.5".to_string(), "head".to_string(), true);
        for i in 0..count {
            index.entries.push(IndexEntry {
                commit: CommitInfo {
                    hash: format!("hash{i:04}"),
                    author: "Alice".to_string(),
                    date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                        .unwrap()
                        .with_timezone(&chrono::Utc),
                    message: format!("commit {i}"),
                    diff_summary: String::new(),
                },
                embedding: (0..32)
                    .map(|d| ((i * 32 + d) as f32 * 0.017).sin())
                    .collect(),
            });
        }
        index.metadata.total_commits = count;
        index
    }

    fn create_git_repo() -> TempDir {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join(".git")).unwrap();
        dir
    }

    fn sample_index() -> SemanticIndex {
        let mut index =
            SemanticIndex::new("bge-small-en-v1.5".to_string(), "abc1234".to_string(), true);
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
        assert!(
            index_file.exists(),
            "index should be stored in .git/semantic-index"
        );
    }

    #[test]
    fn test_ann_path_sits_beside_the_index() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();
        assert_eq!(
            storage.ann_path(),
            dir.path().join(".git").join("semantic-index.hnsw")
        );
    }

    #[test]
    fn test_load_or_build_ann_builds_then_caches() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();
        let index = index_with(40);

        let (graph, rebuilt) = storage.load_or_build_ann(&index, HnswParams::default());
        assert!(rebuilt, "first call has nothing to load");
        assert_eq!(graph.len(), 40);
        assert!(
            storage.ann_path().exists(),
            "graph should be cached to disk"
        );

        let (cached, rebuilt) = storage.load_or_build_ann(&index, HnswParams::default());
        assert!(!rebuilt, "second call should hit the cache");
        assert_eq!(cached.len(), 40);
    }

    #[test]
    fn test_load_or_build_ann_rebuilds_when_index_grows() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();

        storage.load_or_build_ann(&index_with(10), HnswParams::default());

        let (graph, rebuilt) = storage.load_or_build_ann(&index_with(15), HnswParams::default());
        assert!(rebuilt, "a changed index must invalidate the sidecar");
        assert_eq!(graph.len(), 15);
    }

    #[test]
    fn test_load_or_build_ann_survives_corrupt_sidecar() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();
        let index = index_with(20);

        fs::write(storage.ann_path(), b"not a bincode payload at all").unwrap();

        let (graph, rebuilt) = storage.load_or_build_ann(&index, HnswParams::default());
        assert!(rebuilt, "corrupt cache is a miss, not an error");
        assert_eq!(graph.len(), 20);
    }

    #[test]
    fn test_refresh_ann_writes_a_usable_sidecar() {
        let dir = create_git_repo();
        let storage = IndexStorage::new(dir.path()).unwrap();
        let index = index_with(25);

        storage.refresh_ann(&index, HnswParams::default()).unwrap();

        let (_, rebuilt) = storage.load_or_build_ann(&index, HnswParams::default());
        assert!(!rebuilt, "refresh should leave a cache the loader accepts");
    }
}
