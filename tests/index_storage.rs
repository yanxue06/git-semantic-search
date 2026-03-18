use git_semantic::git::CommitInfo;
use git_semantic::index::{IndexEntry, IndexStorage, SemanticIndex};
use git2::{Repository, Signature};
use std::fs;
use tempfile::TempDir;

fn create_test_repo() -> TempDir {
    let dir = TempDir::new().unwrap();
    let repo = Repository::init(dir.path()).unwrap();
    let sig = Signature::now("Test", "test@example.com").unwrap();
    fs::write(dir.path().join("file.txt"), "content").unwrap();
    let mut idx = repo.index().unwrap();
    idx.add_path(std::path::Path::new("file.txt")).unwrap();
    idx.write().unwrap();
    let tree_id = idx.write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();
    repo.commit(Some("HEAD"), &sig, &sig, "initial", &tree, &[])
        .unwrap();
    dir
}

fn make_index(num_entries: usize, include_diffs: bool) -> SemanticIndex {
    let mut index = SemanticIndex::new(
        "bge-small-en-v1.5".to_string(),
        "abc1234".to_string(),
        include_diffs,
    );
    for i in 0..num_entries {
        index.entries.push(IndexEntry {
            commit: CommitInfo {
                hash: format!("{i:040x}"),
                author: format!("Author {i}"),
                date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                    .unwrap()
                    .with_timezone(&chrono::Utc),
                message: format!("commit message {i}"),
                diff_summary: if include_diffs {
                    format!("+line {i}")
                } else {
                    String::new()
                },
            },
            embedding: vec![0.1; 384],
        });
    }
    index.metadata.total_commits = num_entries;
    index
}

#[test]
fn test_roundtrip_with_real_git_repo() {
    let dir = create_test_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();

    let original = make_index(10, true);
    storage.save(&original).unwrap();

    let loaded = storage.load().unwrap();
    assert_eq!(loaded.entries.len(), 10);
    assert_eq!(loaded.model_version, "bge-small-en-v1.5");
    assert!(loaded.metadata.include_diffs);
}

#[test]
fn test_quick_mode_index_roundtrip() {
    let dir = create_test_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();

    let original = make_index(5, false);
    storage.save(&original).unwrap();

    let loaded = storage.load().unwrap();
    assert_eq!(loaded.entries.len(), 5);
    assert!(!loaded.metadata.include_diffs);
    for entry in &loaded.entries {
        assert!(entry.commit.diff_summary.is_empty());
    }
}

#[test]
fn test_large_index_roundtrip() {
    let dir = create_test_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();

    let original = make_index(1000, true);
    storage.save(&original).unwrap();

    let loaded = storage.load().unwrap();
    assert_eq!(loaded.entries.len(), 1000);

    let size = storage.index_size_mb().unwrap();
    // ~3KB per commit as claimed in README -> ~3MB for 1000
    assert!(
        size > 0.5,
        "1000-entry index should be substantial, got {size} MB"
    );
}

#[test]
fn test_stats_fields_present() {
    let dir = create_test_repo();
    let storage = IndexStorage::new(dir.path()).unwrap();
    let index = make_index(42, true);
    storage.save(&index).unwrap();

    let loaded = storage.load().unwrap();
    assert_eq!(loaded.entries.len(), 42);
    assert_eq!(loaded.metadata.total_commits, 42);
    assert_eq!(loaded.model_version, "bge-small-en-v1.5");
    assert_eq!(loaded.last_commit, "abc1234");
    assert!(loaded.metadata.include_diffs);
    // created_at and updated_at should be set
    assert!(loaded.metadata.created_at <= loaded.metadata.updated_at);
}
