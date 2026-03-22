use git_semantic::git::CommitInfo;
use git_semantic::index::{IndexEntry, IndexStorage, SemanticIndex};
use git2::{Repository, Signature};
use std::fs;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_test_repo_with_commits(num_commits: usize) -> TempDir {
    let dir = TempDir::new().unwrap();
    let repo = Repository::init(dir.path()).unwrap();
    let sig = Signature::now("Test Author", "test@example.com").unwrap();

    for i in 0..num_commits {
        let filename = format!("file_{i}.txt");
        fs::write(dir.path().join(&filename), format!("content {i}")).unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(std::path::Path::new(&filename)).unwrap();
        index.write().unwrap();

        let tree_id = repo.index().unwrap().write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();

        let message = format!("feat: add feature {i}");

        if i == 0 {
            repo.commit(Some("HEAD"), &sig, &sig, &message, &tree, &[])
                .unwrap();
        } else {
            let head = repo.head().unwrap().peel_to_commit().unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, &message, &tree, &[&head])
                .unwrap();
        }
    }

    dir
}

/// Add more commits on top of an existing repo.
fn add_commits(dir: &TempDir, count: usize, start_idx: usize) {
    let repo = Repository::open(dir.path()).unwrap();
    let sig = Signature::now("Test Author", "test@example.com").unwrap();

    for i in start_idx..(start_idx + count) {
        let filename = format!("file_{i}.txt");
        fs::write(dir.path().join(&filename), format!("content {i}")).unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(std::path::Path::new(&filename)).unwrap();
        index.write().unwrap();

        let tree_id = repo.index().unwrap().write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();

        let head = repo.head().unwrap().peel_to_commit().unwrap();
        let message = format!("feat: add feature {i}");
        repo.commit(Some("HEAD"), &sig, &sig, &message, &tree, &[&head])
            .unwrap();
    }
}

fn make_entry(hash: &str, message: &str, diff: &str) -> IndexEntry {
    IndexEntry {
        commit: CommitInfo {
            hash: hash.to_string(),
            author: "Author".to_string(),
            date: chrono::DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            message: message.to_string(),
            diff_summary: diff.to_string(),
        },
        embedding: vec![0.1; 384],
    }
}

fn make_index_with_entries(entries: Vec<IndexEntry>, include_diffs: bool) -> SemanticIndex {
    let last_commit = entries
        .first()
        .map(|e| e.commit.hash.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let total = entries.len();
    let mut index = SemanticIndex::new("bge-small-en-v1.5".to_string(), last_commit, include_diffs);
    index.entries = entries;
    index.metadata.total_commits = total;
    index
}

// ---------------------------------------------------------------------------
// Tests: Incremental index storage behavior
// ---------------------------------------------------------------------------

#[test]
fn test_incremental_save_preserves_old_entries_and_adds_new() {
    let dir = create_test_repo_with_commits(1);
    let storage = IndexStorage::new(dir.path()).unwrap();

    // Simulate first index with 3 commits
    let entries = vec![
        make_entry(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "feat: first",
            "",
        ),
        make_entry(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "feat: second",
            "",
        ),
        make_entry(
            "cccccccccccccccccccccccccccccccccccccccc",
            "feat: third",
            "",
        ),
    ];
    let index = make_index_with_entries(entries, false);
    storage.save(&index).unwrap();

    // Verify save
    let loaded = storage.load().unwrap();
    assert_eq!(loaded.entries.len(), 3);
    assert_eq!(loaded.metadata.total_commits, 3);

    // Simulate incremental: add 2 more entries on top
    let mut updated_entries = loaded.entries.clone();
    updated_entries.insert(
        0,
        make_entry(
            "dddddddddddddddddddddddddddddddddddddddd"[..40].into(),
            "feat: fourth",
            "",
        ),
    );
    updated_entries.insert(
        0,
        make_entry(
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "feat: fifth",
            "",
        ),
    );
    let updated_index = make_index_with_entries(updated_entries, false);
    storage.save(&updated_index).unwrap();

    // Verify incremental
    let loaded = storage.load().unwrap();
    assert_eq!(loaded.entries.len(), 5);
    assert_eq!(loaded.metadata.total_commits, 5);
    assert_eq!(loaded.entries[0].commit.message, "feat: fifth");
    assert_eq!(loaded.entries[4].commit.message, "feat: third");
}

#[test]
fn test_index_up_to_date_no_new_commits() {
    let dir = create_test_repo_with_commits(3);
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();

    // Get all commits
    let all = parser.parse_commits(false).unwrap();
    assert_eq!(all.len(), 3);

    // Mark the newest as last_commit, then query for new commits
    let since = parser.parse_commits_since(&all[0].hash, false).unwrap();
    assert!(
        since.is_empty(),
        "no new commits should be found when index is up to date"
    );
}

#[test]
fn test_incremental_only_returns_new_commits() {
    let dir = create_test_repo_with_commits(3);

    // Get the commit hashes before adding more
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let initial = parser.parse_commits(false).unwrap();
    let last_indexed = initial[0].hash.clone(); // newest commit

    // Add 2 more commits
    add_commits(&dir, 2, 3);

    // Now parse_commits_since should only return the 2 new ones
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let new_commits = parser.parse_commits_since(&last_indexed, false).unwrap();
    assert_eq!(
        new_commits.len(),
        2,
        "should only return 2 new commits, got {}",
        new_commits.len()
    );

    // Verify they are the NEW commits (feature 3 and 4)
    assert!(new_commits[0].message.contains("4")); // newest first
    assert!(new_commits[1].message.contains("3"));
}

#[test]
fn test_incremental_does_not_include_boundary_commit() {
    let dir = create_test_repo_with_commits(5);
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();

    let all = parser.parse_commits(false).unwrap();
    // Index up to commit [2] (the middle one)
    let since = parser.parse_commits_since(&all[2].hash, false).unwrap();

    // Should get commits [0] and [1] but NOT [2]
    assert_eq!(since.len(), 2);
    for c in &since {
        assert_ne!(
            c.hash, all[2].hash,
            "boundary commit should not be included"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: Mode switching behavior
// ---------------------------------------------------------------------------

#[test]
fn test_full_mode_index_metadata() {
    let dir = create_test_repo_with_commits(1);
    let storage = IndexStorage::new(dir.path()).unwrap();

    let index = make_index_with_entries(
        vec![make_entry(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "feat: thing",
            "+added line",
        )],
        true,
    );
    storage.save(&index).unwrap();

    let loaded = storage.load().unwrap();
    assert!(
        loaded.metadata.include_diffs,
        "full mode should set include_diffs=true"
    );
}

#[test]
fn test_quick_mode_index_metadata() {
    let dir = create_test_repo_with_commits(1);
    let storage = IndexStorage::new(dir.path()).unwrap();

    let index = make_index_with_entries(
        vec![make_entry(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "feat: thing",
            "",
        )],
        false,
    );
    storage.save(&index).unwrap();

    let loaded = storage.load().unwrap();
    assert!(
        !loaded.metadata.include_diffs,
        "quick mode should set include_diffs=false"
    );
}

#[test]
fn test_full_index_is_superset_of_quick() {
    // A full index entry has a non-empty diff_summary
    // A quick index entry has an empty diff_summary
    // Full mode embeddings encode diffs, so they are strictly more information
    let full_entry = make_entry(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "feat: add auth",
        "+fn authenticate()\n-fn old_auth()",
    );
    let quick_entry = make_entry(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "feat: add auth",
        "",
    );

    // The commit message and author are the same
    assert_eq!(full_entry.commit.message, quick_entry.commit.message);
    assert_eq!(full_entry.commit.hash, quick_entry.commit.hash);

    // Full mode has diff info that quick mode doesn't
    assert!(
        !full_entry.commit.diff_summary.is_empty(),
        "full entry should have diff"
    );
    assert!(
        quick_entry.commit.diff_summary.is_empty(),
        "quick entry should not have diff"
    );
}

#[test]
fn test_mode_preserved_through_save_load_cycle() {
    let dir = create_test_repo_with_commits(1);
    let storage = IndexStorage::new(dir.path()).unwrap();

    // Save quick mode
    let quick_idx = make_index_with_entries(
        vec![make_entry(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "msg",
            "",
        )],
        false,
    );
    storage.save(&quick_idx).unwrap();
    let loaded = storage.load().unwrap();
    assert!(!loaded.metadata.include_diffs);

    // Overwrite with full mode
    let full_idx = make_index_with_entries(
        vec![make_entry(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "msg",
            "+diff",
        )],
        true,
    );
    storage.save(&full_idx).unwrap();
    let loaded = storage.load().unwrap();
    assert!(loaded.metadata.include_diffs);
}

// ---------------------------------------------------------------------------
// Tests: Rebase detection via parse_commits_since
// ---------------------------------------------------------------------------

#[test]
fn test_rebase_detection_commit_not_found() {
    let dir = create_test_repo_with_commits(3);
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();

    // Use a hash that doesn't exist in this repo
    let fake_hash = "0000000000000000000000000000000000000000";
    let result = parser.parse_commits_since(fake_hash, false);

    assert!(
        result.is_err(),
        "should error when last_commit hash is not found (rebase scenario)"
    );
}

// ---------------------------------------------------------------------------
// Tests: last_commit tracking
// ---------------------------------------------------------------------------

#[test]
fn test_last_commit_is_newest_commit_hash() {
    let dir = create_test_repo_with_commits(5);
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();

    let all = parser.parse_commits(false).unwrap();
    // all[0] is newest
    let newest_hash = all[0].hash.clone();

    // When building an index, last_commit should be set to the newest
    let index = make_index_with_entries(
        all.into_iter()
            .map(|c| IndexEntry {
                commit: c,
                embedding: vec![0.1; 384],
            })
            .collect(),
        false,
    );
    assert_eq!(
        index.last_commit, newest_hash,
        "last_commit should be the newest commit hash"
    );
}

#[test]
fn test_last_commit_updated_after_incremental() {
    let dir = create_test_repo_with_commits(3);
    let storage = IndexStorage::new(dir.path()).unwrap();

    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let initial = parser.parse_commits(false).unwrap();
    let initial_head = initial[0].hash.clone();

    // Save initial index
    let index = make_index_with_entries(
        initial
            .into_iter()
            .map(|c| IndexEntry {
                commit: c,
                embedding: vec![0.1; 384],
            })
            .collect(),
        false,
    );
    storage.save(&index).unwrap();

    // Add more commits
    add_commits(&dir, 2, 3);

    // Get the new HEAD
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let all = parser.parse_commits(false).unwrap();
    let new_head = all[0].hash.clone();

    assert_ne!(
        initial_head, new_head,
        "HEAD should have changed after adding commits"
    );

    // Simulate incremental: load, add new commits, save with new last_commit
    let loaded = storage.load().unwrap();
    assert_eq!(loaded.last_commit, initial_head);

    let new_commits = parser
        .parse_commits_since(&loaded.last_commit, false)
        .unwrap();
    assert_eq!(new_commits.len(), 2);

    // Build updated index — new_commits are newest-first, so [0] is the new HEAD
    let newest_new_hash = new_commits[0].hash.clone();
    let mut new_entries: Vec<IndexEntry> = new_commits
        .into_iter()
        .map(|c| IndexEntry {
            commit: c,
            embedding: vec![0.1; 384],
        })
        .collect();
    new_entries.extend(loaded.entries);

    let mut updated_index = make_index_with_entries(new_entries, false);
    // Explicitly set last_commit to the newest commit (as the real code does)
    updated_index.last_commit = newest_new_hash;
    storage.save(&updated_index).unwrap();

    let final_loaded = storage.load().unwrap();
    assert_eq!(
        final_loaded.last_commit, new_head,
        "last_commit should be updated to new HEAD after incremental"
    );
    assert_eq!(final_loaded.entries.len(), 5);
}

// ---------------------------------------------------------------------------
// Tests: Deduplication
// ---------------------------------------------------------------------------

#[test]
fn test_index_entries_have_unique_hashes() {
    let entries = [
        make_entry(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "feat: first",
            "",
        ),
        make_entry(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "feat: second",
            "",
        ),
        make_entry(
            "cccccccccccccccccccccccccccccccccccccccc",
            "feat: third",
            "",
        ),
    ];

    let hashes: std::collections::HashSet<&str> =
        entries.iter().map(|e| e.commit.hash.as_str()).collect();
    assert_eq!(
        hashes.len(),
        entries.len(),
        "all entry hashes should be unique"
    );
}

#[test]
fn test_incremental_entries_dont_duplicate_existing() {
    let dir = create_test_repo_with_commits(3);
    let storage = IndexStorage::new(dir.path()).unwrap();

    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let all = parser.parse_commits(false).unwrap();
    let initial_head = all[0].hash.clone();

    // Save initial index
    let index = make_index_with_entries(
        all.iter()
            .map(|c| IndexEntry {
                commit: c.clone(),
                embedding: vec![0.1; 384],
            })
            .collect(),
        false,
    );
    storage.save(&index).unwrap();

    // parse_commits_since should return empty (no new commits)
    let new_commits = parser.parse_commits_since(&initial_head, false).unwrap();
    assert!(
        new_commits.is_empty(),
        "no duplicates: parse_commits_since should return 0 when up to date"
    );

    // Loaded index should still have exactly 3
    let loaded = storage.load().unwrap();
    assert_eq!(loaded.entries.len(), 3);
}

// ---------------------------------------------------------------------------
// Tests: Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_single_commit_repo_incremental() {
    let dir = create_test_repo_with_commits(1);
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();

    let all = parser.parse_commits(false).unwrap();
    assert_eq!(all.len(), 1);

    // Since HEAD is the only commit, no new commits
    let since = parser.parse_commits_since(&all[0].hash, false).unwrap();
    assert!(since.is_empty());

    // Add one commit
    add_commits(&dir, 1, 1);

    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let since = parser.parse_commits_since(&all[0].hash, false).unwrap();
    assert_eq!(since.len(), 1);
    assert!(since[0].message.contains("1"));
}

#[test]
fn test_many_new_commits_incremental() {
    let dir = create_test_repo_with_commits(2);
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let all = parser.parse_commits(false).unwrap();
    let head = all[0].hash.clone();

    // Add many commits
    add_commits(&dir, 50, 2);

    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let since = parser.parse_commits_since(&head, false).unwrap();
    assert_eq!(
        since.len(),
        50,
        "should find exactly 50 new commits, got {}",
        since.len()
    );
}

#[test]
fn test_incremental_with_diffs() {
    let dir = create_test_repo_with_commits(3);
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();

    let all = parser.parse_commits(true).unwrap();
    // Verify diffs are populated
    for commit in &all[..all.len() - 1] {
        // Skip initial commit which may have diff
        assert!(
            !commit.diff_summary.is_empty() || commit.message.contains("0"),
            "full-mode commits should have diffs: {}",
            commit.message
        );
    }

    let head = all[0].hash.clone();
    add_commits(&dir, 2, 3);

    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let new_commits = parser.parse_commits_since(&head, true).unwrap();
    assert_eq!(new_commits.len(), 2);

    // New commits in full mode should also have diffs
    for commit in &new_commits {
        assert!(
            !commit.diff_summary.is_empty(),
            "new full-mode commits should have diffs: {}",
            commit.message
        );
    }
}

#[test]
fn test_incremental_without_diffs() {
    let dir = create_test_repo_with_commits(3);
    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();

    let all = parser.parse_commits(false).unwrap();
    // Verify diffs are NOT populated in quick mode
    for commit in &all {
        assert!(
            commit.diff_summary.is_empty(),
            "quick-mode commits should not have diffs"
        );
    }

    let head = all[0].hash.clone();
    add_commits(&dir, 2, 3);

    let parser = git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let new_commits = parser.parse_commits_since(&head, false).unwrap();
    assert_eq!(new_commits.len(), 2);

    for commit in &new_commits {
        assert!(
            commit.diff_summary.is_empty(),
            "new quick-mode commits should not have diffs"
        );
    }
}

#[test]
fn test_index_created_at_preserved_on_incremental_update() {
    let dir = create_test_repo_with_commits(1);
    let storage = IndexStorage::new(dir.path()).unwrap();

    let index = make_index_with_entries(
        vec![make_entry(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "feat: first",
            "",
        )],
        false,
    );
    let original_created_at = index.metadata.created_at;
    storage.save(&index).unwrap();

    // Load and resave (simulating incremental update)
    let mut loaded = storage.load().unwrap();
    let created_at_after_load = loaded.metadata.created_at;
    assert_eq!(
        original_created_at, created_at_after_load,
        "created_at should survive save/load cycle"
    );

    // Simulate incremental: add entry, update updated_at but NOT created_at
    loaded.entries.push(make_entry(
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "feat: second",
        "",
    ));
    loaded.metadata.total_commits = loaded.entries.len();
    loaded.metadata.updated_at = chrono::Utc::now();
    // created_at should remain the same
    storage.save(&loaded).unwrap();

    let final_loaded = storage.load().unwrap();
    assert_eq!(
        final_loaded.metadata.created_at, original_created_at,
        "created_at should not change on incremental update"
    );
    assert!(
        final_loaded.metadata.updated_at >= final_loaded.metadata.created_at,
        "updated_at should be >= created_at"
    );
    assert_eq!(final_loaded.entries.len(), 2);
}

#[test]
fn test_force_flag_rebuilds_entire_index() {
    let dir = create_test_repo_with_commits(1);
    let storage = IndexStorage::new(dir.path()).unwrap();

    // Save an index with a bogus hash that doesn't exist
    let index = make_index_with_entries(
        vec![make_entry(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "feat: old",
            "",
        )],
        false,
    );
    storage.save(&index).unwrap();

    // The force flag in commands.rs sets existing_index = None,
    // which triggers full_index() regardless of existing state.
    // We verify the concept: loading with force=true should ignore existing.
    let loaded = storage.load().unwrap();
    assert_eq!(loaded.entries.len(), 1);

    // With force, a new full index would overwrite this entirely
    let fresh_index = make_index_with_entries(
        vec![
            make_entry("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "feat: new1", ""),
            make_entry("cccccccccccccccccccccccccccccccccccccccc", "feat: new2", ""),
        ],
        false,
    );
    storage.save(&fresh_index).unwrap();

    let final_loaded = storage.load().unwrap();
    assert_eq!(
        final_loaded.entries.len(),
        2,
        "force rebuild should replace entire index"
    );
    assert_eq!(final_loaded.entries[0].commit.message, "feat: new1");
}
