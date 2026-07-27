//! Changed-file paths must survive into the stored diff summary.
//!
//! These run against real `git2` repositories, because the bug they cover was
//! invisible to synthetic fixtures: `format_diff` keeps only `+`/`-` content
//! lines, and libgit2 reports paths in the file-header class, so every
//! hand-written test summary containing a path passed while real ones never
//! had one.

use git_semantic::git::RepositoryParser;
use git2::{Repository, Signature};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Commit `files` (path, contents) as a single revision.
fn commit_files(repo: &Repository, root: &Path, message: &str, files: &[(&str, &str)]) {
    let sig = Signature::now("Test Author", "test@example.com").unwrap();

    for (path, contents) in files {
        let full = root.join(path);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full, contents).unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(Path::new(path)).unwrap();
        index.write().unwrap();
    }

    let tree_id = repo.index().unwrap().write_tree().unwrap();
    let tree = repo.find_tree(tree_id).unwrap();

    match repo.head().ok().and_then(|h| h.peel_to_commit().ok()) {
        Some(parent) => repo
            .commit(Some("HEAD"), &sig, &sig, message, &tree, &[&parent])
            .unwrap(),
        None => repo
            .commit(Some("HEAD"), &sig, &sig, message, &tree, &[])
            .unwrap(),
    };
}

fn repo_with_commits() -> TempDir {
    let dir = TempDir::new().unwrap();
    let repo = Repository::init(dir.path()).unwrap();

    commit_files(
        &repo,
        dir.path(),
        "feat: add authentication",
        &[("src/auth.rs", "pub fn login() {}\n")],
    );
    commit_files(
        &repo,
        dir.path(),
        "chore: bump dependency",
        &[("Cargo.toml", "[package]\nversion = \"0.2.0\"\n")],
    );
    commit_files(
        &repo,
        dir.path(),
        "refactor: split modules",
        &[
            ("src/index/storage.rs", "pub struct Storage;\n"),
            ("src/index/mod.rs", "pub mod storage;\n"),
        ],
    );

    dir
}

#[test]
fn diff_summary_records_the_changed_path() {
    let dir = repo_with_commits();
    let parser = RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(true).unwrap();

    let auth = commits
        .iter()
        .find(|c| c.message.starts_with("feat: add authentication"))
        .expect("commit should be present");

    assert!(
        auth.diff_summary.contains("src/auth.rs"),
        "path missing from summary:\n{}",
        auth.diff_summary
    );
    assert_eq!(auth.changed_files(), Some(vec!["src/auth.rs"]));
}

#[test]
fn file_filter_now_matches_a_dependency_bump() {
    // The original report: a Cargo.toml bump's summary contained only version
    // and checksum lines, so `--file Cargo.toml` matched nothing.
    let dir = repo_with_commits();
    let parser = RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(true).unwrap();

    let bump = commits
        .iter()
        .find(|c| c.message.starts_with("chore: bump"))
        .unwrap();

    assert!(
        bump.touches_path("Cargo.toml"),
        "--file Cargo.toml should match this commit; summary:\n{}",
        bump.diff_summary
    );
}

#[test]
fn all_paths_of_a_multi_file_commit_are_recorded() {
    let dir = repo_with_commits();
    let parser = RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(true).unwrap();

    let split = commits
        .iter()
        .find(|c| c.message.starts_with("refactor: split"))
        .unwrap();

    let paths = split.changed_files().unwrap();
    assert!(paths.contains(&"src/index/mod.rs"), "got {paths:?}");
    assert!(paths.contains(&"src/index/storage.rs"), "got {paths:?}");
    assert!(split.touches_path("src/index/"), "prefix match should work");
}

#[test]
fn paths_are_recorded_for_the_root_commit() {
    // The root commit has no parent, which is a separate code path in the
    // extractor — `commit.parent(0)` fails and the diff is against an empty tree.
    let dir = repo_with_commits();
    let parser = RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(true).unwrap();

    let root = commits.last().unwrap();
    assert_eq!(
        root.changed_files(),
        Some(vec!["src/auth.rs"]),
        "root commit summary:\n{}",
        root.diff_summary
    );
}

#[test]
fn quick_mode_still_stores_no_diff_summary() {
    let dir = repo_with_commits();
    let parser = RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(false).unwrap();

    for commit in &commits {
        assert!(
            commit.diff_summary.is_empty(),
            "quick mode must not pay for diffs"
        );
        assert_eq!(commit.changed_files(), None);
    }
}

#[test]
fn diff_content_is_still_captured_alongside_paths() {
    let dir = repo_with_commits();
    let parser = RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(true).unwrap();

    let auth = commits
        .iter()
        .find(|c| c.message.starts_with("feat: add authentication"))
        .unwrap();

    assert!(
        auth.diff_summary.contains("pub fn login()"),
        "adding paths must not drop the diff body:\n{}",
        auth.diff_summary
    );
}

#[test]
fn summary_stays_within_the_size_budget() {
    let dir = TempDir::new().unwrap();
    let repo = Repository::init(dir.path()).unwrap();

    // A commit with far more content than the 10 KB budget.
    let big: String = (0..5_000).map(|i| format!("line {i}\n")).collect();
    commit_files(&repo, dir.path(), "feat: big change", &[("big.txt", &big)]);

    let parser = RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(true).unwrap();

    let summary = &commits[0].diff_summary;
    assert!(
        summary.len() <= 10_000 + "\n... (truncated)".len(),
        "summary grew to {} bytes",
        summary.len()
    );
    assert!(
        summary.starts_with("Files: big.txt"),
        "paths must survive truncation by being written first"
    );
}
