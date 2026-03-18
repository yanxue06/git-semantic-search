use git2::{Repository, Signature};
use std::fs;
use tempfile::TempDir;

fn create_test_repo_with_commits(num_commits: usize, include_file_changes: bool) -> TempDir {
    let dir = TempDir::new().unwrap();
    let repo = Repository::init(dir.path()).unwrap();

    let sig = Signature::now("Test Author", "test@example.com").unwrap();

    for i in 0..num_commits {
        if include_file_changes {
            let filename = format!("file_{i}.txt");
            fs::write(dir.path().join(&filename), format!("content {i}")).unwrap();
            let mut index = repo.index().unwrap();
            index.add_path(std::path::Path::new(&filename)).unwrap();
            index.write().unwrap();
        }

        let tree_id = repo.index().unwrap().write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();

        let message = match i % 3 {
            0 => format!("feat: add feature {i}"),
            1 => format!("fix: resolve bug {i}"),
            _ => format!("refactor: clean up module {i}"),
        };

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

#[test]
fn test_parse_commits_returns_all() {
    let dir = create_test_repo_with_commits(5, true);
    let parser =
        git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(false).unwrap();
    assert_eq!(commits.len(), 5);
}

#[test]
fn test_parse_commits_order_is_newest_first() {
    let dir = create_test_repo_with_commits(3, true);
    let parser =
        git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(false).unwrap();
    // Revwalk with TIME sorting returns newest first
    assert!(commits[0].message.contains("2"), "first commit should be newest: {}", commits[0].message);
}

#[test]
fn test_parse_commits_includes_author() {
    let dir = create_test_repo_with_commits(1, true);
    let parser =
        git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(false).unwrap();
    assert_eq!(commits[0].author, "Test Author");
}

#[test]
fn test_parse_commits_includes_diff_when_requested() {
    let dir = create_test_repo_with_commits(2, true);
    let parser =
        git_semantic::git::RepositoryParser::new(dir.path()).unwrap();

    let without_diff = parser.parse_commits(false).unwrap();
    assert!(without_diff[0].diff_summary.is_empty());

    let with_diff = parser.parse_commits(true).unwrap();
    assert!(!with_diff[0].diff_summary.is_empty());
}

#[test]
fn test_parse_commits_since() {
    let dir = create_test_repo_with_commits(5, true);
    let parser =
        git_semantic::git::RepositoryParser::new(dir.path()).unwrap();

    let all = parser.parse_commits(false).unwrap();
    // all[0] is newest, all[4] is oldest
    // parse_commits_since(all[2].hash) should return commits newer than all[2]
    let since = parser
        .parse_commits_since(&all[2].hash, false)
        .unwrap();
    assert_eq!(since.len(), 2); // all[0] and all[1]
}

#[test]
fn test_parse_commits_since_nonexistent_hash() {
    let dir = create_test_repo_with_commits(2, true);
    let parser =
        git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let result = parser.parse_commits_since(
        "0000000000000000000000000000000000000000",
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_parse_commits_since_head_returns_empty() {
    let dir = create_test_repo_with_commits(3, true);
    let parser =
        git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let all = parser.parse_commits(false).unwrap();
    // Since the newest commit, there should be 0 new commits
    let since = parser
        .parse_commits_since(&all[0].hash, false)
        .unwrap();
    assert!(since.is_empty());
}

#[test]
fn test_parser_invalid_path() {
    let result =
        git_semantic::git::RepositoryParser::new(std::path::Path::new("/nonexistent/path"));
    assert!(result.is_err());
}

#[test]
fn test_commit_hash_is_full_length() {
    let dir = create_test_repo_with_commits(1, true);
    let parser =
        git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(false).unwrap();
    assert_eq!(commits[0].hash.len(), 40, "hash should be full 40-char SHA");
}

#[test]
fn test_commit_date_is_set() {
    let dir = create_test_repo_with_commits(1, true);
    let parser =
        git_semantic::git::RepositoryParser::new(dir.path()).unwrap();
    let commits = parser.parse_commits(false).unwrap();
    // Date should be recent (within the last minute)
    let now = chrono::Utc::now();
    let diff = now - commits[0].date;
    assert!(diff.num_seconds() < 60, "commit date should be recent");
}
