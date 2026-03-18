use std::process::Command;

fn git_semantic_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_git-semantic"))
}

#[test]
fn test_version_flag() {
    let output = git_semantic_bin().arg("--version").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("git-semantic"),
        "should print version: {stdout}"
    );
}

#[test]
fn test_help_flag() {
    let output = git_semantic_bin().arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Semantic search for git history"));
}

#[test]
fn test_search_help() {
    let output = git_semantic_bin()
        .args(["search", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--author"));
    assert!(stdout.contains("--after"));
    assert!(stdout.contains("--before"));
    assert!(stdout.contains("--file"));
    assert!(stdout.contains("-n"));
}

#[test]
fn test_index_help() {
    let output = git_semantic_bin()
        .args(["index", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--quick"));
    assert!(stdout.contains("--full"));
    assert!(stdout.contains("--force"));
}

#[test]
fn test_unknown_subcommand_fails() {
    let output = git_semantic_bin().arg("foobar").output().unwrap();
    assert!(!output.status.success());
}

#[test]
fn test_search_missing_query_fails() {
    let output = git_semantic_bin().arg("search").output().unwrap();
    assert!(!output.status.success());
}

#[test]
fn test_search_n_flag_in_help() {
    let output = git_semantic_bin()
        .args(["search", "--help"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    // -n flag should control number of results
    assert!(stdout.contains("-n") || stdout.contains("--results"));
}

#[test]
fn test_stats_help() {
    let output = git_semantic_bin()
        .args(["stats", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
}

#[test]
fn test_update_help() {
    let output = git_semantic_bin()
        .args(["update", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
}

#[test]
fn test_init_help() {
    let output = git_semantic_bin()
        .args(["init", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--force"));
}
