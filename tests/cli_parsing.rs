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

#[test]
fn test_completions_emit_a_script_per_shell() {
    // Each shell gets a marker only its own format contains, so a silently
    // wrong generator would not pass by emitting the same script five times.
    for (shell, marker) in [
        ("zsh", "#compdef git-semantic"),
        ("bash", "complete -F _git__semantic"),
        ("fish", "complete -c git-semantic"),
        ("elvish", "edit:completion:arg-completer"),
        ("powershell", "Register-ArgumentCompleter"),
    ] {
        let output = git_semantic_bin()
            .args(["completions", shell])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{shell} completions should succeed"
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(marker),
            "{shell} script should contain {marker}, got: {}",
            &stdout[..stdout.len().min(200)]
        );
    }
}

#[test]
fn test_completions_cover_every_subcommand() {
    let output = git_semantic_bin()
        .args(["completions", "zsh"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for subcommand in ["init", "index", "update", "search", "stats", "completions"] {
        assert!(
            stdout.contains(subcommand),
            "completion script should know about `{subcommand}`"
        );
    }
}

#[test]
fn test_completions_rejects_an_unknown_shell() {
    let output = git_semantic_bin()
        .args(["completions", "tcsh"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("bash"), "should list the shells it knows");
}

/// A repository with no commits fails before any model work, which keeps these
/// tests offline while still getting a tracing record emitted.
fn empty_repo() -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().unwrap();
    git2::Repository::init(dir.path()).unwrap();
    dir
}

#[test]
fn test_log_records_never_land_on_stdout() {
    let dir = empty_repo();
    let output = git_semantic_bin()
        .args(["index", "--path", dir.path().to_str().unwrap()])
        .env("RUST_LOG", "info")
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("Parsing git repository"),
        "log records belong on stderr; stdout was: {stdout}"
    );
}

#[test]
fn test_logging_is_off_unless_rust_log_asks_for_it() {
    let dir = empty_repo();
    let output = git_semantic_bin()
        .args(["index", "--path", dir.path().to_str().unwrap()])
        .env_remove("RUST_LOG")
        .output()
        .unwrap();

    let both = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !both.contains("Parsing git repository"),
        "a default run should print no log records, got: {both}"
    );
}

#[test]
fn test_rust_log_still_turns_logging_on() {
    let dir = empty_repo();
    let output = git_semantic_bin()
        .args(["index", "--path", dir.path().to_str().unwrap()])
        .env("RUST_LOG", "info")
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Parsing git repository"),
        "RUST_LOG=info should still produce logs, stderr was: {stderr}"
    );
}
