//! Switching embedding models, end to end through the binary.
//!
//! The models are the two tiny ONNX encoders in `tests/fixtures`, built by
//! `make_tiny_models.py`. They load and run through the same path a Hugging
//! Face model does, so none of this touches the network.
//!
//! Every run gets its own HOME, so the developer's real model choice and
//! downloads are never touched. Windows looks up those folders without HOME,
//! hence Unix only.
#![cfg(unix)]

use git2::{Repository, Signature};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const COMMITS: [&str; 4] = [
    "fix memory leak in the parser",
    "add login page",
    "update readme with install steps",
    "speed up startup by caching config",
];

/// A home directory and a repository to run the binary against.
struct Sandbox {
    home: TempDir,
    repo: TempDir,
}

impl Sandbox {
    fn new() -> Self {
        let sandbox = Self {
            home: TempDir::new().unwrap(),
            repo: TempDir::new().unwrap(),
        };
        commit_all(sandbox.repo.path(), &COMMITS);
        sandbox
    }

    /// Run the binary with this sandbox as HOME, inside the repository.
    fn run(&self, args: &[&str]) -> Output {
        let home = self.home.path();
        Command::new(env!("CARGO_BIN_EXE_git-semantic"))
            .args(args)
            .current_dir(self.repo.path())
            .env("HOME", home)
            .env("XDG_CONFIG_HOME", home.join(".config"))
            .env("XDG_DATA_HOME", home.join(".local").join("share"))
            .env_remove("RUST_LOG")
            .output()
            .unwrap()
    }

    /// Run, and fail the test with both streams if the command failed.
    fn ok(&self, args: &[&str]) -> String {
        let output = self.run(args);
        assert!(
            output.status.success(),
            "`{}` failed\nstdout: {}\nstderr: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).to_string()
    }

    /// Search in semantic mode and return the subject of each hit, best first.
    fn subjects(&self, query: &str) -> Vec<String> {
        let stdout = self.ok(&["search", query, "--mode", "semantic", "--json"]);
        let document: serde_json::Value = serde_json::from_str(&stdout).unwrap();

        let mut subjects = Vec::new();
        for result in document["results"].as_array().unwrap() {
            subjects.push(result["subject"].as_str().unwrap().to_string());
        }
        subjects
    }
}

fn commit_all(dir: &Path, messages: &[&str]) {
    let repo = Repository::init(dir).unwrap();
    let signature = Signature::now("Test Author", "test@example.com").unwrap();

    for (i, message) in messages.iter().enumerate() {
        let file = format!("file_{i}.txt");
        fs::write(dir.join(&file), message).unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(Path::new(&file)).unwrap();
        index.write().unwrap();
        let tree = repo.find_tree(index.write_tree().unwrap()).unwrap();

        let mut parents = Vec::new();
        if let Ok(head) = repo.head() {
            parents.push(head.peel_to_commit().unwrap());
        }
        let parent_refs: Vec<&git2::Commit> = parents.iter().collect();
        repo.commit(
            Some("HEAD"),
            &signature,
            &signature,
            message,
            &tree,
            &parent_refs,
        )
        .unwrap();
    }
}

fn fixture(name: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", name]
        .iter()
        .collect();
    path.to_str().unwrap().to_string()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

#[test]
fn switching_reports_what_was_read_from_the_model() {
    let sandbox = Sandbox::new();

    let mean = sandbox.ok(&["model", &fixture("tiny-mean")]);
    assert!(mean.contains("16 dimensions, mean pooling"), "{mean}");

    // tiny-cls declares prompts and a short context in its settings files.
    let cls = sandbox.ok(&["model", &fixture("tiny-cls")]);
    assert!(
        cls.contains("24 dimensions, cls pooling, up to 128 tokens"),
        "{cls}"
    );
    assert!(cls.contains("query prefix: \"query: \""), "{cls}");
    assert!(cls.contains("document prefix: \"passage: \""), "{cls}");

    let shown = sandbox.ok(&["model"]);
    assert!(shown.contains(&fixture("tiny-cls")), "{shown}");
}

#[test]
fn search_runs_on_the_chosen_model() {
    let sandbox = Sandbox::new();
    sandbox.ok(&["model", &fixture("tiny-mean")]);

    // No index yet: the first search builds one with the chosen model.
    let subjects = sandbox.subjects("memory leak");
    assert_eq!(subjects[0], "fix memory leak in the parser");

    let stats = sandbox.ok(&["stats"]);
    assert!(stats.contains("tiny-mean (mean pooling)"), "{stats}");
}

#[test]
fn the_next_search_after_a_switch_reembeds_instead_of_mixing_models() {
    let sandbox = Sandbox::new();
    sandbox.ok(&["model", &fixture("tiny-mean")]);
    sandbox.subjects("memory leak");

    sandbox.ok(&["model", &fixture("tiny-cls")]);
    let stats = sandbox.ok(&["stats"]);
    assert!(
        stats.contains("the next search re-embeds this index"),
        "stats should flag the stale index: {stats}"
    );

    // 16-dim document vectors against a 24-dim query would be garbage, so
    // this search has to re-embed before it can answer.
    let output = sandbox.run(&["search", "caching startup", "--mode", "semantic", "--json"]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("Re-embedding its 4 commits"),
        "{}",
        stderr(&output)
    );

    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        document["results"][0]["subject"],
        "speed up startup by caching config"
    );

    let stats = sandbox.ok(&["stats"]);
    assert!(stats.contains("tiny-cls (cls pooling"), "{stats}");
    assert!(
        !stats.contains("re-embeds"),
        "index is current again: {stats}"
    );
}

#[test]
fn index_rebuilds_for_a_new_model_rather_than_appending() {
    let sandbox = Sandbox::new();
    sandbox.ok(&["model", &fixture("tiny-mean")]);
    sandbox.ok(&["index"]);

    sandbox.ok(&["model", &fixture("tiny-cls")]);
    let rebuilt = sandbox.ok(&["index"]);
    assert!(rebuilt.contains("Re-embedding all 4 commits"), "{rebuilt}");

    let stats = sandbox.ok(&["stats"]);
    assert!(stats.contains("Total commits indexed: 4"), "{stats}");
    assert!(stats.contains("tiny-cls (cls pooling"), "{stats}");
}

#[test]
fn keyword_search_does_not_reembed() {
    let sandbox = Sandbox::new();
    sandbox.ok(&["model", &fixture("tiny-mean")]);
    sandbox.ok(&["index"]);
    sandbox.ok(&["model", &fixture("tiny-cls")]);

    let output = sandbox.run(&["search", "login", "--mode", "lexical", "--json"]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(
        !stderr(&output).contains("Re-embedding"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_model_that_does_not_declare_pooling_needs_the_flag() {
    let sandbox = Sandbox::new();

    // tiny-mean's files minus its 1_Pooling/config.json.
    let bare = TempDir::new().unwrap();
    for file in ["model.onnx", "tokenizer.json"] {
        fs::copy(
            Path::new(&fixture("tiny-mean")).join(file),
            bare.path().join(file),
        )
        .unwrap();
    }
    let bare_path = bare.path().to_str().unwrap();

    let refused = sandbox.run(&["model", bare_path]);
    assert!(!refused.status.success());
    assert!(stderr(&refused).contains("E1014"), "{}", stderr(&refused));

    let accepted = sandbox.ok(&["model", bare_path, "--pooling", "mean"]);
    assert!(accepted.contains("mean pooling"), "{accepted}");
}

#[test]
fn a_failed_switch_keeps_the_previous_model() {
    let sandbox = Sandbox::new();
    sandbox.ok(&["model", &fixture("tiny-mean")]);

    let empty = TempDir::new().unwrap();
    let refused = sandbox.run(&["model", empty.path().to_str().unwrap()]);
    assert!(!refused.status.success());
    assert!(stderr(&refused).contains("E1016"), "{}", stderr(&refused));

    let shown = sandbox.ok(&["model"]);
    assert!(shown.contains(&fixture("tiny-mean")), "{shown}");
}

#[test]
fn a_name_that_is_neither_a_repo_nor_a_directory_fails_offline() {
    let sandbox = Sandbox::new();
    let refused = sandbox.run(&["model", "nomic-embed-text"]);
    assert!(!refused.status.success());
    assert!(stderr(&refused).contains("E1011"), "{}", stderr(&refused));
}

#[test]
fn default_switches_back_without_downloading_anything() {
    let sandbox = Sandbox::new();
    sandbox.ok(&["model", &fixture("tiny-mean")]);

    let reset = sandbox.ok(&["model", "default"]);
    assert!(reset.contains("bge-small-en-v1.5 (default)"), "{reset}");

    let shown = sandbox.ok(&["model"]);
    assert!(shown.contains("bge-small-en-v1.5 (default)"), "{shown}");
    assert!(shown.contains("not downloaded yet"), "{shown}");
}

#[test]
fn settings_flags_need_a_model_to_apply_to() {
    let sandbox = Sandbox::new();
    let refused = sandbox.run(&["model", "--pooling", "mean"]);
    assert!(!refused.status.success());
}
