use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tracing::info;

use crate::embedding::ModelManager;
use crate::git::{GitError, RepositoryParser};
use crate::index::{IndexBuilder, IndexError, IndexStorage, SemanticIndex};
use crate::search::SearchEngine;

use super::SearchFilters;

pub fn init(force: bool) -> Result<()> {
    println!("🚀 Initializing git-semantic...\n");

    let model_manager = ModelManager::new()?;

    if force || !model_manager.is_model_downloaded() {
        println!("📥 Downloading embedding model (bge-small-en-v1.5, ~130MB)...");
        println!("This is a one-time setup and may take a few minutes.\n");

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message("Downloading model...");

        model_manager.download_model()?;

        pb.finish_with_message("✅ Model downloaded successfully!");
    } else {
        println!("✅ Model already downloaded");
    }

    println!("\n🎉 git-semantic is ready to use!");
    println!("\nNext steps:");
    println!("  1. Navigate to a git repository");
    println!("  2. Run: git-semantic index");
    println!("  3. Run: git-semantic search \"your query\"");

    Ok(())
}

pub fn index(repo_path: &str, include_diffs: bool, force: bool) -> Result<()> {
    let path = Path::new(repo_path);
    let storage = IndexStorage::new(path)?;

    let existing_index = match storage.load() {
        Ok(idx) => Some(idx),
        Err(IndexError::IndexNotFound) => None,
        Err(e) => return Err(e).context("Failed to load existing index"),
    };

    match existing_index {
        Some(existing) => {
            let existing_mode = existing.metadata.include_diffs;

            if force {
                // Force rebuild — warn if downgrading from full to quick
                if existing_mode && !include_diffs {
                    println!(
                        "⚠️  Downgrading from full mode to quick mode. This will discard \
                         diff embeddings for {} commits.\n\
                         Switching back to full mode later will require re-embedding all commits.\n",
                        existing.entries.len()
                    );
                }
                return full_index(path, &storage, include_diffs);
            }

            if existing_mode != include_diffs {
                if !include_diffs && existing_mode {
                    // Full index already exists, quick is a superset — just do incremental with full mode
                    println!(
                        "ℹ️  Index was built in full mode (with diffs), which is a superset of quick mode.\n\
                         Keeping full mode and checking for new commits.\n\
                         To downgrade to quick mode (smaller index), run with --force.\n\
                         Note: switching back to full mode later will require re-embedding all commits.\n"
                    );
                    incremental_index(path, &storage, existing, existing_mode)?;
                } else {
                    // Quick index exists, full requested — requires re-embedding everything
                    println!(
                        "⚠️  Index was built in quick mode (messages only). Switching to full mode \
                         (with diffs) requires re-embedding all {} commits.\n\
                         Run with --force to rebuild the index.",
                        existing.entries.len()
                    );
                }
                return Ok(());
            }

            incremental_index(path, &storage, existing, include_diffs)?;
        }
        None => {
            full_index(path, &storage, include_diffs)?;
        }
    }

    Ok(())
}

fn full_index(path: &Path, storage: &IndexStorage, include_diffs: bool) -> Result<()> {
    let mode = if include_diffs { "full" } else { "quick" };
    println!(
        "📚 Indexing repository ({} mode): {}\n",
        mode,
        path.display()
    );

    info!("Parsing git repository...");
    let parser = RepositoryParser::new(path)?;
    let commits = parser.parse_commits(include_diffs)?;

    println!("Found {} commits to index\n", commits.len());

    let model_manager = ModelManager::new()?;
    let mut builder = IndexBuilder::new(model_manager, include_diffs)?;

    // Commits are in newest-first order from revwalk; track HEAD as last_commit
    if let Some(first) = commits.first() {
        builder.set_last_commit(first.hash.clone());
    }

    let pb = make_progress_bar(commits.len() as u64);

    for commit in commits {
        builder.add_commit(commit)?;
        pb.inc(1);
    }

    pb.finish_with_message("✅ Commits indexed");

    println!("\n💾 Saving index...");
    let index = builder.build();
    storage.save(&index)?;

    print_index_stats(&index, storage)?;

    Ok(())
}

fn incremental_index(
    path: &Path,
    storage: &IndexStorage,
    existing: SemanticIndex,
    include_diffs: bool,
) -> Result<()> {
    let parser = RepositoryParser::new(path)?;

    let new_commits = match parser.parse_commits_since(&existing.last_commit, include_diffs) {
        Ok(commits) => commits,
        Err(GitError::CommitNotFound(_)) => {
            println!(
                "⚠️  Previously indexed commit {} not found in history (was the branch rebased?).",
                &existing.last_commit[..7.min(existing.last_commit.len())]
            );
            println!("Re-indexing from scratch...\n");
            return full_index(path, storage, include_diffs);
        }
        Err(err) => {
            return Err(err.into());
        }
    };

    if new_commits.is_empty() {
        println!(
            "✅ Index is already up to date! ({} commits indexed)",
            existing.entries.len()
        );
        return Ok(());
    }

    let mode = if include_diffs { "full" } else { "quick" };
    println!(
        "📚 Updating index ({} mode): {} ({} new commits)\n",
        mode,
        path.display(),
        new_commits.len()
    );

    let model_manager = ModelManager::new()?;
    let mut builder = IndexBuilder::from_existing(existing, model_manager)?;

    // New commits are newest-first; update last_commit to the newest
    if let Some(first) = new_commits.first() {
        builder.set_last_commit(first.hash.clone());
    }

    let pb = make_progress_bar(new_commits.len() as u64);

    for commit in new_commits {
        builder.add_commit(commit)?;
        pb.inc(1);
    }

    pb.finish_with_message("✅ New commits indexed");

    println!("\n💾 Saving index...");
    let index = builder.build();
    storage.save(&index)?;

    print_index_stats(&index, storage)?;

    Ok(())
}

pub fn update(repo_path: &str) -> Result<()> {
    println!(
        "Note: `git-semantic index` now automatically handles incremental updates.\n\
         The `update` command will be removed in a future release.\n"
    );
    index(repo_path, true, false)
}

pub fn search(
    repo_path: &str,
    query: &str,
    num_results: usize,
    filters: SearchFilters,
) -> Result<()> {
    let path = Path::new(repo_path);

    let storage = IndexStorage::new(path)?;
    let index = storage.load()?;

    let model_manager = ModelManager::new()?;
    let mut engine = SearchEngine::new(model_manager)?;
    let results = engine.search(&index, query, num_results, filters)?;

    if results.is_empty() {
        println!("No results found for: \"{}\"", query);
        return Ok(());
    }

    println!("🎯 Most Relevant Commits for: \"{}\"\n", query);

    for result in results {
        println!(
            "{}. {} - {} ({:.2} similarity)",
            result.rank,
            &result.commit.hash[..7],
            result.commit.message.lines().next().unwrap_or(""),
            result.similarity
        );
        println!(
            "   Author: {}, {}",
            result.commit.author, result.commit.date
        );

        if !result.commit.diff_summary.is_empty() {
            let preview: String = result
                .commit
                .diff_summary
                .lines()
                .take(2)
                .collect::<Vec<_>>()
                .join("\n   ");
            if !preview.is_empty() {
                println!("   {}", preview);
            }
        }

        println!();
    }

    Ok(())
}

pub fn stats(repo_path: &str) -> Result<()> {
    let path = Path::new(repo_path);

    let storage = IndexStorage::new(path)?;
    let index = storage.load()?;

    println!("📊 Index Statistics\n");
    println!("Repository: {}", path.display());
    println!("Total commits indexed: {}", index.entries.len());
    println!("Model version: {}", index.model_version);
    println!("Last indexed commit: {}", index.last_commit);
    println!(
        "Index mode: {}",
        if index.metadata.include_diffs {
            "full (with diffs)"
        } else {
            "quick (messages only)"
        }
    );
    println!("Index size: ~{:.2} MB", storage.index_size_mb()?);
    println!(
        "Created: {}",
        index.metadata.created_at.format("%Y-%m-%d %H:%M:%S")
    );
    println!(
        "Last updated: {}",
        index.metadata.updated_at.format("%Y-%m-%d %H:%M:%S")
    );

    Ok(())
}

fn make_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb
}

fn print_index_stats(index: &SemanticIndex, storage: &IndexStorage) -> Result<()> {
    println!("✅ Index saved successfully!");
    println!("\n📊 Index statistics:");
    println!("  - Total commits: {}", index.entries.len());
    println!(
        "  - Mode: {}",
        if index.metadata.include_diffs {
            "full (with diffs)"
        } else {
            "quick (messages only)"
        }
    );
    println!("  - Model: {}", index.model_version);
    println!("  - Index size: ~{:.2} MB", storage.index_size_mb()?);
    Ok(())
}
