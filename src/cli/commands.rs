use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tracing::info;

use crate::embedding::ModelManager;
use crate::git::RepositoryParser;
use crate::index::{IndexBuilder, IndexStorage};
use crate::search::SearchEngine;

use super::SearchFilters;

pub fn init(force: bool) -> Result<()> {
    println!("🚀 Initializing git-semantic...\n");

    let model_manager = ModelManager::new()?;

    if force || !model_manager.is_model_downloaded()? {
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

pub fn index(repo_path: &str, include_diffs: bool) -> Result<()> {
    let path = Path::new(repo_path);
    println!("📚 Indexing repository: {}\n", path.display());

    // Parse git repository
    info!("Parsing git repository...");
    let parser = RepositoryParser::new(path)?;
    let commits = parser.parse_commits(include_diffs)?;

    println!("Found {} commits to index\n", commits.len());

    // Build index
    info!("Building semantic index...");
    let pb = ProgressBar::new(commits.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let model_manager = ModelManager::new()?;
    let mut builder = IndexBuilder::new(model_manager)?;

    for commit in commits {
        builder.add_commit(commit)?;
        pb.inc(1);
    }

    pb.finish_with_message("✅ Commits indexed");

    // Save index
    println!("\n💾 Saving index...");
    let index = builder.build()?;
    let storage = IndexStorage::new(path)?;
    storage.save(&index)?;

    println!("✅ Index saved successfully!");
    println!("\n📊 Index statistics:");
    println!("  - Total commits: {}", index.entries.len());
    println!("  - Model: {}", index.model_version);
    println!("  - Index size: ~{:.2} MB", storage.index_size_mb()?);

    Ok(())
}

pub fn update(repo_path: &str) -> Result<()> {
    let path = Path::new(repo_path);
    println!("🔄 Updating index for: {}\n", path.display());

    // Load existing index
    let storage = IndexStorage::new(path)?;
    let index = storage
        .load()
        .context("No index found. Run 'git-semantic index' first.")?;

    // Parse new commits
    let parser = RepositoryParser::new(path)?;
    let new_commits = parser.parse_commits_since(&index.last_commit)?;

    if new_commits.is_empty() {
        println!("✅ Index is already up to date!");
        return Ok(());
    }

    println!("Found {} new commits\n", new_commits.len());

    // Update index
    let model_manager = ModelManager::new()?;
    let mut builder = IndexBuilder::from_existing(index, model_manager)?;

    let pb = ProgressBar::new(new_commits.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    for commit in new_commits {
        builder.add_commit(commit)?;
        pb.inc(1);
    }

    pb.finish_with_message("✅ New commits indexed");

    // Save updated index
    let updated_index = builder.build()?;
    storage.save(&updated_index)?;

    println!("\n✅ Index updated successfully!");

    Ok(())
}

pub fn search(
    repo_path: &str,
    query: &str,
    num_results: usize,
    filters: SearchFilters,
) -> Result<()> {
    let path = Path::new(repo_path);

    // Load index
    let storage = IndexStorage::new(path)?;
    let index = storage
        .load()
        .context("No index found. Run 'git-semantic index' first.")?;

    // Perform search
    let model_manager = ModelManager::new()?;
    let mut engine = SearchEngine::new(model_manager)?;
    let results = engine.search(&index, query, num_results, filters)?;

    // Display results
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
        println!("   Author: {}, {}", result.commit.author, result.commit.date);

        // Show first few lines of diff summary if available
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
    let index = storage
        .load()
        .context("No index found. Run 'git-semantic index' first.")?;

    println!("📊 Index Statistics\n");
    println!("Repository: {}", path.display());
    println!("Total commits indexed: {}", index.entries.len());
    println!("Model version: {}", index.model_version);
    println!("Last indexed commit: {}", index.last_commit);
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

