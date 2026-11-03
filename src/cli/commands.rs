use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::Path;
use std::time::Instant;
use tracing::info;

use super::output::JsonOutput;
use super::{ModelFlags, Pooling};
use crate::embedding::{
    DeclaredPooling, EmbeddingError, ModelChoice, ModelManager, ModelSource, ModelSpec,
    clear_active_model, download_repo, downloaded_models, model_dir, model_files_present,
    parse_model_choice, read_model_settings, save_active_model,
};
use crate::git::{GitError, RepositoryParser};
use crate::index::{EXACT_SCAN_THRESHOLD, IndexBuilder, IndexError, IndexStorage, SemanticIndex};
use crate::search::{SearchEngine, SearchOptions, SearchStrategy};
use crate::text::Bm25Params;
use crate::vector::HnswParams;

use super::SearchRequest;

/// Where indexing progress is written.
///
/// `index` is the user asking for indexing, so it reports on stdout. `search`
/// can now trigger the same work on its own, and that run might be emitting
/// `--json` — a progress line in the middle of a JSON document is not
/// parseable, so bootstrap chatter goes to stderr instead.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Progress {
    Stdout,
    Stderr,
}

impl Progress {
    fn say(self, line: &str) {
        match self {
            Self::Stdout => println!("{line}"),
            Self::Stderr => eprintln!("{line}"),
        }
    }
}

pub fn init(force: bool) -> Result<()> {
    println!("🚀 Initializing git-semantic...\n");

    let model_manager = ModelManager::new()?;

    if let ModelSource::Local(dir) = &model_manager.spec().source {
        // A local model is used where it sits, so there is nothing to fetch.
        if !model_manager.is_model_downloaded() {
            return Err(EmbeddingError::local_model_missing(dir).into());
        }
        println!("✅ Using the local model in {}", dir.display());
    } else if force || !model_manager.is_model_downloaded() {
        download_model(&model_manager, Progress::Stdout)?;
    } else {
        println!("✅ Model already downloaded");
    }

    println!("\n🎉 git-semantic is ready to use!");
    println!("\nJust search — the index builds itself on first use:");
    println!("  git sem search \"your query\"");

    Ok(())
}

/// A model manager with the model on disk, downloading it if this is the first
/// run on this machine.
///
/// `init` used to be the only thing that could download, which made it a
/// mandatory step rather than an optional warm-up.
fn ensure_model(progress: Progress) -> Result<ModelManager> {
    let manager = ModelManager::new()?;
    if !manager.is_model_downloaded() {
        download_model(&manager, progress)?;
    }
    Ok(manager)
}

fn download_model(manager: &ModelManager, progress: Progress) -> Result<()> {
    progress.say(&format!(
        "📥 Downloading embedding model ({})...",
        manager.spec().name()
    ));
    progress.say("This is a one-time setup and may take a few minutes.\n");

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message("Downloading model...");

    manager.download_model()?;

    pb.finish_with_message("✅ Model downloaded successfully!");
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
                full_index(path, &storage, include_diffs, Progress::Stdout)?;
                return Ok(());
            }

            let active_model = ModelManager::new()?.model_version();
            if existing.model_version != active_model {
                println!(
                    "🔁 This index was built with {}, but the active model is {}.\n\
                     Re-embedding all {} commits.\n",
                    existing.model_version,
                    active_model,
                    existing.entries.len()
                );
                full_index(path, &storage, include_diffs, Progress::Stdout)?;
                return Ok(());
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
            full_index(path, &storage, include_diffs, Progress::Stdout)?;
        }
    }

    Ok(())
}

fn full_index(
    path: &Path,
    storage: &IndexStorage,
    include_diffs: bool,
    progress: Progress,
) -> Result<SemanticIndex> {
    let mode = if include_diffs { "full" } else { "quick" };
    progress.say(&format!(
        "📚 Indexing repository ({} mode): {}\n",
        mode,
        path.display()
    ));

    info!("Parsing git repository...");
    let parser = RepositoryParser::new(path)?;
    let commits = parser.parse_commits(include_diffs)?;

    progress.say(&format!("Found {} commits to index\n", commits.len()));

    let model_manager = ensure_model(progress)?;
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

    progress.say("\n💾 Saving index...");
    let index = builder.build();
    storage.save(&index)?;

    print_index_stats(&index, storage, progress)?;
    refresh_search_graph(&index, storage, progress);

    Ok(index)
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
            full_index(path, storage, include_diffs, Progress::Stdout)?;
            return Ok(());
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

    let model_manager = ensure_model(Progress::Stdout)?;
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

    print_index_stats(&index, storage, Progress::Stdout)?;
    refresh_search_graph(&index, storage, Progress::Stdout);

    Ok(())
}

pub fn update(repo_path: &str) -> Result<()> {
    println!(
        "Note: `git sem index` now automatically handles incremental updates.\n\
         The `update` command will be removed in a future release.\n"
    );
    index(repo_path, true, false)
}

pub fn search(repo_path: &str, request: SearchRequest) -> Result<()> {
    let SearchRequest {
        query,
        num_results,
        filters,
        exact,
        ef,
        mode,
        diversity,
        json,
    } = request;
    let query = query.as_str();

    let path = Path::new(repo_path);

    let storage = IndexStorage::new(path)?;
    let index = match storage.load() {
        Ok(index) => index,
        // Nothing indexed yet. Everything needed to fix that is already known,
        // so do it rather than send the user off to run a different command.
        Err(IndexError::IndexNotFound) => bootstrap(path, &storage)?,
        Err(err) => return Err(err.into()),
    };

    // Vectors from different models can't be compared, so an index another
    // model built is re-embedded before it is searched. Keyword-only search
    // reads no vectors and skips that.
    let model_manager = ModelManager::new()?;
    let active_model = model_manager.model_version();
    let index = if mode.uses_semantic() && index.model_version != active_model {
        reembed(path, &storage, &index, &active_model)?
    } else {
        index
    };

    // BM25 is cheap to build and independent of repository size, so it is
    // loaded whenever lexical retrieval is in play.
    let lexical = if mode.uses_lexical() && !index.entries.is_empty() {
        let build_started = Instant::now();
        let (lexical, rebuilt) = storage.load_or_build_lexical(&index, Bm25Params::default());
        if rebuilt && !json {
            println!(
                "🔧 Built keyword index for {} commits in {:.1}s (cached for next time)\n",
                lexical.len(),
                build_started.elapsed().as_secs_f64()
            );
        }
        Some(lexical)
    } else {
        None
    };

    // Below the exact-scan threshold the graph is never consulted, so don't pay
    // to load or build it either.
    let graph = if exact || index.entries.len() <= EXACT_SCAN_THRESHOLD {
        None
    } else {
        let build_started = Instant::now();
        let (graph, rebuilt) = storage.load_or_build_ann(&index, HnswParams::default());
        if rebuilt && !json {
            println!(
                "🔧 Built search graph for {} commits in {:.1}s (cached for next time)\n",
                graph.len(),
                build_started.elapsed().as_secs_f64()
            );
        }
        Some(graph)
    };

    let mut engine = SearchEngine::new(model_manager)?;

    let started = Instant::now();
    let outcome = engine.search(
        &index,
        graph.as_ref(),
        lexical.as_ref(),
        query,
        SearchOptions {
            num_results,
            filters,
            exact,
            ef,
            mode,
            diversity,
        },
    )?;
    let elapsed = started.elapsed();

    if json {
        let document = JsonOutput::new(
            query,
            &outcome,
            outcome.diversified,
            elapsed.as_secs_f64() * 1000.0,
        );
        println!("{}", serde_json::to_string_pretty(&document)?);
        return Ok(());
    }

    if outcome.results.is_empty() {
        println!("No results found for: \"{}\"", query);
        return Ok(());
    }

    println!("🎯 Most Relevant Commits for: \"{}\"\n", query);

    for result in &outcome.results {
        // A fused or lexical hit has no cosine to report; showing a BM25 score
        // in a column labelled "similarity" would be a lie.
        let score = if result.similarity.is_nan() {
            "keyword match".to_string()
        } else {
            format!("{:.2} similarity", result.similarity)
        };
        println!(
            "{}. {} - {} ({})",
            result.rank,
            &result.commit.hash[..7],
            result.commit.message.lines().next().unwrap_or(""),
            score
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

    let how = match outcome.strategy {
        SearchStrategy::Exact => "exhaustive scan",
        SearchStrategy::Approximate => "graph search",
        SearchStrategy::ApproximateThenExact => "graph search, verified exhaustively",
    };
    let diversified = if outcome.diversified {
        ", diversified"
    } else {
        ""
    };
    println!(
        "Searched {} commits via {} {}{} in {:.0}ms",
        outcome.candidate_count,
        outcome.mode.as_str(),
        how,
        diversified,
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}

/// Index a repository the first time it is searched.
///
/// Onboarding used to be three commands, and the first one a new user ran was
/// the wrong one — `search` answered with an error telling them to run `index`,
/// which answered with an error telling them to run `init`. Nothing about that
/// sequence needed a human: the model location is fixed and the index mode has
/// a default. So `search` does it.
///
/// Full mode, matching `git sem index`. Quick mode is faster but its index
/// is not a superset, so bootstrapping into it would quietly commit the user to
/// the weaker option and a full re-embed to leave it.
fn bootstrap(path: &Path, storage: &IndexStorage) -> Result<SemanticIndex> {
    eprintln!("No index for this repository yet — building one (one-time).");
    eprintln!("For a faster, message-only index instead: git sem index --quick\n");

    full_index(path, storage, true, Progress::Stderr)
}

/// Rebuild an index that a different model built, keeping its quick or full
/// mode. Progress goes to stderr for the same reason as [`bootstrap`].
fn reembed(
    path: &Path,
    storage: &IndexStorage,
    stale: &SemanticIndex,
    active_model: &str,
) -> Result<SemanticIndex> {
    eprintln!(
        "This index was built with {}, but the active model is {}.",
        stale.model_version, active_model
    );
    eprintln!(
        "Re-embedding its {} commits with the active model (one-time).\n",
        stale.entries.len()
    );

    full_index(
        path,
        storage,
        stale.metadata.include_diffs,
        Progress::Stderr,
    )
}

/// `git sem model`: show the active model, or switch to another.
pub fn model(choice: Option<&str>, flags: ModelFlags) -> Result<()> {
    match choice {
        None => show_model(),
        Some(arg) => switch_model(arg, flags),
    }
}

fn show_model() -> Result<()> {
    let manager = ModelManager::new()?;
    let spec = manager.spec();

    println!("Active model: {}", display_name(spec));
    print_model_settings(spec);
    if manager.is_model_downloaded() {
        println!("  files: {}", manager.model_dir().display());
    } else if let ModelSource::Local(dir) = &spec.source {
        println!("  files: missing from {}", dir.display());
    } else {
        println!("  files: not downloaded yet (they download on first use)");
    }

    let downloaded = downloaded_models()?;
    if !downloaded.is_empty() {
        println!("\nDownloaded: {}", downloaded.join(", "));
    }
    println!("\nSwitch with: git sem model <hugging-face-id | directory | default>");
    Ok(())
}

fn switch_model(arg: &str, flags: ModelFlags) -> Result<()> {
    let spec = resolve_model(arg, flags)?;

    // Saving nothing for the plain default lets a future release change what
    // the default is without this machine staying pinned to the old one.
    if spec.is_default() {
        clear_active_model()?;
    } else {
        save_active_model(&spec)?;
    }

    println!("✅ Active model: {}", display_name(&spec));
    print_model_settings(&spec);
    println!("\nIndexes built with another model re-embed the next time you search or index them.");
    Ok(())
}

/// Turn the argument to `git sem model` into a spec that is known to run.
fn resolve_model(arg: &str, flags: ModelFlags) -> Result<ModelSpec> {
    let source = match parse_model_choice(arg)? {
        ModelChoice::Default => {
            // Known to work, so nothing is downloaded or checked here. It
            // downloads on first use, as it always has.
            let mut spec = ModelSpec::default_model();
            if let Some(pooling) = flags.pooling {
                spec.pooling = pooling;
            }
            if let Some(prefix) = flags.query_prefix {
                spec.query_prefix = prefix;
            }
            if let Some(prefix) = flags.document_prefix {
                spec.document_prefix = prefix;
            }
            return Ok(spec);
        }
        ModelChoice::HuggingFace(repo) => ModelSource::HuggingFace(repo),
        ModelChoice::Local(dir) => ModelSource::Local(dir),
    };

    let dir = model_dir(&source)?;
    if let ModelSource::HuggingFace(repo) = &source
        && !model_files_present(&dir)
    {
        println!("📥 Downloading {repo}...\n");
        if let Err(err) = download_repo(repo, &dir) {
            // Nothing usable arrived, so don't leave its folder behind. The
            // repo id was validated, so this is always one folder in models/.
            let _ = fs::remove_dir_all(&dir);
            return Err(err.into());
        }
    }
    if !model_files_present(&dir) {
        return Err(EmbeddingError::local_model_missing(&dir).into());
    }

    let name = source.name();
    let settings = read_model_settings(&dir);
    let pooling = choose_pooling(flags.pooling, settings.pooling, &name)?;

    let mut spec = ModelSpec {
        source,
        pooling,
        query_prefix: flags.query_prefix.unwrap_or(settings.query_prefix),
        document_prefix: flags.document_prefix.unwrap_or(settings.document_prefix),
        max_tokens: settings.max_tokens,
        dimension: 0,
    };
    spec.dimension = check_model_runs(&spec)?;
    Ok(spec)
}

/// The `--pooling` flag wins. Otherwise use what the model declares, as long
/// as it is a kind of pooling this tool can do.
fn choose_pooling(
    flag: Option<Pooling>,
    declared: DeclaredPooling,
    name: &str,
) -> Result<Pooling, EmbeddingError> {
    if let Some(pooling) = flag {
        return Ok(pooling);
    }

    match declared {
        DeclaredPooling::Supported(pooling) => Ok(pooling),
        DeclaredPooling::Missing => Err(EmbeddingError::UnknownPooling(name.to_string())),
        DeclaredPooling::Unsupported(mode) => Err(EmbeddingError::UnsupportedModel {
            model: name.to_string(),
            reason: format!("it pools with `{mode}`, and only cls and mean are supported"),
        }),
    }
}

/// Load the model and embed one sample, so a model that can't run is turned
/// away now instead of partway through indexing a repository. Returns the
/// vector length.
fn check_model_runs(spec: &ModelSpec) -> Result<usize> {
    println!("🔎 Checking that {} runs...", spec.name());

    let mut manager = ModelManager::with_spec(spec.clone())?;
    manager.init()?;
    let vector = manager.encode_document("fix: handle an empty config file without crashing")?;

    let mut usable = !vector.is_empty();
    for value in vector.iter() {
        if !value.is_finite() {
            usable = false;
        }
    }
    if !usable {
        return Err(EmbeddingError::UnsupportedModel {
            model: spec.name(),
            reason: "it produced an empty or non-finite vector".to_string(),
        }
        .into());
    }

    Ok(vector.len())
}

fn display_name(spec: &ModelSpec) -> String {
    if spec.source == ModelSpec::default_model().source {
        format!("{} (default)", spec.name())
    } else {
        spec.name()
    }
}

fn print_model_settings(spec: &ModelSpec) {
    println!(
        "  {} dimensions, {} pooling, up to {} tokens",
        spec.dimension,
        spec.pooling.as_str(),
        spec.max_tokens
    );
    if !spec.query_prefix.is_empty() {
        println!("  query prefix: {:?}", spec.query_prefix);
    }
    if !spec.document_prefix.is_empty() {
        println!("  document prefix: {:?}", spec.document_prefix);
    }
}

pub fn stats(repo_path: &str) -> Result<()> {
    let path = Path::new(repo_path);

    let storage = IndexStorage::new(path)?;
    let index = storage.load()?;

    println!("📊 Index Statistics\n");
    println!("Repository: {}", path.display());
    println!("Total commits indexed: {}", index.entries.len());
    println!("Model version: {}", index.model_version);
    let active_model = ModelManager::new()?.model_version();
    if active_model != index.model_version {
        println!("Active model: {active_model} (the next search re-embeds this index)");
    }
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
        "Keyword index: {}",
        if storage.lexical_path().exists() {
            "BM25 (cached)"
        } else {
            "BM25 (builds on next search)"
        }
    );
    println!(
        "Search strategy: {}",
        if index.entries.len() <= EXACT_SCAN_THRESHOLD {
            format!("exhaustive scan (under the {EXACT_SCAN_THRESHOLD}-commit graph threshold)")
        } else if storage.ann_path().exists() {
            "HNSW graph (cached)".to_string()
        } else {
            "HNSW graph (builds on next search)".to_string()
        }
    );
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

fn print_index_stats(
    index: &SemanticIndex,
    storage: &IndexStorage,
    progress: Progress,
) -> Result<()> {
    progress.say("✅ Index saved successfully!");
    progress.say("\n📊 Index statistics:");
    progress.say(&format!("  - Total commits: {}", index.entries.len()));
    progress.say(&format!(
        "  - Mode: {}",
        if index.metadata.include_diffs {
            "full (with diffs)"
        } else {
            "quick (messages only)"
        }
    ));
    progress.say(&format!("  - Model: {}", index.model_version));
    progress.say(&format!(
        "  - Index size: ~{:.2} MB",
        storage.index_size_mb()?
    ));
    Ok(())
}

/// Build the ANN graph now so the first search doesn't pay for it.
///
/// Skipped for small repositories, which never consult the graph. A failure here
/// is reported, not propagated — the index itself saved fine and search falls
/// back to an exhaustive scan.
fn refresh_search_graph(index: &SemanticIndex, storage: &IndexStorage, progress: Progress) {
    if !index.entries.is_empty()
        && let Err(err) = storage.refresh_lexical(index, Bm25Params::default())
    {
        info!("keyword index will be rebuilt on first search: {err}");
    }

    if index.entries.len() <= EXACT_SCAN_THRESHOLD {
        return;
    }

    let started = Instant::now();
    progress.say(&format!(
        "\n🔧 Building search graph for {} commits...",
        index.entries.len()
    ));

    match storage.refresh_ann(index, HnswParams::default()) {
        Ok(()) => progress.say(&format!(
            "  done in {:.1}s",
            started.elapsed().as_secs_f64()
        )),
        Err(err) => {
            progress.say(&format!("  skipped ({err})"));
            info!("ANN graph will be rebuilt on first search");
        }
    }
}
