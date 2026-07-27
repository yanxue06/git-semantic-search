use clap::{Parser, Subcommand};
use git_semantic::{cli, embedding, git, index, search};
use std::process;

#[derive(Parser)]
#[command(name = "git-semantic")]
#[command(about = "Semantic search for git history", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize git-semantic (download models and prepare environment)
    Init {
        /// Force re-download of models
        #[arg(long)]
        force: bool,
    },

    /// Index the git repository
    Index {
        /// Only index commit messages (faster)
        #[arg(long)]
        quick: bool,

        /// Index messages and diffs (more thorough)
        #[arg(long)]
        full: bool,

        /// Force full re-index from scratch (required to change between quick/full modes)
        #[arg(long)]
        force: bool,

        /// Repository path (defaults to current directory)
        #[arg(short, long)]
        path: Option<String>,
    },

    /// Update the index with new commits
    Update {
        /// Repository path (defaults to current directory)
        #[arg(short, long)]
        path: Option<String>,
    },

    /// Search the git history
    Search {
        /// Search query in natural language
        query: String,

        /// Number of results to return
        #[arg(short = 'n', long, default_value = "10")]
        results: usize,

        /// Filter by author
        #[arg(long)]
        author: Option<String>,

        /// Filter by commits after this date (YYYY-MM-DD)
        #[arg(long)]
        after: Option<String>,

        /// Filter by commits before this date (YYYY-MM-DD)
        #[arg(long)]
        before: Option<String>,

        /// Filter by file path
        #[arg(long)]
        file: Option<String>,

        /// Score every commit instead of using the approximate graph
        #[arg(long)]
        exact: bool,

        /// Graph candidate-list width — higher is slower and more accurate
        #[arg(long, value_name = "N")]
        ef: Option<usize>,

        /// Repository path (defaults to current directory)
        #[arg(long)]
        path: Option<String>,
    },

    /// Show index statistics
    Stats {
        /// Repository path (defaults to current directory)
        #[arg(short, long)]
        path: Option<String>,
    },
}

fn main() {
    // Initialize tracing — only show logs when RUST_LOG is explicitly set
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Init { force } => {
            tracing::info!("Initializing git-semantic...");
            cli::commands::init(force)
        }
        Commands::Index {
            quick,
            full,
            force,
            path,
        } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            let include_diffs = full || !quick;
            cli::commands::index(&repo_path, include_diffs, force)
        }
        Commands::Update { path } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            cli::commands::update(&repo_path)
        }
        Commands::Search {
            query,
            results,
            author,
            after,
            before,
            file,
            exact,
            ef,
            path,
        } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            let filters = cli::SearchFilters {
                author,
                after,
                before,
                file,
            };
            cli::commands::search(&repo_path, &query, results, filters, exact, ef)
        }
        Commands::Stats { path } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            cli::commands::stats(&repo_path)
        }
    };

    if let Err(err) = result {
        print_error(&err);
        process::exit(1);
    }
}

/// Format and print errors in a user-friendly way, with hints when available.
/// Walks the full error chain to surface root causes without exposing internals.
fn print_error(err: &anyhow::Error) {
    eprintln!("\n  error: {err}");

    // Walk the error chain for context
    let mut source = err.source();
    while let Some(cause) = source {
        eprintln!("  caused by: {cause}");
        source = std::error::Error::source(cause);
    }

    // Try to extract a hint from known error types
    if let Some(hint) = extract_hint(err) {
        eprintln!("\n  hint: {hint}");
    }

    // Extract error code if available
    if let Some(code) = extract_code(err) {
        eprintln!("  code: {code}");
    }

    eprintln!();
}

/// Extract a hint from any of our domain error types.
fn extract_hint(err: &anyhow::Error) -> Option<&'static str> {
    if let Some(e) = err.downcast_ref::<embedding::EmbeddingError>() {
        return e.hint();
    }
    if let Some(e) = err.downcast_ref::<git::GitError>() {
        return e.hint();
    }
    if let Some(e) = err.downcast_ref::<index::IndexError>() {
        return e.hint();
    }
    if let Some(e) = err.downcast_ref::<search::SearchError>() {
        return e.hint();
    }
    None
}

/// Extract an error code from any of our domain error types.
fn extract_code(err: &anyhow::Error) -> Option<&'static str> {
    if let Some(e) = err.downcast_ref::<embedding::EmbeddingError>() {
        return Some(e.code());
    }
    if let Some(e) = err.downcast_ref::<git::GitError>() {
        return Some(e.code());
    }
    if let Some(e) = err.downcast_ref::<index::IndexError>() {
        return Some(e.code());
    }
    if let Some(e) = err.downcast_ref::<search::SearchError>() {
        return Some(e.code());
    }
    None
}
