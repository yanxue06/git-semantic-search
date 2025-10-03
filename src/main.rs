use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;

mod cli;
mod embedding;
mod git;
mod index;
mod search;

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

        /// Repository path (defaults to current directory)
        #[arg(short, long)]
        path: Option<String>,
    },

    /// Show index statistics
    Stats {
        /// Repository path (defaults to current directory)
        #[arg(short, long)]
        path: Option<String>,
    },
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { force } => {
            info!("Initializing git-semantic...");
            cli::commands::init(force)?;
        }
        Commands::Index { quick, full, path } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            let include_diffs = full || !quick;
            cli::commands::index(&repo_path, include_diffs)?;
        }
        Commands::Update { path } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            cli::commands::update(&repo_path)?;
        }
        Commands::Search {
            query,
            results,
            author,
            after,
            before,
            file,
            path,
        } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            let filters = cli::SearchFilters {
                author,
                after,
                before,
                file,
            };
            cli::commands::search(&repo_path, &query, results, filters)?;
        }
        Commands::Stats { path } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            cli::commands::stats(&repo_path)?;
        }
    }

    Ok(())
}

