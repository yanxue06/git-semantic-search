use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use git_semantic::{cli, embedding, git, index, search};
use std::process;

#[derive(Parser)]
#[command(name = "git-sem")]
#[command(about = "Semantic search for git history", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// CLI surface for [`cli::RetrievalMode`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Fuse semantic and keyword results (default)
    Hybrid,
    /// Embedding similarity only
    Semantic,
    /// Keyword (BM25) matching only
    Lexical,
}

impl From<Mode> for cli::RetrievalMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::Hybrid => Self::Hybrid,
            Mode::Semantic => Self::Semantic,
            Mode::Lexical => Self::Lexical,
        }
    }
}

/// CLI surface for [`cli::Pooling`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum PoolingArg {
    /// Use the first token's vector (BGE, Arctic, mxbai)
    Cls,
    /// Average all token vectors (MiniLM, mpnet, nomic, e5, gte)
    Mean,
}

impl From<PoolingArg> for cli::Pooling {
    fn from(pooling: PoolingArg) -> Self {
        match pooling {
            PoolingArg::Cls => Self::Cls,
            PoolingArg::Mean => Self::Mean,
        }
    }
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

        /// Which retrievers to use
        #[arg(long, value_enum, default_value_t = Mode::Hybrid)]
        mode: Mode,

        /// Spread results across distinct changes instead of near-duplicates
        #[arg(long)]
        diverse: bool,

        /// Diversity balance: 1.0 is pure relevance, 0.0 pure novelty
        #[arg(long, value_name = "L", default_value_t = git_semantic::search::DEFAULT_LAMBDA)]
        lambda: f32,

        /// Emit machine-readable JSON instead of formatted text
        #[arg(long)]
        json: bool,

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

    /// Show the embedding model, or switch to another one
    ///
    /// With no argument, prints the active model. With one, switches to it:
    ///
    ///   git sem model BAAI/bge-base-en-v1.5   any Hugging Face repo with an ONNX export
    ///   git sem model ./my-model              a directory with model.onnx and tokenizer.json
    ///   git sem model default                 back to bge-small-en-v1.5
    ///
    /// Pooling and prompts are read from the model's sentence-transformers
    /// files when it has them. The flags below cover models that don't.
    /// Indexes built with another model re-embed on their next search.
    #[command(verbatim_doc_comment)]
    Model {
        /// Hugging Face id, local directory, or `default`
        model: Option<String>,

        /// How token vectors become one vector, if the model doesn't say
        #[arg(long, value_enum, requires = "model")]
        pooling: Option<PoolingArg>,

        /// Text put before each search query, e.g. "search_query: " for nomic
        #[arg(long, value_name = "TEXT", requires = "model")]
        query_prefix: Option<String>,

        /// Text put before each commit when indexing, e.g. "search_document: " for nomic
        #[arg(long, value_name = "TEXT", requires = "model")]
        document_prefix: Option<String>,
    },

    /// Print a shell completion script to stdout
    ///
    /// Source it from your shell config, or drop it somewhere on the
    /// completion path:
    ///
    ///   bash    git-sem completions bash > /etc/bash_completion.d/git-sem
    ///   zsh     git-sem completions zsh > ~/.zfunc/_git-sem
    ///   fish    git-sem completions fish > ~/.config/fish/completions/git-sem.fish
    #[command(verbatim_doc_comment)]
    Completions {
        /// Shell to generate the script for
        #[arg(value_enum)]
        shell: Shell,
    },
}

/// Diagnostics on stderr, silent unless `RUST_LOG` asks for them.
///
/// Two separate things used to send log records to stdout on every run.
/// `add_directive(INFO)` made INFO the floor rather than a fallback, so the
/// filter was on whether or not `RUST_LOG` was set; and `fmt()` writes to
/// stdout by default. Together they put several hundred lines of ONNX Runtime
/// session log in front of every result — and inside `--json`, which made the
/// document unparseable.
fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
}

/// Let a closed pipe kill the process the way every other Unix tool dies.
///
/// Rust ignores `SIGPIPE` at startup, so `… | head -5` turns a write into an
/// `EPIPE`, and `println!` panics on it — the user asks for five results and
/// gets a Rust backtrace. Restoring the default disposition makes the process
/// exit quietly instead, which matters most for the two outputs people
/// actually pipe: `--json` and `completions`.
#[cfg(unix)]
fn die_quietly_on_closed_pipe() {
    // SAFETY: runs before any other thread exists, and only resets a signal
    // disposition to the operating system's default.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
}

#[cfg(not(unix))]
fn die_quietly_on_closed_pipe() {}

fn main() {
    die_quietly_on_closed_pipe();

    init_logging();

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
            mode,
            diverse,
            lambda,
            json,
            path,
        } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            let filters = cli::SearchFilters {
                author,
                after,
                before,
                file,
            };
            cli::commands::search(
                &repo_path,
                cli::SearchRequest {
                    query,
                    num_results: results,
                    filters,
                    exact,
                    ef,
                    mode: mode.into(),
                    diversity: diverse.then_some(lambda),
                    json,
                },
            )
        }
        Commands::Stats { path } => {
            let repo_path = path.unwrap_or_else(|| ".".to_string());
            cli::commands::stats(&repo_path)
        }
        Commands::Model {
            model,
            pooling,
            query_prefix,
            document_prefix,
        } => cli::commands::model(
            model.as_deref(),
            cli::ModelFlags {
                pooling: pooling.map(Into::into),
                query_prefix,
                document_prefix,
            },
        ),
        Commands::Completions { shell } => {
            print_completions(shell);
            Ok(())
        }
    };

    if let Err(err) = result {
        print_error(&err);
        process::exit(1);
    }
}

/// Write a completion script for `shell` to stdout.
///
/// Generated from the same [`Cli`] definition the parser uses, so the flags a
/// shell offers cannot drift from the flags that exist. Named for the binary
/// (`git-sem`) rather than the git alias — `git sem …` completes through
/// git's own subcommand machinery.
fn print_completions(shell: Shell) {
    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();
    clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
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
