# git-semantic

**Semantic search for git history using natural language**

Find commits by what they mean, not just what they say.

## The Problem

Traditional git search requires exact keywords:
```bash
git log --grep="race"           # 847 results, none relevant
git log -S "mutex"              # 12 results, but not the one you need
```

You spend 20 minutes scrolling through `git log`, trying different keywords.

## The Solution

```bash
git-semantic search "fixed race condition in authentication"
```

Results in < 100ms:
```
🎯 Most Relevant Commits:

1. abc1234 - Resolve concurrent login session handling (0.89 similarity)
   Author: Alice Chen, 6 months ago
   
2. def5678 - Synchronize user token refresh logic (0.84 similarity)
   Author: Bob Martinez, 4 months ago
```

You found it in 3 seconds instead of 20 minutes.

## Features

- 🔍 **Semantic Search** - Find commits by meaning, not just keywords
- 🚀 **Fast** - Search returns in < 100ms
- 🔒 **Private** - Everything runs locally, no data leaves your machine
- 📦 **Zero Config** - Works out of the box, no API keys needed
- 🎯 **Smart Filtering** - Filter by author, date, file, and more

## Installation

### From Source (Current)

```bash
git clone https://github.com/yanxue06/git-semantic-search
cd git-semantic-search
cargo build --release
cargo install --path .
```

### Package Managers (Coming Soon)

```bash
# Cargo
cargo install git-semantic

# Homebrew (macOS)
brew install git-semantic
```

## Quick Start

### 1. Initialize (one-time setup)

```bash
git-semantic init
```

This downloads the BGE-small-en-v1.5 ONNX model (~130MB) - only needed once.

### 2. Index your repository

```bash
cd /path/to/your/repo
git-semantic index
```

### 3. Search!

```bash
git-semantic search "fix memory leak"
git-semantic search "refactor authentication"
git-semantic search "add new feature"
```

## Usage

### Basic Search

```bash
# Search for commits
git-semantic search "your natural language query"

# Limit results
git-semantic search "bug fix" -n 5
```

### Advanced Filtering

```bash
# Filter by author
git-semantic search "refactor" --author=alice

# Filter by date
git-semantic search "bug" --after=2024-01-01 --before=2024-12-31

# Filter by file
git-semantic search "optimization" --file=src/auth.rs
```

### Index Management

```bash
# Quick index (messages only - faster)
git-semantic index --quick

# Full index (messages + diffs - more thorough)
git-semantic index --full

# Update index with new commits
git-semantic update

# Show index statistics
git-semantic stats
```

## How It Works

1. **Indexing**: Parses your git history and generates semantic embeddings for each commit using the BGE-small-en-v1.5 ONNX model
2. **Storage**: Stores embeddings in `.git/semantic-index` using efficient binary serialization (automatically ignored by git)
3. **Search**: Converts your query to an embedding and finds the most similar commits using cosine similarity
4. **ONNX Runtime**: Uses ONNX Runtime for fast, local AI inference without external dependencies

## Development Status

**✅ MVP Complete - Fully Functional!**

- [x] Project structure and dependencies
- [x] CLI interface with clap
- [x] Git history parser implementation
- [x] ONNX embedding model integration (BGE-small-en-v1.5)
- [x] Vector storage and search with binary serialization
- [x] Progress bars and user feedback
- [x] HuggingFace model download with progress tracking
- [x] Tokenization and L2 normalization
- [x] Semantic search with similarity scoring

**Coming Soon:**
- Phase 2: Performance optimizations (parallel processing, compression)
- Phase 3: Interactive TUI, advanced features
- Phase 4: Distribution and polish

## Requirements

- Git repository
- ~130MB disk space for the model
- Rust 1.70+ (for building from source)

## Technical Implementation

### Core Technologies
- **ONNX Runtime**: Fast, local AI inference with the `ort` crate
- **BGE Embeddings**: BAAI's BGE-small-en-v1.5 model for high-quality semantic embeddings
- **Tokenization**: BERT-style tokenization with `tokenizers` crate
- **Binary Storage**: Efficient `bincode` serialization for index storage
- **Git Integration**: Uses `git2` crate for robust git history parsing

### Model Details
- **Model**: BGE-small-en-v1.5 (384-dimensional embeddings)
- **Size**: ~130MB ONNX model + tokenizer
- **Performance**: < 100ms inference time per query
- **Storage**: ~0.04MB per 7 commits (highly efficient)

## Project Structure

```
git-semantic/
├── src/
│   ├── main.rs           # CLI entry point
│   ├── git/              # Git parsing and history extraction
│   ├── embedding/        # ONNX model integration and tokenization
│   ├── index/            # Binary storage and indexing
│   ├── search/           # Semantic search engine
│   └── cli/              # Command handlers and user interface
├── .git/semantic-index/  # Generated index storage (auto-ignored)
└── tests/                # Integration tests
```

## Contributing

Contributions are welcome! This is an early-stage project with lots of room for improvements.

## License

MIT

## Roadmap

- [x] Phase 1: MVP (Core functionality) ✅
- [ ] Phase 2: Performance & usability
- [ ] Phase 3: Advanced features (TUI, clustering, exports)
- [ ] Phase 4: Polish & distribution

## Real Example Output

```bash
$ git-semantic search "ONNX integration"

🎯 Most Relevant Commits for: "ONNX integration"

1. 28e9c31 - Implement ONNX model inference and HuggingFace download (0.69 similarity)
   Author: yan, 2025-10-13 06:50:59 UTC
   +use indicatif::{ProgressBar, ProgressStyle};
   +use ndarray::Array1;

2. 079773d - Initialize model in IndexBuilder and SearchEngine (0.59 similarity)
   Author: yan, 2025-10-13 06:51:05 UTC
   -    pub fn new(model_manager: ModelManager) -> Result<Self> {
   +    pub fn new(mut model_manager: ModelManager) -> Result<Self> {
```

---

**Status**: ✅ **Fully functional MVP ready for use!**

