# git-semantic

[![Release](https://img.shields.io/github/v/release/yanxue06/git-semantic-search?style=flat-square)](https://github.com/yanxue06/git-semantic-search/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/yanxue06/git-semantic-search/total?style=flat-square&label=binary%20downloads)](https://github.com/yanxue06/git-semantic-search/releases)
[![Crates.io Downloads](https://img.shields.io/crates/d/git-semantic?style=flat-square&label=cargo%20installs)](https://crates.io/crates/git-semantic)
[![Crates.io](https://img.shields.io/crates/v/git-semantic?style=flat-square)](https://crates.io/crates/git-semantic)
[![License](https://img.shields.io/github/license/yanxue06/git-semantic-search?style=flat-square)](LICENSE)

**Search your git history using natural language - find commits by what they mean, not just what they say.**

```bash
$ git-semantic search "fixed race condition in authentication"

🎯 Most Relevant Commits:

1. abc1234 - Resolved concurrent login session handling (0.89 similarity)
   Author: Alice Chen, 6 months ago
   
2. def5678 - Synchronized user token refresh logic (0.84 similarity)
   Author: Bob Martinez, 4 months ago
```

Stop scrolling through hundreds of commits with `git log --grep`. Just describe what you're looking for in plain English.

## Why?

Traditional git search is **keyword-based**. You need to guess the exact words the author used:

```bash
git log --grep="race"     # 847 results 😵
git log -S "mutex"        # Maybe? 🤷
```

**git-semantic** understands **meaning**. Search for "race condition" and find commits about "concurrent access" or "synchronization bugs" - even if those exact words aren't in the message.

## Features

- 🔍 **Natural language search** - "fix memory leak" finds more than just those exact words
- 🚀 **Fast** - Results in < 100ms
- 🔒 **Private** - Everything runs locally with ONNX, no API keys or cloud services
- 📦 **Zero config** - Works out of the box
- 🎯 **Smart filtering** - By author, date, file, and more

## Installation

### Using Cargo (Recommended)

```bash
cargo install git-semantic
```

Alternatively, you can also install from the latest release compatible with your OS on the [releases page](https://github.com/yanxue06/git-semantic-search/releases). 

## Quick Start

```bash
# 1. One-time setup (downloads AI model, ~130MB)
git-semantic init

# 2. Index your repository
cd /path/to/your/repo
git-semantic index

# 3. Search!
git-semantic search "your query here"
```

## Usage

### Basic Search

```bash
git-semantic search "fix memory leak"
git-semantic search "add authentication feature"
git-semantic search "refactor payment logic"
```

### Filters

```bash
# By author
git-semantic search "refactor" --author=alice

# By date
git-semantic search "bug fix" --after=2024-01-01

# By file
git-semantic search "optimization" --file=src/auth.rs

# Manually decide number of matches with the -n flag 
git-semantic search "feature" -n 5
```

### Index Management

```bash
# Update index with new commits
git-semantic update

# Show index statistics
git-semantic stats

# Quick index (messages only, faster)
git-semantic index --quick

# Full index (messages + diffs, more context)
git-semantic index --full
```

## How It Works

1. **Downloads BGE-small-en-v1.5** - A compact AI model (130MB) for semantic embeddings
2. **Indexes your repo** - Converts each commit into a 384-dimensional vector
3. **Stores locally** - Binary index saved in `.git/semantic-index` (ignored by git)
4. **Searches by meaning** - Your query becomes a vector, finds similar commit vectors using cosine similarity
5. **ONNX Runtime** - Fast local inference, no cloud services needed

**Stored locations:**
- Model: `~/Library/Application Support/com.git-semantic.git-semantic/models/` (macOS)
- Index: `.git/semantic-index` (per repository)

## Technical Details

- **Model**: BGE-small-en-v1.5 (BAAI)
- **Runtime**: ONNX Runtime for fast local inference
- **Storage**: Bincode serialization (~3KB per Commit)
- **Search**: Cosine similarity with L2 normalization
- **Inference**: < 100ms per query

## Real Example

```bash
$ git-semantic search "ONNX integration"

🎯 Most Relevant Commits for: "ONNX integration"

1. 4d8acb9 - docs: Update README with complete ONNX integration details (0.73 similarity)
   Author: yan, 2025-10-13 08:17:23 UTC
   -# git-semantic (IN DEVELOPMENT)
   +# git-semantic

2. 776ff32 - feat: Complete ONNX integration with real BGE embeddings (0.73 similarity)
   Author: yan, 2025-10-13 07:24:37 UTC
   -    let engine = SearchEngine::new(model_manager)?;
   +    let mut engine = SearchEngine::new(model_manager)?;

3. 28e9c31 - Implement ONNX model inference and HuggingFace download (0.69 similarity)
   Author: yan, 2025-10-13 06:50:59 UTC
   +use indicatif::{ProgressBar, ProgressStyle};
   +use ndarray::Array1;
```

## Contributing

Contributions welcome! Please use [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
feat: add new search feature
fix: resolve memory leak in indexing
docs: update installation instructions
```

## Requirements

- Git repository (obviously!)
- ~130MB disk space for the AI model
- Rust 1.70+ (if building from source)

## License

MIT

---

**Built with:** Rust 🦀 and ❤️
