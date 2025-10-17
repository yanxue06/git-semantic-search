# git-semantic

**Search your git history using natural language - find commits by what they mean, not just what they say.**

```bash
$ git-semantic search "fixed race condition in authentication"

🎯 Most Relevant Commits:

1. abc1234 - Resolve concurrent login session handling (0.89 similarity)
   Author: Alice Chen, 6 months ago
   
2. def5678 - Synchronize user token refresh logic (0.84 similarity)
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

### From Source (Recommended)

```bash
cargo install --git https://github.com/yanxue06/git-semantic-search
```

### From Release 

**macOS (Intel):**
```bash
curl -L -o git-semantic https://github.com/yanxue06/git-semantic-search/releases/latest/download/git-semantic-macos-x86_64
chmod +x git-semantic
sudo mv git-semantic /usr/local/bin/
```

**macOS (Apple Silicon):**
```bash
curl -L -o git-semantic https://github.com/yanxue06/git-semantic-search/releases/latest/download/git-semantic-macos-arm64
chmod +x git-semantic
sudo mv git-semantic /usr/local/bin/
```

**Linux:**
```bash
curl -L -o git-semantic https://github.com/yanxue06/git-semantic-search/releases/latest/download/git-semantic-linux-x86_64
chmod +x git-semantic
sudo mv git-semantic /usr/local/bin/
```

**Windows (PowerShell):**
```powershell
curl -L -o git-semantic.exe https://github.com/yanxue06/git-semantic-search/releases/latest/download/git-semantic-windows-x86_64.exe
# Move to a directory in your PATH, e.g., C:\Program Files\
```

If these don't work, you can also click into the [releases page](https://github.com/yanxue06/git-semantic-search/releases) and download the binaries manually. 

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

# Limit results
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
- **Storage**: Bincode serialization (~0.04MB per 7 commits)
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

Releases are automated via semantic-release. Push to `main` and let CI handle versioning and binary builds.

## Requirements

- Git repository (obviously!)
- ~130MB disk space for the AI model
- Rust 1.70+ (if building from source)

## License

MIT

---

**Built with:** Rust 🦀 and ❤️
