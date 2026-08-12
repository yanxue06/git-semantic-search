# git-semantic

[![Release](https://img.shields.io/github/v/release/yanxue06/git-semantic-search?style=flat-square)](https://github.com/yanxue06/git-semantic-search/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/yanxue06/git-semantic-search/total?style=flat-square&label=binary%20downloads)](https://github.com/yanxue06/git-semantic-search/releases)
[![Crates.io Downloads](https://img.shields.io/crates/d/git-semantic?style=flat-square&label=cargo%20installs)](https://crates.io/crates/git-semantic)
[![Crates.io](https://img.shields.io/crates/v/git-semantic?style=flat-square)](https://crates.io/crates/git-semantic)
[![License](https://img.shields.io/github/license/yanxue06/git-semantic-search?style=flat-square)](https://github.com/yanxue06/git-semantic-search/blob/main/LICENSE)

**Search your git history using natural language - find commits by what they mean, not just what they say.**

Both of these run against a fresh clone of [ripgrep](https://github.com/BurntSushi/ripgrep), 2,287 commits:

```console
$ git log --grep="large files"
139f186 crates/ignore: switch to depth first traversal
714ae82 Add `--max-filesize` option to cli
d06f84c Get rid of special mmap decision on Windows.

$ git-semantic search "searching very large files without loading them into memory"
🎯 Most Relevant Commits for: "searching very large files without loading them into memory"

1. ca058d7 - Add support for memory maps. (0.75 similarity)
   Author: Andrew Gallant, 2016-09-07 01:47:33 UTC

2. 139f186 - crates/ignore: switch to depth first traversal (0.70 similarity)
   Author: Andrew Gallant, 2020-04-18 15:33:03 UTC

3. 6b2efd4 - If a file is empty, still try to search it. (0.72 similarity)
   Author: Andrew Gallant, 2016-09-25 00:45:06 UTC

Searched 2287 commits via hybrid graph search in 3ms
```

The commit that answers the question is called *Add support for memory maps*. `--grep` never surfaces it — you can only grep for "memory maps" once you already know that's the answer.

Example:

https://github.com/user-attachments/assets/91d33745-24ac-47ef-8a82-7ad6510eb17d


## Why?

Traditional git search is **keyword-based**. You need to guess the exact words the author used:

```bash
git log --grep="race"     # 847 results 😵
git log -S "mutex"        # Maybe? 🤷
```

**git-semantic** understands **meaning**. Search for "race condition" and find commits about "concurrent access" or "synchronization bugs" - even if those exact words aren't in the message.

## Features

- 🔍 **Natural language search** - "fix memory leak" finds more than just those exact words
- 🎯 **Hybrid retrieval** - meaning *and* exact tokens: `CVE-2024-1234`, `src/auth.rs`, a commit hash
- 🧩 **Diverse results** - `--diverse` stops ten near-identical dependency bumps from filling the page
- 🤖 **Scriptable** - `--json` for piping into `jq`, a script, or an LLM
- 🚀 **Fast** - Millisecond searches that stay flat as history grows (HNSW graph index)
- 🔒 **Private** - Everything runs locally with ONNX, no API keys or cloud services
- 📦 **Zero config** - One command. The model downloads and the index builds on first search
- 🎯 **Smart filtering** - By author, date, file, and more
- 🐚 **Shell completions** - bash, zsh, fish, elvish, powershell

## Installation

### Using Cargo (Recommended)

```bash
cargo install git-semantic
```

Alternatively, you can also install from the latest release compatible with your OS on the [releases page](https://github.com/yanxue06/git-semantic-search/releases). 

Because the binary is named `git-semantic`, git picks it up as a subcommand for free:

```bash
git semantic search "the commit that broke the build"
```

### Shell completions

```bash
git-semantic completions zsh  > ~/.zfunc/_git-semantic
git-semantic completions bash > /etc/bash_completion.d/git-semantic
git-semantic completions fish > ~/.config/fish/completions/git-semantic.fish
```

## Quick Start

```bash
cd /path/to/your/repo
git-semantic search "your query here"
```

That's it. The first search in a repository downloads the embedding model
(~130MB, once per machine) and indexes the history, saying so as it goes.
Every search after that answers in about 100ms.

To do that work ahead of time instead — on a new machine, or before a demo:

```bash
git-semantic init     # download the model
git-semantic index    # build the index (also picks up new commits incrementally)
```

Any directory inside the repository works, not just the root. Worktrees get
their own index; submodules index into their own git dir.

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

# By date — either bound, or both, inclusive
git-semantic search "bug fix" --after=2024-01-01
git-semantic search "bug fix" --after=2024-01-01 --before=2024-06-30

# By file — matches the commit's changed paths
git-semantic search "optimization" --file=src/auth.rs
git-semantic search "dependency bump" --file=Cargo.toml
git-semantic search "refactor" --file=src/index/      # prefix works too

# Manually decide number of matches with the -n flag 
git-semantic search "feature" -n 5
```

### Semantic, keyword, or both

Embeddings are good at meaning and bad at exact strings — a 384-dimensional
vector cannot reliably tell `CVE-2024-1234` from `CVE-2024-5678`. So search
runs **both** an embedding search and a BM25 keyword search, then fuses the two
rankings. That is the default; you can pin either side:

```bash
git-semantic search "race condition"      # hybrid (default)
git-semantic search "CVE-2024-1234"       # hybrid — BM25 nails the exact token
git-semantic search "auth" --mode semantic  # embeddings only (pre-1.5 behaviour)
git-semantic search "Cargo.toml" --mode lexical  # keywords only
```

Fusion uses [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
rather than a weighted score blend. Cosine similarity sits in a narrow band
while BM25 is unbounded and corpus-dependent, so any `α` tuned on one repository
is wrong on the next. RRF discards the magnitudes and keeps only the ranks —
nothing to calibrate, nothing to re-tune as the repo grows.

### Diverse results

Relevance ranking has no opinion about redundancy. Ask a busy repo for
"dependency update" and the top ten are ten renovate commits that differ only in
a crate name — technically the ten best answers, practically one answer repeated
ten times.

```bash
git-semantic search "dependency update" --diverse
git-semantic search "refactor" --diverse --lambda 0.5   # push harder for novelty
```

`--diverse` reranks with [Maximal Marginal Relevance](https://dl.acm.org/doi/10.1145/290941.291025),
picking each result on relevance *minus* similarity to what is already shown.
`--lambda` balances the two: `1.0` is pure relevance, `0.0` pure novelty,
default `0.7`. The top result never moves.

### Scripting

```bash
git-semantic search "race condition" --json | jq -r '.results[].hash'
git-semantic search "auth" --json | jq '.results[] | {subject, files}'
```

Actual output, again from ripgrep's history:

```json
{
  "query": "memory maps",
  "mode": "hybrid",
  "strategy": "approximate",
  "candidates": 2287,
  "diversified": false,
  "took_ms": 1.802042,
  "results": [
    {
      "rank": 1,
      "hash": "5a9883d27c018256c45e450764cf711fe53ce0f3",
      "author": "Andrew Gallant",
      "date": "2016-09-22T00:47:40+00:00",
      "subject": "Try to use memory maps more aggressively on Windows.",
      "message": "Try to use memory maps more aggressively on Windows.\n\nSome brief playing around suggests that it is faster.",
      "similarity": 0.76922643
    }
  ]
}
```

Full 40-character hashes, RFC 3339 dates, and `similarity` omitted entirely on
keyword-only hits — JSON cannot represent NaN, so the field is absent rather
than null. Progress and diagnostics go to stderr, so stdout is always a single
parseable document, even on the run that builds the index.

### Tuning search

Repositories above 2,048 commits are searched through an approximate
nearest-neighbor graph. Two escape hatches let you trade speed for accuracy:

```bash
# Score every commit — exact, and the baseline the graph is measured against
git-semantic search "race condition" --exact

# Widen the graph's candidate list: slower, higher recall (default 64)
git-semantic search "race condition" --ef 256
```

Every search prints how it ran, so the tradeoff is never invisible:

```
Searched 48213 commits via graph search in 1ms
```

### Index Management

```bash
# Build, or pick up new commits incrementally
git-semantic index

# Quick index (messages only, ~5x faster to build)
git-semantic index --quick

# Full index (messages + diffs, more context) — the default
git-semantic index --full

# Rebuild from scratch, e.g. to switch modes
git-semantic index --force

# What's indexed, how big, and which search strategy it will use
git-semantic stats
```

## How It Works

1. **Downloads BGE-small-en-v1.5** - A compact AI model (130MB) for semantic embeddings
2. **Indexes your repo** - Converts each commit into a 384-dimensional vector
3. **Stores locally** - Binary index saved in `.git/semantic-index` (ignored by git)
4. **Builds a proximity graph** - An HNSW index over those vectors, cached in `.git/semantic-index.hnsw`
5. **Searches by meaning** - Your query becomes a vector; the graph finds its nearest commit vectors without touching most of the repository
6. **ONNX Runtime** - Fast local inference, no cloud services needed

**Stored locations:**
- Model: `~/Library/Application Support/com.git-semantic.git-semantic/models/` (macOS)
- Index: `.git/semantic-index` (per repository, inside the git dir, so it is never committed)
- Search graph: `.git/semantic-index.hnsw` (rebuilt automatically when stale)
- Keyword index: `.git/semantic-index.bm25` (same)

## Technical Details

- **Model**: BGE-small-en-v1.5 (BAAI)
- **Runtime**: ONNX Runtime for fast local inference
- **Storage**: Bincode serialization (~3KB per Commit)
- **Similarity**: Cosine, computed as a dot product over L2-normalized vectors
- **Retrieval**: hybrid — HNSW vector search fused with BM25 via Reciprocal Rank Fusion
- **Search**: [HNSW](https://arxiv.org/abs/1603.09320) graph traversal above 2,048 commits; exhaustive scan below, where it is genuinely faster
- **Tokenizer**: code-aware — splits paths, `snake_case`, `camelCase`, and letter/digit boundaries, keeping the whole identifier too
- **Diversification**: optional MMR rerank over the retrieved pool
- **Recall**: ≥0.95 recall@10 against exhaustive search, asserted in CI

### Search performance

Top-10 query latency over 384-dimensional embeddings, Apple silicon, `cargo bench`:

| Commits | Exhaustive scan | HNSW graph | Speedup |
|--------:|----------------:|-----------:|--------:|
|   1,000 |          24 µs  |     34 µs  |  0.7x   |
|  10,000 |         235 µs  |     72 µs  |  3.3x   |
|  50,000 |        1180 µs  |     76 µs  | 15.5x   |

Exhaustive scan wins below ~2k commits, which is exactly where the graph is
skipped. Graph latency is near-flat in repository size; the scan is linear.

Building the graph costs ~0.5s per 1k commits, once, and is cached — a rounding
error next to embedding the same commits through ONNX.

### End to end

The table above times retrieval in isolation. What a query actually costs,
measured on a fresh clone of [ripgrep](https://github.com/BurntSushi/ripgrep) at
2,287 commits on an Apple M5 Pro:

| | `--quick` | `--full` |
|---|---|---|
| One-time index build | 16 s | 83 s |
| Index on disk | 4.1 MB | 7.8 MB |
| Graph build | 0.4 s | 0.4 s |

A search is **100 ms wall clock**. The few milliseconds the CLI reports are
embedding the query plus retrieval; the rest is process start and loading the
ONNX model. The microbenchmark above isolates retrieval alone, which is why its
numbers are two orders of magnitude smaller.

## Full-mode output

A full index embeds diffs as well as messages, and results carry a couple of
lines of diff context with them:

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

- A git repository — any directory inside it, including worktrees and submodules
- ~130MB disk space for the AI model
- Rust 1.88+ (if building from source) — declared as `rust-version`, so cargo checks it for you

## License

MIT

---

**Built with:** Rust 🦀 and ❤️
