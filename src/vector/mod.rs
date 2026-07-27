//! Vector primitives: similarity kernels and the approximate nearest-neighbor
//! index they feed.
//!
//! Deliberately independent of the rest of the crate — nothing here knows about
//! git, commits, or the CLI — so it stays unit-testable against synthetic
//! corpora and reusable by both the index builder and the search engine.

pub mod hnsw;
pub mod scoring;

pub use hnsw::{HnswIndex, HnswParams};
pub use scoring::{Scored, TopK, cosine, dot, normalize};
