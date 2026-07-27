//! Lexical retrieval primitives: tokenization and BM25.
//!
//! A leaf module in the same spirit as [`crate::vector`] — no knowledge of git,
//! commits, or the CLI — so it can be tested against plain strings.

pub mod bm25;
pub mod tokenize;

pub use bm25::{Bm25Index, Bm25Params};
pub use tokenize::tokenize;
