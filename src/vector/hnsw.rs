//! Hierarchical Navigable Small World graph for sublinear nearest-neighbor search.
//!
//! Implements Malkov & Yashunin, *Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs* (2016).
//!
//! # Why a graph
//!
//! Exact search compares the query against every commit: O(N·D) dot products,
//! where D is 384. A repository with 50k commits burns ~19M multiply-adds per
//! query. HNSW instead walks a layered proximity graph, touching
//! `O(ef · log N)` nodes — a few thousand comparisons regardless of N.
//!
//! # Layout
//!
//! Layer 0 holds every node. Each node is promoted to layer `l` with
//! probability `1/M^l`, so upper layers are exponentially sparser and act as
//! express lanes. A search greedily descends the express lanes to land near the
//! query, then does a wider best-first expansion on layer 0.
//!
//! # Determinism
//!
//! Level assignment is the only randomized part of construction, and it draws
//! from a seeded SplitMix64 stored in the struct. Same vectors plus same seed
//! produce an identical graph, which is what makes recall regressions
//! reproducible in CI.

use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

use super::scoring::{Scored, dot, normalize};

/// Hard ceiling on layer count. `1/M^l` makes anything past this
/// astronomically unlikely; the cap just bounds worst-case memory.
const MAX_LEVEL: usize = 16;

/// Construction and query tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswParams {
    /// Target out-degree on layers 1 and above. Layer 0 gets `2 * m`.
    pub m: usize,
    /// Candidate-list width while inserting. Higher builds slower, recalls better.
    pub ef_construction: usize,
    /// Candidate-list width at query time. Higher is slower and more accurate.
    pub ef_search: usize,
    /// Seed for level assignment.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        // M=16 is the paper's recommended starting point for embeddings of this
        // dimensionality. efC=128 measured the same recall@10 as 200 on
        // bge-small-shaped vectors while building ~35% faster.
        Self {
            m: 16,
            ef_construction: 128,
            ef_search: 64,
            seed: 0x5EED_1234_ABCD_9876,
        }
    }
}

impl HnswParams {
    /// Out-degree cap for a given layer.
    fn max_degree(&self, level: usize) -> usize {
        if level == 0 { self.m * 2 } else { self.m }
    }

    /// Level-generation normalization factor, `1 / ln(M)`.
    fn level_factor(&self) -> f64 {
        1.0 / (self.m.max(2) as f64).ln()
    }
}

/// One node's adjacency lists, indexed by layer with layer 0 first.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    /// `links[l]` are the layer-`l` neighbors. Length is the node's level + 1.
    links: Vec<Vec<u32>>,
}

impl Node {
    fn new(level: usize) -> Self {
        Self {
            links: vec![Vec::new(); level + 1],
        }
    }

    fn level(&self) -> usize {
        self.links.len() - 1
    }
}

/// A navigable small-world graph over unit-length vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    params: HnswParams,
    dim: usize,
    /// Row-major `len * dim`, L2-normalized so distance is `1 - dot`.
    vectors: Vec<f32>,
    nodes: Vec<Node>,
    /// Node on the topmost layer; entry point for every descent.
    entry: Option<u32>,
    rng: u64,
}

impl HnswIndex {
    /// Empty graph expecting `dim`-dimensional vectors.
    pub fn new(dim: usize, params: HnswParams) -> Self {
        Self {
            rng: params.seed,
            params,
            dim,
            vectors: Vec::new(),
            nodes: Vec::new(),
            entry: None,
        }
    }

    /// Build a graph over `vectors`, assigning ids by insertion order.
    ///
    /// Vectors shorter than `dim` are zero-padded and longer ones truncated, so
    /// a corrupt row degrades one result instead of failing the whole build.
    pub fn build<'a, I>(dim: usize, params: HnswParams, vectors: I) -> Self
    where
        I: IntoIterator<Item = &'a [f32]>,
    {
        let mut graph = Self::new(dim, params);
        // One scratch set reused across every insert. Allocating and zeroing a
        // per-insert set would cost O(N²) writes over a build — gigabytes of
        // memset at 50k commits.
        let mut visited = VisitedSet::new(0);
        for v in vectors {
            graph.insert_with(v, &mut visited);
        }
        graph
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn params(&self) -> HnswParams {
        self.params
    }

    /// Add one vector and return its id.
    pub fn insert(&mut self, vector: &[f32]) -> u32 {
        let mut visited = VisitedSet::new(self.nodes.len() + 1);
        self.insert_with(vector, &mut visited)
    }

    /// Insert reusing a caller-owned visited set (paper Algorithm 1).
    fn insert_with(&mut self, vector: &[f32], visited: &mut VisitedSet) -> u32 {
        let id = self.nodes.len() as u32;

        let mut owned = vec![0.0f32; self.dim];
        let copy_len = vector.len().min(self.dim);
        owned[..copy_len].copy_from_slice(&vector[..copy_len]);
        normalize(&mut owned);
        self.vectors.extend_from_slice(&owned);

        let level = self.random_level();
        self.nodes.push(Node::new(level));

        let Some(entry) = self.entry else {
            self.entry = Some(id);
            return id;
        };

        let entry_level = self.nodes[entry as usize].level();

        // Express-lane descent: greedy single-best hops down to the highest
        // layer the new node will actually join.
        let mut cursor = entry;
        for level_above in ((level + 1)..=entry_level).rev() {
            cursor = self.greedy_descend(&owned, cursor, level_above);
        }

        let mut entry_points = vec![cursor];

        for current in (0..=level.min(entry_level)).rev() {
            let mut candidates = self.search_layer(
                &owned,
                &entry_points,
                self.params.ef_construction,
                current,
                visited,
                usize::MAX,
                |_| true,
            );
            candidates.sort_unstable();
            // Never link a node to itself; it is already in `nodes` by now.
            candidates.retain(|c| c.id != id);

            let degree = self.params.max_degree(current);
            let selected = self.select_neighbors(&candidates, degree);

            self.nodes[id as usize].links[current] = selected.clone();

            for neighbor in selected {
                self.link_back(neighbor, current, id, degree);
            }

            entry_points = candidates.iter().map(|c| c.id).collect();
            if entry_points.is_empty() {
                entry_points.push(cursor);
            }
        }

        if level > entry_level {
            self.entry = Some(id);
        }

        id
    }

    /// `k` nearest neighbors of `query`, best first, as `(id, similarity)`.
    pub fn search(&self, query: &[f32], k: usize, ef: Option<usize>) -> Vec<(u32, f32)> {
        self.search_filtered(query, k, ef, |_| true)
    }

    /// `k` nearest neighbors restricted to ids where `allow` returns true.
    ///
    /// The predicate gates *admission to the result set*, not graph traversal —
    /// hopping through non-matching nodes is what keeps the graph connected. If
    /// a selective filter starves the result set, callers should widen `ef` or
    /// fall back to an exact scan over the matching subset; the engine decides
    /// that from measured selectivity.
    pub fn search_filtered<P>(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        allow: P,
    ) -> Vec<(u32, f32)>
    where
        P: Fn(u32) -> bool,
    {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut normalized = vec![0.0f32; self.dim];
        let copy_len = query.len().min(self.dim);
        normalized[..copy_len].copy_from_slice(&query[..copy_len]);
        normalize(&mut normalized);

        let ef = ef.unwrap_or(self.params.ef_search).max(k);
        let entry = self.entry.unwrap_or(0);
        let entry_level = self.nodes[entry as usize].level();

        let mut cursor = entry;
        for level in (1..=entry_level).rev() {
            cursor = self.greedy_descend(&normalized, cursor, level);
        }

        // Cap total node expansions so a pathological filter cannot turn one
        // query into a full graph walk. Generous enough to be unreachable for
        // ordinary filters.
        let budget = (ef * self.params.m * 8).max(1024);
        let mut visited = VisitedSet::new(self.nodes.len());
        let mut found =
            self.search_layer(&normalized, &[cursor], ef, 0, &mut visited, budget, &allow);

        found.sort_unstable();
        found.truncate(k);
        found.into_iter().map(|s| (s.id, 1.0 - s.dist)).collect()
    }

    /// Cosine similarity between `query` and the stored vector for `id`.
    pub fn similarity(&self, id: u32, query: &[f32]) -> f32 {
        dot(self.vector(id), query)
    }

    fn vector(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    /// Cosine distance in `[0, 2]`. Both operands are unit length, so the dot
    /// product is already the cosine and no norms are recomputed.
    fn distance(&self, id: u32, query: &[f32]) -> f32 {
        1.0 - dot(self.vector(id), query)
    }

    fn distance_between(&self, a: u32, b: u32) -> f32 {
        1.0 - dot(self.vector(a), self.vector(b))
    }

    /// Walk one layer taking the single best hop until no neighbor improves.
    fn greedy_descend(&self, query: &[f32], start: u32, level: usize) -> u32 {
        let mut current = start;
        let mut current_dist = self.distance(current, query);

        loop {
            let mut improved = false;

            for &neighbor in self.links(current, level) {
                let dist = self.distance(neighbor, query);
                if dist < current_dist {
                    current_dist = dist;
                    current = neighbor;
                    improved = true;
                }
            }

            if !improved {
                return current;
            }
        }
    }

    /// Best-first expansion on one layer (paper Algorithm 2).
    ///
    /// Returns up to `ef` admitted results, unordered. `budget` bounds how many
    /// nodes may be expanded; `allow` gates admission only.
    #[allow(clippy::too_many_arguments)]
    fn search_layer<P>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: usize,
        visited: &mut VisitedSet,
        budget: usize,
        allow: P,
    ) -> Vec<Scored>
    where
        P: Fn(u32) -> bool,
    {
        visited.begin();

        // Frontier ordered nearest-first; `Reverse` turns the max-heap into a
        // min-heap without a second comparator.
        let mut frontier: BinaryHeap<std::cmp::Reverse<Scored>> = BinaryHeap::new();
        // Results ordered farthest-first so the weakest entry is at the top.
        let mut results: BinaryHeap<Scored> = BinaryHeap::new();

        for &ep in entry_points {
            if ep as usize >= self.nodes.len() || !visited.insert(ep) {
                continue;
            }
            let scored = Scored::new(self.distance(ep, query), ep);
            frontier.push(std::cmp::Reverse(scored));
            if allow(ep) {
                results.push(scored);
            }
        }

        let mut expansions = 0usize;

        while let Some(std::cmp::Reverse(nearest)) = frontier.pop() {
            // Stop once the frontier can no longer beat a full result set.
            if results.len() >= ef
                && let Some(worst) = results.peek()
                && nearest.dist > worst.dist
            {
                break;
            }

            expansions += 1;
            if expansions > budget {
                break;
            }

            for &neighbor in self.links(nearest.id, level) {
                if !visited.insert(neighbor) {
                    continue;
                }

                let dist = self.distance(neighbor, query);
                let worst = results.peek().map(|s| s.dist);

                // Expand when the result set has room or this node could
                // improve it. With a filter active `results` fills slowly, so
                // this stays true longer and the walk widens automatically.
                let promising = results.len() < ef || worst.is_none_or(|w| dist < w);
                if !promising {
                    continue;
                }

                frontier.push(std::cmp::Reverse(Scored::new(dist, neighbor)));

                if allow(neighbor) {
                    results.push(Scored::new(dist, neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results.into_vec()
    }

    /// Diversity-aware neighbor pruning (paper Algorithm 4).
    ///
    /// Taking the M closest candidates clusters every link in one direction and
    /// leaves the graph poorly navigable. This keeps a candidate only when it is
    /// closer to the new node than to any neighbor already chosen, spreading
    /// links across directions. Pruned candidates backfill any shortfall so no
    /// node ends up under-connected.
    fn select_neighbors(&self, candidates: &[Scored], max: usize) -> Vec<u32> {
        if candidates.len() <= max {
            return candidates.iter().map(|c| c.id).collect();
        }

        let mut selected: Vec<u32> = Vec::with_capacity(max);
        let mut discarded: Vec<u32> = Vec::new();

        for candidate in candidates {
            if selected.len() >= max {
                break;
            }

            let closer_to_base = selected
                .iter()
                .all(|&chosen| self.distance_between(candidate.id, chosen) > candidate.dist);

            if closer_to_base {
                selected.push(candidate.id);
            } else {
                discarded.push(candidate.id);
            }
        }

        for id in discarded {
            if selected.len() >= max {
                break;
            }
            selected.push(id);
        }

        selected
    }

    /// Add a reverse edge, re-pruning the target if it exceeds its degree cap.
    fn link_back(&mut self, node: u32, level: usize, new_neighbor: u32, max: usize) {
        let links = &mut self.nodes[node as usize].links[level];
        if links.contains(&new_neighbor) {
            return;
        }
        links.push(new_neighbor);

        if links.len() <= max {
            return;
        }

        let mut candidates: Vec<Scored> = self.nodes[node as usize].links[level]
            .iter()
            .map(|&id| Scored::new(self.distance_between(node, id), id))
            .collect();
        candidates.sort_unstable();

        let kept = self.select_neighbors(&candidates, max);
        self.nodes[node as usize].links[level] = kept;
    }

    fn links(&self, node: u32, level: usize) -> &[u32] {
        self.nodes[node as usize]
            .links
            .get(level)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Draw a level from a geometric distribution: `floor(-ln(U) / ln(M))`.
    fn random_level(&mut self) -> usize {
        let uniform = self.next_uniform().max(f64::MIN_POSITIVE);
        let level = (-uniform.ln() * self.params.level_factor()).floor();
        (level.max(0.0) as usize).min(MAX_LEVEL)
    }

    /// SplitMix64 — small, fast, no dependency, and stable across toolchains so
    /// serialized graphs stay reproducible.
    fn next_u64(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_uniform(&mut self) -> f64 {
        // Top 53 bits give a uniform double in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Membership set with generation stamps instead of clearing.
///
/// A query touches a few thousand of N nodes, so zeroing an N-element bitmap
/// per layer would dominate the work. Bumping a generation counter is O(1).
struct VisitedSet {
    stamps: Vec<u32>,
    generation: u32,
}

impl VisitedSet {
    fn new(capacity: usize) -> Self {
        Self {
            stamps: vec![0; capacity],
            generation: 0,
        }
    }

    /// Start a fresh epoch, invalidating every prior mark.
    fn begin(&mut self) {
        self.generation = match self.generation.checked_add(1) {
            Some(next) => next,
            None => {
                // Wrapped after 4 billion layer searches; pay the one reset.
                self.stamps.fill(0);
                1
            }
        };
    }

    /// Mark `id` visited. Returns false if it was already marked this epoch.
    fn insert(&mut self, id: u32) -> bool {
        let slot = id as usize;
        if slot >= self.stamps.len() {
            self.stamps.resize(slot + 1, 0);
        }
        if self.stamps[slot] == self.generation {
            return false;
        }
        self.stamps[slot] = self.generation;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::scoring::dot;

    /// Deterministic pseudo-random unit vectors, no rand dependency.
    fn corpus(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        };

        (0..count)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
                crate::vector::scoring::normalize(&mut v);
                v
            })
            .collect()
    }

    fn brute_force(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (dot(v, query), i as u32))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn build(corpus: &[Vec<f32>], dim: usize) -> HnswIndex {
        HnswIndex::build(dim, HnswParams::default(), corpus.iter().map(Vec::as_slice))
    }

    #[test]
    fn empty_graph_returns_nothing() {
        let graph = HnswIndex::new(4, HnswParams::default());
        assert!(graph.is_empty());
        assert!(graph.search(&[1.0, 0.0, 0.0, 0.0], 5, None).is_empty());
    }

    #[test]
    fn single_node_graph_returns_that_node() {
        let mut graph = HnswIndex::new(3, HnswParams::default());
        graph.insert(&[1.0, 0.0, 0.0]);

        let hits = graph.search(&[0.9, 0.1, 0.0], 5, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn k_larger_than_corpus_returns_everything() {
        let data = corpus(7, 16, 42);
        let graph = build(&data, 16);
        let hits = graph.search(&data[0], 100, None);
        assert_eq!(hits.len(), 7);
    }

    #[test]
    fn indexed_vector_retrieves_itself_first() {
        let data = corpus(500, 64, 7);
        let graph = build(&data, 64);

        for probe in [0usize, 137, 499] {
            let hits = graph.search(&data[probe], 5, None);
            assert_eq!(
                hits[0].0, probe as u32,
                "querying an indexed vector must return it first"
            );
            assert!(
                (hits[0].1 - 1.0).abs() < 1e-4,
                "self-similarity should be ~1.0, got {}",
                hits[0].1
            );
        }
    }

    #[test]
    fn recall_at_10_matches_brute_force() {
        let dim = 128;
        let data = corpus(2_000, dim, 99);
        let graph = build(&data, dim);
        let queries = corpus(50, dim, 12_345);

        let mut hits = 0usize;
        let mut total = 0usize;

        for query in &queries {
            let exact = brute_force(&data, query, 10);
            let approx: Vec<u32> = graph
                .search(query, 10, None)
                .into_iter()
                .map(|h| h.0)
                .collect();

            total += exact.len();
            hits += exact.iter().filter(|id| approx.contains(id)).count();
        }

        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.95,
            "recall@10 regressed to {recall:.3} (want >= 0.95)"
        );
    }

    #[test]
    fn higher_ef_never_lowers_recall() {
        let dim = 64;
        let data = corpus(1_000, dim, 5);
        let graph = build(&data, dim);
        let query = &corpus(1, dim, 777)[0];

        let exact = brute_force(&data, query, 10);
        let narrow: Vec<u32> = graph
            .search(query, 10, Some(10))
            .into_iter()
            .map(|h| h.0)
            .collect();
        let wide: Vec<u32> = graph
            .search(query, 10, Some(200))
            .into_iter()
            .map(|h| h.0)
            .collect();

        let narrow_hits = exact.iter().filter(|id| narrow.contains(id)).count();
        let wide_hits = exact.iter().filter(|id| wide.contains(id)).count();
        assert!(
            wide_hits >= narrow_hits,
            "ef=200 recalled {wide_hits} vs ef=10 recalling {narrow_hits}"
        );
    }

    #[test]
    fn results_are_sorted_by_descending_similarity() {
        let data = corpus(300, 32, 3);
        let graph = build(&data, 32);
        let hits = graph.search(&corpus(1, 32, 8)[0], 20, None);

        for pair in hits.windows(2) {
            assert!(
                pair[0].1 >= pair[1].1,
                "similarities out of order: {} then {}",
                pair[0].1,
                pair[1].1
            );
        }
    }

    #[test]
    fn filter_admits_only_matching_ids() {
        let data = corpus(800, 32, 21);
        let graph = build(&data, 32);
        let query = &corpus(1, 32, 4)[0];

        let hits = graph.search_filtered(query, 10, Some(128), |id| id % 2 == 0);
        assert!(!hits.is_empty());
        assert!(
            hits.iter().all(|(id, _)| id % 2 == 0),
            "filter leaked non-matching ids"
        );
    }

    #[test]
    fn filter_still_fills_k_when_half_the_corpus_matches() {
        let data = corpus(800, 32, 21);
        let graph = build(&data, 32);
        let query = &corpus(1, 32, 4)[0];

        let hits = graph.search_filtered(query, 10, Some(128), |id| id % 2 == 0);
        assert_eq!(
            hits.len(),
            10,
            "a 50%-selective filter should still fill k results"
        );
    }

    #[test]
    fn filter_matching_nothing_returns_empty() {
        let data = corpus(200, 32, 21);
        let graph = build(&data, 32);
        let hits = graph.search_filtered(&data[0], 10, None, |_| false);
        assert!(hits.is_empty());
    }

    #[test]
    fn same_seed_builds_identical_graphs() {
        let data = corpus(400, 32, 2);
        let a = build(&data, 32);
        let b = build(&data, 32);

        let query = &corpus(1, 32, 9)[0];
        assert_eq!(a.search(query, 10, None), b.search(query, 10, None));
    }

    #[test]
    fn serialization_roundtrip_preserves_results() {
        let data = corpus(400, 32, 11);
        let graph = build(&data, 32);
        let query = &corpus(1, 32, 6)[0];
        let before = graph.search(query, 10, None);

        let bytes = bincode::serialize(&graph).unwrap();
        let restored: HnswIndex = bincode::deserialize(&bytes).unwrap();

        assert_eq!(restored.len(), graph.len());
        assert_eq!(restored.dim(), graph.dim());
        assert_eq!(restored.search(query, 10, None), before);
    }

    #[test]
    fn insert_normalizes_unnormalized_input() {
        let mut graph = HnswIndex::new(3, HnswParams::default());
        graph.insert(&[3.0, 4.0, 0.0]);

        let stored = graph.vector(0);
        assert!((dot(stored, stored).sqrt() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn short_input_is_zero_padded() {
        let mut graph = HnswIndex::new(4, HnswParams::default());
        graph.insert(&[1.0, 0.0]);
        assert_eq!(graph.vector(0).len(), 4);
        assert_eq!(graph.search(&[1.0, 0.0, 0.0, 0.0], 1, None).len(), 1);
    }

    #[test]
    fn incremental_insert_matches_batch_build() {
        let data = corpus(300, 32, 44);
        let batch = build(&data, 32);

        let mut incremental = HnswIndex::new(32, HnswParams::default());
        for v in &data {
            incremental.insert(v);
        }

        let query = &corpus(1, 32, 55)[0];
        assert_eq!(
            incremental.search(query, 10, None),
            batch.search(query, 10, None),
            "reusing a scratch visited set must not change results"
        );
    }

    #[test]
    fn layer_zero_degree_stays_within_cap() {
        let data = corpus(600, 32, 31);
        let graph = build(&data, 32);
        let cap = graph.params.max_degree(0);

        for (id, node) in graph.nodes.iter().enumerate() {
            assert!(
                node.links[0].len() <= cap,
                "node {id} has {} layer-0 links, cap is {cap}",
                node.links[0].len()
            );
        }
    }

    #[test]
    fn no_node_links_to_itself() {
        let data = corpus(400, 32, 17);
        let graph = build(&data, 32);

        for (id, node) in graph.nodes.iter().enumerate() {
            for level in &node.links {
                assert!(!level.contains(&(id as u32)), "node {id} has a self-loop");
            }
        }
    }

    #[test]
    fn graph_stays_connected_at_layer_zero() {
        let data = corpus(500, 32, 23);
        let graph = build(&data, 32);

        for (id, node) in graph.nodes.iter().enumerate() {
            assert!(
                !node.links[0].is_empty(),
                "node {id} is isolated on layer 0"
            );
        }
    }

    #[test]
    fn visited_set_generation_isolates_epochs() {
        let mut visited = VisitedSet::new(4);
        visited.begin();
        assert!(visited.insert(2));
        assert!(!visited.insert(2));

        visited.begin();
        assert!(visited.insert(2), "new epoch must forget prior marks");
    }

    #[test]
    fn visited_set_grows_for_out_of_range_ids() {
        let mut visited = VisitedSet::new(1);
        visited.begin();
        assert!(visited.insert(9));
        assert!(!visited.insert(9));
    }

    #[test]
    fn random_level_distribution_is_top_heavy() {
        let mut graph = HnswIndex::new(8, HnswParams::default());
        let mut counts = [0usize; MAX_LEVEL + 1];
        for _ in 0..10_000 {
            counts[graph.random_level()] += 1;
        }

        assert!(
            counts[0] > 8_000,
            "expected ~1-1/M of nodes on layer 0 only, got {}",
            counts[0]
        );
        assert!(counts[1] > 0, "no nodes promoted to layer 1");
        assert!(counts[0] > counts[1], "layer sizes must shrink upward");
    }
}
