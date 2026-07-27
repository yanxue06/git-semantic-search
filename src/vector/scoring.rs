//! Similarity kernels and bounded top-k selection.
//!
//! Embeddings are L2-normalized when they are written to the index, so cosine
//! similarity collapses to a plain dot product. That removes two square roots
//! and two full passes over the vector per candidate — the single hottest
//! operation in the whole search path.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Width of one accumulator block: two 4-wide SIMD registers on both NEON and
/// SSE, which is what lets the multiplies issue without waiting on each other.
const LANES: usize = 8;

/// Dot product of two equal-length slices.
///
/// `chunks_exact` hands LLVM fixed-size slices, so indexing inside the block is
/// provably in bounds and the whole block lowers to packed multiply-adds across
/// `LANES` independent accumulators — no bounds checks, no serial dependency
/// chain on a single register. The remainder is summed scalar.
///
/// # Panics
/// Debug builds assert equal lengths. Release builds score the common prefix.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot product needs equal-length vectors");

    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);

    let mut acc = [0.0f32; LANES];
    let mut blocks_a = a.chunks_exact(LANES);
    let mut blocks_b = b.chunks_exact(LANES);

    for (x, y) in blocks_a.by_ref().zip(blocks_b.by_ref()) {
        for lane in 0..LANES {
            acc[lane] += x[lane] * y[lane];
        }
    }

    let mut sum = 0.0f32;
    for value in acc {
        sum += value;
    }

    for (x, y) in blocks_a.remainder().iter().zip(blocks_b.remainder()) {
        sum += x * y;
    }

    sum
}

/// Cosine similarity that does not assume normalized inputs.
///
/// Used for query vectors and any legacy index whose embeddings predate
/// write-time normalization.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot(a, b) / (norm_a * norm_b)
}

/// Scale a vector to unit length in place. No-op for the zero vector.
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// A scored candidate ordered by distance, ascending.
///
/// Distance rather than similarity so every heap and comparison in the HNSW
/// traversal reads "smaller is better" without inverted-comparator bugs. Ties
/// break on id, which makes results stable across runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    pub dist: f32,
    pub id: u32,
}

impl Scored {
    pub fn new(dist: f32, id: u32) -> Self {
        Self { dist, id }
    }
}

impl Eq for Scored {}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        // NaN distances sort last so a poisoned vector can never win a slot.
        match (self.dist.is_nan(), other.dist.is_nan()) {
            (true, true) => self.id.cmp(&other.id),
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => self
                .dist
                .partial_cmp(&other.dist)
                .unwrap_or(Ordering::Equal)
                .then_with(|| self.id.cmp(&other.id)),
        }
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bounded max-heap that retains the `k` smallest distances seen.
///
/// The previous implementation sorted every candidate in the repository to
/// return ten rows — O(N log N) work and an N-element allocation for a
/// K-element answer. This is O(N log K) with a K-element allocation.
#[derive(Debug)]
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Scored>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    /// Offer a candidate. Kept only if it beats the current worst retained one.
    pub fn push(&mut self, item: Scored) {
        if self.k == 0 {
            return;
        }

        if self.heap.len() < self.k {
            self.heap.push(item);
            return;
        }

        // `peek` is the largest distance retained, i.e. the weakest result.
        if let Some(worst) = self.heap.peek()
            && item < *worst
        {
            self.heap.pop();
            self.heap.push(item);
        }
    }

    /// The worst retained distance, or `None` until `k` items have been offered.
    pub fn worst(&self) -> Option<f32> {
        if self.heap.len() < self.k {
            None
        } else {
            self.heap.peek().map(|s| s.dist)
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drain into best-first order.
    pub fn into_sorted_vec(self) -> Vec<Scored> {
        self.heap.into_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_naive_sum() {
        let a: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..384).map(|i| ((384 - i) as f32) * 0.02).collect();

        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let fast = dot(&a, &b);

        assert!(
            (naive - fast).abs() < naive.abs() * 1e-4,
            "unrolled dot drifted: naive={naive} fast={fast}"
        );
    }

    #[test]
    fn dot_handles_non_multiple_of_four_lengths() {
        for len in 0..13 {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|_| 2.0).collect();
            let expected: f32 = a.iter().map(|x| x * 2.0).sum();
            assert!(
                (dot(&a, &b) - expected).abs() < 1e-5,
                "len {len} mismatched"
            );
        }
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert!(cosine(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_is_negative_one() {
        let sim = cosine(&[1.0, 2.0, 3.0], &[-1.0, -2.0, -3.0]);
        assert!((sim + 1.0).abs() < 1e-6, "got {sim}");
    }

    #[test]
    fn cosine_zero_vector_is_zero() {
        assert_eq!(cosine(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn normalize_produces_unit_length() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((dot(&v, &v).sqrt() - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn normalize_leaves_zero_vector_alone() {
        let mut v = vec![0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn normalized_dot_equals_cosine() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let mut b = vec![4.0, 1.0, 0.5, 2.0];
        let expected = cosine(&a, &b);
        normalize(&mut a);
        normalize(&mut b);
        assert!((dot(&a, &b) - expected).abs() < 1e-6);
    }

    #[test]
    fn topk_keeps_smallest_distances() {
        let mut top = TopK::new(3);
        for (i, dist) in [0.9, 0.1, 0.5, 0.05, 0.7].iter().enumerate() {
            top.push(Scored::new(*dist, i as u32));
        }

        let out = top.into_sorted_vec();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].id, 3);
        assert_eq!(out[1].id, 1);
        assert_eq!(out[2].id, 2);
    }

    #[test]
    fn topk_handles_fewer_items_than_k() {
        let mut top = TopK::new(10);
        top.push(Scored::new(0.4, 0));
        top.push(Scored::new(0.2, 1));

        assert_eq!(top.worst(), None, "worst is undefined until k items land");
        let out = top.into_sorted_vec();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, 1);
    }

    #[test]
    fn topk_zero_capacity_accepts_nothing() {
        let mut top = TopK::new(0);
        top.push(Scored::new(0.1, 0));
        assert!(top.is_empty());
    }

    #[test]
    fn topk_worst_tracks_weakest_retained() {
        let mut top = TopK::new(2);
        top.push(Scored::new(0.1, 0));
        top.push(Scored::new(0.9, 1));
        assert_eq!(top.worst(), Some(0.9));

        top.push(Scored::new(0.2, 2));
        assert_eq!(top.worst(), Some(0.2));
    }

    #[test]
    fn scored_orders_nan_last() {
        let mut top = TopK::new(2);
        top.push(Scored::new(f32::NAN, 0));
        top.push(Scored::new(0.5, 1));
        top.push(Scored::new(0.6, 2));

        let out = top.into_sorted_vec();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, 1);
        assert_eq!(out[1].id, 2, "NaN must never displace a real distance");
    }

    #[test]
    fn scored_ties_break_on_id_for_stable_output() {
        assert!(Scored::new(0.5, 1) < Scored::new(0.5, 2));
    }
}
