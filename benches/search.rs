//! Exact scan vs. HNSW traversal across realistic repository sizes.
//!
//! Run with `cargo bench`. Vectors are 384-dimensional to match
//! bge-small-en-v1.5, and generated from a fixed seed so numbers are
//! comparable between runs.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use git_semantic::vector::{HnswIndex, HnswParams, Scored, TopK, dot, normalize};
use std::hint::black_box;

const DIM: usize = 384;

fn corpus(count: usize, seed: u64) -> Vec<Vec<f32>> {
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
            let mut v: Vec<f32> = (0..DIM).map(|_| next()).collect();
            normalize(&mut v);
            v
        })
        .collect()
}

/// What the engine's exhaustive path does: one dot product per commit into a
/// bounded top-k heap.
fn exact_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<Scored> {
    let mut top = TopK::new(k);
    for (i, v) in corpus.iter().enumerate() {
        top.push(Scored::new(1.0 - dot(v, query), i as u32));
    }
    top.into_sorted_vec()
}

fn bench_query_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("search/top10");

    for size in [1_000usize, 10_000, 50_000] {
        let data = corpus(size, 0xC0FF_EE00);
        let query = corpus(1, 0xBEEF)[0].clone();
        let graph = HnswIndex::build(DIM, HnswParams::default(), data.iter().map(Vec::as_slice));

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("exact", size), &size, |b, _| {
            b.iter(|| black_box(exact_top_k(&data, &query, 10)));
        });

        group.bench_with_input(BenchmarkId::new("hnsw", size), &size, |b, _| {
            b.iter(|| black_box(graph.search(&query, 10, None)));
        });
    }

    group.finish();
}

fn bench_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    group.sample_size(10);

    for size in [1_000usize, 10_000] {
        let data = corpus(size, 0xC0FF_EE00);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(HnswIndex::build(
                    DIM,
                    HnswParams::default(),
                    data.iter().map(Vec::as_slice),
                ))
            });
        });
    }

    group.finish();
}

fn bench_dot_kernel(c: &mut Criterion) {
    let a = corpus(1, 1)[0].clone();
    let b_vec = corpus(1, 2)[0].clone();

    c.bench_function("dot/384", |b| {
        b.iter(|| black_box(dot(black_box(&a), black_box(&b_vec))));
    });
}

criterion_group!(
    benches,
    bench_query_latency,
    bench_graph_construction,
    bench_dot_kernel
);
criterion_main!(benches);
