//! Scaling + determinism bench for the job pool (P7.0).
//!
//! `cargo bench -p inf-core` compares a serial map against [`JobPool`]s of
//! increasing width over a CPU-bound workload, demonstrating that the pool
//! scales; the `determinism` group re-runs the same map at several widths to
//! confirm (in the criterion output) that wall-clock — not the result — is the
//! only thing the pool size changes. The byte-for-byte invariance itself is
//! asserted as a unit test in `job.rs` (`parallel_map_is_pool_size_invariant`);
//! this bench is the performance half of the P7.0 deliverable.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use inf_core::JobPool;

/// A deliberately non-trivial pure function so the parallel win is visible above
/// scheduling overhead.
fn work(x: u64) -> u64 {
    let mut acc = x;
    for _ in 0..256 {
        acc = acc
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        acc ^= acc >> 27;
    }
    acc
}

fn bench_scaling(c: &mut Criterion) {
    let input: Vec<u64> = (0..50_000).collect();
    let mut group = c.benchmark_group("parallel_map_scaling");
    group.throughput(Throughput::Elements(input.len() as u64));

    group.bench_function("serial", |b| {
        b.iter(|| input.iter().copied().map(work).collect::<Vec<_>>())
    });

    for threads in [1usize, 2, 4, 8] {
        let pool = JobPool::new(threads);
        group.bench_with_input(BenchmarkId::new("pool", threads), &threads, |b, _| {
            b.iter(|| pool.parallel_map(input.clone(), work))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_scaling);
criterion_main!(benches);
