//! Criterion benchmark for `sample_derangement` on large inputs.
//!
//! Run with `cargo bench`. `Throughput::Elements` makes criterion report a
//! per-element time, which is the interesting figure as `n` grows.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use derangements::sample_derangement;

fn bench_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_derangement");
    // Large inputs are slow per iteration, so keep the sampling budget modest.
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for n in [1_000usize, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| sample_derangement(black_box(n)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sampling);
criterion_main!(benches);
