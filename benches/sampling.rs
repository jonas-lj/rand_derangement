//! Simple wall-clock benchmark for `sample_derangement` on large inputs.
//!
//! Run with `cargo bench` (release mode). Uses a custom harness (no criterion),
//! so it works on stable Rust.

use std::hint::black_box;
use std::time::Instant;

use derangements::sample_derangement;

fn bench_n(n: usize, iters: usize) {
    // Warm up once, and sanity-check we're producing a real derangement.
    let warm = sample_derangement(n);
    assert!(warm.iter().enumerate().all(|(i, &pi)| i != pi));

    let start = Instant::now();
    for _ in 0..iters {
        black_box(sample_derangement(black_box(n)));
    }
    let elapsed = start.elapsed();

    let per_sample = elapsed / iters as u32;
    let ns_per_elem = elapsed.as_secs_f64() / (iters as f64 * n as f64) * 1e9;
    println!(
        "n = {n:>10}  iters = {iters:>5}  per sample = {per_sample:>12.3?}  ({ns_per_elem:5.2} ns/element)"
    );
}

fn main() {
    println!("derangement sampling benchmark\n");
    for &(n, iters) in &[
        (1_000usize, 5_000usize),
        (10_000, 1_000),
        (100_000, 100),
        (1_000_000, 20),
        (10_000_000, 3),
    ] {
        bench_n(n, iters);
    }
}
