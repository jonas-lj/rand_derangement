//! Estimates the Golomb–Dickman constant λ ≈ 0.6243299885…
//!
//! λ is the limiting ratio `E[longest cycle length] / n` for a uniformly random
//! permutation of `n` elements. We sample many permutations, take the longest
//! cycle of each (via `Permutation::cycles`), and average the ratio.
//!
//! Run with: `cargo run --release --example golomb_dickman`

use rand_derangement::Permutation;

fn main() {
    let n = 1_000usize;
    let trials = 100_000usize;

    let mut rng = rand::rng();
    let mut sum_ratio = 0.0f64;
    for _ in 0..trials {
        let p = Permutation::sample_permutation_with(n, &mut rng);
        let longest = p.cycles().map(|c| c.len()).max().unwrap_or(0);
        sum_ratio += longest as f64 / n as f64;
    }
    let estimate = sum_ratio / trials as f64;

    // Golomb–Dickman constant.
    const LAMBDA: f64 = 0.6243299885435508;

    println!("Golomb–Dickman constant estimate");
    println!("  n           = {n}");
    println!("  trials      = {trials}");
    println!("  estimate    = {estimate:.5}");
    println!("  known value = {LAMBDA:.5}");
    println!("  abs. error  = {:.5}", (estimate - LAMBDA).abs());
}
