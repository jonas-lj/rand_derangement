//! Demonstrates that the fraction of random permutations that are derangements
//! tends to `1/e ≈ 0.3679` (since `!n / n! -> 1/e`).
//!
//! This is the classic hat-check (coat-check) problem: `n` diners check their
//! coats and get them back at random — the chance that nobody receives their own
//! tends to `1/e`, essentially independent of `n`. See
//! <https://proofwiki.org/wiki/Hat-Check_Problem>.
//!
//! We sample many random permutations, count how many have no fixed point, and
//! compare the fraction against `1/e`.
//!
//! Run with: `cargo run --release --example derangement_ratio`

use rand_combinatorics::permutation::Permutation;

fn main() {
    let n = 10usize;
    let trials = 1_000_000usize;

    let mut rng = rand::rng();
    let derangements = (0..trials)
        .filter(|_| Permutation::sample_permutation_with(n, &mut rng).is_derangement())
        .count();

    let fraction = derangements as f64 / trials as f64;
    let one_over_e = 1.0 / std::f64::consts::E;

    println!("fraction of random permutations that are derangements");
    println!("  n           = {n}");
    println!("  trials      = {trials}");
    println!("  fraction    = {fraction:.5}");
    println!("  1/e         = {one_over_e:.5}");
    println!("  abs. error  = {:.5}", (fraction - one_over_e).abs());
}
