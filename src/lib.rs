//! Sampling of uniformly random derangements.
//!
//! A derangement is a permutation with no fixed points. This is a variant of
//! the Martínez–Panholzer–Prodinger algorithm that avoids index-selection
//! rejection sampling.
//!
//! The only randomness that depends on the subfactorials is a single Bernoulli
//! trial per step. With `u + 1` elements still in play, the element being placed
//! either forms a **2-cycle** (a transposition with its partner) or splices into
//! a longer cycle. It forms a 2-cycle with probability
//!
//! ```text
//! two_cycle(u) = d[u-1] / (d[u-1] + d[u]),
//! ```
//!
//! where `d[k]` is the number of derangements of `k` elements (the subfactorial
//! `!k = round(k!/e)`). This follows from the derangement recurrence
//! `d[u+1] = u*(d[u] + d[u-1])`, whose two terms `u*d[u-1]` and `u*d[u]` count
//! exactly the 2-cycle and longer-cycle cases; it is the original
//! `u * d[u-1] / d[u+1]` in simplified form.
//!
//! We precompute these probabilities once, in `f64`, with the stable recursion
//!
//! ```text
//! two_cycle(1) = 1,   two_cycle(u) = (1 - two_cycle(u-1)) / (u - two_cycle(u-1)),
//! ```
//!
//! which never forms the subfactorials themselves — so there are no big integers
//! and no overflow for any `n` — and feed them to a plain Bernoulli trial.

use rand::RngExt;

/// Precomputes `two_cycle(u) = d[u-1] / (d[u-1] + d[u])` for `u = 0..n`: the
/// probability that, with `u + 1` elements left to place, the current one closes
/// a 2-cycle rather than extending into a longer cycle.
///
/// Uses the stable float recursion `two_cycle(u) = (1 - two_cycle(u-1)) / (u -
/// two_cycle(u-1))`, seeded by `two_cycle(1) = 1`. Entry `[0]` is unused (the
/// loop never queries `u = 0`).
fn two_cycle_probabilities(n: usize) -> Vec<f64> {
    let mut p = vec![0.0f64; n];
    if n > 1 {
        p[1] = 1.0;
    }
    for u in 2..n {
        p[u] = (1.0 - p[u - 1]) / (u as f64 - p[u - 1]);
    }
    p
}

/// Samples a uniformly random derangement of `{0, 1, ..., n-1}`.
///
/// # Panics
/// Panics if `n == 1`, since no derangement of a single element exists.
pub fn sample_derangement(n: usize) -> Vec<usize> {
    assert!(n != 1, "no derangement exists for n = 1");

    let mut rng = rand::rng();
    sample_derangement_with(n, &mut rng)
}

/// Samples a uniformly random derangement of `{0, 1, ..., n-1}` using the given
/// random number generator.
pub fn sample_derangement_with<R: RngExt + ?Sized>(n: usize, rng: &mut R) -> Vec<usize> {
    let mut permutation = (0..n).collect::<Vec<usize>>();
    if n == 0 {
        return permutation;
    }

    let two_cycle_prob = two_cycle_probabilities(n);
    let mut unmarked = (0..n).collect::<Vec<usize>>();

    let mut u = n - 1;
    while u > 0 {
        let i = unmarked.pop().unwrap();
        let j = rng.random_range(0..unmarked.len());
        permutation.swap(i, unmarked[j]);

        // Close a 2-cycle with the current element, or leave it in a longer cycle.
        if rng.random_bool(two_cycle_prob[u]) {
            unmarked.remove(j);
            u -= 1;
            if u == 0 {
                break;
            }
        }
        u -= 1;
    }

    permutation
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A permutation is a derangement iff it has no fixed points.
    fn is_derangement(p: &[usize]) -> bool {
        p.iter().enumerate().all(|(i, &pi)| i != pi)
    }

    /// A slice is a permutation of 0..n iff every index appears exactly once.
    fn is_permutation(p: &[usize]) -> bool {
        let mut seen = vec![false; p.len()];
        for &x in p {
            if x >= p.len() || seen[x] {
                return false;
            }
            seen[x] = true;
        }
        true
    }

    #[test]
    fn samples_are_valid_derangements() {
        let mut rng = rand::rng();
        for n in [2, 3, 4, 5, 8, 13, 21, 34, 50, 100] {
            for _ in 0..1000 {
                let d = sample_derangement_with(n, &mut rng);
                assert!(is_permutation(&d), "not a permutation for n = {n}: {d:?}");
                assert!(is_derangement(&d), "not a derangement for n = {n}: {d:?}");
            }
        }
    }

    #[test]
    fn empty_input() {
        assert_eq!(sample_derangement_with(0, &mut rand::rng()), Vec::<usize>::new());
    }

    /// For n = 3 there are exactly two derangements: [1,2,0] and [2,0,1].
    /// A uniform sampler should hit each roughly half the time.
    #[test]
    fn distribution_is_uniform_for_n3() {
        let mut rng = rand::rng();
        let mut counts: HashMap<Vec<usize>, u32> = HashMap::new();
        let trials = 200_000;
        for _ in 0..trials {
            *counts.entry(sample_derangement_with(3, &mut rng)).or_default() += 1;
        }

        assert_eq!(counts.len(), 2, "expected exactly two derangements of 3 elements");
        for (d, &c) in &counts {
            let freq = c as f64 / trials as f64;
            assert!((freq - 0.5).abs() < 0.02, "derangement {d:?} had frequency {freq}");
        }
    }

    /// All 9 derangements of 4 elements should appear with frequency ~1/9.
    #[test]
    fn distribution_is_uniform_for_n4() {
        let mut rng = rand::rng();
        let mut counts: HashMap<Vec<usize>, u32> = HashMap::new();
        let trials = 900_000;
        for _ in 0..trials {
            *counts.entry(sample_derangement_with(4, &mut rng)).or_default() += 1;
        }

        assert_eq!(counts.len(), 9, "expected exactly nine derangements of 4 elements");
        let expected = 1.0 / 9.0;
        for (d, &c) in &counts {
            let freq = c as f64 / trials as f64;
            assert!((freq - expected).abs() < 0.01, "derangement {d:?} had frequency {freq}");
        }
    }
}
