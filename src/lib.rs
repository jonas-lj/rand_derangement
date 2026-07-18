//! Sampling of uniformly random derangements (permutations with no fixed points),
//! via a variant of the Martínez–Panholzer–Prodinger algorithm. See
//! [`two_cycle_probabilities`] for the per-step probabilities.

use rand::RngExt;

/// Infinite iterator over the 2-cycle probabilities `two_cycle(u)` for
/// `u = 0, 1, 2, ...`.
///
/// `two_cycle(u) = d[u-1] / (d[u-1] + d[u])` is the probability that, with
/// `u + 1` elements left to place, the current one closes a 2-cycle rather than
/// extending into a longer cycle. It is generated with the stable float
/// recursion `two_cycle(u) = (1 - two_cycle(u-1)) / (u - two_cycle(u-1))`, which
/// runs off the single seed `two_cycle(0) = 0` (giving `two_cycle(1) = 1`,
/// `two_cycle(2) = 0`, `two_cycle(3) = 1/3`, ...).
///
/// The sequence is infinite; use [`Iterator::take`] to get a prefix.
pub fn two_cycle_probabilities() -> impl Iterator<Item = f64> {
    // State is `(u, two_cycle(u))`; the recursion needs the index alongside the
    // previous value. Seed at `u = 0` and map the tuple down to the probability.
    std::iter::successors(Some((0usize, 0.0f64)), |&(u, prev)| {
        let u = u + 1;
        Some((u, (1.0 - prev) / (u as f64 - prev)))
    })
    .map(|(_, prob)| prob)
}

/// Samples a uniformly random derangement of `{0, 1, ..., n-1}`.
///
/// # Panics
/// Panics if `n == 1`, since no derangement of a single element exists.
pub fn sample_derangement(n: usize) -> Vec<usize> {
    sample_derangement_with(n, &mut rand::rng())
}

/// Samples a uniformly random derangement of `{0, 1, ..., n-1}` using the given
/// random number generator.
///
/// # Panics
/// Panics if `n == 1`, since no derangement of a single element exists.
pub fn sample_derangement_with<R: RngExt + ?Sized>(n: usize, rng: &mut R) -> Vec<usize> {
    assert!(n != 1, "no derangement exists for n = 1");

    let mut permutation = (0..n).collect::<Vec<usize>>();
    if n == 0 {
        return permutation;
    }

    let two_cycle_prob = two_cycle_probabilities().take(n).collect::<Vec<f64>>();
    let mut unmarked = (0..n).collect::<Vec<usize>>();

    while unmarked.len() > 1 {
        let i = unmarked.pop().unwrap();
        let j = rng.random_range(..unmarked.len());
        permutation.swap(i, unmarked[j]);
        if rng.random_bool(two_cycle_prob[unmarked.len()]) {
            unmarked.swap_remove(j);
        }
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

    #[test]
    #[should_panic(expected = "no derangement exists for n = 1")]
    fn n1_panics() {
        sample_derangement_with(1, &mut rand::rng());
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
