//! Sampling of uniformly random derangements (permutations with no fixed points),
//! via a variant of the Martínez–Panholzer–Prodinger algorithm.
//!
//! # Reference
//! Conrado Martínez, Alois Panholzer, and Helmut Prodinger, "Generating Random
//! Derangements", *Proc. 5th Workshop on Analytic Algorithmics and Combinatorics
//! (ANALCO)*, SIAM, 2008.
//! <https://epubs.siam.org/doi/pdf/10.1137/1.9781611972986.7>

use std::iter::successors;
use rand::RngExt;

/// Infinite iterator over the 2-cycle probabilities `two_cycle(u)` for
/// `u = 0, 1, 2, ...` where `two_cycle(u) = d[u-1] / (d[u-1] + d[u])` is the probability that, with
/// `u + 1` elements left to place, the current one closes a 2-cycle rather than
/// extending into a longer cycle. It is generated with the stable float
/// recursion `two_cycle(u) = (1 - two_cycle(u-1)) / (u - two_cycle(u-1))`.
fn two_cycle_probabilities() -> impl Iterator<Item = f64> {
    successors(Some((0usize, 0.0f64)), |&(mut u, prev)| {
        u += 1;
        Some((u, (1.0 - prev) / (u as f64 - prev)))
    })
    .map(|(_, p)| p)
}

/// Rearranges `data` in place into a uniformly random derangement of its
/// elements so every element ends up at a position different from where it started.
///
/// # Panics
/// Panics if `data.len() == 1`, since no derangement of a single element exists.
pub fn derange<T, R: RngExt + ?Sized>(data: &mut [T], rng: &mut R) {
    let n = data.len();
    if n == 0 {
        return;
    }
    assert!(n != 1, "no derangement exists for n = 1");

    let two_cycle_prob = two_cycle_probabilities().take(n).collect::<Vec<f64>>();
    let mut unmarked = (0..n).collect::<Vec<usize>>();

    while unmarked.len() > 1 {
        let i = unmarked.pop().unwrap();
        let j = rng.random_range(..unmarked.len());
        data.swap(i, unmarked[j]);
        if rng.random_bool(two_cycle_prob[unmarked.len()]) {
            unmarked.swap_remove(j);
        }
    }
}

/// Samples a uniformly random derangement of `{0, 1, ..., n-1}` using the given
/// random number generator.
///
/// # Panics
/// Panics if `n == 1`, since no derangement of a single element exists.
pub fn sample_derangement_with<R: RngExt + ?Sized>(n: usize, rng: &mut R) -> Vec<usize> {
    let mut permutation = (0..n).collect::<Vec<usize>>();
    derange(&mut permutation, rng);
    permutation
}

/// Samples a uniformly random derangement of `{0, 1, ..., n-1}`.
///
/// # Panics
/// Panics if `n == 1`, since no derangement of a single element exists.
pub fn sample_derangement(n: usize) -> Vec<usize> {
    sample_derangement_with(n, &mut rand::rng())
}

/// Returns `true` iff `p` is a permutation of `{0, 1, ..., p.len()-1}`, i.e. every
/// index in that range appears exactly once.
pub fn is_permutation(p: &[usize]) -> bool {
    let mut seen = vec![false; p.len()];
    for &x in p {
        if x >= p.len() || seen[x] {
            return false;
        }
        seen[x] = true;
    }
    true
}

/// Returns `true` iff `p` is a derangement: a permutation of
/// `{0, 1, ..., p.len()-1}` with no fixed point (`p[i] != i` for all `i`).
pub fn is_derangement(p: &[usize]) -> bool {
    is_permutation(p) && p.iter().enumerate().all(|(i, &pi)| i != pi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn samples_are_valid_derangements() {
        let mut rng = rand::rng();
        for n in [2, 3, 4, 5, 8, 13, 21, 34, 50, 100] {
            for _ in 0..1000 {
                let d = sample_derangement_with(n, &mut rng);
                assert!(is_derangement(&d), "not a derangement for n = {n}: {d:?}");
            }
        }
    }

    #[test]
    fn empty_input() {
        assert_eq!(sample_derangement_with(0, &mut rand::rng()), Vec::<usize>::new());
    }

    #[test]
    fn predicates() {
        assert!(is_permutation(&[2, 0, 1]));
        assert!(!is_permutation(&[0, 0, 1])); // repeat
        assert!(!is_permutation(&[0, 1, 3])); // out of range

        assert!(is_derangement(&[1, 2, 0]));
        assert!(!is_derangement(&[0, 2, 1])); // fixed point at 0
        assert!(!is_derangement(&[1, 1, 0])); // not a permutation
        assert!(is_derangement(&[]) && is_permutation(&[])); // vacuously true
    }

    #[test]
    fn derange_moves_every_element() {
        let mut rng = rand::rng();
        // Derange arbitrary (distinct) values and check none stays put.
        let original: Vec<char> = ('a'..='z').collect();
        for _ in 0..1000 {
            let mut data = original.clone();
            derange(&mut data, &mut rng);
            assert!(
                data.iter().zip(&original).all(|(now, before)| now != before),
                "element left in place: {data:?}"
            );
            // still the same multiset of elements
            let mut sorted = data.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, original);
        }
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
