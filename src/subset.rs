//! Uniform random `k`-subsets (combinations) of `{0, 1, ..., n-1}`.
//!
//! [`sample`] draws one of the `C(n, k)` subsets uniformly at random, returned in
//! ascending order, via **Floyd's algorithm** — `O(k)` time and no rejection, so it
//! stays cheap even when `n` is huge and `k` is small. [`subset`] does the same
//! over the elements of a given `Vec`.

use rand::RngExt;
use std::collections::HashSet;

/// Samples a uniformly random `k`-element subset of `{0, 1, ..., n-1}`, in
/// ascending order.
///
/// # Panics
/// Panics if `k > n`.
pub fn sample(n: usize, k: usize) -> Vec<usize> {
    sample_with(n, k, &mut rand::rng())
}

/// Samples a uniformly random `k`-element subset of `{0, 1, ..., n-1}` using the
/// given random number generator, in ascending order, via Floyd's algorithm.
///
/// # Panics
/// Panics if `k > n`.
pub fn sample_with<R: RngExt + ?Sized>(n: usize, k: usize, rng: &mut R) -> Vec<usize> {
    assert!(k <= n, "cannot choose k = {k} elements from a set of {n}");

    // Floyd's algorithm: for each j from n-k to n-1, draw t uniformly in 0..=j and
    // add t if new, otherwise add j. Yields a uniform k-subset.
    let mut chosen = HashSet::with_capacity(k);
    for j in (n - k)..n {
        let t = rng.random_range(0..=j);
        if !chosen.insert(t) {
            chosen.insert(j);
        }
    }

    let mut subset: Vec<usize> = chosen.into_iter().collect();
    subset.sort_unstable();
    subset
}

/// Samples a uniformly random `k`-element subset of `elements`, consuming the
/// input and keeping the chosen elements in their original order.
///
/// # Panics
/// Panics if `k > elements.len()`.
pub fn subset<T>(elements: Vec<T>, k: usize) -> Vec<T> {
    subset_with(elements, k, &mut rand::rng())
}

/// Samples a uniformly random `k`-element subset of `elements` using the given
/// random number generator, consuming the input and keeping the chosen elements in
/// their original order.
///
/// # Panics
/// Panics if `k > elements.len()`.
pub fn subset_with<T, R: RngExt + ?Sized>(elements: Vec<T>, k: usize, rng: &mut R) -> Vec<T> {
    let indices = sample_with(elements.len(), k, rng); // sorted ascending
    let mut wanted = indices.into_iter().peekable();
    let mut subset = Vec::with_capacity(k);
    for (i, element) in elements.into_iter().enumerate() {
        if wanted.peek() == Some(&i) {
            wanted.next();
            subset.push(element);
        }
    }
    subset
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn samples_are_valid_subsets() {
        let mut rng = rand::rng();
        for n in [0usize, 1, 2, 5, 10, 50] {
            for k in 0..=n {
                let s = sample_with(n, k, &mut rng);
                assert_eq!(s.len(), k);
                // Ascending and distinct (strictly increasing), all in range.
                assert!(
                    s.windows(2).all(|w| w[0] < w[1]),
                    "not sorted/distinct: {s:?}"
                );
                assert!(s.iter().all(|&x| x < n));
            }
        }
    }

    #[test]
    fn edge_cases() {
        let mut rng = rand::rng();
        assert_eq!(sample_with(5, 0, &mut rng), Vec::<usize>::new());
        assert_eq!(sample_with(5, 5, &mut rng), vec![0, 1, 2, 3, 4]);
        assert_eq!(sample_with(0, 0, &mut rng), Vec::<usize>::new());
    }

    #[test]
    #[should_panic(expected = "cannot choose")]
    fn too_large_panics() {
        sample_with(3, 5, &mut rand::rng());
    }

    #[test]
    fn subset_selects_elements() {
        let mut rng = rand::rng();
        let items: Vec<char> = ('a'..='j').collect(); // 10 distinct, ascending
        for k in 0..=items.len() {
            let sub = subset_with(items.clone(), k, &mut rng);
            assert_eq!(sub.len(), k);
            // Chosen from `items`, distinct, in original (ascending) order.
            assert!(sub.iter().all(|&c| items.contains(&c)));
            assert!(
                sub.windows(2).all(|w| w[0] < w[1]),
                "not in original order: {sub:?}"
            );
        }
    }

    /// All `C(5, 2) = 10` two-subsets of a 5-set should be equiprobable.
    #[test]
    fn uniform_for_5_choose_2() {
        let mut rng = rand::rng();
        let mut counts: HashMap<Vec<usize>, u32> = HashMap::new();
        let trials = 500_000;
        for _ in 0..trials {
            *counts.entry(sample_with(5, 2, &mut rng)).or_default() += 1;
        }

        assert_eq!(counts.len(), 10, "expected all C(5,2) = 10 subsets");
        let expected = 1.0 / 10.0;
        for (s, &c) in &counts {
            let freq = c as f64 / trials as f64;
            assert!(
                (freq - expected).abs() < 0.01,
                "subset {s:?} had frequency {freq}"
            );
        }
    }
}
