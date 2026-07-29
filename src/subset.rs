//! Uniform random `k`-subsets (combinations) of `{0, 1, ..., n-1}`.
//!
//! [`Subset`] is a lazy iterator: [`Subset::sample_with`] picks one of the
//! `C(n, k)` subsets uniformly at random, and iterating it yields the `k` chosen
//! indices one at a time via **Floyd's algorithm** — `O(k)` time, no rejection, so
//! it stays cheap even when `n` is huge and `k` is small.
//!
//! The indices come out in Floyd's selection order, **not sorted**; collect and
//! sort if you need them ordered.

use rand::RngExt;
use std::collections::HashSet;
use std::iter::FusedIterator;

/// A uniformly random `k`-subset of `{0, 1, ..., n-1}`, produced lazily.
///
/// Iterating yields the `k` distinct chosen indices in Floyd's selection order
/// (not sorted). See the [module docs](self) for the algorithm.
pub struct Subset<'a, R: ?Sized> {
    rng: &'a mut R,
    /// Indices chosen so far (needed for Floyd's membership test).
    chosen: HashSet<usize>,
    /// Next value of Floyd's loop counter `j`.
    j: usize,
    /// One past the last `j` (i.e. `n`).
    end: usize,
}

impl<'a, R: RngExt + ?Sized> Subset<'a, R> {
    /// Samples a uniformly random `k`-subset of `{0, 1, ..., n-1}` using the given
    /// random number generator, via Floyd's algorithm.
    ///
    /// # Panics
    /// Panics if `k > n`.
    pub fn sample_with(n: usize, k: usize, rng: &'a mut R) -> Subset<'a, R> {
        assert!(k <= n, "cannot choose k = {k} elements from a set of {n}");
        Subset {
            rng,
            chosen: HashSet::with_capacity(k),
            j: n - k,
            end: n,
        }
    }
}

impl<R: RngExt + ?Sized> Iterator for Subset<'_, R> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.j == self.end {
            return None;
        }
        let j = self.j;
        self.j += 1;

        // Floyd step: draw t in 0..=j; take t if new, else take j (always new,
        // since every chosen index so far is < j).
        let t = self.rng.random_range(0..=j);
        Some(if self.chosen.insert(t) {
            t
        } else {
            self.chosen.insert(j);
            j
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.j;
        (remaining, Some(remaining))
    }
}

impl<R: RngExt + ?Sized> ExactSizeIterator for Subset<'_, R> {}
impl<R: RngExt + ?Sized> FusedIterator for Subset<'_, R> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn samples_are_valid_subsets() {
        let mut rng = rand::rng();
        for n in [0usize, 1, 2, 5, 10, 50] {
            for k in 0..=n {
                let mut s: Vec<usize> = Subset::sample_with(n, k, &mut rng).collect();
                assert_eq!(s.len(), k);
                // Distinct and in range (after sorting, strictly increasing).
                s.sort_unstable();
                assert!(
                    s.windows(2).all(|w| w[0] < w[1]),
                    "duplicate indices: {s:?}"
                );
                assert!(s.iter().all(|&x| x < n));
            }
        }
    }

    #[test]
    fn edge_cases() {
        let mut rng = rand::rng();
        assert_eq!(Subset::sample_with(5, 0, &mut rng).count(), 0);
        assert_eq!(Subset::sample_with(0, 0, &mut rng).count(), 0);

        let mut full: Vec<usize> = Subset::sample_with(5, 5, &mut rng).collect();
        full.sort_unstable();
        assert_eq!(full, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "cannot choose")]
    fn too_large_panics() {
        let _ = Subset::sample_with(3, 5, &mut rand::rng());
    }

    /// All `C(5, 2) = 10` two-subsets of a 5-set should be equiprobable.
    #[test]
    fn uniform_for_5_choose_2() {
        let mut rng = rand::rng();
        let mut counts: HashMap<Vec<usize>, u32> = HashMap::new();
        let trials = 500_000;
        for _ in 0..trials {
            let mut s: Vec<usize> = Subset::sample_with(5, 2, &mut rng).collect();
            s.sort_unstable(); // canonicalize (Floyd's order is not sorted)
            *counts.entry(s).or_default() += 1;
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
