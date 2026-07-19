//! Sampling of uniformly random derangements (permutations with no fixed points),
//! via a variant of the Martínez–Panholzer–Prodinger algorithm.
//!
//! # Reference
//! Conrado Martínez, Alois Panholzer, and Helmut Prodinger, "Generating Random
//! Derangements", *Proc. 5th Workshop on Analytic Algorithmics and Combinatorics
//! (ANALCO)*, SIAM, 2008.
//! <https://epubs.siam.org/doi/pdf/10.1137/1.9781611972986.7>

use std::iter::successors;
use std::ops::Index;
use rand::RngExt;

/// Infinite iterator over the 2-cycle probabilities `two_cycle(u)` for
/// `u = 0, 1, 2, ...` where `two_cycle(u) = d[u-1] / (d[u-1] + d[u])` is the probability that, with
/// `u + 1` elements left to place, the current one closes a 2-cycle rather than
/// extending into a longer cycle.
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
pub fn sample_derangement_with<R: RngExt + ?Sized>(n: usize, rng: &mut R) -> Permutation {
    let mut permutation = (0..n).collect::<Vec<usize>>();
    derange(&mut permutation, rng);
    Permutation(permutation)
}

/// Samples a uniformly random derangement of `{0, 1, ..., n-1}`.
///
/// # Panics
/// Panics if `n == 1`, since no derangement of a single element exists.
pub fn sample_derangement(n: usize) -> Permutation {
    sample_derangement_with(n, &mut rand::rng())
}

/// Samples a uniformly random permutation of `{0, 1, ..., n-1}` using the given
/// random number generator, via a Fisher–Yates shuffle.
pub fn sample_permutation_with<R: RngExt + ?Sized>(n: usize, rng: &mut R) -> Permutation {
    let mut permutation = (0..n).collect::<Vec<usize>>();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        permutation.swap(i, j);
    }
    Permutation(permutation)
}

/// Samples a uniformly random permutation of `{0, 1, ..., n-1}`.
pub fn sample_permutation(n: usize) -> Permutation {
    sample_permutation_with(n, &mut rand::rng())
}

/// Returns `true` iff `p` is a permutation of `{0, 1, ..., p.len()-1}`, i.e. every
/// index in that range appears exactly once.
fn is_permutation(p: &[usize]) -> bool {
    let mut seen = vec![false; p.len()];
    // `x < len` first so the index is in bounds; `replace` returns the previous
    // bit, so a repeat (already `true`) fails the check.
    p.iter().all(|&x| x < p.len() && !std::mem::replace(&mut seen[x], true))
}

/// Returns `true` iff `p` is a derangement: a permutation of
/// `{0, 1, ..., p.len()-1}` with no fixed point (`p[i] != i` for all `i`).
fn is_derangement(p: &[usize]) -> bool {
    is_permutation(p) && p.iter().enumerate().all(|(i, &pi)| i != pi)
}

/// A permutation of `{0, 1, ..., n-1}`, represented by its map: element `i` maps
/// to `self[i]`. Valid by construction — built by [`sample_derangement`] and
/// friends, or checked via [`Permutation::try_new`] / `TryFrom<Vec<usize>>`.
///
/// Derefs to `[usize]`, so slice methods and indexing (`perm[i]`) work directly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    /// Wraps `map` after checking it is a permutation of `{0, ..., map.len()-1}`.
    pub fn try_new(map: Vec<usize>) -> Result<Self, NotAPermutation> {
        if is_permutation(&map) {
            Ok(Self(map))
        } else {
            Err(NotAPermutation)
        }
    }

    /// The inverse permutation, satisfying `self.inverse()[self[i]] == i`.
    pub fn inverse(&self) -> Permutation {
        let mut inverse = vec![0usize; self.0.len()];
        for (i, &pi) in self.0.iter().enumerate() {
            inverse[pi] = i;
        }
        Permutation(inverse)
    }

    /// Applies the permutation to `data`.
    ///
    /// # Panics
    /// Panics if `data.len() != self.len()`.
    pub fn apply<T: Clone>(&self, data: &[T]) -> Vec<T> {
        assert_eq!(
            data.len(),
            self.len(),
            "data length must match permutation length"
        );
        self.0.iter().map(|&i| data[i].clone()).collect()
    }

    /// Applies the permutation to `data` in place
    ///
    /// # Panics
    /// Panics if `data.len() != self.len()`.
    pub fn apply_mut<T>(&self, data: &mut [T]) {
        assert_eq!(
            data.len(),
            self.len(),
            "data length must match permutation length"
        );
        let n = self.len();
        let mut seen = vec![false; n];
        for start in 0..n {
            if seen[start] {
                continue;
            }
            // Rotate the cycle through `start` by one, so each position ends up
            // with the value of its successor `self[cur]`.
            let mut cur = start;
            seen[cur] = true;
            loop {
                let next = self.0[cur];
                if next == start {
                    break;
                }
                data.swap(cur, next);
                seen[next] = true;
                cur = next;
            }
        }
    }

    /// Returns `true` iff this permutation has no fixed point (`self[i] != i`).
    pub fn is_derangement(&self) -> bool {
        is_derangement(&self.0)
    }

    /// Consumes the permutation, returning the underlying map.
    pub fn into_vec(self) -> Vec<usize> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Index<usize> for Permutation {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::Deref for Permutation {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        &self.0
    }
}

impl TryFrom<Vec<usize>> for Permutation {
    type Error = NotAPermutation;
    fn try_from(map: Vec<usize>) -> Result<Self, NotAPermutation> {
        Self::try_new(map)
    }
}

/// Formats the permutation in cycle notation, e.g. `(0 2 1)(3 4)`. Fixed points
/// appear as singleton cycles, so the identity of size 3 is `(0)(1)(2)`.
impl std::fmt::Display for Permutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut seen = vec![false; self.0.len()];
        for start in 0..self.0.len() {
            if seen[start] {
                continue;
            }
            write!(f, "(")?;
            let mut cur = start;
            loop {
                seen[cur] = true;
                write!(f, "{cur}")?;
                cur = self.0[cur];
                if cur == start {
                    break;
                }
                write!(f, " ")?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

/// Error returned when a `Vec<usize>` is not a valid permutation of `{0, ..., n-1}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotAPermutation;

impl std::fmt::Display for NotAPermutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("not a valid permutation of {0, ..., n-1}")
    }
}

impl std::error::Error for NotAPermutation {}

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
                assert!(d.is_derangement(), "not a derangement for n = {n}: {d:?}");
            }
        }
    }

    #[test]
    fn empty_input() {
        assert!(sample_derangement_with(0, &mut rand::rng()).is_empty());
    }

    #[test]
    fn samples_are_valid_permutations() {
        let mut rng = rand::rng();
        // Unlike derangements, permutations exist for n = 0 and n = 1.
        for n in [0, 1, 2, 3, 5, 8, 50, 100] {
            for _ in 0..500 {
                let p = sample_permutation_with(n, &mut rng);
                assert_eq!(p.len(), n);
                assert!(is_permutation(&p), "not a permutation for n = {n}: {p:?}");
            }
        }
    }

    /// All 6 permutations of 3 elements should appear with frequency ~1/6.
    #[test]
    fn sample_permutation_is_uniform_for_n3() {
        let mut rng = rand::rng();
        let mut counts: HashMap<Permutation, u32> = HashMap::new();
        let trials = 600_000;
        for _ in 0..trials {
            *counts.entry(sample_permutation_with(3, &mut rng)).or_default() += 1;
        }

        assert_eq!(counts.len(), 6, "expected all six permutations of 3 elements");
        let expected = 1.0 / 6.0;
        for (p, &c) in &counts {
            let freq = c as f64 / trials as f64;
            assert!((freq - expected).abs() < 0.01, "permutation {p:?} had frequency {freq}");
        }
    }

    #[test]
    fn permutation_type() {
        // Validation via try_new / TryFrom.
        assert!(Permutation::try_new(vec![2, 0, 1]).is_ok());
        assert_eq!(Permutation::try_new(vec![0, 0]), Err(NotAPermutation));
        assert!(Permutation::try_from(vec![1, 2, 3]).is_err()); // out of range

        let p = Permutation::try_new(vec![1, 2, 0]).unwrap();

        // Deref: indexing and slice methods.
        assert_eq!(p[0], 1);
        assert_eq!(p.len(), 3);
        assert!(p.is_derangement());

        // inverse ∘ p == identity.
        let inv = p.inverse();
        for i in 0..p.len() {
            assert_eq!(inv[p[i]], i);
            assert_eq!(p[inv[i]], i);
        }

        // apply: out[i] = data[p[i]].
        let data = ['a', 'b', 'c'];
        assert_eq!(p.apply(&data), vec!['b', 'c', 'a']);

        // Display in cycle notation.
        assert_eq!(p.to_string(), "(0 1 2)");
        let two_cycles = Permutation::try_new(vec![1, 0, 3, 2]).unwrap();
        assert_eq!(two_cycles.to_string(), "(0 1)(2 3)");

        // into_inner round-trips.
        assert_eq!(p.into_vec(), vec![1, 2, 0]);
    }

    #[test]
    #[should_panic(expected = "data length must match permutation length")]
    fn apply_length_mismatch_panics() {
        let p = Permutation::try_new(vec![1, 2, 0]).unwrap();
        p.apply(&[1, 2]);
    }

    #[test]
    fn apply_mut_matches_apply() {
        let mut rng = rand::rng();
        for n in [2usize, 3, 5, 8, 50] {
            let p = sample_derangement_with(n, &mut rng);
            let data: Vec<usize> = (0..n).map(|i| i * 10).collect();

            let out = p.apply(&data);
            let mut in_place = data.clone();
            p.apply_mut(&mut in_place);

            assert_eq!(in_place, out, "apply_mut disagrees with apply for n = {n}");
        }

        // Concrete check, incl. a 2-cycle + a longer cycle + a fixed point.
        let p = Permutation::try_new(vec![0, 2, 3, 1]).unwrap();
        let mut data = ['a', 'b', 'c', 'd'];
        p.apply_mut(&mut data);
        assert_eq!(data, ['a', 'c', 'd', 'b']); // out[i] = old[p[i]]
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
        let mut counts: HashMap<Permutation, u32> = HashMap::new();
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
        let mut counts: HashMap<Permutation, u32> = HashMap::new();
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
