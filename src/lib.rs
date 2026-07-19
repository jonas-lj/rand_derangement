//! A small library for random permutations and derangements of
//! `{0, 1, ..., n-1}` (a *derangement* is a permutation with no fixed points).
//!
//! - [`sample_permutation`] / [`sample_derangement`] draw a uniformly random
//!   permutation / derangement.
//! - [`shuffle`] / [`derange`] do the same in place on an arbitrary slice.
//! - [`Permutation`] is a validated wrapper offering [`apply`](Permutation::apply),
//!   [`inverse`](Permutation::inverse), cycle-notation `Display`, and more.
//!
//! Permutations use a Fisher–Yates shuffle, and derangements use a variant of the
//! Martínez–Panholzer–Prodinger algorithm (see [`derange`] for the reference).

use std::iter::successors;
use std::ops::Index;
use rand::RngExt;

/// A permutation of `{0, 1, ..., n-1}`, represented by its map: element `i` maps
/// to `self[i]`. Valid by construction — built by [`sample_derangement`] and
/// friends, or checked via [`Permutation::try_new`] / `TryFrom<Vec<usize>>`.
///
/// Derefs to `[usize]`, so slice methods and indexing (`perm[i]`) work directly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    /// The identity permutation of `{0, 1, ..., n-1}`, mapping every element to
    /// itself (`self[i] == i`).
    pub fn identity(n: usize) -> Permutation {
        Permutation((0..n).collect())
    }

    /// Wraps `map` after checking it is a permutation of `{0, ..., map.len()-1}`.
    pub fn try_new(map: Vec<usize>) -> Result<Self, NotAPermutation> {
        is_permutation(&map).then_some(Self(map)).ok_or(NotAPermutation)
    }

    /// The inverse permutation, satisfying `self.inverse()[self[i]] == i`.
    pub fn inverse(&self) -> Permutation {
        let mut inverse = vec![0usize; self.0.len()];
        for (i, &pi) in self.0.iter().enumerate() {
            inverse[pi] = i;
        }
        Permutation(inverse)
    }

    /// The composition `self ∘ other`.
    ///
    /// # Panics
    /// Panics if `self.len() != other.len()`.
    pub fn compose(&self, other: &Permutation) -> Permutation {
        assert_eq!(
            self.symbols(),
            other.symbols(),
            "permutations must have the same length"
        );
        Permutation(other.0.iter().map(|&i| self.0[i]).collect())
    }

    /// Walks the cycle decomposition, calling `f` once per cycle with the cycle's
    /// elements in cyclic order (starting at its smallest). A single buffer is
    /// reused across cycles, so nothing is allocated per cycle. This is the shared
    /// engine behind [`cycles`](Permutation::cycles), [`parity`](Permutation::parity),
    /// [`order`](Permutation::order), and [`apply_mut`](Permutation::apply_mut).
    fn for_each_cycle(&self, mut f: impl FnMut(&[usize])) {
        let n = self.0.len();
        let mut seen = vec![false; n];
        let mut cycle = Vec::new();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            cycle.clear();
            let mut cur = start;
            while !seen[cur] {
                seen[cur] = true;
                cycle.push(cur);
                cur = self.0[cur];
            }
            f(&cycle);
        }
    }

    /// Iterates over the cycles of the permutation, each beginning at its smallest
    /// element. Fixed points appear as singleton cycles, so the cycles partition
    /// `{0, ..., n-1}`.
    pub fn cycles(&self) -> impl Iterator<Item = Cycle> {
        let mut cycles = Vec::new();
        self.for_each_cycle(|elements| cycles.push(Cycle { elements: elements.to_vec() }));
        cycles.into_iter()
    }

    /// The parity (sign) of the permutation.
    pub fn parity(&self) -> Parity {
        let mut cycle_count = 0;
        self.for_each_cycle(|_| cycle_count += 1);
        if (self.symbols() - cycle_count).is_multiple_of(2) {
            Parity::Even
        } else {
            Parity::Odd
        }
    }

    /// The order of the permutation: the least `k >= 1` such that applying it `k`
    /// times gives the identity. It equals the least common multiple of the cycle
    /// lengths; the identity and the empty permutation have order 1.
    ///
    /// # Panics
    /// Panics if the order overflows `usize`. The order can be astronomically
    /// large (up to Landau's function `g(n)`) for large permutations.
    pub fn order(&self) -> usize {
        let mut order = 1usize;
        self.for_each_cycle(|cycle| {
            order = (order / gcd(order, cycle.len()))
                .checked_mul(cycle.len())
                .expect("permutation order overflows usize");
        });
        order
    }

    /// Applies the permutation to `data`.
    ///
    /// # Panics
    /// Panics if `data.len() != self.len()`.
    pub fn apply<T: Clone>(&self, data: &[T]) -> Vec<T> {
        assert_eq!(
            data.len(),
            self.symbols(),
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
            self.symbols(),
            "data length must match permutation length"
        );
        // Rotate each cycle by one via swaps down consecutive elements.
        self.for_each_cycle(|cycle| {
            for pair in cycle.windows(2) {
                data.swap(pair[0], pair[1]);
            }
        });
    }

    /// Returns `true` iff this permutation has no fixed point (`self[i] != i`).
    pub fn is_derangement(&self) -> bool {
        is_derangement(&self.0)
    }

    /// Returns `true` iff this permutation is an involution: its own inverse
    /// (`self[self[i]] == i` for all `i`, equivalently every cycle has length ≤ 2).
    pub fn is_involution(&self) -> bool {
        self.0.iter().enumerate().all(|(i, &pi)| self.0[pi] == i)
    }

    /// Consumes the permutation, returning the underlying map.
    pub fn into_vec(self) -> Vec<usize> {
        self.0
    }

    pub fn symbols(&self) -> usize {
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

/// Formats the permutation in one-line notation: the images `self[0], self[1],
/// ...` in order, e.g. `[1 2 0]`.
impl std::fmt::Display for Permutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, x) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{x}")?;
        }
        write!(f, "]")
    }
}

/// A single cycle of a permutation: the elements it moves, in cyclic order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Cycle {
    elements: Vec<usize>,
}

impl Cycle {
    /// The number of elements the cycle moves. A cycle is never empty (a fixed
    /// point is a cycle of length 1), so there is no `is_empty`.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Consumes the cycle, returning its elements in cyclic order.
    pub fn into_vec(self) -> Vec<usize> {
        self.elements
    }

    /// Applies the cycle to `data` in place, rotating the entries at its element
    /// positions by one so each ends up with the value of its successor. Only
    /// those positions are touched, so `data` may be longer than the cycle needs.
    ///
    /// # Panics
    /// Panics if any element of the cycle is out of bounds for `data`.
    pub fn apply_mut<T>(&self, data: &mut [T]) {
        assert!(
            self.elements.iter().all(|&i| i < data.len()),
            "cycle indices must be within the data length"
        );
        for pair in self.elements.windows(2) {
            data.swap(pair[0], pair[1]);
        }
    }
}

impl std::ops::Deref for Cycle {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        &self.elements
    }
}

impl IntoIterator for Cycle {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;
    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

/// Formats the cycle in cycle notation, e.g. `(0 2 1)`.
impl std::fmt::Display for Cycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, x) in self.elements.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{x}")?;
        }
        write!(f, ")")
    }
}

/// The parity (sign) of a permutation, i.e. whether it decomposes into an even or
/// odd number of transpositions. Returned by [`Permutation::parity`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Parity {
    Even,
    Odd,
}

impl Parity {
    /// The sign of the permutation: `+1` if even, `-1` if odd.
    pub fn sign(self) -> i32 {
        match self {
            Parity::Even => 1,
            Parity::Odd => -1,
        }
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
///
/// # Reference
/// Conrado Martínez, Alois Panholzer, and Helmut Prodinger, "Generating Random
/// Derangements", *Proc. 5th Workshop on Analytic Algorithmics and Combinatorics
/// (ANALCO)*, SIAM, 2008.
/// <https://epubs.siam.org/doi/pdf/10.1137/1.9781611972986.7>
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

/// Shuffles `data` in place into a uniformly random permutation of its elements,
/// via a Fisher–Yates shuffle.
pub fn shuffle<T, R: RngExt + ?Sized>(data: &mut [T], rng: &mut R) {
    for i in (1..data.len()).rev() {
        let j = rng.random_range(0..=i);
        data.swap(i, j);
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
    shuffle(&mut permutation, rng);
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
    p.iter().all(|&x| x < p.len() && !std::mem::replace(&mut seen[x], true))
}

/// Returns `true` iff `p` is a derangement: a permutation of
/// `{0, 1, ..., p.len()-1}` with no fixed point (`p[i] != i` for all `i`).
fn is_derangement(p: &[usize]) -> bool {
    is_permutation(p) && p.iter().enumerate().all(|(i, &pi)| i != pi)
}

/// Greatest common divisor, via the Euclidean algorithm.
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
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
                assert_eq!(p.symbols(), n);
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
    fn identity_maps_each_element_to_itself() {
        let id = Permutation::identity(4);
        assert_eq!(id.to_vec(), vec![0, 1, 2, 3]);
        assert!((0..4).all(|i| id[i] == i));
        assert!(!id.is_derangement());
        // identity ∘ data == data
        assert_eq!(id.apply(&['a', 'b', 'c', 'd']), vec!['a', 'b', 'c', 'd']);
        // empty identity is valid.
        assert!(Permutation::identity(0).is_empty());
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
        assert_eq!(p.symbols(), 3);
        assert!(p.is_derangement());

        // inverse ∘ p == identity.
        let inv = p.inverse();
        for i in 0..p.symbols() {
            assert_eq!(inv[p[i]], i);
            assert_eq!(p[inv[i]], i);
        }

        // apply: out[i] = data[p[i]].
        let data = ['a', 'b', 'c'];
        assert_eq!(p.apply(&data), vec!['b', 'c', 'a']);

        // Display in one-line notation (the images in order).
        assert_eq!(p.to_string(), "[1 2 0]");
        let q = Permutation::try_new(vec![1, 0, 3, 2]).unwrap();
        assert_eq!(q.to_string(), "[1 0 3 2]");
        assert_eq!(Permutation::identity(3).to_string(), "[0 1 2]");

        // into_inner round-trips.
        assert_eq!(p.into_vec(), vec![1, 2, 0]);
    }

    #[test]
    fn cycles_decomposition() {
        // Cycles compared by their element vectors.
        let collect = |p: &Permutation| p.cycles().map(Cycle::into_vec).collect::<Vec<_>>();

        // one long cycle
        assert_eq!(collect(&Permutation::try_new(vec![1, 2, 0]).unwrap()), vec![vec![0, 1, 2]]);
        // two 2-cycles
        assert_eq!(
            collect(&Permutation::try_new(vec![1, 0, 3, 2]).unwrap()),
            vec![vec![0, 1], vec![2, 3]]
        );
        // fixed point + 2-cycle
        assert_eq!(
            collect(&Permutation::try_new(vec![0, 2, 1]).unwrap()),
            vec![vec![0], vec![1, 2]]
        );
        // identity => all singletons; empty => no cycles
        assert_eq!(collect(&Permutation::identity(3)), vec![vec![0], vec![1], vec![2]]);
        assert_eq!(Permutation::identity(0).cycles().count(), 0);

        // Cycle Display uses cycle notation; Deref gives slice access.
        let c = Permutation::try_new(vec![1, 2, 0]).unwrap().cycles().next().unwrap();
        assert_eq!(c.to_string(), "(0 1 2)");
        assert_eq!(c.len(), 3);

        // A cycle can be applied on its own, and to any slice long enough to
        // contain its indices (extra tail entries are left untouched).
        let cyc = Permutation::try_new(vec![1, 2, 0, 3]).unwrap().cycles().next().unwrap();
        assert_eq!(cyc.len(), 3);
        let mut data = ['a', 'b', 'c', 'd', 'e'];
        cyc.apply_mut(&mut data);
        assert_eq!(data, ['b', 'c', 'a', 'd', 'e']); // positions 0,1,2 rotated; 3,4 untouched

        // Cycles partition {0, ..., n-1} for a random permutation (IntoIterator).
        let p = sample_permutation_with(50, &mut rand::rng());
        let mut all: Vec<usize> = p.cycles().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, (0..50).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "data length must match permutation length")]
    fn apply_length_mismatch_panics() {
        let p = Permutation::try_new(vec![1, 2, 0]).unwrap();
        p.apply(&[1, 2]);
    }

    #[test]
    #[should_panic(expected = "cycle indices must be within the data length")]
    fn cycle_apply_mut_out_of_bounds_panics() {
        let c = Permutation::try_new(vec![1, 2, 0]).unwrap().cycles().next().unwrap();
        c.apply_mut(&mut [1, 2]); // cycle touches index 2, but data.len() == 2
    }

    #[test]
    fn compose_and_inverse() {
        let p = Permutation::try_new(vec![1, 2, 0, 3]).unwrap();
        let q = Permutation::try_new(vec![3, 0, 1, 2]).unwrap();

        // (p ∘ q)[i] == p[q[i]]
        let pq = p.compose(&q);
        for i in 0..p.symbols() {
            assert_eq!(pq[i], p[q[i]]);
        }

        // p ∘ p⁻¹ == p⁻¹ ∘ p == identity.
        let id = Permutation::identity(p.symbols());
        assert_eq!(p.compose(&p.inverse()), id);
        assert_eq!(p.inverse().compose(&p), id);

        // Composing with the identity is a no-op on either side.
        assert_eq!(p.compose(&id), p);
        assert_eq!(id.compose(&p), p);
    }

    #[test]
    #[should_panic(expected = "permutations must have the same length")]
    fn compose_length_mismatch_panics() {
        Permutation::identity(3).compose(&Permutation::identity(2));
    }

    #[test]
    fn parity_of_known_permutations() {
        let parity = |v: Vec<usize>| Permutation::try_new(v).unwrap().parity();

        assert_eq!(Permutation::identity(5).parity(), Parity::Even); // identity is even
        assert_eq!(parity(vec![1, 0, 2]), Parity::Odd); // one transposition
        assert_eq!(parity(vec![1, 2, 0]), Parity::Even); // a 3-cycle = 2 transpositions
        assert_eq!(parity(vec![1, 0, 3, 2]), Parity::Even); // two transpositions
        assert_eq!(parity(vec![]), Parity::Even); // empty is even

        assert_eq!(Parity::Even.sign(), 1);
        assert_eq!(Parity::Odd.sign(), -1);
    }

    #[test]
    fn order_is_lcm_of_cycle_lengths() {
        let order = |v: Vec<usize>| Permutation::try_new(v).unwrap().order();

        assert_eq!(Permutation::identity(5).order(), 1);
        assert_eq!(order(vec![1, 0, 2]), 2); // transposition
        assert_eq!(order(vec![1, 2, 0]), 3); // 3-cycle
        assert_eq!(order(vec![1, 0, 3, 2]), 2); // two 2-cycles: lcm(2,2) = 2
        assert_eq!(order(vec![1, 2, 0, 4, 3]), 6); // 3-cycle + 2-cycle: lcm(3,2) = 6
        assert_eq!(Permutation::identity(0).order(), 1); // empty

        // Applying p exactly order(p) times yields the identity.
        let mut rng = rand::rng();
        for _ in 0..100 {
            let p = sample_permutation_with(7, &mut rng);
            let id = Permutation::identity(7);
            let mut acc = id.clone();
            for _ in 0..p.order() {
                acc = p.compose(&acc);
            }
            assert_eq!(acc, id);
        }
    }

    #[test]
    fn is_involution_detects_self_inverse() {
        let involution = |v: Vec<usize>| Permutation::try_new(v).unwrap().is_involution();

        assert!(Permutation::identity(4).is_involution());
        assert!(involution(vec![1, 0, 2])); // single transposition
        assert!(involution(vec![1, 0, 3, 2])); // two transpositions
        assert!(!involution(vec![1, 2, 0])); // 3-cycle
        assert!(Permutation::identity(0).is_involution()); // empty, vacuously

        // Equivalent to `p == p⁻¹`.
        let mut rng = rand::rng();
        for _ in 0..200 {
            let p = sample_permutation_with(8, &mut rng);
            assert_eq!(p.is_involution(), p == p.inverse());
        }
    }

    #[test]
    fn parity_is_multiplicative() {
        // sign(p ∘ q) == sign(p) * sign(q), and inverse has the same parity.
        let mut rng = rand::rng();
        for _ in 0..200 {
            let p = sample_permutation_with(9, &mut rng);
            let q = sample_permutation_with(9, &mut rng);
            assert_eq!(
                p.compose(&q).parity().sign(),
                p.parity().sign() * q.parity().sign()
            );
            assert_eq!(p.inverse().parity(), p.parity());
        }
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
    fn shuffle_preserves_multiset() {
        let mut rng = rand::rng();
        let original: Vec<char> = ('a'..='z').collect();
        for _ in 0..1000 {
            let mut data = original.clone();
            shuffle(&mut data, &mut rng);
            let mut sorted = data.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, original, "shuffle lost or duplicated elements");
        }
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
