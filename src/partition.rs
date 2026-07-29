//! Uniform random *set* partitions of `{0, 1, ..., n-1}`.
//!
//! A partition splits the set into non-empty, disjoint blocks; there are `B_n`
//! (the Bell number) of them, and [`Partition::sample`] draws one uniformly.
//!
//! # Algorithm
//! Stam's urn method: pick a number of colors `k` with probability proportional
//! to `k^n / k!` (the distribution behind Dobinski's formula
//! `B_n = (1/e) * sum_{k>=0} k^n / k!`), color the `n` elements independently and
//! uniformly with one of `k` colors, and let the blocks be the color classes.
//! Each partition then arises with probability exactly `1 / B_n`.
//!
//! # Reference
//! A. J. Stam, "Generation of a random partition of a finite set by an urn model",
//! *J. Combin. Theory Ser. A* 35 (1983); see also Knuth, *TAOCP* Vol. 4A, and
//! <https://djalil.chafai.net/blog/2012/05/03/generating-uniform-random-partitions/>.

use rand::RngExt;

/// A partition of the set `{0, 1, ..., n-1}` into non-empty blocks.
///
/// Stored in restricted-growth form: [`assignment`](Partition::assignment)`[i]` is
/// the block index of element `i`, with blocks numbered `0, 1, ...` in order of
/// their smallest element. This form is canonical, so two `Partition`s are equal
/// iff they describe the same partition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Partition {
    block: Vec<usize>,
    blocks: usize,
}

impl Partition {
    /// Samples a uniformly random partition of `{0, 1, ..., n-1}`.
    pub fn sample(n: usize) -> Partition {
        Self::sample_with(n, &mut rand::rng())
    }

    /// Samples a uniformly random partition of `{0, 1, ..., n-1}` using the given
    /// random number generator, via Stam's urn algorithm.
    pub fn sample_with<R: RngExt + ?Sized>(n: usize, rng: &mut R) -> Partition {
        if n == 0 {
            return Partition {
                block: Vec::new(),
                blocks: 0,
            };
        }
        let colors = sample_color_count(n, rng);

        // Color each element uniformly in `0..colors`, relabeling to restricted-
        // growth form (blocks numbered by first appearance) in a single pass.
        let mut label = vec![usize::MAX; colors];
        let mut block = Vec::with_capacity(n);
        let mut blocks = 0;
        for _ in 0..n {
            let color = rng.random_range(0..colors);
            if label[color] == usize::MAX {
                label[color] = blocks;
                blocks += 1;
            }
            block.push(label[color]);
        }
        Partition { block, blocks }
    }

    /// The number of elements, `n`.
    pub fn size(&self) -> usize {
        self.block.len()
    }

    /// Returns `true` iff there are no elements (`n == 0`).
    pub fn is_empty(&self) -> bool {
        self.block.is_empty()
    }

    /// The number of (non-empty) blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks
    }

    /// The restricted-growth assignment: `assignment()[i]` is the block index of
    /// element `i`, with blocks numbered in order of their smallest element.
    pub fn assignment(&self) -> &[usize] {
        &self.block
    }

    /// Materializes the blocks, each as the ascending list of its elements.
    pub fn blocks(&self) -> Vec<Vec<usize>> {
        let mut result = vec![Vec::new(); self.blocks];
        for (element, &b) in self.block.iter().enumerate() {
            result[b].push(element);
        }
        result
    }
}

/// Iterating a `Partition` yields the block index of each element `0, 1, ..., n-1`
/// in order (its [`assignment`](Partition::assignment)). Together with
/// [`num_blocks`](Partition::num_blocks) — known up front — this lets you scatter a
/// set into pre-created buckets without ever materializing the blocks.
impl IntoIterator for Partition {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;
    fn into_iter(self) -> Self::IntoIter {
        self.block.into_iter()
    }
}

impl<'a> IntoIterator for &'a Partition {
    type Item = usize;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, usize>>;
    fn into_iter(self) -> Self::IntoIter {
        self.block.iter().copied()
    }
}

/// Randomly partitions `elements` into non-empty blocks, uniformly over all set
/// partitions of them, consuming the input.
///
/// Equivalent to sampling a [`Partition`] of `{0, ..., elements.len()-1}` and
/// grouping the elements by its blocks; each element keeps its position within its
/// block. Returns no blocks for an empty input.
pub fn partition_elements<T, R: RngExt + ?Sized>(elements: Vec<T>, rng: &mut R) -> Vec<Vec<T>> {
    let partition = Partition::sample_with(elements.len(), rng);
    let mut blocks: Vec<Vec<T>> = (0..partition.num_blocks()).map(|_| Vec::new()).collect();
    for (element, &block) in elements.into_iter().zip(partition.assignment()) {
        blocks[block].push(element);
    }
    blocks
}

/// Samples the number of colors `k >= 1` with probability proportional to
/// `k^n / k!` (Stam's distribution). Works in log-space and stops once the weight
/// has fallen negligibly below the peak, so the truncated tail is far below the
/// resolution of `f64`.
fn sample_color_count<R: RngExt + ?Sized>(n: usize, rng: &mut R) -> usize {
    debug_assert!(n >= 1);
    let n = n as f64;

    // log_weight(k) = n*ln(k) - ln(k!); accumulate ln(k!) incrementally.
    let mut log_weights = Vec::new();
    let mut ln_factorial = 0.0; // ln(1!) = 0
    let mut max = f64::NEG_INFINITY;
    let mut k = 1usize;
    loop {
        let log_weight = n * (k as f64).ln() - ln_factorial;
        log_weights.push(log_weight);
        max = max.max(log_weight);
        // The mode is below k = n, and beyond it the weight decays super-
        // exponentially; stop once it is negligible relative to the peak.
        if k as f64 > n && log_weight < max - 50.0 {
            break;
        }
        k += 1;
        ln_factorial += (k as f64).ln();
    }

    // Sample k proportional to exp(log_weight - max).
    let total: f64 = log_weights.iter().map(|&w| (w - max).exp()).sum();
    let mut u = rng.random_range(0.0..total);
    for (i, &w) in log_weights.iter().enumerate() {
        u -= (w - max).exp();
        if u < 0.0 {
            return i + 1;
        }
    }
    log_weights.len() // numerical fallback (essentially unreachable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Checks the restricted-growth invariant and internal consistency.
    fn is_valid(p: &Partition) -> bool {
        let a = p.assignment();
        let mut max_seen = 0usize;
        for (i, &b) in a.iter().enumerate() {
            let bound = if i == 0 { 0 } else { max_seen + 1 };
            if b > bound {
                return false;
            }
            max_seen = max_seen.max(b);
        }
        let expected_blocks = if a.is_empty() { 0 } else { max_seen + 1 };
        p.num_blocks() == expected_blocks
    }

    #[test]
    fn samples_are_valid_partitions() {
        let mut rng = rand::rng();
        for n in [0usize, 1, 2, 3, 5, 10, 30] {
            for _ in 0..300 {
                let p = Partition::sample_with(n, &mut rng);
                assert_eq!(p.size(), n);
                assert!(is_valid(&p), "invalid partition for n = {n}: {p:?}");

                // Blocks are non-empty and partition {0, ..., n-1} exactly.
                let blocks = p.blocks();
                assert_eq!(blocks.len(), p.num_blocks());
                assert!(blocks.iter().all(|b| !b.is_empty()));
                let mut all: Vec<usize> = blocks.into_iter().flatten().collect();
                all.sort_unstable();
                assert_eq!(all, (0..n).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn edge_cases() {
        let mut rng = rand::rng();

        let empty = Partition::sample_with(0, &mut rng);
        assert_eq!(empty.size(), 0);
        assert_eq!(empty.num_blocks(), 0);
        assert!(empty.is_empty());

        // {0} has a single partition: one block.
        let single = Partition::sample_with(1, &mut rng);
        assert_eq!(single.num_blocks(), 1);
        assert_eq!(single.blocks(), vec![vec![0]]);
    }

    #[test]
    fn partition_elements_groups_all() {
        let mut rng = rand::rng();
        for n in [0usize, 1, 2, 5, 12] {
            let elements: Vec<usize> = (0..n).map(|i| i * 10).collect();
            let blocks = partition_elements(elements.clone(), &mut rng);

            // No empty blocks; blocks exist iff there are elements.
            assert!(
                blocks.iter().all(|b| !b.is_empty()),
                "empty block for n = {n}"
            );
            assert_eq!(blocks.is_empty(), n == 0);

            // Every element appears exactly once, across all blocks.
            let mut all: Vec<usize> = blocks.into_iter().flatten().collect();
            all.sort_unstable();
            assert_eq!(all, elements);
        }
    }

    #[test]
    fn partition_iterates_block_indices() {
        let mut rng = rand::rng();
        for n in [0usize, 1, 4, 20] {
            let p = Partition::sample_with(n, &mut rng);

            // By-ref iteration yields the assignment and leaves `p` usable.
            let via_iter: Vec<usize> = (&p).into_iter().collect();
            assert_eq!(via_iter, p.assignment());

            // Scatter element indices into `num_blocks` pre-created buckets;
            // this reproduces `blocks()`.
            let mut buckets = vec![Vec::new(); p.num_blocks()];
            for (element, block) in (0..n).zip(&p) {
                buckets[block].push(element);
            }
            assert_eq!(buckets, p.blocks());

            // By-value iteration consumes and yields the same sequence.
            assert_eq!(p.into_iter().collect::<Vec<_>>(), via_iter);
        }
    }

    /// The 5 partitions of a 3-element set (`B_3 = 5`) should be equiprobable.
    #[test]
    fn uniform_for_n3() {
        let mut rng = rand::rng();
        let mut counts: HashMap<Partition, u32> = HashMap::new();
        let trials = 600_000;
        for _ in 0..trials {
            *counts
                .entry(Partition::sample_with(3, &mut rng))
                .or_default() += 1;
        }

        assert_eq!(counts.len(), 5, "expected all five partitions of a 3-set");
        let expected = 1.0 / 5.0;
        for (p, &c) in &counts {
            let freq = c as f64 / trials as f64;
            assert!(
                (freq - expected).abs() < 0.01,
                "partition {p:?} had frequency {freq}"
            );
        }
    }
}
