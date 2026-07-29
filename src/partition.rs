//! Uniform random *set* partitions of `{0, 1, ..., n-1}`.
//!
//! [`Partition`] is a lazy iterator. Sampling — [`Partition::sample_with`] — draws
//! a partition uniformly at random (one of the `B_n` Bell-number many), and
//! iterating it yields the block index of each element `0, 1, ..., n-1` in order.
//!
//! Indices are assigned in restricted-growth form: a new block appears only when
//! its first element does, so they come out contiguous (`0`, then `1`, ...), every
//! block is non-empty, and nothing is materialized. Group as you like by collecting
//! or scattering into buckets on the fly.
//!
//! # Algorithm
//! Stam's urn method: pick a number of colors `k` with probability proportional to
//! `k^n / k!` (the distribution behind Dobinski's formula
//! `B_n = (1/e) * sum_{k>=0} k^n / k!`), then color the `n` elements independently
//! and uniformly with one of `k` colors; the blocks are the color classes. Each
//! partition then arises with probability exactly `1 / B_n`.
//!
//! # Reference
//! A. J. Stam, "Generation of a random partition of a finite set by an urn model",
//! *J. Combin. Theory Ser. A* 35 (1983); see also Knuth, *TAOCP* Vol. 4A, and
//! <https://djalil.chafai.net/blog/2012/05/03/generating-uniform-random-partitions/>.

use rand::RngExt;
use std::iter::FusedIterator;

/// A uniformly random set partition of `{0, 1, ..., n-1}`, produced lazily.
///
/// Iterating yields the block index of each element in order, as restricted-growth
/// labels (contiguous from `0`, one new label per new block). Nothing is stored per
/// element — only the color-to-block relabeling map, of size `k` (the sampled color
/// count). See the [module docs](self) for the algorithm.
pub struct Partition<'a, R: ?Sized> {
    rng: &'a mut R,
    /// Elements still to be colored.
    remaining: usize,
    /// Number of colors `k` sampled up front.
    colors: usize,
    /// `label[color]` is the block index assigned to `color`, or `usize::MAX` until
    /// the color is first used.
    label: Vec<usize>,
    /// Next block index to hand out.
    next_block: usize,
}

impl<'a, R: RngExt + ?Sized> Partition<'a, R> {
    /// Samples a uniformly random partition of `{0, 1, ..., n-1}` using the given
    /// random number generator, via Stam's urn algorithm.
    pub fn sample_with(n: usize, rng: &'a mut R) -> Partition<'a, R> {
        if n == 0 {
            return Partition {
                rng,
                remaining: 0,
                colors: 0,
                label: Vec::new(),
                next_block: 0,
            };
        }
        let colors = sample_color_count(n, rng);
        Partition {
            rng,
            remaining: n,
            colors,
            label: vec![usize::MAX; colors],
            next_block: 0,
        }
    }
}

impl<R: RngExt + ?Sized> Iterator for Partition<'_, R> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        // Color this element, minting a fresh block index the first time a color
        // is used (keeping the labels in restricted-growth form).
        let color = self.rng.random_range(0..self.colors);
        if self.label[color] == usize::MAX {
            self.label[color] = self.next_block;
            self.next_block += 1;
        }
        Some(self.label[color])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

// The number of remaining elements is known exactly, even though the block count
// is not known until iteration finishes.
impl<R: RngExt + ?Sized> ExactSizeIterator for Partition<'_, R> {}
impl<R: RngExt + ?Sized> FusedIterator for Partition<'_, R> {}

/// Randomly partitions `elements` into non-empty blocks, uniformly over all set
/// partitions, consuming the input. Each element keeps its position within its
/// block; an empty input yields no blocks.
pub fn partition<T>(elements: Vec<T>) -> Vec<Vec<T>> {
    partition_with(elements, &mut rand::rng())
}

/// Randomly partitions `elements` into non-empty blocks, uniformly over all set
/// partitions, using the given random number generator and consuming the input.
/// Each element keeps its position within its block; an empty input yields no blocks.
pub fn partition_with<T, R: RngExt + ?Sized>(elements: Vec<T>, rng: &mut R) -> Vec<Vec<T>> {
    let n = elements.len();
    let mut blocks: Vec<Vec<T>> = Vec::new();
    for (element, block) in elements.into_iter().zip(Partition::sample_with(n, rng)) {
        // Block indices arrive in restricted-growth order, so a new block is always
        // exactly `blocks.len()`.
        if block == blocks.len() {
            blocks.push(Vec::new());
        }
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

    #[test]
    fn samples_are_valid_partitions() {
        let mut rng = rand::rng();
        for n in [0usize, 1, 2, 3, 5, 10, 30] {
            for _ in 0..300 {
                let assignment: Vec<usize> = Partition::sample_with(n, &mut rng).collect();
                assert_eq!(assignment.len(), n);

                // Restricted-growth invariant: `block[0] == 0` and each label is at
                // most one past the running max, giving contiguous block indices.
                let mut max = 0usize;
                for (i, &b) in assignment.iter().enumerate() {
                    let bound = if i == 0 { 0 } else { max + 1 };
                    assert!(
                        b <= bound,
                        "not restricted-growth for n = {n}: {assignment:?}"
                    );
                    max = max.max(b);
                }

                // Every block index `0..num_blocks` appears, so all blocks are non-empty.
                let num_blocks = if n == 0 { 0 } else { max + 1 };
                let mut used = vec![false; num_blocks];
                for &b in &assignment {
                    used[b] = true;
                }
                assert!(
                    used.iter().all(|&u| u),
                    "empty block for n = {n}: {assignment:?}"
                );
            }
        }
    }

    #[test]
    fn edge_cases() {
        let mut rng = rand::rng();
        // Empty set: no elements, no blocks.
        assert_eq!(Partition::sample_with(0, &mut rng).count(), 0);
        // Singleton: one element in block 0.
        assert_eq!(
            Partition::sample_with(1, &mut rng).collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn scatters_into_non_empty_buckets() {
        let mut rng = rand::rng();
        let items = vec![10, 20, 30, 40, 50];

        // Grow buckets on the fly: a new block index is always exactly `buckets.len()`.
        let mut buckets: Vec<Vec<i32>> = Vec::new();
        for (item, block) in items
            .clone()
            .into_iter()
            .zip(Partition::sample_with(items.len(), &mut rng))
        {
            if block == buckets.len() {
                buckets.push(Vec::new());
            }
            buckets[block].push(item);
        }

        assert!(buckets.iter().all(|b| !b.is_empty()));
        let mut all: Vec<i32> = buckets.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, items);
    }

    #[test]
    fn partition_groups_all() {
        let mut rng = rand::rng();
        for n in [0usize, 1, 2, 5, 12] {
            let elements: Vec<usize> = (0..n).map(|i| i * 10).collect();
            let blocks = partition_with(elements.clone(), &mut rng);

            // Non-empty blocks; there are blocks iff there are elements.
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

    /// The 5 partitions of a 3-element set (`B_3 = 5`) should be equiprobable.
    /// In restricted-growth form they are the strings [000], [001], [010], [011], [012].
    #[test]
    fn uniform_for_n3() {
        let mut rng = rand::rng();
        let mut counts: HashMap<Vec<usize>, u32> = HashMap::new();
        let trials = 600_000;
        for _ in 0..trials {
            let assignment: Vec<usize> = Partition::sample_with(3, &mut rng).collect();
            *counts.entry(assignment).or_default() += 1;
        }

        assert_eq!(counts.len(), 5, "expected all five partitions of a 3-set");
        let expected = 1.0 / 5.0;
        for (a, &c) in &counts {
            let freq = c as f64 / trials as f64;
            assert!(
                (freq - expected).abs() < 0.01,
                "partition {a:?} had frequency {freq}"
            );
        }
    }
}
