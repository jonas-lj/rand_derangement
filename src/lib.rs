//! Sampling of uniformly random derangements.
//!
//! A derangement is a permutation with no fixed points. This is a variant of
//! the Martínez–Panholzer–Prodinger algorithm that avoids index-selection
//! rejection sampling.
//!
//! The only randomness that depends on the subfactorials is a single Bernoulli
//! trial per step: while `u` elements are still unmarked, we "close" the current
//! element with probability
//!
//! ```text
//! p(u) = d[u-1] / (d[u-1] + d[u]),
//! ```
//!
//! where `d[k]` is the number of derangements of `k` elements (the subfactorial
//! `!k = round(k!/e)`). This is the same probability as the original
//! `u * d[u-1] / d[u+1]`, simplified with the recursion `d[u+1] = u*(d[u] + d[u-1])`.
//!
//! We evaluate that Bernoulli trial with **integer arithmetic only** — no floating
//! point — in two regimes:
//!
//! * For small `u` we draw a uniform integer and compare, which is exact.
//! * For large `u` we use the identity `d[u] = u*d[u-1] + (-1)^u`, giving
//!   `p(u) = d[u-1] / ((u+1)*d[u-1] + (-1)^u)`, which differs from `1/(u+1)` by
//!   less than `1/d[u+1]`. By `u = 20` that gap is below `2^-64`, i.e. below the
//!   resolution of the random number generator, so sampling `1/(u+1)` is exact
//!   in practice.

use rand::RngExt;

/// Subfactorials `d[k] = !k` for `k = 0..=20`. All fit in a `u64`
/// (`d[20] ≈ 8.95e17 < 2^63`), which is exactly the range in which we still
/// sample the Bernoulli trial exactly.
const SUBFACTORIALS: [u64; 21] = [
    1,
    0,
    1,
    2,
    9,
    44,
    265,
    1854,
    14833,
    133496,
    1334961,
    14684570,
    176214841,
    2290792932,
    32071101049,
    481066515734,
    7697064251745,
    130850092279664,
    2355301661033953,
    44750731559645106,
    895014631192902121,
];

/// Largest `u` for which we still sample `p(u)` exactly from the table above.
/// Beyond this, `p(u)` and `1/(u+1)` differ by less than `2^-64`.
const EXACT_THRESHOLD: usize = 20;

/// Returns `true` with probability `p(u) = d[u-1] / (d[u-1] + d[u])`, using
/// integer arithmetic only.
fn should_mark<R: RngExt + ?Sized>(rng: &mut R, u: usize) -> bool {
    if u <= EXACT_THRESHOLD {
        // Exact integer Bernoulli trial: accept the first d[u-1] of the
        // d[u-1] + d[u] equally likely outcomes.
        let lo = SUBFACTORIALS[u - 1];
        let hi = SUBFACTORIALS[u]; // u <= 20  =>  u + 1 <= 21, in bounds
        rng.random_range(0..lo + hi) < lo
    } else {
        // p(u) is within 1/d[u+1] < 2^-64 of 1/(u+1), below RNG resolution.
        rng.random_range(0..=u) == 0
    }
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

    let mut unmarked = (0..n).collect::<Vec<usize>>();

    let mut u = n - 1;
    while u > 0 {
        let i = unmarked.pop().unwrap();
        let j = rng.random_range(0..unmarked.len());
        permutation.swap(i, unmarked[j]);

        if should_mark(rng, u) {
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
