//! Uniform random sampling of combinatorial structures.
//!
//! - [`permutation`] — random permutations and derangements of `{0, ..., n-1}`,
//!   plus a [`Permutation`](permutation::Permutation) type with the usual
//!   operations (inverse, composition, cycles, parity, order).
//! - [`partition`] — uniform random set partitions of `{0, ..., n-1}` via Stam's
//!   urn algorithm, produced lazily as a per-element block-index iterator.
//! - [`subset`] — uniform random `k`-subsets (combinations) of `{0, ..., n-1}`.

pub mod partition;
pub mod permutation;
pub mod subset;
