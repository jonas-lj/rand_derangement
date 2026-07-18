# derangements

Sampling of uniformly random derangements (permutations with no fixed points),
using **integer arithmetic only** for the probabilistic decisions.

This is a variant of the Martínez–Panholzer–Prodinger algorithm. At each step,
while `u` elements remain unmarked, the current element is "closed" with
probability

```
p(u) = d[u-1] / (d[u-1] + d[u])
```

where `d[k] = !k` is the number of derangements of `k` elements (the
subfactorial). This is the original `u * d[u-1] / d[u+1]` simplified with the
recursion `d[u+1] = u * (d[u] + d[u-1])`.

Rather than evaluate that probability in floating point, the Bernoulli trial is
done with integers:

- **Small `u` (`u <= 20`):** draw a uniform integer in `0..d[u-1] + d[u]` and
  accept if it is `< d[u-1]`. Exact, and the subfactorials still fit in a `u64`.
- **Large `u`:** using `d[u] = u * d[u-1] + (-1)^u`, we get
  `p(u) = d[u-1] / ((u+1) * d[u-1] + (-1)^u)`, which differs from `1/(u+1)` by
  less than `1/d[u+1]`. By `u = 20` that gap is below `2^-64`, i.e. below the
  resolution of the RNG, so sampling `1/(u+1)` (a plain `random_range(0..=u) == 0`)
  is exact in practice.

The result: no floating point, no big integers, no overflow for any `n`, and a
tiny fixed lookup table.

## Usage

```rust
use derangements::sample_derangement;

let d = sample_derangement(10);
assert!(d.iter().enumerate().all(|(i, &pi)| i != pi));
```

## Development

```
cargo test    # correctness + uniformity checks
cargo run     # small demo
```
