# derangements

Sampling of uniformly random derangements (permutations with no fixed points).

This is a variant of the Martínez–Panholzer–Prodinger algorithm. At each step,
while `u` elements remain unmarked, the current element is "closed" with
probability

```
p(u) = d[u-1] / (d[u-1] + d[u])
```

where `d[k] = !k` is the number of derangements of `k` elements (the
subfactorial). This is the original `u * d[u-1] / d[u+1]` simplified with the
recursion `d[u+1] = u * (d[u] + d[u-1])`.

The probabilities are precomputed once, in `f64`, with the stable recursion

```
p(1) = 1,   p(u) = (1 - p(u-1)) / (u - p(u-1))
```

and fed to a plain Bernoulli trial (`random_bool`). This never forms the
subfactorials themselves, so there are no big integers and no overflow for any
`n`.

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
