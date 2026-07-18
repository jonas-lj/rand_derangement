# rand_derangement

Fast, uniformly random sampling of derangements (permutations with no fixed points).

This is a variant of the Martínez–Panholzer–Prodinger algorithm. With `u + 1`
elements still in play, the element being placed either forms a **2-cycle** (a
transposition with its partner) or splices into a longer cycle. It forms a
2-cycle with probability

```
two_cycle(u) = d[u-1] / (d[u-1] + d[u])
```

where `d[k] = !k` is the number of derangements of `k` elements (the
subfactorial). This follows from the recurrence `d[u+1] = u * (d[u] + d[u-1])`,
whose two terms count exactly the 2-cycle and longer-cycle cases — it is the
original `u * d[u-1] / d[u+1]` in simplified form. (This is the general
counting-to-sampling / self-reducibility pattern: each branch is taken with
probability proportional to the number of derangements that complete it.)

The probabilities are precomputed once, in `f64`, with the stable recursion

```
two_cycle(1) = 1,   two_cycle(u) = (1 - two_cycle(u-1)) / (u - two_cycle(u-1))
```

and fed to a plain Bernoulli trial (`random_bool`). This never forms the
subfactorials themselves, so there are no big integers and no overflow for any
`n`. The recursion is only stable in this direction (increasing `u`); it is the
minimal solution of the underlying linear recurrence, so it must be built up
front rather than generated downward inside the loop.

## Usage

```rust
use rand_derangement::sample_derangement;

let d = sample_derangement(10);
assert!(d.iter().enumerate().all(|(i, &pi)| i != pi));
```

## Development

```
cargo test    # correctness + uniformity checks
cargo run     # small demo
```
