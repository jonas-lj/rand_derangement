# rand_combinatorics

Uniform random sampling of combinatorial structures over `{0, 1, …, n-1}`:

- **`permutation`** — random permutations, derangements, and involutions, plus a
  `Permutation` type with the usual operations (inverse, composition, cycles,
  parity, order).
- **`partition`** — uniform random *set* partitions (Stam's urn algorithm).
- **`subset`** — uniform random `k`-subsets / combinations (Floyd's algorithm).

A *derangement* is a permutation with no fixed points. Unlike some existing
crates, the derangement sampler here is provably uniform and runs in `O(n)` time
with no big-integer arithmetic and no overflow ceiling on `n`.

## Usage

Sample a random derangement or permutation:

```rust
use rand_combinatorics::permutation::Permutation;

let d = Permutation::sample_derangement(10);
assert!(d.is_derangement());

let p = Permutation::sample_permutation(10);

// An involution is its own inverse (only fixed points and 2-cycles).
let inv = Permutation::sample_involution(10);
assert!(inv.is_involution());
```

Derangements and involutions are both special cases of one sampler: a uniform
permutation whose cycle lengths satisfy an arbitrary predicate. Its per-step
probabilities come from the recurrence for how many such permutations exist,
evaluated in log-space so nothing overflows however large `n` gets.

```rust
use rand_combinatorics::permutation::Permutation;

let derangement = Permutation::sample_cycle_type(10, |k| k >= 2);   // no fixed points
let involution  = Permutation::sample_cycle_type(10, |k| k <= 2);   // fixed points + 2-cycles
let matching    = Permutation::sample_cycle_type(10, |k| k == 2);   // perfect matching (even n)
let short       = Permutation::sample_cycle_type(10, |k| k <= 3);   // cycles of length at most 3
```

`Permutation` derefs to `[usize]` (its one-line map), so slice methods and
indexing work directly, and it offers the usual group operations:

```rust
use rand_combinatorics::permutation::{Parity, Permutation};

let p = Permutation::try_new(vec![1, 2, 0, 3]).unwrap();

assert_eq!(p[0], 1);                       // Deref + indexing
assert_eq!(p.inverse().compose(&p), Permutation::identity(4));
assert_eq!(p.order(), Ok(3));              // lcm of cycle lengths
assert_eq!(p.parity(), Parity::Even);      // decomposes into 2 transpositions

// cycles: a 3-cycle and the fixed point 3.
let cycles: Vec<Vec<usize>> = p.cycles().map(|c| c.into_vec()).collect();
assert_eq!(cycles, vec![vec![0, 1, 2], vec![3]]);

let permuted = p.apply(&['a', 'b', 'c', 'd']); // out[i] = data[p[i]]
```

Derange or shuffle an arbitrary slice in place (no `Permutation` produced, no
`Clone` bound):

```rust
use rand_combinatorics::permutation::{derange, involute, shuffle};

let mut rng = rand::rng();
let mut data = ['a', 'b', 'c', 'd', 'e'];

shuffle(&mut data, &mut rng);   // uniform random permutation, in place
derange(&mut data, &mut rng);   // no element stays where it was
involute(&mut data, &mut rng);  // a random self-inverse rearrangement
```

All samplers have an `_with(…, rng)` variant that takes an explicit RNG.

Sample a uniform random set partition (one of the `Bₙ` ways to split the set into
non-empty blocks):

```rust
use rand_combinatorics::partition;

// Group a `Vec` into random non-empty blocks (elements keep their order within a block):
let blocks: Vec<Vec<i32>> = partition::partition(vec![10, 20, 30, 40, 50]);
```

For large sets you can avoid materializing the blocks: `Partition` is a lazy
iterator yielding each element's block index in order, as restricted-growth labels
(`0`, then `1`, …), so you can grow buckets on the fly.

```rust
use rand_combinatorics::partition::Partition;

let items = vec![10, 20, 30, 40, 50];
let mut rng = rand::rng();

let mut buckets: Vec<Vec<i32>> = Vec::new();
for (item, block) in items.into_iter().zip(Partition::sample_with(5, &mut rng)) {
    if block == buckets.len() {
        buckets.push(Vec::new());
    }
    buckets[block].push(item);
}
```

Sample a uniform random `k`-subset (combination), in ascending order — either as
indices, or directly from a `Vec` of elements:

```rust
use rand_combinatorics::subset;

let picks = subset::sample(52, 5); // 5 distinct indices from {0, ..., 51}
assert_eq!(picks.len(), 5);

let hand = subset::subset(vec!['A', 'K', 'Q', 'J', 'T', '9', '8', '7'], 5);
assert_eq!(hand.len(), 5); // 5 of the given cards, in their original order
```

## Development

```
cargo test                              # correctness + uniformity checks
cargo bench                             # criterion sampling benchmark
cargo run --release --example golomb_dickman   # estimates the Golomb–Dickman constant
cargo run --release --example derangement_ratio  # fraction of permutations that are derangements -> 1/e
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
