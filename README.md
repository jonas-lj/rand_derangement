# rand_combinatorics

Uniform random sampling of combinatorial structures over `{0, 1, …, n-1}`:

- **`permutation`** — random permutations and derangements, plus a `Permutation`
  type with the usual operations (inverse, composition, cycles, parity, order).
- **`partition`** — uniform random *set* partitions (Stam's urn algorithm).

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
use rand_combinatorics::permutation::{derange, shuffle};

let mut rng = rand::rng();
let mut data = ['a', 'b', 'c', 'd', 'e'];

shuffle(&mut data, &mut rng);   // uniform random permutation, in place
derange(&mut data, &mut rng);   // no element stays where it was
```

All samplers have an `_with(…, rng)` variant that takes an explicit RNG.

Sample a uniform random set partition (one of the `Bₙ` ways to split the set into
non-empty blocks):

```rust
use rand_combinatorics::partition::Partition;

let p = Partition::sample(5);
println!("{} blocks: {:?}", p.num_blocks(), p.blocks());
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
