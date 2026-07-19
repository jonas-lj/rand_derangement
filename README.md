# rand_derangement

Fast, **uniformly** random derangements and permutations of `{0, 1, …, n-1}`,
plus a small `Permutation` type with the usual operations.

A *derangement* is a permutation with no fixed points. Unlike some existing
crates, the sampler here is provably uniform and runs in `O(n)` time with no
big-integer arithmetic and no overflow ceiling on `n`.

## Usage

Sample a random derangement or permutation:

```rust
use rand_derangement::Permutation;

let d = Permutation::sample_derangement(10);
assert!(d.is_derangement());

let p = Permutation::sample_permutation(10);
```

`Permutation` derefs to `[usize]` (its one-line map), so slice methods and
indexing work directly, and it offers the usual group operations:

```rust
use rand_derangement::{Parity, Permutation};

let p = Permutation::try_new(vec![1, 2, 0, 3]).unwrap();

assert_eq!(p[0], 1);                       // Deref + indexing
assert_eq!(p.inverse().compose(&p), Permutation::identity(4));
assert_eq!(p.order(), Ok(3));              // lcm of cycle lengths
assert_eq!(p.parity(), Parity::Even);      // decomposes into 2 transpositions

// cycles: a 3-cycle and the fixed point 3, in cycle notation.
let cycles: String = p.cycles().iter().map(|c| c.to_string()).collect();
assert_eq!(cycles, "(0 1 2)(3)");

assert_eq!(format!("{p}"), "[1 2 0 3]");   // one-line notation

let permuted = p.apply(&['a', 'b', 'c', 'd']); // out[i] = data[p[i]]
```

Derange or shuffle an arbitrary slice in place (no `Permutation` produced, no
`Clone` bound):

```rust
use rand_derangement::{derange, shuffle};

let mut rng = rand::rng();
let mut data = ['a', 'b', 'c', 'd', 'e'];

shuffle(&mut data, &mut rng);   // uniform random permutation, in place
derange(&mut data, &mut rng);   // no element stays where it was
```

All samplers have an `_with(…, rng)` variant that takes an explicit RNG.

## Development

```
cargo test                              # correctness + uniformity checks
cargo bench                             # criterion sampling benchmark
cargo run --release --example golomb_dickman   # estimates the Golomb–Dickman constant
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
