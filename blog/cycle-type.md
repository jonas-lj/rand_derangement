## Sampling permutations by cycle type

In an [earlier post](https://www.jonaslindstrom.dk/?p=1328) I described a way to sample a uniformly random *derangement* — a permutation with no fixed points — in linear time and without any big-integer arithmetic. Since then I have added samplers for a couple of neighbouring objects: *involutions*, which are permutations equal to their own inverse, and perfect matchings, which are involutions with no fixed points either. Writing the third one it dawned on me that they are all the same algorithm.

The thing they have in common is the cycle structure. A derangement is a permutation whose cycles all have length [latex]\geq 2[/latex]; an involution has cycles of length [latex]\leq 2[/latex]; a matching has cycles of length exactly [latex]2[/latex]. In every case we pick a set [latex]S[/latex] of permitted cycle lengths and ask for a uniformly random permutation built from those alone.

Counting such permutations is a single recurrence. Let [latex]a_n[/latex] be the number of permutations of [latex]n[/latex] elements whose cycle lengths all lie in [latex]S[/latex]. The largest element must sit in a cycle of some length [latex]k \in S[/latex]; there are [latex](n-1)(n-2)\cdots(n-k+1)[/latex] ways to line up its [latex]k-1[/latex] cycle-mates in order, and the remaining elements form a smaller instance of the same problem:

[latex]a_n = \sum_{k \in S} (n-1)(n-2)\cdots(n-k+1)\, a_{n-k}.[/latex]

For [latex]S = \{2, 3, \dots\}[/latex] this is the derangement recurrence from last time (the algorithm is due to Martínez, Panholzer and Prodinger); for [latex]S = \{1, 2\}[/latex] it collapses to [latex]a_n = a_{n-1} + (n-1)a_{n-2}[/latex], the involution numbers.

The recurrence is also the sampler. Repeatedly take the largest remaining element and give it a cycle of length [latex]k[/latex] with probability proportional to the [latex]k[/latex]-th term above, then draw its [latex]k-1[/latex] partners uniformly at random. Each valid permutation then turns up with probability exactly [latex]1/a_n[/latex].

The only nuisance is that [latex]a_n[/latex] grows like [latex]n![/latex], so forming those weights directly overflows almost immediately. As before I sidestep this by never building the numbers at all — here I keep everything in log-space and normalise, so no big integers are needed however large [latex]n[/latex] gets.

```rust
/// Sample a uniform permutation of {0, ..., n-1} whose cycle lengths all satisfy `allowed`.
pub fn sample_cycle_type_with<R: RngExt + ?Sized, F: Fn(usize) -> bool>(
    n: usize,
    allowed: F,
    rng: &mut R,
) -> Permutation {
    // ln of the count a[m] for every m <= n, from the recurrence above (kept in log-space).
    let log_count = log_cycle_type_counts(n, &allowed);

    let mut permutation = (0..n).collect::<Vec<usize>>();
    let mut pool = (0..n).collect::<Vec<usize>>();
    while !pool.is_empty() {
        // Give the largest remaining element a cycle of length k, chosen with
        // probability proportional to (r-1)(r-2)...(r-k+1) * a[r-k], then splice in
        // its k-1 random partners with the same swap-chain trick used for derangements.
        let k = sample_cycle_length(pool.len(), &allowed, &log_count, rng);
        let mut prev = pool.pop().unwrap();
        for _ in 1..k {
            let next = pool.swap_remove(rng.random_range(..pool.len()));
            permutation.swap(prev, next);
            prev = next;
        }
    }
    Permutation(permutation)
}
```

The two helpers — one for the log-counts, one for sampling a cycle length — are a handful of lines each; the whole thing lives in my [`rand_combinatorics`](https://github.com/jonas-lj/rand_combinatorics) crate. The special cases keep their own dedicated linear-time samplers, but it is quite satisfying that a single recurrence covers the entire family.
