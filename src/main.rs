use derangements::sample_derangement;

fn main() {
    for n in [5, 10, 20, 50] {
        let d = sample_derangement(n);
        let fixed_points = d.iter().enumerate().filter(|&(i, &pi)| i == pi).count();
        println!("n = {n:>3}  fixed points = {fixed_points}  {d:?}");
    }
}
