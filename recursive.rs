fn factorial(n: u32) -> u32 {
    if n == 0 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn main() {
    let num = 5;
    println!("The factorial of {} is {}", num, factorial(num));
}

