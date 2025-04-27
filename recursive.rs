fn factorial(n: u32) -> u32 {
    if n == 0 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn main() {
    let num = 5;
    let result = factorial(num);
    println!("The factorial of {} is {}", num, result);
}

