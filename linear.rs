fn main() {
    println!("Starting...");
    a();
    d(); // Call d directly
    println!("Finished.");
}

fn a() {
    println!("In a");
    let mut x = 1;
    x = x + 1;
    println!("{}", x);
    b();
}

fn b() {
    println!("In b");
    c();
}

fn c() {
    println!("In c");
    // No further calls within this path
}

fn d() {
    println!("In d");
    // Another separate function
}

// Example of recursion (will be detected as a cycle)
/*
fn recursive_a() {
    println!("Recursing A");
    recursive_b();
}

fn recursive_b() {
    println!("Recursing B");
    recursive_a(); // Cycle!
}
*/
