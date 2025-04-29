use std::arch::x86_64::_rdtsc;

fn main() {
    let start = unsafe { _rdtsc() };
    a();
    let end = unsafe { _rdtsc() };
    println!("{}", end - start);
}

fn a() {
    let a = 1;
    let b = 2;
    let _c = a + b;
}