fn main() {
    // Arithmetic operations
    let a = 10;
    let b = 3;
    println!("Addition: {}", a + b);
    println!("Subtraction: {}", a - b);
    println!("Multiplication: {}", a * b);
    println!("Division: {}", a / b);
    println!("Remainder: {}", a % b);

    // Bitwise operations
    println!("Bitwise AND: {}", a & b);
    println!("Bitwise OR: {}", a | b);
    println!("Bitwise XOR: {}", a ^ b);
    println!("Bitwise NOT: {}", !a);
    println!("Left shift: {}", a << 2);
    println!("Right shift: {}", a >> 1);

    // Logical operations
    let t = true;
    let f = false;
    println!("Logical AND: {}", t && f);
    println!("Logical OR: {}", t || f);
    println!("Logical NOT: {}", !t);

    // Comparison operations
    println!("Equal: {}", a == b);
    println!("Not equal: {}", a != b);
    println!("Greater than: {}", a > b);
    println!("Less than: {}", a < b);
    println!("Greater or equal: {}", a >= b);
    println!("Less or equal: {}", a <= b);

    // Assignment operations
    let mut x = 5;
    x += 2;
    println!("x += 2: {}", x);
    x -= 1;
    println!("x -= 1: {}", x);
    x *= 3;
    println!("x *= 3: {}", x);
    x /= 2;
    println!("x /= 2: {}", x);
    x %= 2;
    println!("x %= 2: {}", x);
    x &= 1;
    println!("x &= 1: {}", x);
    x |= 2;
    println!("x |= 2: {}", x);
    x ^= 3;
    println!("x ^= 3: {}", x);
    x <<= 1;
    println!("x <<= 1: {}", x);
    x >>= 1;
    println!("x >>= 1: {}", x);

    // Range operations
    for i in 0..3 {
        println!("Range (exclusive): {}", i);
    }
    for i in 0..=3 {
        println!("Range (inclusive): {}", i);
    }

    // Tuple operations
    let tup = (1, "hello", 4.5);
    println!("Tuple: {:?}", tup);
    let (x, y, z) = tup;
    println!("Destructured tuple: {}, {}, {}", x, y, z);

    // Array operations
    let arr = [1, 2, 3];
    println!("Array: {:?}", arr);
    println!("Array element: {}", arr[1]);

    // Vector operations
    let mut vec = vec![1, 2, 3];
    vec.push(4);
    println!("Vector: {:?}", vec);

    // Struct operations
    struct Point {
        x: i32,
        y: i32,
    }
    let p = Point { x: 1, y: 2 };
    println!("Struct: ({}, {})", p.x, p.y);

    // Enum operations
    enum Color {
        Red,
        Green,
        Blue,
    }
    let c = Color::Green;
    match c {
        Color::Red => println!("Color is Red"),
        Color::Green => println!("Color is Green"),
        Color::Blue => println!("Color is Blue"),
    }

    // Option and Result operations
    let some_val: Option<i32> = Some(5);
    let none_val: Option<i32> = None;
    println!("Option Some: {:?}", some_val);
    println!("Option None: {:?}", none_val);

    let ok_val: Result<i32, &str> = Ok(10);
    let err_val: Result<i32, &str> = Err("error");
    println!("Result Ok: {:?}", ok_val);
    println!("Result Err: {:?}", err_val);

    // Function pointer and closure
    fn add(a: i32, b: i32) -> i32 { a + b }
    let f: fn(i32, i32) -> i32 = add;
    println!("Function pointer: {}", f(2, 3));
    let closure = |x: i32| x * 2;
    println!("Closure: {}", closure(4));

    // Reference and dereference
    let y = 10;
    let y_ref = &y;
    println!("Reference: {}", y_ref);
    println!("Dereference: {}", *y_ref);

    // Type casting
    let f = 4.7f32;
    let i = f as i32;
    println!("Type casting: {}", i);

    // String operations
    let s = String::from("hello");
    let s2 = s + " world";
    println!("String addition: {}", s2);

    // Pattern matching
    let num = 2;
    match num {
        1 => println!("One"),
        2 | 3 => println!("Two or Three"),
        _ => println!("Other"),
    }

    // If let and while let
    let opt = Some(7);
    if let Some(v) = opt {
        println!("If let: {}", v);
    }
    let mut opt2 = Some(0);
    while let Some(v) = opt2 {
        println!("While let: {}", v);
        opt2 = None;
    }

    // Indexing and slicing
    let arr = [10, 20, 30, 40];
    println!("Indexing: {}", arr[2]);
    println!("Slicing: {:?}", &arr[1..3]);

    // Vector operations
    let mut v = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    println!("Vector: {:?}", v);
    println!("Vector sum: {}", v.iter().sum::<i32>());
}
