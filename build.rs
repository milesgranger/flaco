extern crate cbindgen;

fn main() {
    cbindgen::generate(&"./".to_string())
        .unwrap()
        .write_to_file("./flaco/libflaco.h");
}
