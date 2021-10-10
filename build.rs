use cbindgen;


fn main() {
    let runner_os = option_env!("RUNNER_OS").map(|v| v.to_lowercase());
    let header_name = match runner_os {
        Some(os) => {
            match os.as_str() {
                "windows" => "flaco.h".to_string(),
                _ => "libflaco.h".to_string()
            }
        },
        None => "libflaco.h".to_string()

    };
    cbindgen::generate(&"./".to_string())
        .unwrap()
        .write_to_file(format!("./flaco/{}", header_name));
}
