//! Embed rpath to libquantumsim in the test/bin binaries so cargo test
//! works without LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=MOONLAB_LIB_DIR");
    if let Ok(lib_dir) = env::var("MOONLAB_LIB_DIR") {
        if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
        }
    }
}
