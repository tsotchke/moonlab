//! Embed rpath to libquantumsim so the tui binary runs without
//! DYLD_LIBRARY_PATH / LD_LIBRARY_PATH.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = PathBuf::from(&manifest_dir)
        .parent().unwrap()
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf();

    let lib_dir = env::var("MOONLAB_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("build"));

    println!("cargo:rerun-if-env-changed=MOONLAB_LIB_DIR");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
