//! Embed rpath to libquantumsim in the test/bin binaries so cargo test
//! works without LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.

use std::env;
use std::path::{Path, PathBuf};

fn quantumsim_library_exists(dir: &Path) -> bool {
    [
        "libquantumsim.a",
        "libquantumsim.dylib",
        "libquantumsim.so",
        "quantumsim.dll",
        "quantumsim.lib",
    ]
    .iter()
    .any(|name| dir.join(name).exists())
}

fn emit_quantumsim_rerun_paths(project_root: &Path) {
    for candidate in [
        project_root.join("build-qgtl-vendor"),
        project_root.join("build"),
        project_root.to_path_buf(),
    ] {
        if candidate.exists() {
            println!("cargo:rerun-if-changed={}", candidate.display());
        }
    }
}

fn default_quantumsim_lib_dir(project_root: &Path) -> PathBuf {
    for candidate in [
        project_root.join("build-qgtl-vendor"),
        project_root.join("build"),
        project_root.to_path_buf(),
    ] {
        if quantumsim_library_exists(&candidate) {
            return candidate;
        }
    }

    project_root.join("build")
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = PathBuf::from(&manifest_dir)
        .parent()
        .unwrap() // rust/
        .parent()
        .unwrap() // bindings/
        .parent()
        .unwrap() // project root
        .to_path_buf();

    let lib_dir = env::var("MOONLAB_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_quantumsim_lib_dir(&project_root));

    println!("cargo:rerun-if-env-changed=MOONLAB_LIB_DIR");
    emit_quantumsim_rerun_paths(&project_root);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
