//! Low-level FFI bindings to the Moonlab quantum simulator.
//!
//! This crate provides raw, unsafe bindings to the C library.
//! For a safe, idiomatic Rust API, use the `moonlab` crate instead.
//!
//! # Safety
//!
//! All functions in this crate are unsafe and require careful handling
//! of pointers and memory management. The safe wrappers in `moonlab`
//! handle this for you.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
