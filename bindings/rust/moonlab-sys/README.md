# moonlab-sys

Raw Rust FFI bindings for the Moonlab quantum simulator.

Install the Moonlab native SDK first, then ensure `pkg-config quantumsim`
resolves it. For custom or relocatable SDK installations, set
`MOONLAB_LIB_DIR` and `MOONLAB_INCLUDE_DIR`.

Most users should depend on the safe `moonlab` crate instead.
