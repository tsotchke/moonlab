# Exact-byte crates.io publication

Moonlab certifies the SHA-256 of each Rust crate tarball in the release
certificate. The release must publish the certified bytes, not a fresh
re-packaging. `cargo publish` re-runs `cargo package` at publish time and emits
a new tarball whose SHA-256 differs from the certified artifact (the tarball
embeds `.cargo_vcs_info.json` and a `Cargo.lock`, so its bytes depend on the
checkout and the resolver state at packaging time). The exact-byte path removes
that gap: the crate is packaged once, its hash is certified, and those exact
bytes are uploaded through the raw registry API.

Two tools implement this:

- `scripts/package_rust_crates.sh <out-dir>` packages the three crates in
  dependency order and writes `SHA256SUMS`. This is the certification point.
- `scripts/publish_crate_exact_bytes.py` verifies a `.crate` against its
  certified SHA-256 and uploads those exact bytes. It never packages.

## Publication order

The crates form a dependency chain and publish in this order, waiting for the
crates.io index between each:

1. `moonlab-sys`
2. `moonlab` (depends on `moonlab-sys`)
3. `moonlab-tui` (depends on `moonlab`)

`moonlab` and `moonlab-tui` depend on siblings that are not yet on crates.io.
`cargo package` cannot resolve their path+version dependencies until the sibling
is published, so `package_rust_crates.sh` injects a throwaway
`--config patch.crates-io.<sibling>.path=...`. Cargo strips the patch from the
packaged manifest, so the published crate still carries the rewritten registry
dependency (for example `moonlab-sys = "1.2.0"`).

## Authentication

The registry token is read from `--token` or, failing that, the
`CARGO_REGISTRY_TOKEN` environment variable -- the same variable `cargo publish`
uses. crates.io wants the token verbatim in the `Authorization` header; it is
**not** prefixed with `Bearer` or `token`. `--dry-run` needs no token.

```bash
export CARGO_REGISTRY_TOKEN=cio...          # a crates.io API token
python3 scripts/publish_crate_exact_bytes.py publish \
    --crate dist/rust/moonlab-sys-1.2.0.crate \
    --sha256 2660d0fab0f47aaa5e9758a729030b93089e1078eb87441b96a5fa082823596f
```

## Request body format

The endpoint and framing are the authoritative Cargo wire format (the Cargo
book, "Registry Web API" -> Publish, and `crates/crates-io/lib.rs` in
`rust-lang/cargo`):

```
PUT {registry}/api/v1/crates/new
  Authorization: <token>                 # raw token, no scheme prefix
  Content-Type:  application/octet-stream
  Accept:        application/json
  User-Agent:    moonlab-exact-byte-publisher/1.0

body =
    u32_le(len(metadata_json))    # 4 bytes, little-endian
    metadata_json                 # JSON object: the crate metadata
    u32_le(len(crate_tarball))    # 4 bytes, little-endian
    crate_tarball                 # the .crate bytes, verbatim
```

The trailing segment is the `.crate` file read from disk, byte for byte. It is
never regenerated. A `User-Agent` is sent because crates.io rejects requests
that omit it.

The metadata JSON object matches Cargo's `NewCrate` structure (`name`, `vers`,
`deps`, `features`, `authors`, `description`, `documentation`, `homepage`,
`readme`, `readme_file`, `keywords`, `categories`, `license`, `license_file`,
`repository`, `badges`, `links`, `rust_version`). By default the publisher
derives it from the crate's own embedded, Cargo-normalized `Cargo.toml` -- the
same manifest Cargo itself uploads, with path dependencies already rewritten to
registry dependencies. Pass `--metadata FILE` to supply an explicit object
instead. Inspect the derived metadata with:

```bash
python3 scripts/publish_crate_exact_bytes.py metadata --crate <path>.crate
```

## Testing against a local registry

`--registry-url` (or `$MOONLAB_REGISTRY_URL`) points the uploader at any base
URL, so the full wire path can be exercised without touching real crates.io.
`--dry-run` validates and prints the request without sending, and `--emit-body`
writes the exact request body to a file for inspection.

```bash
python3 scripts/publish_crate_exact_bytes.py publish \
    --crate <path>.crate --sha256 <hex> \
    --registry-url http://127.0.0.1:8099 --dry-run --emit-body /tmp/body.bin
```

`scripts/test_publish_crate_exact_bytes.py` packages the three crates, asserts
the emitted body ends with the exact `.crate` bytes, asserts the publisher fails
closed on a wrong hash, and uploads to an in-process registry to confirm the
bytes arrive unchanged over the wire with the documented headers.

## Failure modes

Every failure is fail-closed: a non-zero exit, nothing uploaded.

- The `.crate` file is missing or unreadable.
- The computed SHA-256 does not equal the expected SHA-256. Checked before the
  request body is built and before the network is touched -- mismatched bytes
  are never sent, and with `--emit-body` no body file is written.
- The expected hash is not a 64-character hex digest.
- The metadata or the tarball is larger than the `u32` length prefix allows.
- The metadata lacks a `name` or `vers`.
- No token is available for a real (non-`--dry-run`) upload.
- The registry cannot be reached, or answers with a non-2xx status or a JSON
  `errors` array. The registry's error detail is surfaced. A `409`/duplicate
  from crates.io means the version is already published; re-running is safe only
  after confirming the certified hash matches what is already live.

## Determinism and residual risk

`cargo package` is byte-reproducible for a **fixed** commit, toolchain, and
dependency-index state: repeated runs from the same checkout produce identical
tarballs. It is **not** reproducible across those inputs, because the tarball
embeds:

- `.cargo_vcs_info.json`, which records the git commit SHA; and
- `Cargo.lock`, which pins exact dependency versions resolved from the index at
  packaging time.

This is exactly why the exact-byte path exists. The crate must be packaged once
and that single artifact both certified and published. The release does this in
one job (`rust-crates` packages and certifies, uploading the `.crate` files and
`SHA256SUMS` as an artifact) and consumes the same artifact downstream
(`publish-rust` downloads it and ships the exact bytes). Do not re-run
`cargo package` between certification and publication -- a new run may resolve a
newer transitive dependency and change the tarball hash.
