# Engineering principles

SEPARATION OF CONCERNS!

Create separate components with clean interfaces that can be easily tested,
swapped out, etc.

Use the LTS (best supported release version) of libraries wherever possible.

Use dependencies that are still supported.

Unless specified, use more popular battle-tested dependencies where possible.

When making changes consider how these changes fit in with the overall
architecture.

Don't take shortcuts to add functionality.

Application design should be fractal, where common patterns and style are
repeated throughout the entire codebase.

Performance is important. Choose algorithms and a concurrency model that
leverage the hardware available. For example, when writing GPU code leverage
the vast concurrency available instead of sequential loops.

## Applied to MoonLab

- Optional acceleration is a swappable backend, never a fork of the algorithm.
  A build without OpenMP, LAPACK, Metal, CUDA, or MPI must still compile and
  produce the same numbers, only slower.
- Bindings adapt the core; they do not reimplement it. When a binding needs
  behaviour the core lacks, add it to the core and export it, rather than
  computing it a second time in TypeScript, Python, or Rust.
- The CPU path is the numerical reference. A GPU or WASM path is correct when
  it matches that reference within a declared tolerance, and the tolerance is
  written down with the contract that claims it.

## Branch-scoped development history

Maintain one detailed development log per Git branch under `plan/log/detail/`.
Replace `/` in the full branch name with `--` for the filename so parallel
branches do not edit the same log. Each branch updates only its own log; keep
the shared summary concise and retain branch logs after merge as history.
