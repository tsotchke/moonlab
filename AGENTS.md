# Agent instructions

## Before starting work

Read:

1. `plan/README.md`
2. `plan/plan.md`
3. `plan/todo/priority.md`

Then read only the task, workflow, architecture, test, reference, and
summary-log files relevant to the work. Do not load `plan/log/detail/` by
default. Decide whether to scan branch logs when the task may overlap prior
attempts, failures, pivots, or unresolved work. Prefer a targeted search before
reading whole logs.

Keep planning documents aligned with the implementation. Do not edit
`plan/plan.md` without explicit user permission. Keep the current branch's
detailed development log updated even though branch logs are excluded from
routine reading. Under `plan/log/detail/`, replace `/` in the branch name with
`--` to form its filename; for example, `feature/login` uses
`feature--login.md`.

## Repository rules

These come from the repository itself and take precedence over any general
convention:

- **`CONTRIBUTING.md` is authoritative** for branch naming, commit format, and
  pull requests. Use Conventional Commits — `<type>(<scope>): <description>`
  with types `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
  `chore`, `security`. Branch names use `feature/`, `fix/`, `docs/`, `perf/`,
  or `refactor/`.
- **Build with CMake, not the bare `Makefile`.** The `Makefile` cannot build a
  clean tree; it needs a header only CMake generates. See `plan/arch/stack.md`.

  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(nproc)"
  cd build && ctest --output-on-failure
  ```

- **Cap CPU on shared machines.** Tests spawn one OpenMP thread per core, so
  `ctest -jN` can use far more than `N` cores. Constrain the whole tree:
  `OMP_NUM_THREADS=2 taskset -c 0-7 ctest -j4`.
- **This is a fork.** `origin` is `tsotchke/moonlab`, which we do not control
  and which force-pushes `master`. `ulg` is rebased onto it, not merged. See
  `plan/workflow/version-control.md` before rebasing.
- **Optional acceleration stays optional.** A build without OpenMP, LAPACK,
  Metal, CUDA, or MPI must still compile and produce the same results.
- **Do not commit build outputs.** Compiled binaries, `dist/`, and virtualenvs
  are artifacts, not source.

## Documentation layout

- `documents/` is the published user and API documentation. Upstream renamed it
  from `docs/`; do not add new files under `docs/`.
- `plan/` is planning context, not user documentation.

When a planning document disagrees with `CONTRIBUTING.md` or `documents/`, the
repository document wins and the planning document is the one to correct.
