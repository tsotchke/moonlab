# Version-control workflow

`CONTRIBUTING.md` is authoritative for branch naming, commit format, and pull
requests. This file records how those rules apply to *this fork*, which is not
a plain trunk-based repository.

## Topology

```
tsotchke/moonlab  master        upstream; we do not control it
                    │
                    └── ulg      our long-lived integration branch
                          │
                          └── exomoonlab   console-interface work
```

`origin` is `tsotchke/moonlab`. `ulg` carries this fork's work — the ULG
quantum-response artifact, the magnetar reference suite, and the browser WebGPU
complex64 parity effort — and is periodically rebased onto `origin/master`.

Because `ulg` is rebased rather than merged, its commits are rewritten. Treat it
as shared-but-rebasing: coordinate before force-pushing, and always branch new
work from the current tip rather than an older copy.

## Working rules

- Integrate small, complete changes frequently. Avoid long-lived side branches
  other than `ulg` itself.
- Start each change from the current tip of its base branch, on a short-lived
  branch named per `CONTRIBUTING.md`: `feature/`, `fix/`, `docs/`, `perf/`, or
  `refactor/`.
- Use Conventional Commits, as `CONTRIBUTING.md` specifies:
  `<type>(<scope>): <description>`, with types `feat`, `fix`, `docs`, `style`,
  `refactor`, `perf`, `test`, `chore`, `security`.
- Start the branch's detail log under `plan/log/detail/` using the branch-name
  filename convention in `plan/README.md`.
- Sync before integrating. Resolve conflicts on the working branch and rerun
  affected checks.
- Preserve unrelated local changes. Do not discard or include someone else's
  work in a commit without authorization.
- Keep credentials, generated runtime state, and large disposable artifacts out
  of version control. Compiled binaries and `dist/` are build outputs, not
  source.

## Rebasing onto upstream

Upstream moves fast and force-pushes `master`. When rebasing `ulg`:

1. Back up first: `git branch -f ulg-prerebase-backup ulg`.
2. `git fetch origin --prune`, then check what is genuinely ours:
   `git cherry -v origin/master ulg`. Lines marked `-` are already upstream and
   will drop out on their own.
3. Enable `rerere` so repeated conflicts resolve once:
   `git config rerere.enabled true`.
4. For each conflict, check whether upstream already implemented the same fix.
   It frequently has, and usually more thoroughly. Prefer upstream's version and
   skip the superseded commit rather than reapplying a weaker duplicate.
5. Watch for changes that auto-merge into a *duplicate* rather than a conflict —
   two registrations of the same callback, two exports of the same name. These
   pass the rebase and fail the build later.
6. Rebuild and run the suite afterwards; a rebase that merges cleanly can still
   break the type build.

## Verifying before integration

Run the checks in `plan/test/test.md`. At minimum, build with CMake and run
`ctest`, capped so the machine stays usable — see `plan/workflow/development.md`.

## Current state and target state

**Current state.** `CONTRIBUTING.md` tells contributors to branch from and open
pull requests against `develop`, but no `develop` branch exists on `origin`; the
default branch is `master`. There is also no deployed test environment, so the
pre-merge "deploy the PR revision and run integrated tests" gate does not apply
here.

**Target state.** Either `CONTRIBUTING.md` is corrected upstream to name the
branch that actually exists, or this fork records the discrepancy for
contributors. Do not follow the `develop` instruction literally.
