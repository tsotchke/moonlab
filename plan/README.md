# Planning guide

This directory holds the durable context needed to understand and continue
MoonLab. Keep it useful and small: record information only when it will affect
future work or prevent a decision from being repeated.

## What to read

For every task, read:

1. `plan.md` for scope and intended outcomes.
2. `todo/priority.md` for the current order of work.
3. The specific task file, when one exists.

Read relevant files from `workflow/`, `arch/`, `test/`, `refs/`, and
`log/log-summary.md` as needed. Do not load `log/detail/` by default. The LLM
should decide when the current task may benefit from prior attempts, failures,
pivots, or unresolved work and scan relevant branch logs then. Prefer targeted
searches to reading entire files. This reading rule does not make detailed
logging optional.

## What belongs where

- `plan.md`: user-owned scope, goals, non-goals, success criteria, and
  high-level direction. An LLM may update it only with explicit permission.
- `todo/priority.md`: a short, ordered list of the next actionable tasks.
- `todo/`: one file for work that spans multiple steps or sessions, requires
  meaningful decisions, or needs durable acceptance criteria. Handle small,
  obvious changes directly without creating a task file.
- `todo/done/`: completed task files worth retaining.
- `todo/hiatus/`: paused task files that may be resumed.
- `arch/`: cross-cutting architecture, stack choices, and decisions that affect
  more than one task. Do not document implementation details already obvious
  from the code. Embed simple Mermaid diagrams in the relevant document; keep
  larger or reusable diagram sources in `arch/diagrams/`.
- `test/`: the shared test strategy and project-wide completion expectations.
  Task-specific acceptance checks belong in the task file.
- `workflow/`: lightweight defaults for the development loop and version
  control. Keep them practical; add process only when it prevents recurring
  mistakes or coordination problems.
- `log/log-summary.md`: concise durable progress, decisions, and pivots.
- `log/detail/`: one complete prompt and development record per branch,
  including responses, strategies, attempts, failures, and pivots. Each branch
  updates only its own file, reducing merge conflicts between parallel work.
  Replace `/` in the branch name with `--` for the filename; for example,
  `feature/login` uses `detail/feature--login.md`. Use `detail/_template.md` to
  start a log. Scan branch logs only when their history may inform current work.
- `refs/`: reference material that is actually used; link each reference from
  the plan or task that needs it.
- `implementation-status.md`: the running checklist of delivered slices. It
  predates this framework and is kept because it is still the fastest way to
  see what has actually shipped on this branch.
- `wasm-spec-changes.md`: the branch specification — everything this branch
  changes relative to upstream `main`, and why. Read it before rebasing or
  before deciding what is safe to drop.

Use `todo/_template.md` when a task file is warranted. Delete unused template
sections rather than filling them with boilerplate.

## Relationship to the repository's own documents

This directory is planning context, not user documentation. It does not replace:

- `CONTRIBUTING.md` — the authoritative branch, commit, and pull-request rules.
  `workflow/version-control.md` defers to it rather than restating it.
- `documents/` — the published user and API documentation.

When a planning document and one of the above disagree, the repository document
wins and the planning document is the one to correct.

## Current and target state

When documentation describes a desired state that differs from the repository,
add brief `Current state` and `Target state` sections to the relevant plan,
architecture, or task file. Remove the distinction once the target is reached.
Do not maintain a separate drift register.

## Maintenance

Update planning files in the same change that makes them inaccurate. Keep the
priority list short. When work finishes or pauses, move its task file and remove
it from the active queue. Delete obsolete references. Keep branch detail logs
as history; summarize their durable outcomes in the summary log.
