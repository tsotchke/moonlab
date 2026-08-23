# exotui

- **Source:** `/home/cos/projects/exotui` — <https://github.com/ubernaut/exotui>,
  published to JSR as `@ubernaut/exotui`
- **Relevant to:** [exomoonlab console interface](../todo/exomoonlab-console-interface.md)
  and [the architecture](../arch/exomoonlab.md)
- **Why it matters:** A Deno TUI toolkit whose presenter seam lets one
  cell-composed application run in a terminal and a browser with no separate
  code paths. exomoonlab is built on it.

The parts exomoonlab depends on:

| Path | What it gives us |
|------|------------------|
| `src/app/shell_presenter.ts` | `ShellPresenter`, `ShellApp`, `runShellApp` — the seam |
| `src/runtime/console_presenter.ts` | Terminal host: alt screen, diffing ANSI, SIGWINCH |
| `src/web/web_presenter.ts` | Browser host: IndexedDB stores, shader layer |
| `src/app/workbench_window_host.ts` | `WorkbenchWindowHostController` — the window manager |
| `src/app/workbench_shell.ts` | Host-neutral painters: chrome, switcher, menus, tabs |
| `src/app/shell_theme.ts` | `ShellThemeSpec` and seventeen themes |
| `src/app/backgrounds/` | Ten animated background fields |
| `examples/web/desktop_app.ts` | **exowebtui** — the reference dual-host desktop |

Public entrypoints are declared in `deno.jsonc` (`.`, `./app`, `./web`,
`./remote`, `./theme`, `./runtime`, `./terminal`, `./viz`, …). exomux is the
standing proof those entrypoints are sufficient for a real application; import
only from them, never from `src/`.
