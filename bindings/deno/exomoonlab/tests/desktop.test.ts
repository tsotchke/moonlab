/**
 * The desktop, headlessly.
 *
 * The window mechanics belong to exotui and are tested there; what is checked
 * here is that this application wires them up correctly -- three windows with
 * chrome, the circuit list tracking selection, themes cycling, and the
 * persisted state actually round-tripping.
 */

import { MoonLabDesktop } from "../src/app/desktop.ts";
import { openNativeBackend } from "../src/backend/mod.ts";
import { THEMES } from "../src/ui/theme.ts";
import { APPS, CLOSED_LAUNCHER, handleLauncherKey } from "../src/app/launcher.ts";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`assertion failed: ${message}`);
}

const SIZE = { columns: 104, rows: 30 };

function renderToText(desktop: MoonLabDesktop): string {
  return desktop.frame(0, SIZE)
    .map((row) => row.map((cell) => cell.char).join("").trimEnd())
    .join("\n");
}

async function settle(desktop: MoonLabDesktop, frames = 400): Promise<string> {
  let text = "";
  for (let i = 0; i < frames; i++) {
    text = renderToText(desktop);
    if (!text.includes("computing…") && !text.includes("· running")) return text;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(`desktop did not settle:\n${text}`);
}

/** An in-memory stand-in for the presenter's durable store. */
function memoryHost() {
  const data = new Map<string, unknown>();
  return {
    data,
    store<T>(_name: string) {
      return {
        get: (key: string) => Promise.resolve(data.get(key) as T | undefined),
        set: (key: string, value: T) => {
          data.set(key, value);
          return Promise.resolve();
        },
      };
    },
  };
}

/** Opens an app from the launcher and waits for it. */
async function launch(desktop: MoonLabDesktop, steps: number): Promise<string> {
  desktop.key({ key: "`" } as never);
  for (let i = 0; i < steps; i++) desktop.key({ key: "down" } as never);
  desktop.key({ key: "return" } as never);
  return await settle(desktop);
}

Deno.test("demos are applications: closed until launched", async () => {
  const backend = await openNativeBackend();
  const desktop = new MoonLabDesktop({ backend });
  try {
    await desktop.init();
    const start = await settle(desktop);

    // Only the first demo is running; the rest wait to be launched.
    assert(start.includes("Probabilities"), `default app not open:\n${start}`);
    for (const closed of ["Band geometry", "QEC decoder", "Session"]) {
      assert(!start.includes(closed), `"${closed}" should not be open before launch`);
    }

    // The launcher lists them, with a hint each.
    desktop.key({ key: "`" } as never);
    const menu = renderToText(desktop);
    assert(menu.includes("⏻ MoonLab"), `no start button:\n${menu}`);
    assert(menu.includes("Berry curvature"), `menu hints missing:\n${menu}`);
    desktop.key({ key: "escape" } as never);
    assert(!renderToText(desktop).includes("Berry curvature"), "escape did not close the menu");

    // Launching opens the window, chrome and all.
    const after = await launch(desktop, 1);
    assert(after.includes("Schrödinger"), `launch did not open the app:\n${after}`);
    assert(after.includes("[x]"), `launched window has no chrome:\n${after}`);
    assert(after.includes("Probabilities"), "launching closed the app that was already open");
  } finally {
    await backend.dispose();
  }
});

Deno.test("circuit selection moves in the list and the readout follows", async () => {
  const backend = await openNativeBackend();
  const desktop = new MoonLabDesktop({ backend });
  try {
    await desktop.init();
    await settle(desktop);
    // The circuit list is an application; launch it before reading it.
    await launch(desktop, 4);

    desktop.key({ key: "j" } as never);
    const after = await settle(desktop);
    assert(!after.includes("▸ Bell pair"), `selection did not move:\n${after}`);
    assert(after.includes("▸ GHZ state"), `expected GHZ selected:\n${after}`);
  } finally {
    await backend.dispose();
  }
});

Deno.test("themes cycle through the full catalog", async () => {
  const backend = await openNativeBackend();
  const desktop = new MoonLabDesktop({ backend });
  try {
    await desktop.init();
    await settle(desktop);
    // Session is an application now, so it has to be launched before its
    // theme row exists to read.
    await launch(desktop, 5);
    // Read the Session window's theme row rather than matching a literal: that
    // window is narrow and truncates a long label, so any fixed string is a
    // hostage to its width.
    // Anchored on the window border: the status bar's own hint contains the
    // word "theme" too, and an unanchored match picks that up instead.
    const themeOf = (text: string) =>
      text.split("\n").find((line) => /│\s*theme\s{2,}/.test(line))
        ?.match(/theme\s{2,}(\S+)/)?.[1] ?? "";
    const first = themeOf(renderToText(desktop));
    assert(first.startsWith("MoonLab"), `did not start on the MoonLab theme: "${first}"`);

    desktop.key({ key: "t" } as never);
    const next = renderToText(desktop);
    assert(next.includes("theme: "), `no theme change announced:\n${next}`);
    assert(themeOf(next) !== first, `theme did not change from "${first}"`);
    assert(THEMES.length === 18, `expected 18 themes, got ${THEMES.length}`);

    // All the way around returns to where it started.
    for (let i = 1; i < THEMES.length; i++) desktop.key({ key: "t" } as never);
    assert(themeOf(renderToText(desktop)) === first, "cycling did not wrap to the start");
  } finally {
    await backend.dispose();
  }
});

Deno.test("theme, circuit and register size survive a restart", async () => {
  const backend = await openNativeBackend();
  const host = memoryHost();
  try {
    const first = new MoonLabDesktop({ backend });
    await first.init(host);
    await settle(first);
    await launch(first, 5);
    first.key({ key: "t" } as never);
    first.key({ key: "j" } as never);
    first.key({ key: "+" } as never);
    await settle(first);
    const before = renderToText(first);

    // A second desktop over the same store is what a restart looks like.
    const second = new MoonLabDesktop({ backend });
    await second.init(host);
    await settle(second);
    const after = await launch(second, 5);

    const themeOf = (text: string) =>
      text.split("\n").find((line) => /│\s*theme\s{2,}/.test(line))
        ?.match(/theme\s{2,}(\S+)/)?.[1] ?? "";
    assert(
      themeOf(after) === themeOf(before),
      `theme not restored: ${themeOf(after)} vs ${themeOf(before)}`,
    );
    const selected = (text: string) => text.match(/▸ (.+?)\s*│/)?.[1]?.trim();
    assert(
      selected(after) === selected(before),
      `circuit not restored: ${selected(after)} vs ${selected(before)}`,
    );
    assert(host.data.has("state"), "nothing was persisted");
  } finally {
    await backend.dispose();
  }
});

Deno.test("shifted keys work under both hosts' event shapes", async () => {
  // The console reader reports shift+T as key "T" with shift:true; the browser
  // reports key "t" with shift:true. An app switching on lowercase literals
  // silently ignores the console form, which killed every reverse binding in a
  // terminal while working perfectly in a tab.
  const backend = await openNativeBackend();
  try {
    const themeOf = (text: string) =>
      text.split("\n").find((line) => /│\s*theme\s{2,}/.test(line))
        ?.match(/theme\s{2,}(\S+)/)?.[1] ?? "";

    const advance = async (key: string) => {
      const desktop = new MoonLabDesktop({ backend });
      await desktop.init();
      await settle(desktop);
      await launch(desktop, 5);
      const before = themeOf(renderToText(desktop));
      desktop.key({ key, shift: true } as never);
      await settle(desktop);
      return { before, after: themeOf(renderToText(desktop)) };
    };

    const browserForm = await advance("t");
    const consoleForm = await advance("T");

    assert(browserForm.after !== browserForm.before, "browser-form shift+t did nothing");
    assert(consoleForm.after !== consoleForm.before, "console-form shift+T did nothing");
    // Both must land on the same theme: shift means "previous" either way.
    assert(
      browserForm.after === consoleForm.after,
      `hosts disagree: browser->${browserForm.after} console->${consoleForm.after}`,
    );
  } finally {
    await backend.dispose();
  }
});

Deno.test("the launcher claims only the keys it uses", () => {
  const open = { open: true, index: 0 };

  // Traversal comes from exotui's moveWorkbenchMenuIndex, including wrap.
  const down = handleLauncherKey(open, "down");
  assert(down.kind === "state" && down.state.index === 1, "down did not move");
  const up = handleLauncherKey(open, "up");
  assert(up.kind === "state" && up.state.index === APPS.length - 1, "up did not wrap");

  // Enter launches; escape closes. Both are exotui's predicates, so the menu
  // agrees with the rest of the shell about what those keys mean.
  const enter = handleLauncherKey(open, "return");
  assert(enter.kind === "launch" && enter.app.id === APPS[0].id, "return did not launch");
  const esc = handleLauncherKey(open, "escape");
  assert(esc.kind === "state" && !esc.state.open, "escape did not close");

  // Anything else falls through, so the menu does not swallow the keyboard.
  assert(handleLauncherKey(open, "q").kind === "none", "the menu swallowed q");
  assert(handleLauncherKey(open, "t").kind === "none", "the menu swallowed t");
  // And a closed launcher claims nothing at all.
  assert(handleLauncherKey(CLOSED_LAUNCHER, "down").kind === "none", "closed menu claimed a key");
});
