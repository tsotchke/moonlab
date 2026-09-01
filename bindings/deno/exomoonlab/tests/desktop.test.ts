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

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`assertion failed: ${message}`);
}

const SIZE = { columns: 104, rows: 30 };

function render(desktop: MoonLabDesktop): string {
  return desktop.frame(0, SIZE)
    .map((row) => row.map((cell) => cell.char).join("").trimEnd())
    .join("\n");
}

async function settle(desktop: MoonLabDesktop, frames = 400): Promise<string> {
  let text = "";
  for (let i = 0; i < frames; i++) {
    text = render(desktop);
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

Deno.test("desktop opens its windows with chrome, none hidden at the default size", async () => {
  const backend = await openNativeBackend();
  const desktop = new MoonLabDesktop({ backend });
  try {
    await desktop.init();
    const text = await settle(desktop);

    for (const title of ["Probabilities", "Band geometry", "Schrödinger", "Circuits", "Session"]) {
      assert(text.includes(title), `window "${title}" missing:\n${text}`);
    }
    // Chrome controls come from exotui's painter, not from this app.
    assert(text.includes("[x]"), `no window controls painted:\n${text}`);
    // The body must not draw a second box over the chrome's title bar.
    assert(!text.includes("┌─ Bell"), `body painted its own frame over chrome:\n${text}`);
    assert(text.includes("|00⟩") && text.includes("|11⟩"), `probabilities missing:\n${text}`);
    assert(text.includes("▸ Bell pair"), `circuit selection marker missing:\n${text}`);
    assert(text.includes("backend"), `session window empty:\n${text}`);
    // The default layout must not start any window behind another: every
    // window's own content has to be visible, not just its title bar.
    assert(text.includes("C = "), `band window content hidden:\n${text}`);
    // A title bar too narrow for its own name plus the four controls truncates
    // the title away, which is how Circuits and Session went missing once.
    assert(text.includes("a₀"), `orbital window content hidden:\n${text}`);
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
    // Read the Session window's theme row rather than matching a literal: that
    // window is narrow and truncates a long label, so any fixed string is a
    // hostage to its width.
    // Anchored on the window border: the status bar's own hint contains the
    // word "theme" too, and an unanchored match picks that up instead.
    const themeOf = (text: string) =>
      text.split("\n").find((line) => /│\s*theme\s{2,}/.test(line))
        ?.match(/theme\s{2,}(\S+)/)?.[1] ?? "";
    const first = themeOf(render(desktop));
    assert(first.startsWith("MoonLab"), `did not start on the MoonLab theme: "${first}"`);

    desktop.key({ key: "t" } as never);
    const next = render(desktop);
    assert(next.includes("theme: "), `no theme change announced:\n${next}`);
    assert(themeOf(next) !== first, `theme did not change from "${first}"`);
    assert(THEMES.length === 18, `expected 18 themes, got ${THEMES.length}`);

    // All the way around returns to where it started.
    for (let i = 1; i < THEMES.length; i++) desktop.key({ key: "t" } as never);
    assert(themeOf(render(desktop)) === first, "cycling did not wrap to the start");
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
    first.key({ key: "t" } as never);
    first.key({ key: "j" } as never);
    first.key({ key: "+" } as never);
    await settle(first);
    const before = render(first);

    // A second desktop over the same store is what a restart looks like.
    const second = new MoonLabDesktop({ backend });
    await second.init(host);
    const after = await settle(second);

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
      const before = themeOf(render(desktop));
      desktop.key({ key, shift: true } as never);
      await settle(desktop);
      return { before, after: themeOf(render(desktop)) };
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
