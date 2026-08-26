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

Deno.test("desktop opens three windows with chrome", async () => {
  const backend = await openNativeBackend();
  const desktop = new MoonLabDesktop({ backend });
  try {
    await desktop.init();
    const text = await settle(desktop);

    for (const title of ["Probabilities", "Circuits", "Session"]) {
      assert(text.includes(title), `window "${title}" missing:\n${text}`);
    }
    // Chrome controls come from exotui's painter, not from this app.
    assert(text.includes("[x]"), `no window controls painted:\n${text}`);
    // The body must not draw a second box over the chrome's title bar.
    assert(!text.includes("┌─ Bell"), `body painted its own frame over chrome:\n${text}`);
    assert(text.includes("|00⟩") && text.includes("|11⟩"), `probabilities missing:\n${text}`);
    assert(text.includes("▸ Bell pair"), `circuit selection marker missing:\n${text}`);
    assert(text.includes("backend"), `session window empty:\n${text}`);
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
    assert(render(desktop).includes("MoonLab (1/"), "did not start on the MoonLab theme");

    desktop.key({ key: "t" } as never);
    const next = render(desktop);
    // Assert on the status bar, not the Session window: that window is narrow
    // and truncates a long theme label, so "(2/18)" may never appear there.
    assert(next.includes("theme: "), `no theme change announced:\n${next}`);
    assert(!next.includes("MoonLab (1/"), `theme did not change:\n${next}`);
    assert(THEMES.length === 18, `expected 18 themes, got ${THEMES.length}`);

    // All the way around returns to where it started.
    for (let i = 1; i < THEMES.length; i++) desktop.key({ key: "t" } as never);
    assert(render(desktop).includes("MoonLab (1/"), "cycling did not wrap to the start");
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

    const themeOf = (text: string) => text.match(/theme\s+(.+?)\s*│/)?.[1]?.trim();
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
