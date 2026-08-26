/**
 * The MoonLab desktop.
 *
 * Phase 3. Windows drag, resize, snap, tile and minimize because exotui's
 * `WorkbenchWindowHostController` does all of that -- the same controller
 * exowebtui uses, so the behaviour is not a reimplementation that will drift
 * from it. This file supplies the window contents, the palette, and the
 * routing; it owns no window mechanics of its own.
 *
 * As with the single-window app, `frame()` is synchronous and never awaits.
 */

import {
  createTiledWorkspaceController,
  createWorkbenchWindowHostController,
  type KeyPressEvent,
  paintShellWindowChrome,
  type PointerInputEvent,
  type Rectangle,
  type ShellPresenterSize,
  solidGround,
  type WorkbenchWindowChromeProjection,
  type WorkbenchWindowHostProjectionOptions,
} from "@ubernaut/exotui/shell";

import type { MoonLabBackend } from "../backend/mod.ts";
import { type Circuit, CIRCUITS } from "./circuits.ts";
import { Job } from "./jobs.ts";
import { computeReadout, DEFAULT_SCAN_LIMIT } from "./readout.ts";
import { type Frame, Surface } from "../ui/cells.ts";
import { paintProbabilitiesBody, type ProbabilityReadout } from "../ui/probabilities.ts";
import { type DesktopPalette, desktopPalette, themeById, themeIndex, THEMES } from "../ui/theme.ts";

/**
 * The only thing the desktop needs from its host: a durable store.
 *
 * `ShellPresenter` satisfies this structurally, so the real hosts pass
 * themselves. Depending on the narrow shape rather than the whole presenter is
 * what lets the persistence round-trip be tested without one.
 */
export interface DesktopHost {
  store<T>(
    name: string,
  ): { get(key: string): Promise<T | undefined>; set(key: string, value: T): Promise<void> };
}

export interface MoonLabDesktopOptions {
  readonly backend: MoonLabBackend;
  readonly onQuit?: () => void;
  readonly scanLimit?: number;
}

const WINDOW_IDS = ["probabilities", "circuits", "session"] as const;
type WindowId = (typeof WINDOW_IDS)[number];

/** What persists between runs. Deliberately small. */
interface PersistedState {
  theme?: string;
  circuit?: string;
  qubits?: number;
}

const THIN_GLYPHS = {
  topLeft: "┌",
  top: "─",
  topRight: "┐",
  left: "│",
  right: "│",
  bottomLeft: "└",
  bottom: "─",
  bottomRight: "┘",
} as const;

export class MoonLabDesktop {
  readonly #backend: MoonLabBackend;
  readonly #onQuit: () => void;
  readonly #scanLimit: number;
  readonly #job = new Job<ProbabilityReadout>();
  readonly #workspace = createTiledWorkspaceController({});
  readonly #host: ReturnType<typeof createWorkbenchWindowHostController<WindowId>>;

  #size: ShellPresenterSize = { columns: 100, rows: 32 };
  #palette: DesktopPalette;
  #themeId = "moonlab";
  #circuitIndex = Math.max(0, CIRCUITS.findIndex((c) => c.id === "bell"));
  #numQubits: number;
  #store?: {
    get(key: string): Promise<PersistedState | undefined>;
    set(key: string, value: PersistedState): Promise<void>;
  };
  #status = "";

  constructor(options: MoonLabDesktopOptions) {
    this.#backend = options.backend;
    this.#onQuit = options.onQuit ?? (() => {});
    this.#scanLimit = options.scanLimit ?? DEFAULT_SCAN_LIMIT;
    this.#palette = desktopPalette(themeById(this.#themeId));
    this.#numQubits = this.#circuit.defaultQubits;
    this.#host = createWorkbenchWindowHostController<WindowId>({
      workspace: this.#workspace,
      ownerId: "exomoonlab",
      snapDistance: 2,
      snapOnRelease: true,
      compactMode: "auto",
      windows: [
        {
          id: "probabilities",
          title: "Probabilities",
          minWidth: 34,
          minHeight: 10,
          placement: "floating",
          state: "normal",
          floatingRect: { column: 2, row: 2, width: 62, height: 18 },
        },
        {
          id: "circuits",
          title: "Circuits",
          minWidth: 24,
          minHeight: 8,
          placement: "floating",
          state: "normal",
          floatingRect: { column: 66, row: 2, width: 32, height: 12 },
        },
        {
          id: "session",
          title: "Session",
          minWidth: 24,
          minHeight: 6,
          placement: "floating",
          state: "normal",
          floatingRect: { column: 66, row: 15, width: 32, height: 9 },
        },
      ],
    });
  }

  get #circuit(): Circuit {
    return CIRCUITS[this.#circuitIndex];
  }

  /** Windows live below the status bar. */
  #bodyBounds(): Rectangle {
    return {
      column: 0,
      row: 1,
      width: this.#size.columns,
      height: Math.max(1, this.#size.rows - 1),
    };
  }
  #barBounds(): Rectangle {
    return { column: 0, row: 0, width: this.#size.columns, height: 1 };
  }
  #projectionOptions(): WorkbenchWindowHostProjectionOptions {
    return { shelfBounds: this.#barBounds(), doubleClickMaximizeMs: 400 };
  }

  async init(host?: DesktopHost): Promise<void> {
    if (host) {
      this.#store = host.store<PersistedState>("exomoonlab");
      try {
        const saved = await this.#store.get("state");
        if (saved?.theme) this.#applyTheme(saved.theme);
        if (saved?.circuit) {
          const found = CIRCUITS.findIndex((c) => c.id === saved.circuit);
          if (found >= 0) this.#circuitIndex = found;
        }
        this.#numQubits = saved?.qubits ?? this.#circuit.defaultQubits;
      } catch {
        // A missing or unreadable store is not a failure: the console starts
        // on its defaults rather than refusing to open.
      }
    }
    this.#run();
  }

  #persist(): void {
    // Fire and forget: persistence must never delay a frame, and losing the
    // last theme change to a crash costs nothing worth blocking for.
    void this.#store?.set("state", {
      theme: this.#themeId,
      circuit: this.#circuit.id,
      qubits: this.#numQubits,
    }).catch(() => {});
  }

  #applyTheme(id: string): void {
    this.#themeId = themeById(id).id;
    this.#palette = desktopPalette(themeById(this.#themeId));
  }

  #cycleTheme(delta: number): void {
    const next = (themeIndex(this.#themeId) + delta + THEMES.length) % THEMES.length;
    this.#applyTheme(THEMES[next].id);
    this.#status = `theme: ${this.#palette.label}`;
    this.#persist();
  }

  #run(): void {
    const circuit = this.#circuit;
    const numQubits = this.#numQubits;
    this.#job.start(() =>
      computeReadout(this.#backend, circuit, { numQubits, scanLimit: this.#scanLimit })
    );
  }

  #selectCircuit(delta: number): void {
    this.#circuitIndex = (this.#circuitIndex + delta + CIRCUITS.length) % CIRCUITS.length;
    this.#numQubits = this.#circuit.defaultQubits;
    this.#status = `circuit: ${this.#circuit.name}`;
    this.#persist();
    this.#run();
  }

  #resizeRegister(delta: number): void {
    const next = Math.max(
      this.#circuit.minQubits,
      Math.min(this.#backend.capabilities.maxQubits, this.#numQubits + delta),
    );
    if (next === this.#numQubits) return;
    this.#numQubits = next;
    this.#persist();
    this.#run();
  }

  resize(size: ShellPresenterSize): void {
    this.#size = size;
  }

  pointer(event: PointerInputEvent): void {
    this.#host.handlePointer(event, this.#bodyBounds(), this.#projectionOptions());
  }

  key(event: KeyPressEvent): void {
    if (event.ctrl && event.key === "c") return this.#onQuit();
    switch (event.key) {
      case "q":
        return this.#onQuit();
      case "r":
        return this.#run();
      case "j":
      case "down":
        return this.#selectCircuit(1);
      case "k":
      case "up":
        return this.#selectCircuit(-1);
      case "+":
      case "=":
      case "right":
        return this.#resizeRegister(1);
      case "-":
      case "_":
      case "left":
        return this.#resizeRegister(-1);
      case "t":
        return this.#cycleTheme(event.shift ? -1 : 1);
      case "tab":
        this.#host.execute(
          { kind: "focus-next", direction: event.shift ? -1 : 1 },
          this.#bodyBounds(),
          this.#projectionOptions(),
        );
        return;
      case "m": {
        const active = this.#host.controller.inspect().activeWindowId;
        if (active) {
          this.#host.execute(
            { kind: "toggle-maximize", id: active },
            this.#bodyBounds(),
            this.#projectionOptions(),
          );
        }
        return;
      }
    }
  }

  frame(_now: number, size: ShellPresenterSize): Frame {
    this.#size = size;
    const surface = new Surface(size.columns, size.rows, { background: this.#palette.background });
    const p = this.#palette;

    this.#paintBar(surface);

    const projection = this.#host.project(this.#bodyBounds(), this.#projectionOptions());
    for (const window of projection.windows) this.#paintWindow(surface, window);
    return surface.frame();
  }

  #paintBar(surface: Surface): void {
    const p = this.#palette;
    surface.fill(this.#barBounds(), " ", { background: p.surfaceStrong });
    const snapshot = this.#job.snapshot;
    const state = snapshot.status === "running"
      ? "running"
      : snapshot.durationMs !== undefined
      ? `${snapshot.durationMs.toFixed(1)}ms`
      : "idle";
    surface.write(1, 0, "MoonLab", {
      foreground: p.accent,
      background: p.surfaceStrong,
      bold: true,
    });
    surface.write(
      9,
      0,
      `${this.#backend.kind} · ${this.#numQubits}q · ${state}`,
      { foreground: p.muted, background: p.surfaceStrong },
    );
    const hint = this.#status || "j/k circuit  ± qubits  t theme  tab focus  m max  q quit";
    surface.writeFitted(
      Math.max(0, surface.columns - hint.length - 2),
      0,
      hint,
      hint.length,
      { foreground: p.muted, background: p.surfaceStrong },
    );
  }

  #paintWindow(surface: Surface, window: WorkbenchWindowChromeProjection): void {
    const p = this.#palette;
    const chromeBase = window.active ? p.surfaceStrong : p.surface;
    paintShellWindowChrome(surface.shellSurface(), window, {
      surfaceFill: { foreground: p.text, background: chromeBase },
      borderGlyphs: THIN_GLYPHS,
      borderForeground: window.active ? p.accent : p.border,
      chromeGround: solidGround(chromeBase),
      titleBarGround: solidGround(window.active ? p.accent : p.surfaceStrong),
      titleBarFillForeground: p.text,
      titleText: window.title,
      titleForeground: window.active ? p.onAccent : p.muted,
      titleBold: window.active,
      // Danger-toned controls (close) stay bold on an inactive window too, so
      // the destructive one is never the hardest to see.
      controlBold: (control) => control.tone === "danger" || window.active,
      controlForeground: window.active ? p.onAccent : p.muted,
    });

    const client = window.clientRect;
    if (client.width <= 0 || client.height <= 0) return;
    switch (window.id as WindowId) {
      case "probabilities":
        return this.#paintProbabilities(surface, client);
      case "circuits":
        return this.#paintCircuits(surface, client);
      case "session":
        return this.#paintSession(surface, client);
    }
  }

  #paintProbabilities(surface: Surface, rect: Rectangle): void {
    const snapshot = this.#job.snapshot;
    // The chrome already drew the border and the title bar; only the contents
    // belong here. Painting the boxed form would overwrite both.
    paintProbabilitiesBody(
      surface,
      { column: rect.column + 1, row: rect.row, width: rect.width - 2, height: rect.height },
      {
        title: this.#circuit.name,
        subtitle: `${this.#circuit.name} · ${this.#numQubits} qubits`,
        readout: snapshot.value,
        busy: snapshot.status === "running",
        error: snapshot.status === "failed" ? snapshot.error : undefined,
      },
      this.#palette,
    );
  }

  #paintCircuits(surface: Surface, rect: Rectangle): void {
    const p = this.#palette;
    CIRCUITS.forEach((circuit, index) => {
      const y = rect.row + index;
      if (y >= rect.row + rect.height) return;
      const selected = index === this.#circuitIndex;
      surface.fill(
        { column: rect.column, row: y, width: rect.width, height: 1 },
        " ",
        { background: selected ? p.accent : p.surface },
      );
      surface.writeFitted(
        rect.column + 1,
        y,
        `${selected ? "▸ " : "  "}${circuit.name}`,
        rect.width - 2,
        {
          foreground: selected ? p.onAccent : p.text,
          background: selected ? p.accent : p.surface,
          bold: selected,
        },
      );
    });
  }

  #paintSession(surface: Surface, rect: Rectangle): void {
    const p = this.#palette;
    const caps = this.#backend.capabilities;
    const lines: Array<[string, string]> = [
      ["backend", this.#backend.kind],
      ["max qubits", String(caps.maxQubits)],
      ["alloc ctor", caps.allocatingConstructor ? "yes" : "no"],
      ["exports", caps.exportedFunctions ? String(caps.exportedFunctions) : "—"],
      ["theme", `${this.#palette.label} (${themeIndex(this.#themeId) + 1}/${THEMES.length})`],
    ];
    lines.forEach(([label, value], index) => {
      const y = rect.row + index;
      if (y >= rect.row + rect.height) return;
      surface.writeFitted(rect.column + 1, y, label, 12, {
        foreground: p.muted,
        background: p.surface,
      });
      surface.writeFitted(rect.column + 13, y, value, rect.width - 14, {
        foreground: p.text,
        background: p.surface,
      });
    });
  }
}
