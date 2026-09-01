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

import type { BerryGrid, MoonLabBackend } from "../backend/mod.ts";
import { type Circuit, CIRCUITS } from "./circuits.ts";
import { Job } from "./jobs.ts";
import { computeReadout, DEFAULT_SCAN_LIMIT } from "./readout.ts";
import { type Frame, Surface } from "../ui/cells.ts";
import { paintProbabilitiesBody, type ProbabilityReadout } from "../ui/probabilities.ts";
import { type BandField, paintBandGeometry } from "../ui/band_geometry.ts";
import {
  activeCorrections,
  NO_CORRECTIONS,
  type Orbital,
  orbitalIsValid,
  orbitalLabel,
  type OrbitalPhysics,
} from "./orbital.ts";
import { densityToRgb, ImageLayer, probeGraphics } from "../ui/graphics.ts";
import { orbitalRamp } from "../ui/orbital_view.ts";
import { ELEMENTS } from "./elements.ts";
import { computeOrbital, type OrbitalResult } from "./orbital_job.ts";
import { paintOrbital } from "../ui/orbital_view.ts";
import { measureThreshold, type ThresholdCurve } from "./repetition_code.ts";
import { paintDecoder } from "../ui/decoder_view.ts";
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

const WINDOW_IDS = ["probabilities", "bands", "orbital", "decoder", "circuits", "session"] as const;
type WindowId = (typeof WINDOW_IDS)[number];

/** What persists between runs. Deliberately small. */
interface PersistedState {
  theme?: string;
  circuit?: string;
  qubits?: number;
  qwzMass?: number;
  orbital?: number;
  n?: number;
  l?: number;
  m?: number;
  element?: number;
  physics?: OrbitalPhysics;
  zoom?: number;
  guides?: boolean;
}

/**
 * The orbitals on offer, in shell order. Every entry satisfies l < n and
 * |m| <= l, asserted in the tests so a typo cannot ship a bogus quantum
 * number that would silently render as an empty slice.
 */
const ORBITALS: readonly Orbital[] = [
  { n: 1, l: 0, m: 0, z: 1 },
  { n: 2, l: 0, m: 0, z: 1 },
  { n: 2, l: 1, m: 0, z: 1 },
  { n: 2, l: 1, m: 1, z: 1 },
  { n: 3, l: 0, m: 0, z: 1 },
  { n: 3, l: 1, m: 0, z: 1 },
  { n: 3, l: 2, m: 0, z: 1 },
  { n: 3, l: 2, m: 2, z: 1 },
  { n: 4, l: 3, m: 0, z: 1 },
];

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
  readonly #bandJob = new Job<BerryGrid>();
  readonly #orbitalJob = new Job<OrbitalResult>();
  readonly #decoderJob = new Job<ThresholdCurve>();
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
  /** QWZ mass. Its sign and magnitude select the topological phase. */
  #qwzMass = -1;
  /** The orbital on show. Cycled through a fixed, valid sequence. */
  #orbitalIndex = 0;
  readonly #renderer: "kitty" | "half-block";
  readonly #image: ImageLayer;
  readonly #graphicsReason: string;
  /** Direct quantum numbers, as the web build exposes them. */
  #n = 3;
  #l = 2;
  #m = 0;
  #elementIndex = 0;
  #physics: OrbitalPhysics = NO_CORRECTIONS;
  #resolution = 48;
  #zoom = 1;
  #showGuides = false;
  /** Shots per threshold point. Enough to resolve the crossing, cheap enough
   * to recompute on demand. */
  readonly #decoderShots = 2000;

  constructor(options: MoonLabDesktopOptions) {
    this.#backend = options.backend;
    this.#onQuit = options.onQuit ?? (() => {});
    this.#scanLimit = options.scanLimit ?? DEFAULT_SCAN_LIMIT;
    this.#palette = desktopPalette(themeById(this.#themeId));
    const graphics = probeGraphics();
    this.#image = new ImageLayer(graphics.surface);
    this.#graphicsReason = graphics.reason;
    // What the terminal can actually do, not what it advertises.
    this.#renderer = this.#image.available ? "kitty" : "half-block";
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
          floatingRect: { column: 2, row: 2, width: 40, height: 14 },
        },
        {
          id: "bands",
          title: "Band geometry",
          minWidth: 30,
          minHeight: 10,
          placement: "floating",
          state: "normal",
          floatingRect: { column: 2, row: 17, width: 40, height: 12 },
        },
        {
          id: "orbital",
          title: "Schrödinger",
          minWidth: 24,
          minHeight: 10,
          placement: "floating",
          state: "normal",
          floatingRect: { column: 44, row: 2, width: 30, height: 27 },
        },
        {
          id: "decoder",
          title: "QEC decoder",
          minWidth: 26,
          minHeight: 10,
          placement: "floating",
          state: "normal",
          floatingRect: { column: 76, row: 19, width: 26, height: 10 },
        },
        {
          id: "circuits",
          title: "Circuits",
          minWidth: 24,
          minHeight: 8,
          placement: "floating",
          state: "normal",
          floatingRect: { column: 76, row: 2, width: 26, height: 9 },
        },
        {
          id: "session",
          title: "Session",
          minWidth: 24,
          minHeight: 6,
          placement: "floating",
          state: "normal",
          floatingRect: { column: 76, row: 12, width: 26, height: 6 },
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
        if (typeof saved?.qwzMass === "number") this.#qwzMass = saved.qwzMass;
        if (
          typeof saved?.orbital === "number" && saved.orbital >= 0 &&
          saved.orbital < ORBITALS.length
        ) {
          this.#orbitalIndex = saved.orbital;
          const preset = ORBITALS[this.#orbitalIndex];
          this.#n = preset.n;
          this.#l = preset.l;
          this.#m = preset.m;
        }
        if (typeof saved?.n === "number") this.#n = saved.n;
        if (typeof saved?.l === "number") this.#l = saved.l;
        if (typeof saved?.m === "number") this.#m = saved.m;
        if (typeof saved?.element === "number") {
          this.#elementIndex = Math.max(0, Math.min(ELEMENTS.length - 1, saved.element));
        }
        if (saved?.physics) this.#physics = saved.physics;
        if (typeof saved?.zoom === "number") this.#zoom = saved.zoom;
        if (typeof saved?.guides === "boolean") this.#showGuides = saved.guides;
        this.#clampQuantumNumbers();
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
    this.#runBands();
    this.#runOrbital();
    this.#runDecoder();
  }

  #persist(): void {
    // Fire and forget: persistence must never delay a frame, and losing the
    // last theme change to a crash costs nothing worth blocking for.
    void this.#store?.set("state", {
      theme: this.#themeId,
      circuit: this.#circuit.id,
      qubits: this.#numQubits,
      qwzMass: this.#qwzMass,
      orbital: this.#orbitalIndex,
      n: this.#n,
      l: this.#l,
      m: this.#m,
      element: this.#elementIndex,
      physics: this.#physics,
      zoom: this.#zoom,
      guides: this.#showGuides,
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

  /** Recomputes the Berry field. Off the frame loop, like everything else. */
  #runBands(): void {
    if (!this.#backend.capabilities.bandGeometry || !this.#backend.berryGrid) return;
    const m = this.#qwzMass;
    const berryGrid = this.#backend.berryGrid.bind(this.#backend);
    this.#bandJob.start(() => berryGrid({ kind: "qwz", m }, 32));
  }

  /**
   * Kitty-class terminals can show a real image; everything else gets the
   * half-block path. Detected once from the environment, because the console
   * presenter exposes no image channel and the host would have to supply one
   * through `ShellCapabilities.extras` for the image path to be usable.
   */
  static detectRenderer(): "kitty" | "half-block" {
    const env = (name: string) => {
      try {
        return Deno.env.get(name) ?? "";
      } catch {
        return "";
      }
    };
    const term = env("TERM").toLowerCase();
    const program = env("TERM_PROGRAM").toLowerCase();
    const kitty = term.includes("kitty") || env("KITTY_WINDOW_ID") !== "" ||
      program.includes("ghostty") || program.includes("wezterm");
    return kitty ? "kitty" : "half-block";
  }

  #runOrbital(): void {
    const orbital = this.#orbital;
    const backend = this.#backend;
    const physics = this.#physics;
    const resolution = this.#resolution;
    const zoom = this.#zoom;
    this.#orbitalJob.start(() => computeOrbital(backend, orbital, resolution, physics, zoom));
  }

  /** Steps through the preset shells, as the web build's dropdown does. */
  #cycleOrbital(delta: number): void {
    const count = ORBITALS.length;
    this.#orbitalIndex = (this.#orbitalIndex + delta + count) % count;
    const preset = ORBITALS[this.#orbitalIndex];
    this.#n = preset.n;
    this.#l = preset.l;
    this.#m = preset.m;
    this.#status = `orbital: ${orbitalLabel(this.#orbital)}`;
    this.#persist();
    this.#runOrbital();
  }

  get #orbital(): Orbital {
    return { n: this.#n, l: this.#l, m: this.#m, z: ELEMENTS[this.#elementIndex].z };
  }

  /** Keeps l < n and |m| <= l after any change to n, l, m or the element. */
  #clampQuantumNumbers(): void {
    this.#n = Math.max(1, Math.min(6, this.#n));
    this.#l = Math.max(0, Math.min(this.#n - 1, this.#l));
    this.#m = Math.max(-this.#l, Math.min(this.#l, this.#m));
  }

  #adjustQuantum(which: "n" | "l" | "m" | "z", delta: number): void {
    if (which === "n") this.#n += delta;
    else if (which === "l") this.#l += delta;
    else if (which === "m") this.#m += delta;
    else {
      this.#elementIndex = Math.max(
        0,
        Math.min(ELEMENTS.length - 1, this.#elementIndex + delta),
      );
    }
    this.#clampQuantumNumbers();
    const element = ELEMENTS[this.#elementIndex];
    this.#status = `${element.symbol}  ${orbitalLabel(this.#orbital)}`;
    this.#persist();
    this.#runOrbital();
  }

  #adjustZoom(factor: number): void {
    this.#zoom = Math.max(0.25, Math.min(6, this.#zoom * factor));
    this.#status = `zoom ×${this.#zoom.toFixed(2)}`;
    this.#persist();
    this.#runOrbital();
  }

  #togglePhysics(which: keyof OrbitalPhysics): void {
    this.#physics = { ...this.#physics, [which]: !this.#physics[which] };
    const active = activeCorrections(this.#physics);
    this.#status = active.length === 0 ? "hydrogenic baseline" : active.join(" + ");
    this.#persist();
    this.#runOrbital();
  }

  #paintOrbital(surface: Surface, rect: Rectangle): void {
    const snapshot = this.#orbitalJob.snapshot;
    const result = snapshot.value;
    const element = ELEMENTS[this.#elementIndex];
    const useImage = this.#image.available && result !== undefined;

    paintOrbital(surface, rect, {
      label: orbitalLabel(this.#orbital),
      detail: `n${this.#n} l${this.#l} m${this.#m}  ·  n/l/, . quantum  z element  1 2 3 physics`,
      element: `${element.symbol} (Z=${element.z})`,
      slice: result?.slice,
      busy: snapshot.status === "running",
      error: snapshot.status === "failed" ? snapshot.error : undefined,
      drift: result?.drift,
      renderer: this.#renderer,
      corrections: activeCorrections(this.#physics),
      resolution: this.#resolution,
      showGuides: this.#showGuides,
      reserveForImage: useImage,
      graphicsReason: this.#renderer === "kitty" ? undefined : this.#graphicsReason,
    }, this.#palette);

    if (useImage) {
      // The picture occupies the window body; the header and the two status
      // rows stay as cells so they keep following the theme.
      const ramp = orbitalRamp(this.#palette);
      const slice = result.slice;
      this.#image.show(snapshot.generation, {
        column: rect.column + 1,
        row: rect.row + 3,
        width: Math.max(1, rect.width - 2),
        height: Math.max(1, rect.height - 6),
      }, () => densityToRgb(slice, ramp, 256));
    }
  }

  #adjustMass(delta: number): void {
    const next = Math.round((this.#qwzMass + delta) * 100) / 100;
    if (next < -3.5 || next > 3.5) return;
    this.#qwzMass = next;
    this.#status = `QWZ m = ${next.toFixed(2)}`;
    this.#persist();
    this.#runBands();
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
    // The two hosts disagree about case: the browser lowercases a shifted
    // letter and reports shift separately, while the console reader leaves the
    // raw character, so shift+T arrives as "T" and would fall straight through
    // a switch on lowercase literals -- every reverse binding silently dead in
    // a terminal. Folding here is exotui's own idiom (workbench_menu,
    // workbench_terminal and workbench_window_host all do it).
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    switch (key) {
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
      case "o":
        return this.#cycleOrbital(event.shift ? -1 : 1);
      // Quantum numbers directly, as the web build's sliders expose them.
      case "n":
        return this.#adjustQuantum("n", event.shift ? -1 : 1);
      case "l":
        return this.#adjustQuantum("l", event.shift ? -1 : 1);
      case ",":
        return this.#adjustQuantum("m", -1);
      case ".":
        return this.#adjustQuantum("m", 1);
      case "z":
        return this.#adjustQuantum("z", event.shift ? -1 : 1);
      // The three multi-electron corrections.
      case "1":
        return this.#togglePhysics("screeningExchange");
      case "2":
        return this.#togglePhysics("relativisticSpinOrbit");
      case "3":
        return this.#togglePhysics("correlationMixing");
      case "g":
        this.#showGuides = !this.#showGuides;
        this.#status = `guides ${this.#showGuides ? "on" : "off"}`;
        this.#persist();
        return;
      case "9":
        return this.#adjustZoom(1 / 1.25);
      case "0":
        return this.#adjustZoom(1.25);
      case "[":
        return this.#adjustMass(-0.25);
      case "]":
        return this.#adjustMass(0.25);
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
    const hint = this.#status ||
      "j/k circuit  ± qubits  [ ] mass  o/n/l/,. orbital  z element  1 2 3 physics  g guides  9 0 zoom  t theme  q quit";
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
      case "bands":
        return this.#paintBands(surface, client);
      case "orbital":
        return this.#paintOrbital(surface, client);
      case "decoder":
        return this.#paintDecoder(surface, client);
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

  #paintBands(surface: Surface, rect: Rectangle): void {
    const snapshot = this.#bandJob.snapshot;
    const capable = this.#backend.capabilities.bandGeometry;
    paintBandGeometry(surface, rect, {
      modelLabel: "Qi-Wu-Zhang",
      paramLabel: `m = ${this.#qwzMass.toFixed(2)}   [ / ] to sweep`,
      field: snapshot.value as BandField | undefined,
      busy: snapshot.status === "running",
      unavailable: capable
        ? undefined
        : `the ${this.#backend.kind} backend does not export the quantum-geometry symbols`,
      error: snapshot.status === "failed" ? snapshot.error : undefined,
    }, this.#palette);
  }

  #runDecoder(): void {
    if (!this.#backend.capabilities.decoder || !this.#backend.decodeBatch) return;
    const backend = this.#backend;
    const shots = this.#decoderShots;
    // Rates straddle the repetition code's p=0.5 threshold, because the
    // crossing is the whole reason to draw several distances at once.
    this.#decoderJob.start(() =>
      measureThreshold(backend, [3, 5, 9], [0.1, 0.3, 0.45, 0.5, 0.55, 0.65], shots)
    );
  }

  #paintDecoder(surface: Surface, rect: Rectangle): void {
    const snapshot = this.#decoderJob.snapshot;
    paintDecoder(surface, rect, {
      curve: snapshot.value,
      busy: snapshot.status === "running",
      error: snapshot.status === "failed" ? snapshot.error : undefined,
      unavailable: this.#backend.capabilities.decoder
        ? undefined
        : `the ${this.#backend.kind} backend does not export the union-find decoder`,
      shots: this.#decoderShots,
    }, this.#palette);
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
    // Ordered by what survives truncation best: this window is narrow and
    // short, and the rows below the fold are the ones nobody misses.
    const lines: Array<[string, string]> = [
      ["backend", this.#backend.kind],
      ["theme", `${this.#palette.label} (${themeIndex(this.#themeId) + 1}/${THEMES.length})`],
      ["max qubits", String(caps.maxQubits)],
      ["gpu/decoder", `${caps.bandGeometry ? "y" : "n"}/${caps.decoder ? "y" : "n"}`],
      ["exports", caps.exportedFunctions ? String(caps.exportedFunctions) : "—"],
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
