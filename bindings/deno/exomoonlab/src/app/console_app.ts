/**
 * The MoonLab console application.
 *
 * Written against the presenter seam and nothing else: it composes cells and
 * takes events, and has no idea whether a terminal or a browser is showing
 * them. That is the whole point -- the same object runs under either host.
 *
 * `frame()` is synchronous and never awaits. Every backend call goes through
 * a {@link Job}; the frame paints whatever the job has most recently produced.
 */

import type { MoonLabBackend } from "../backend/mod.ts";
import { type Circuit, CIRCUITS } from "./circuits.ts";
import { Job } from "./jobs.ts";
import { computeReadout, DEFAULT_SCAN_LIMIT } from "./readout.ts";
import { type Frame, Surface } from "../ui/cells.ts";
import {
  DEFAULT_PALETTE,
  paintProbabilities,
  type ProbabilityReadout,
  type WindowPalette,
} from "../ui/probabilities.ts";

/**
 * Structural stand-in for exotui's `KeyPressEvent`. Only these fields are
 * used, and declaring them here keeps `deno check` offline; see
 * `src/ui/cells.ts` for why the seam's types are mirrored rather than
 * imported at 0.6.0.
 */
export interface KeyEvent {
  readonly key: string;
  readonly ctrl?: boolean;
  readonly shift?: boolean;
  readonly meta?: boolean;
}

export interface Size {
  readonly columns: number;
  readonly rows: number;
}

export interface MoonLabAppOptions {
  readonly backend: MoonLabBackend;
  /** Called when the user asks to leave; the host stops the loop. */
  readonly onQuit?: () => void;
  readonly palette?: WindowPalette;
  readonly scanLimit?: number;
}

const MIN_QUBITS = 1;

export class MoonLabApp {
  readonly #backend: MoonLabBackend;
  readonly #onQuit: () => void;
  readonly #palette: WindowPalette;
  readonly #scanLimit: number;
  readonly #job = new Job<ProbabilityReadout>();

  #circuitIndex = CIRCUITS.findIndex((c) => c.id === "bell");
  #numQubits: number;
  #lastKey = "";

  constructor(options: MoonLabAppOptions) {
    this.#backend = options.backend;
    this.#onQuit = options.onQuit ?? (() => {});
    this.#palette = options.palette ?? DEFAULT_PALETTE;
    this.#scanLimit = options.scanLimit ?? DEFAULT_SCAN_LIMIT;
    if (this.#circuitIndex < 0) this.#circuitIndex = 0;
    this.#numQubits = this.#circuit.defaultQubits;
  }

  get #circuit(): Circuit {
    return CIRCUITS[this.#circuitIndex];
  }

  /** Largest register this backend will attempt. */
  get #maxQubits(): number {
    return this.#backend.capabilities.maxQubits;
  }

  init(): Promise<void> {
    this.#run();
    return Promise.resolve();
  }

  #run(): void {
    const circuit = this.#circuit;
    const numQubits = this.#numQubits;
    this.#job.start(() =>
      computeReadout(this.#backend, circuit, { numQubits, scanLimit: this.#scanLimit })
    );
  }

  #selectCircuit(delta: number): void {
    const count = CIRCUITS.length;
    this.#circuitIndex = (this.#circuitIndex + delta + count) % count;
    this.#numQubits = this.#circuit.defaultQubits;
    this.#run();
  }

  #resize(delta: number): void {
    const next = Math.max(
      Math.max(MIN_QUBITS, this.#circuit.minQubits),
      Math.min(this.#maxQubits, this.#numQubits + delta),
    );
    if (next === this.#numQubits) return;
    this.#numQubits = next;
    this.#run();
  }

  key(event: KeyEvent): void {
    this.#lastKey = event.key;
    if (event.ctrl && event.key === "c") return this.#onQuit();
    switch (event.key) {
      case "q":
      case "escape":
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
        return this.#resize(1);
      case "-":
      case "_":
      case "left":
        return this.#resize(-1);
    }
  }

  frame(_now: number, size: Size): Frame {
    const surface = new Surface(size.columns, size.rows);
    const snapshot = this.#job.snapshot;
    const circuit = this.#circuit;

    const status = snapshot.status === "running"
      ? "running"
      : snapshot.durationMs !== undefined
      ? `${snapshot.durationMs.toFixed(1)}ms`
      : "idle";

    paintProbabilities(
      surface,
      { column: 0, row: 0, width: size.columns, height: Math.max(0, size.rows - 1) },
      {
        title: `MoonLab · ${circuit.name}`,
        subtitle: `${this.#backend.kind} backend · ${this.#numQubits} qubits · ${status}` +
          (snapshot.status === "running" && snapshot.value ? " · showing previous" : ""),
        readout: snapshot.value,
        busy: snapshot.status === "running",
        error: snapshot.status === "failed" ? snapshot.error : undefined,
      },
      this.#palette,
    );

    const help = " j/k circuit   ±  qubits   r rerun   q quit ";
    surface.writeFitted(1, size.rows - 1, help, Math.max(0, size.columns - 2), {
      foreground: this.#palette.muted,
    });
    if (this.#lastKey) {
      const tag = ` ${this.#lastKey} `;
      surface.writeFitted(
        Math.max(0, size.columns - tag.length - 1),
        size.rows - 1,
        tag,
        tag.length,
        { foreground: this.#palette.border },
      );
    }
    return surface.frame();
  }
}
