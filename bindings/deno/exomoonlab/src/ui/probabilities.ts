/**
 * The amplitudes/probabilities window.
 *
 * A pure painter: readout in, cells out. It touches no backend, no terminal,
 * and no clock, which is what lets it be tested headlessly and rendered
 * identically by either host.
 *
 * It renders only the strongest basis states. At 20 qubits the state vector
 * has a million entries; carrying them all into the frame path would stall the
 * loop this design exists to protect. Selecting the top entries is the job's
 * work, done off-thread -- see `src/app/console_app.ts`.
 */

import type { Rect, Rgb, Style, Surface } from "./cells.ts";

export interface BasisEntry {
  readonly index: number;
  readonly probability: number;
}

/** Everything the window draws, already reduced to a renderable size. */
export interface ProbabilityReadout {
  readonly numQubits: number;
  readonly stateDim: number;
  /** Basis states actually examined; may be less than `stateDim`. */
  readonly scanned: number;
  /** Strongest basis states, descending. */
  readonly top: readonly BasisEntry[];
  /** Probability mass covered by `top`. */
  readonly coveredProbability: number;
  readonly entropy: number;
  readonly purity: number;
  /** P(qubit = 1), one per qubit. */
  readonly qubitOnes: readonly number[];
}

export interface WindowPalette {
  readonly text: Rgb;
  readonly muted: Rgb;
  readonly accent: Rgb;
  readonly border: Rgb;
  readonly warning: Rgb;
}

export const DEFAULT_PALETTE: WindowPalette = {
  text: [220, 223, 228],
  muted: [122, 132, 148],
  accent: [126, 200, 227],
  border: [72, 82, 98],
  warning: [226, 178, 106],
};

export interface WindowState {
  readonly title: string;
  readonly subtitle: string;
  readonly readout?: ProbabilityReadout;
  /** Shown instead of a readout while the first run is in flight. */
  readonly busy: boolean;
  readonly error?: string;
}

const BAR_GLYPHS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"] as const;

/** A fractional-width bar, so small probabilities stay visible. */
export function bar(fraction: number, width: number): string {
  if (!(fraction > 0) || width <= 0) return "";
  const eighths = Math.round(Math.max(0, Math.min(1, fraction)) * width * 8);
  const full = Math.floor(eighths / 8);
  const remainder = eighths % 8;
  return "█".repeat(Math.min(full, width)) +
    (full < width && remainder > 0 ? BAR_GLYPHS[remainder] : "");
}

/** `|0110⟩`, padded to the register width. */
export function basisLabel(index: number, numQubits: number): string {
  return `|${index.toString(2).padStart(numQubits, "0")}⟩`;
}

/**
 * The boxed form: draws its own border and title.
 *
 * Used by the single-window app. On the desktop the window chrome already
 * supplies a border and a title bar, so that path calls
 * {@link paintProbabilitiesBody} instead -- drawing a second box inside the
 * chrome overwrote the title bar and its controls.
 */
export function paintProbabilities(
  surface: Surface,
  rect: Rect,
  state: WindowState,
  palette: WindowPalette = DEFAULT_PALETTE,
): void {
  surface.box(rect.column, rect.row, rect.width, rect.height, { foreground: palette.border });
  paintProbabilitiesBody(
    surface,
    { column: rect.column + 2, row: rect.row, width: rect.width - 4, height: rect.height },
    state,
    palette,
    { title: true },
  );
}

export interface BodyOptions {
  /** Draw the title and subtitle lines. Off when chrome already shows them. */
  readonly title?: boolean;
}

/** The contents alone: no border, no assumption about who drew one. */
export function paintProbabilitiesBody(
  surface: Surface,
  rect: Rect,
  state: WindowState,
  palette: WindowPalette = DEFAULT_PALETTE,
  options: BodyOptions = {},
): void {
  const text: Style = { foreground: palette.text };
  const muted: Style = { foreground: palette.muted };
  const accent: Style = { foreground: palette.accent };

  const inner = rect.column;
  const innerWidth = rect.width;
  if (innerWidth < 8 || rect.height < 4) return;

  let y = rect.row;
  if (options.title) {
    surface.writeFitted(inner, y, ` ${state.title} `, innerWidth, {
      foreground: palette.accent,
      bold: true,
    });
    surface.writeFitted(inner, y + 1, state.subtitle, innerWidth, muted);
    y += 3;
  } else {
    surface.writeFitted(inner, y, state.subtitle, innerWidth, muted);
    y += 2;
  }
  const lastRow = rect.row + rect.height - 2;

  if (state.error) {
    surface.writeFitted(inner, y, `error: ${state.error}`, innerWidth, {
      foreground: palette.warning,
    });
    return;
  }
  if (!state.readout) {
    surface.writeFitted(inner, y, state.busy ? "computing…" : "no result yet", innerWidth, muted);
    return;
  }

  const readout = state.readout;

  // Layout: label, bar, percentage. The bar takes whatever is left.
  const labelWidth = Math.min(readout.numQubits + 2, Math.max(6, innerWidth - 20));
  const percentWidth = 9;
  const barWidth = Math.max(1, innerWidth - labelWidth - percentWidth - 2);

  for (const entry of readout.top) {
    if (y > lastRow - 3) break;
    surface.writeFitted(inner, y, basisLabel(entry.index, readout.numQubits), labelWidth, text);
    surface.write(inner + labelWidth + 1, y, bar(entry.probability, barWidth), accent);
    const percent = `${(entry.probability * 100).toFixed(3)}%`;
    surface.write(
      inner + labelWidth + 1 + barWidth + 1,
      y,
      percent.padStart(percentWidth - 1),
      entry.probability > 0 ? text : muted,
    );
    y += 1;
  }

  if (readout.top.length < readout.stateDim && y <= lastRow - 2) {
    const partial = readout.scanned < readout.stateDim;
    // Say plainly when the scan was bounded. Claiming a share "of the mass"
    // from a partial survey would be a number we did not actually measure.
    const others = (readout.stateDim - readout.top.length).toLocaleString();
    // Clamp: summing floats can put the covered mass a hair above 1, and
    // "-0.000% of the mass" is a worse lie than a rounded zero.
    const remaining = Math.max(0, 1 - readout.coveredProbability);
    const note = partial
      ? `… showing ${readout.top.length} of the first ${readout.scanned.toLocaleString()} ` +
        `basis states scanned (of ${readout.stateDim.toLocaleString()}; scan bounded)`
      : remaining < 5e-6
      ? `… ${others} more basis states, none with measurable probability`
      : `… ${others} more basis states, ${(remaining * 100).toFixed(3)}% of the mass`;
    surface.writeFitted(inner, y, note, innerWidth, muted);
    y += 1;
  }

  // Footer: the scalars, and the per-qubit marginals when they fit.
  const footer = lastRow;
  surface.writeFitted(
    inner,
    footer - 1,
    `entropy ${readout.entropy.toFixed(6)}   purity ${readout.purity.toFixed(6)}   ` +
      `dim ${readout.stateDim.toLocaleString()}`,
    innerWidth,
    muted,
  );
  const marginals = readout.qubitOnes
    .map((p, q) => `q${q}:${p.toFixed(3)}`)
    .join("  ");
  surface.writeFitted(inner, footer, `P(1)  ${marginals}`, innerWidth, muted);
}
