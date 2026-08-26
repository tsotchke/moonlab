/**
 * The band-geometry window: Berry curvature over the Brillouin zone.
 *
 * A pure painter, like the probabilities one. Two things are shown, and the
 * order matters: the Chern number is the headline -- a single integer that
 * says which topological phase the model is in -- and the curvature field is
 * the evidence for it. A reader who takes only the number away has taken away
 * the right thing.
 *
 * Each grid point is painted two columns wide, because a terminal cell is
 * roughly twice as tall as it is wide and a Brillouin zone is square.
 */

import type { Rect, Rgb, Style, Surface } from "./cells.ts";
import type { DesktopPalette } from "./theme.ts";
import { type DivergingRamp, divergingRamp, neutralMidpoint, rampColor } from "./colormap.ts";

export interface BandField {
  readonly n: number;
  readonly curvature: Float64Array;
  readonly chern: number;
  readonly min: number;
  readonly max: number;
}

export interface BandWindowState {
  readonly modelLabel: string;
  readonly paramLabel: string;
  readonly field?: BandField;
  readonly busy: boolean;
  /** Set when the backend cannot do band geometry at all. */
  readonly unavailable?: string;
  readonly error?: string;
}

/** The theme's diverging pair: its two signed-meaning hues, neutral centre. */
export function paletteRamp(palette: DesktopPalette): DivergingRamp {
  return divergingRamp(
    palette.accent,
    neutralMidpoint(palette.surface, palette.muted),
    palette.danger,
  );
}

/** Grid points that fit, given two columns per point and room for the frame. */
export function fittedGridSize(rect: Rect, requested: number): number {
  const byWidth = Math.floor((rect.width - 2) / 2);
  const byHeight = rect.height - 5;
  return Math.max(2, Math.min(requested, byWidth, byHeight));
}

export function paintBandGeometry(
  surface: Surface,
  rect: Rect,
  state: BandWindowState,
  palette: DesktopPalette,
): void {
  const muted: Style = { foreground: palette.muted, background: palette.surface };
  const text: Style = { foreground: palette.text, background: palette.surface };

  const left = rect.column + 1;
  const width = rect.width - 2;
  if (width < 12 || rect.height < 6) return;

  surface.writeFitted(left, rect.row, `${state.modelLabel} · ${state.paramLabel}`, width, muted);

  if (state.unavailable) {
    surface.writeFitted(left, rect.row + 2, "band geometry unavailable", width, {
      foreground: palette.warning,
      background: palette.surface,
    });
    surface.writeFitted(left, rect.row + 3, state.unavailable, width, muted);
    return;
  }
  if (state.error) {
    surface.writeFitted(left, rect.row + 2, `error: ${state.error}`, width, {
      foreground: palette.danger,
      background: palette.surface,
    });
    return;
  }
  if (!state.field) {
    surface.writeFitted(
      left,
      rect.row + 2,
      state.busy ? "computing…" : "no field yet",
      width,
      muted,
    );
    return;
  }

  const field = state.field;
  const ramp = paletteRamp(palette);
  // Symmetric about zero so the two signs are comparable at a glance.
  const extent = Math.max(Math.abs(field.min), Math.abs(field.max));
  const size = fittedGridSize(rect, field.n);
  const stride = field.n / size;

  const gridTop = rect.row + 2;
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      // Nearest-sample when the field is finer than the cells available.
      const sourceRow = Math.min(field.n - 1, Math.floor(row * stride));
      const sourceColumn = Math.min(field.n - 1, Math.floor(column * stride));
      const value = field.curvature[sourceRow * field.n + sourceColumn];
      const background: Rgb = rampColor(ramp, value, extent);
      const x = left + column * 2;
      surface.set(x, gridTop + row, " ", { background });
      surface.set(x + 1, gridTop + row, " ", { background });
    }
  }

  // The headline: a topological invariant is an integer, so show it as one.
  const chern = Math.round(field.chern);
  const chernText = `C = ${chern > 0 ? "+" : ""}${chern}`;
  surface.writeFitted(
    left + Math.max(0, width - chernText.length),
    rect.row,
    chernText,
    chernText.length,
    { foreground: palette.accent, background: palette.surface, bold: true },
  );

  // Colour bar, then its ends in text ink -- never in the ramp's colours.
  const barRow = gridTop + size + 1;
  if (barRow < rect.row + rect.height) {
    const barWidth = Math.min(width, ramp.steps.length * 2);
    for (let i = 0; i < barWidth; i++) {
      const t = (i / Math.max(1, barWidth - 1)) * 2 - 1;
      surface.set(left + i, barRow, " ", { background: rampColor(ramp, t, 1) });
    }
    const legend = `  Ω  ${(-extent).toExponential(1)} … 0 … ${extent.toExponential(1)}`;
    surface.writeFitted(left + barWidth, barRow, legend, width - barWidth, muted);
  }
  void text;
}
