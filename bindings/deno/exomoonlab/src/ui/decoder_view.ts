/**
 * The decoder window: a repetition code's threshold curve.
 *
 * Logical error rate against physical error rate, one series per code
 * distance. The shape is the point: below threshold the longer code is
 * better, above it the longer code is worse, and the series cross where they
 * meet. A single curve would show none of that, which is why this is the one
 * window here with several series and a legend.
 *
 * Series colour is categorical -- distance is an identity, not a magnitude --
 * so the hues come from fixed slots in the theme and never from a ramp.
 */

import type { Rect, Rgb, Style, Surface } from "./cells.ts";
import type { DesktopPalette } from "./theme.ts";
import type { ThresholdCurve } from "../app/repetition_code.ts";

export interface DecoderWindowState {
  readonly curve?: ThresholdCurve;
  readonly busy: boolean;
  readonly error?: string;
  readonly unavailable?: string;
  readonly shots: number;
}

/** Fixed slots, assigned in order and never cycled. */
export function seriesColors(palette: DesktopPalette): readonly Rgb[] {
  return [palette.accent, palette.success, palette.warning, palette.danger];
}

const MARKS = ["●", "▲", "■", "◆"] as const;

/**
 * Plots the curve on a log-ish vertical scale.
 *
 * Logical error spans several decades below threshold, so a linear axis would
 * flatten every interesting difference into the bottom row. sqrt is used
 * rather than log because a measured rate of exactly zero is common at small
 * p and has no logarithm.
 */
function plotRow(logical: number, rows: number): number {
  const t = Math.max(0, Math.min(1, Math.sqrt(logical)));
  return Math.round((1 - t) * (rows - 1));
}

export function paintDecoder(
  surface: Surface,
  rect: Rect,
  state: DecoderWindowState,
  palette: DesktopPalette,
): void {
  const muted: Style = { foreground: palette.muted, background: palette.surface };
  const left = rect.column + 1;
  const width = rect.width - 2;
  if (width < 16 || rect.height < 8) return;

  surface.writeFitted(left, rect.row, "Repetition code · union-find decoder", width, {
    foreground: palette.text,
    background: palette.surface,
    bold: true,
  });

  if (state.unavailable) {
    surface.writeFitted(left, rect.row + 2, "decoder unavailable", width, {
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
  if (!state.curve) {
    surface.writeFitted(left, rect.row + 2, state.busy ? "decoding…" : "no run yet", width, muted);
    return;
  }

  const curve = state.curve;
  const colors = seriesColors(palette);
  const plotTop = rect.row + 2;
  const plotRows = Math.max(3, rect.height - 6);
  const gutter = 6;
  const plotLeft = left + gutter;
  const plotWidth = Math.max(2, width - gutter);

  // Axis labels: the sqrt scale's midpoint is 0.25, worth naming so the
  // spacing is not mistaken for linear.
  surface.writeFitted(left, plotTop, "1.0", gutter - 1, muted);
  surface.writeFitted(left, plotTop + Math.floor(plotRows / 2), "0.25", gutter - 1, muted);
  surface.writeFitted(left, plotTop + plotRows - 1, "0", gutter - 1, muted);
  for (let r = 0; r < plotRows; r++) surface.set(plotLeft - 1, plotTop + r, "│", muted);

  // Series are dodged by one column each. Without it they land on identical
  // cells at every shared x and the last drawn simply hides the rest -- which
  // is exactly what a threshold plot must not do, since the whole point is
  // comparing distances at the same p.
  const dodge = curve.series.length > 1 ? 1 : 0;
  const span = Math.max(1, plotWidth - dodge * (curve.series.length - 1) - 1);

  curve.series.forEach((points, si) => {
    const style: Style = { foreground: colors[si % colors.length], background: palette.surface };
    points.forEach((point, pi) => {
      const base = points.length === 1 ? 0 : Math.round((pi / (points.length - 1)) * span);
      const x = base + si * dodge;
      if (x < 0 || x >= plotWidth) return;
      const y = plotRow(point.logical, plotRows);
      surface.set(plotLeft + x, plotTop + y, MARKS[si % MARKS.length], style);
    });
  });

  // The x axis, then the legend: identity is never colour alone, so each
  // series carries its mark and its distance in text.
  const axisRow = plotTop + plotRows;
  for (let c = 0; c < plotWidth; c++) surface.set(plotLeft + c, axisRow, "─", muted);
  const first = curve.rates[0];
  const last = curve.rates[curve.rates.length - 1];
  surface.writeFitted(plotLeft, axisRow + 1, `p=${first.toFixed(2)}`, 8, muted);
  const rightLabel = `p=${last.toFixed(2)}`;
  surface.writeFitted(
    plotLeft + Math.max(0, plotWidth - rightLabel.length),
    axisRow + 1,
    rightLabel,
    rightLabel.length,
    muted,
  );

  const legend = curve.distances
    .map((d, i) => `${MARKS[i % MARKS.length]} d=${d}`)
    .join("  ");
  surface.writeFitted(
    left,
    rect.row + rect.height - 1,
    `${legend}   ${state.shots} shots`,
    width,
    muted,
  );
}
