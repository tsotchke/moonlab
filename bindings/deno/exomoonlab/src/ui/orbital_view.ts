/**
 * The Schrödinger window: |psi|^2 for a hydrogenic orbital.
 *
 * Two render paths, chosen by what the host can do:
 *
 *  - **Kitty graphics.** When the host supplies an image channel through
 *    `ShellCapabilities.extras`, the slice goes out as a real picture at the
 *    terminal's own resolution.
 *  - **Half-block cells.** Otherwise each cell carries two vertical samples
 *    via `▀` with a foreground for the upper half and a background for the
 *    lower, which doubles the vertical resolution for free and is what makes
 *    a 2p nodal plane visible at all in a small window.
 *
 * The scale is sequential, not diverging: a probability density is magnitude,
 * and there is no sign to encode.
 */

import type { Rect, Rgb, Style, Surface } from "./cells.ts";
import type { DesktopPalette } from "./theme.ts";
import { sequentialColor, sequentialRamp } from "./colormap.ts";
import type { DensitySlice } from "../app/orbital.ts";

export interface OrbitalWindowState {
  readonly label: string;
  readonly detail: string;
  readonly slice?: DensitySlice;
  readonly busy: boolean;
  readonly error?: string;
  /** L1 distance between MoonLab's probabilities and the analytic ones. */
  readonly drift?: number;
  /** Set when the image path is in use, for the status line. */
  readonly renderer: "kitty" | "half-block";
}

export function orbitalRamp(palette: DesktopPalette): readonly Rgb[] {
  return sequentialRamp(palette.background, palette.accent);
}

export function paintOrbital(
  surface: Surface,
  rect: Rect,
  state: OrbitalWindowState,
  palette: DesktopPalette,
): void {
  const muted: Style = { foreground: palette.muted, background: palette.surface };
  const left = rect.column + 1;
  const width = rect.width - 2;
  if (width < 12 || rect.height < 6) return;

  surface.writeFitted(left, rect.row, `${state.label}  ${state.detail}`, width, muted);

  if (state.error) {
    surface.writeFitted(left, rect.row + 2, `error: ${state.error}`, width, {
      foreground: palette.danger,
      background: palette.surface,
    });
    return;
  }
  if (!state.slice) {
    surface.writeFitted(left, rect.row + 2, state.busy ? "solving…" : "no slice yet", width, muted);
    return;
  }

  const ramp = orbitalRamp(palette);
  const slice = state.slice;
  const top = rect.row + 2;
  const rows = Math.max(1, rect.height - 4);
  // Two columns per sample keeps the plane square; two samples per cell row
  // via the half-block doubles the vertical detail.
  const columns = Math.min(Math.floor(width / 2), slice.size);
  const sampleRows = Math.min(rows * 2, slice.size);

  for (let cell = 0; cell < Math.min(rows, Math.ceil(sampleRows / 2)); cell++) {
    for (let column = 0; column < columns; column++) {
      const sx = Math.min(slice.size - 1, Math.floor((column * slice.size) / columns));
      const upperY = Math.min(slice.size - 1, Math.floor(((cell * 2) * slice.size) / sampleRows));
      const lowerY = Math.min(
        slice.size - 1,
        Math.floor(((cell * 2 + 1) * slice.size) / sampleRows),
      );
      const upper = sequentialColor(ramp, slice.density[upperY * slice.size + sx], slice.max);
      const lower = sequentialColor(ramp, slice.density[lowerY * slice.size + sx], slice.max);
      const x = left + column * 2;
      surface.set(x, top + cell, "▀", { foreground: upper, background: lower });
      surface.set(x + 1, top + cell, "▀", { foreground: upper, background: lower });
    }
  }

  const footer = rect.row + rect.height - 1;
  const scale = `±${slice.extent.toFixed(1)} a₀`;
  const drift = state.drift === undefined
    ? "MoonLab round-trip pending"
    : `MoonLab L1 drift ${state.drift.toExponential(2)}`;
  surface.writeFitted(left, footer, `${scale}   ${drift}   [${state.renderer}]`, width, muted);
}
