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
import type { DensitySlice, Projection } from "../app/orbital.ts";

export interface OrbitalWindowState {
  readonly label: string;
  readonly detail: string;
  readonly slice?: DensitySlice;
  /** Rotated column density; when present it is drawn instead of the slice. */
  readonly projection?: Projection;
  /** Shown in the status row so the orientation is readable, not guessed. */
  readonly yaw?: number;
  readonly pitch?: number;
  readonly busy: boolean;
  readonly error?: string;
  /** L1 distance between MoonLab's probabilities and the analytic ones. */
  readonly drift?: number;
  /** Which render path is actually in use, not which one is wanted. */
  readonly renderer: "kitty" | "half-block";
  /** Element symbol and Z, from the periodic-table picker. */
  readonly element: string;
  /** Corrections currently switched on; empty means the hydrogenic baseline. */
  readonly corrections: readonly string[];
  /** Grid samples per axis. */
  readonly resolution: number;
  readonly showGuides: boolean;
  /** Rows the caller should leave for the image; the picture is not painted. */
  readonly reserveForImage?: boolean;
  /** Why images are unavailable, when they are. */
  readonly graphicsReason?: string;
}

/** Cartesian axes through the nucleus, as the web build's guides do. */
function paintGuides(
  surface: Surface,
  left: number,
  top: number,
  columns: number,
  rows: number,
  palette: DesktopPalette,
): void {
  const style: Style = { foreground: palette.border, background: palette.background };
  const midRow = top + Math.floor(rows / 2);
  const midColumn = left + Math.floor(columns / 2) * 2;
  for (let c = 0; c < columns * 2; c++) surface.set(left + c, midRow, "─", style);
  for (let r = 0; r < rows; r++) surface.set(midColumn, top + r, "│", style);
  surface.set(midColumn, midRow, "┼", style);
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

  surface.writeFitted(left, rect.row, `${state.element}  ${state.label}`, width, {
    foreground: palette.text,
    background: palette.surface,
    bold: true,
  });
  surface.writeFitted(left, rect.row + 1, state.detail, width, muted);

  if (state.error) {
    surface.writeFitted(left, rect.row + 2, `error: ${state.error}`, width, {
      foreground: palette.danger,
      background: palette.surface,
    });
    return;
  }
  if (!state.slice) {
    surface.writeFitted(left, rect.row + 3, state.busy ? "solving…" : "no slice yet", width, muted);
    return;
  }

  const ramp = orbitalRamp(palette);
  const slice = state.slice;
  const proj = state.projection;
  // One accessor over either source, so the half-block loop below does not
  // care which of the two it is drawing.
  const sampleSize = proj ? Math.min(proj.width, proj.height) : slice.size;
  const sampleAt = (sx: number, sy: number): number =>
    proj ? proj.image[sy * proj.width + sx] : slice.density[sy * slice.size + sx];
  const sampleMax = proj ? 1 : slice.max;
  const top = rect.row + 3;
  const rows = Math.max(1, rect.height - 6);
  // Two columns per sample keeps the plane square; two samples per cell row
  // via the half-block doubles the vertical detail.
  const columns = Math.min(Math.floor(width / 2), sampleSize);
  const sampleRows = Math.min(rows * 2, sampleSize);

  // With a real image over this area the cells stay clear, or the half-block
  // fallback would show through the picture.
  if (state.reserveForImage) {
    surface.fill(
      {
        column: left,
        row: top,
        width: columns * 2,
        height: Math.min(rows, Math.ceil(sampleRows / 2)),
      },
      " ",
      { background: palette.background },
    );
  } else {
    for (let cell = 0; cell < Math.min(rows, Math.ceil(sampleRows / 2)); cell++) {
      for (let column = 0; column < columns; column++) {
        const sx = Math.min(sampleSize - 1, Math.floor((column * sampleSize) / columns));
        const upperY = Math.min(
          sampleSize - 1,
          Math.floor(((cell * 2) * sampleSize) / sampleRows),
        );
        const lowerY = Math.min(
          sampleSize - 1,
          Math.floor(((cell * 2 + 1) * sampleSize) / sampleRows),
        );
        const upper = sequentialColor(ramp, sampleAt(sx, upperY), sampleMax);
        const lower = sequentialColor(ramp, sampleAt(sx, lowerY), sampleMax);
        const x = left + column * 2;
        surface.set(x, top + cell, "▀", { foreground: upper, background: lower });
        surface.set(x + 1, top + cell, "▀", { foreground: upper, background: lower });
      }
    }
  }

  if (state.showGuides && !state.reserveForImage) {
    paintGuides(surface, left, top, columns, Math.min(rows, Math.ceil(sampleRows / 2)), palette);
  }

  // Two status rows: what the physics is, then what the render is.
  const footer = rect.row + rect.height - 1;
  const corrections = state.corrections.length === 0
    ? "hydrogenic baseline"
    : state.corrections.join(" + ");
  surface.writeFitted(left, footer - 1, corrections, width, {
    foreground: state.corrections.length === 0 ? palette.muted : palette.success,
    background: palette.surface,
  });
  const drift = state.drift === undefined
    ? "round-trip pending"
    : `L1 ${state.drift.toExponential(1)}`;
  surface.writeFitted(
    left,
    footer,
    `±${slice.extent.toFixed(1)} a₀  ` +
      (proj
        ? `yaw ${Math.round(((state.yaw ?? 0) * 180) / Math.PI)}° pitch ${
          Math.round(((state.pitch ?? 0) * 180) / Math.PI)
        }°  `
        : `${state.resolution}²  `) +
      `${drift}  [${state.renderer}]` +
      (state.graphicsReason ? `  ${state.graphicsReason}` : ""),
    width,
    muted,
  );
}
