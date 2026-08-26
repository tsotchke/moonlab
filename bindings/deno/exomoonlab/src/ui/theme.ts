/**
 * Themes.
 *
 * exotui ships seventeen `ShellThemeSpec`s and the console uses all of them
 * rather than inventing a palette: a theme is a property of the shell, not of
 * this application, and duplicating one here would guarantee they drift.
 *
 * One theme is added -- `moonlab` -- because the console should be able to
 * look like the project it belongs to.
 */

import {
  SHELL_THEMES,
  shellActiveTitlebarForeground,
  type ShellThemeSpec,
} from "@ubernaut/exotui/shell";
import type { Rgb } from "./cells.ts";
import type { WindowPalette } from "./probabilities.ts";

/** The console's own theme, in the project's colours. */
export const MOONLAB_THEME: ShellThemeSpec = {
  id: "moonlab",
  label: "MoonLab",
  background: [5, 7, 13],
  surface: [12, 16, 26],
  surfaceStrong: [20, 26, 40],
  border: [46, 56, 78],
  text: [219, 234, 254],
  muted: [122, 132, 148],
  accent: [126, 200, 227],
  success: [126, 217, 163],
  warning: [226, 178, 106],
  danger: [232, 122, 122],
};

/** Every theme the console offers, the MoonLab one first. */
export const THEMES: readonly ShellThemeSpec[] = [MOONLAB_THEME, ...SHELL_THEMES];

export function themeById(id: string): ShellThemeSpec {
  return THEMES.find((theme) => theme.id === id) ?? MOONLAB_THEME;
}

export function themeIndex(id: string): number {
  const found = THEMES.findIndex((theme) => theme.id === id);
  return found < 0 ? 0 : found;
}

/**
 * The colours the desktop paints with, resolved once per theme.
 *
 * `onAccent` comes from exotui's own luminance rule rather than a guess, so a
 * pale accent gets dark title text and a dark one gets light -- the thing that
 * goes wrong first when an application picks title colours by eye.
 */
export interface DesktopPalette extends WindowPalette {
  readonly id: string;
  readonly label: string;
  readonly background: Rgb;
  readonly surface: Rgb;
  readonly surfaceStrong: Rgb;
  readonly success: Rgb;
  readonly danger: Rgb;
  /** Legible foreground on top of `accent`. */
  readonly onAccent: Rgb;
}

export function desktopPalette(theme: ShellThemeSpec): DesktopPalette {
  return {
    id: theme.id,
    label: theme.label,
    background: theme.background,
    surface: theme.surface,
    surfaceStrong: theme.surfaceStrong,
    border: theme.border,
    text: theme.text,
    muted: theme.muted,
    accent: theme.accent,
    success: theme.success,
    warning: theme.warning,
    danger: theme.danger,
    onAccent: shellActiveTitlebarForeground(theme),
  };
}
