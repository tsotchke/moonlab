/**
 * The application launcher.
 *
 * Each demo is an application: closed until launched, opened from a menu,
 * present on the shelf once running. That framing is what the window host
 * already models -- a window's `closed` state is not "destroyed", it is "not
 * launched" -- so launching is a `restore` command and nothing here has to
 * invent a lifecycle.
 *
 * Menu behaviour comes from exotui: `moveWorkbenchMenuIndex` for traversal
 * with wrap-around, and the activation/close key predicates. Reimplementing
 * those is how a launcher ends up subtly disagreeing with the rest of the
 * shell about what Enter and Escape mean.
 */

import {
  isWorkbenchMenuActivationKey,
  isWorkbenchMenuCloseKey,
  moveWorkbenchMenuIndex,
} from "@ubernaut/exotui/shell";

/** One launchable demo. `id` is the window it opens. */
export interface LauncherApp {
  readonly id: string;
  readonly label: string;
  /** One line describing what the demo shows. */
  readonly hint: string;
}

export const APPS: readonly LauncherApp[] = [
  { id: "probabilities", label: "Probabilities", hint: "amplitudes, entropy, purity" },
  { id: "orbital", label: "Schrödinger", hint: "3-D orbital cloud, orbitable" },
  { id: "bands", label: "Band geometry", hint: "Berry curvature and Chern numbers" },
  { id: "decoder", label: "QEC decoder", hint: "repetition-code threshold" },
  { id: "circuits", label: "Circuits", hint: "the circuit catalog" },
  { id: "session", label: "Session", hint: "backend and capabilities" },
];

export interface LauncherState {
  readonly open: boolean;
  readonly index: number;
}

export const CLOSED_LAUNCHER: LauncherState = { open: false, index: 0 };

/** What a key did to the launcher, so the caller knows whether to act. */
export type LauncherAction =
  | { readonly kind: "none" }
  | { readonly kind: "state"; readonly state: LauncherState }
  | { readonly kind: "launch"; readonly app: LauncherApp; readonly state: LauncherState };

/**
 * Routes one key while the menu is open.
 *
 * Returns "none" for keys the launcher does not claim, so the caller can pass
 * them on rather than swallowing every keystroke whenever the menu is up.
 */
export function handleLauncherKey(state: LauncherState, key: string): LauncherAction {
  if (!state.open) return { kind: "none" };
  if (isWorkbenchMenuCloseKey(key)) return { kind: "state", state: CLOSED_LAUNCHER };
  if (isWorkbenchMenuActivationKey(key)) {
    const app = APPS[state.index];
    return app ? { kind: "launch", app, state: CLOSED_LAUNCHER } : { kind: "none" };
  }
  const moved = moveWorkbenchMenuIndex(state.index, APPS.length, { key }, { pageSize: 4 });
  if (moved !== state.index) return { kind: "state", state: { open: true, index: moved } };
  // j/k are the console's own list keys; accept them here too so the menu
  // does not feel like a different application.
  if (key === "j") {
    return { kind: "state", state: { open: true, index: (state.index + 1) % APPS.length } };
  }
  if (key === "k") {
    return {
      kind: "state",
      state: { open: true, index: (state.index - 1 + APPS.length) % APPS.length },
    };
  }
  return { kind: "none" };
}

/** The panel's size, from the longest entry it has to hold. */
export function launcherPanelSize(): { width: number; height: number } {
  const widest = APPS.reduce((w, a) => Math.max(w, a.label.length + a.hint.length + 5), 0);
  return { width: Math.min(52, widest + 4), height: APPS.length + 2 };
}
