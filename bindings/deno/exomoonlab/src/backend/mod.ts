/**
 * The backend seam and its two implementations.
 *
 * Callers should reach for {@link selectBackend}; it probes rather than
 * guesses, and reports which implementation answered. The architecture note
 * for this seam is `plan/arch/exomoonlab.md`.
 */

export type { BackendCapabilities, BackendKind, MoonLabBackend, StateHandle } from "./types.ts";
export { BackendUnavailableError } from "./types.ts";
export { nativeLibraryCandidates, openNativeBackend } from "./native.ts";
export { openWasmBackend, wasmGlueCandidates } from "./wasm.ts";

import { type BackendKind, BackendUnavailableError, type MoonLabBackend } from "./types.ts";
import { openNativeBackend } from "./native.ts";
import { openWasmBackend } from "./wasm.ts";

export interface SelectBackendOptions {
  /**
   * Force one implementation. Without this the probe prefers `native`, which
   * is faster and carries the fuller C surface.
   */
  prefer?: BackendKind;
  /** Probe only this one; a failure throws rather than falling back. */
  only?: BackendKind;
}

/**
 * Picks a backend at startup rather than at build time, so one binary can run
 * wherever it lands. Order is native, then WASM, unless overridden.
 */
export async function selectBackend(
  options: SelectBackendOptions = {},
): Promise<MoonLabBackend> {
  const openers: Record<BackendKind, () => Promise<MoonLabBackend>> = {
    native: openNativeBackend,
    wasm: openWasmBackend,
  };

  if (options.only) return await openers[options.only]();

  const order: BackendKind[] = options.prefer === "wasm" ? ["wasm", "native"] : ["native", "wasm"];

  const failures: string[] = [];
  for (const kind of order) {
    try {
      return await openers[kind]();
    } catch (cause) {
      failures.push(cause instanceof Error ? cause.message : String(cause));
    }
  }
  throw new BackendUnavailableError(
    order[0],
    `no backend could be opened.\n\n${failures.join("\n\n")}`,
  );
}
