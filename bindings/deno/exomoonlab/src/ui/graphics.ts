/**
 * Real images, where the terminal can show them.
 *
 * exotui publishes a `GraphicsSurface` abstraction with a kitty implementation
 * and a no-op one, so this file does not speak the graphics protocol itself --
 * it decides *what* to draw and hands over pixels. Terminals that cannot show
 * an image get the no-op surface and the caller falls back to half-blocks,
 * which is why nothing here throws when graphics are unavailable.
 *
 * Placement is reconciled, not painted. `frame()` is synchronous and
 * `putImage` is not, so the frame records the placement it wants and a
 * guarded async pass makes the terminal match it. The same discipline the
 * backend calls follow.
 */

import {
  createKittyGraphicsSurface,
  createNoopGraphicsSurface,
  detectKittyGraphicsCapability,
  type GraphicsHandle,
  type GraphicsPlacement,
  type GraphicsSurface,
} from "@ubernaut/exotui/runtime";
import type { Rgb } from "./cells.ts";
import type { DensitySlice } from "../app/orbital.ts";
import { sequentialColor } from "./colormap.ts";

export interface GraphicsProbe {
  readonly surface: GraphicsSurface;
  /** Why images are or are not available, in the terminal's own terms. */
  readonly reason: string;
  readonly mode: string;
}

/**
 * Builds the surface this terminal can actually drive.
 *
 * tmux passthrough is requested by default. Ghostty and kitty both implement
 * the protocol, and inside tmux the escapes have to be wrapped to reach them;
 * without asking, a perfectly capable terminal reports "passthrough must be
 * enabled" and silently gets half-blocks. tmux still needs
 * `set -g allow-passthrough on` at its end -- if that is missing the wrapped
 * escapes are dropped rather than printed as garbage, and the window says
 * which mode it is in.
 *
 * `DENO_TUI_KITTY=0` disables images; `DENO_TUI_KITTY=1` forces them.
 */
export function probeGraphics(): GraphicsProbe {
  try {
    const capability = detectKittyGraphicsCapability({ tmuxPassthrough: true });
    if (!capability.supported) {
      return {
        surface: createNoopGraphicsSurface(),
        reason: capability.reason,
        mode: capability.mode,
      };
    }
    const encoder = new TextEncoder();
    return {
      surface: createKittyGraphicsSurface({
        capability,
        mode: capability.mode,
        writer: {
          write: (data: string) => {
            Deno.stdout.writeSync(encoder.encode(data));
          },
        },
      }),
      reason: capability.reason,
      mode: capability.mode,
    };
  } catch (cause) {
    // Detection touches the environment and the tty; a console that cannot
    // probe simply does not get images, and says why.
    return {
      surface: createNoopGraphicsSurface(),
      reason: cause instanceof Error ? cause.message : "graphics probe failed",
      mode: "unknown",
    };
  }
}

/** Backwards-compatible shorthand. */
export function createGraphicsSurface(): GraphicsSurface {
  return probeGraphics().surface;
}

/** Renders a density slice to packed RGB, one pixel per sample. */
export function densityToRgb(
  slice: DensitySlice,
  ramp: readonly Rgb[],
  pixels: number,
): { data: Uint8Array; width: number; height: number } {
  const size = Math.max(1, Math.min(pixels, slice.size));
  const data = new Uint8Array(size * size * 3);
  for (let row = 0; row < size; row++) {
    const sourceRow = Math.min(slice.size - 1, Math.floor((row * slice.size) / size));
    for (let column = 0; column < size; column++) {
      const sourceColumn = Math.min(slice.size - 1, Math.floor((column * slice.size) / size));
      const [r, g, b] = sequentialColor(
        ramp,
        slice.density[sourceRow * slice.size + sourceColumn],
        slice.max,
      );
      const offset = (row * size + column) * 3;
      data[offset] = r;
      data[offset + 1] = g;
      data[offset + 2] = b;
    }
  }
  return { data, width: size, height: size };
}

function samePlacement(a: GraphicsPlacement, b: GraphicsPlacement): boolean {
  return a.column === b.column && a.row === b.row && a.width === b.width && a.height === b.height;
}

/**
 * One image on screen, kept in step with what the frame wants.
 *
 * `show` is safe to call every frame: it returns immediately and only touches
 * the terminal when the picture or its placement actually changed.
 */
export class ImageLayer {
  readonly #surface: GraphicsSurface;
  #handle?: GraphicsHandle;
  #placement?: GraphicsPlacement;
  #generation = -1;
  #busy = false;
  #failed = false;

  constructor(surface: GraphicsSurface) {
    this.#surface = surface;
  }

  get available(): boolean {
    return this.#surface.kind !== "none" && !this.#failed;
  }

  get kind(): string {
    return this.#surface.kind;
  }

  show(
    generation: number,
    placement: GraphicsPlacement,
    render: () => { data: Uint8Array; width: number; height: number },
  ): void {
    if (!this.available || this.#busy) return;
    const unchanged = generation === this.#generation && this.#placement &&
      samePlacement(this.#placement, placement);
    if (unchanged) return;

    this.#busy = true;
    const image = render();
    void (async () => {
      try {
        if (this.#handle) await this.#surface.deleteImage(this.#handle, "placement");
        this.#handle = await this.#surface.putImage({
          data: image.data,
          encoding: "bytes",
          format: 24,
          pixelWidth: image.width,
          pixelHeight: image.height,
        }, placement);
        this.#placement = placement;
        this.#generation = generation;
      } catch {
        // One failure is enough: a terminal that rejects a placement will
        // reject the next one too, and retrying every frame would flood it.
        this.#failed = true;
      } finally {
        this.#busy = false;
      }
    })();
  }

  /** Removes the image, e.g. when its window closes. */
  hide(): void {
    const handle = this.#handle;
    if (!handle) return;
    this.#handle = undefined;
    this.#placement = undefined;
    this.#generation = -1;
    void this.#surface.deleteImage(handle, "placement").catch(() => {});
  }
}
