/**
 * Lightweight bridge to the libquantumsim WASM build for live
 * topological-invariant computation.
 *
 * The TopologyDemo page primarily uses closed-form analytical phase
 * boundaries, which are exact for every model it visualises.  This
 * bridge layers a numerical cross-check on top: when the WASM module
 * exposes `moonlab_qwz_chern` (i.e. the symbol is in
 * `bindings/javascript/packages/core/emscripten/exports.txt`), the
 * page calls into it and reports the WASM-computed Chern alongside
 * the closed-form result.  When the symbol is absent (older WASM
 * build, or environments without WebAssembly), the bridge returns
 * `null` and the page silently uses only the closed-form path.
 *
 * The WASM module ships in `public/moonlab.js` + `public/moonlab.wasm`
 * and is loaded once on demand via the global `MoonlabModule`
 * factory installed by `moonlab.js`.
 */

declare global {
  interface Window {
    MoonlabModule?: (mod?: Record<string, unknown>) => Promise<MoonlabRuntime>;
  }
}

interface MoonlabRuntime {
  cwrap?: (
    name: string,
    returnType: string | null,
    argTypes: ReadonlyArray<string>,
  ) => (...args: unknown[]) => unknown;
}

export type QwzChernFn = (m: number, n: number) => number;

type LoadState =
  | { kind: 'unloaded' }
  | { kind: 'loading'; promise: Promise<QwzChernFn | null> }
  | { kind: 'ready'; fn: QwzChernFn | null };

let state: LoadState = { kind: 'unloaded' };

function injectScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(
      `script[data-moonlab-wasm="${src}"]`,
    );
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error('wasm load failed')), {
        once: true,
      });
      return;
    }
    const tag = document.createElement('script');
    tag.src = src;
    tag.async = true;
    tag.dataset.moonlabWasm = src;
    tag.addEventListener('load', () => resolve(), { once: true });
    tag.addEventListener('error', () => reject(new Error('wasm script load failed')), {
      once: true,
    });
    document.head.appendChild(tag);
  });
}

async function load(): Promise<QwzChernFn | null> {
  if (typeof window === 'undefined' || typeof WebAssembly === 'undefined') {
    return null;
  }
  const moonUrl = `${import.meta.env.BASE_URL}moonlab.js`;
  try {
    if (!window.MoonlabModule) {
      await injectScript(moonUrl);
    }
    const factory = window.MoonlabModule;
    if (!factory) return null;
    const mod = await factory({ locateFile: (p: string) => `${import.meta.env.BASE_URL}${p}` });
    if (!mod.cwrap) return null;
    // moonlab_qwz_chern(double m, size_t N, double *raw_chern) -> int
    // We pass NULL (0) for raw_chern: the integer Chern is the return.
    const wrapped = mod.cwrap('moonlab_qwz_chern', 'number', [
      'number',
      'number',
      'number',
    ]) as (m: number, n: number, rawPtr: number) => number;
    return (m: number, n: number) => {
      const c = wrapped(m, n, 0);
      // The C ABI returns INT_MIN on failure; treat that as "no value".
      if (c === -2147483648) return Number.NaN;
      return c;
    };
  } catch {
    return null;
  }
}

/**
 * Returns the WASM-backed `moonlab_qwz_chern` if available, or `null`
 * if the WASM module is unavailable / does not export the symbol.
 *
 * The first call kicks off WASM initialisation.  Subsequent calls
 * return the cached result.
 */
export function getQwzChern(): Promise<QwzChernFn | null> {
  if (state.kind === 'ready') return Promise.resolve(state.fn);
  if (state.kind === 'loading') return state.promise;
  const promise = load().then((fn) => {
    state = { kind: 'ready', fn };
    return fn;
  });
  state = { kind: 'loading', promise };
  return promise;
}
