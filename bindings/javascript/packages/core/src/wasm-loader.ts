/**
 * WASM Module Loader
 *
 * Handles loading the Moonlab WASM module in both browser and Node.js environments.
 */

import type { MoonlabModule } from './memory';

// Singleton module instance
let modulePromise: Promise<MoonlabModule> | null = null;
let moduleInstance: MoonlabModule | null = null;
const globalScope =
  typeof globalThis !== 'undefined'
    ? (globalThis as Record<string, unknown>)
    : typeof window !== 'undefined'
      ? ((window as unknown as Record<string, unknown>))
      : {};

/**
 * Load options for the WASM module
 */
export interface LoadOptions {
  /**
   * Custom path to the WASM file
   * Default: auto-detected based on environment
   */
  wasmPath?: string;

  /**
   * Custom path to the JS glue code
   * Default: auto-detected based on environment
   */
  jsPath?: string;

  /**
   * Whether to use streaming instantiation (browser only)
   * Default: true
   */
  streaming?: boolean;
}

/**
 * Get the WASM module, loading it if necessary
 */
export async function getModule(options?: LoadOptions): Promise<MoonlabModule> {
  if (globalScope.__moonlabModuleInstance && typeof globalScope.__moonlabModuleInstance === 'object') {
    return globalScope.__moonlabModuleInstance as MoonlabModule;
  }

  if (moduleInstance) {
    return moduleInstance;
  }

  if (globalScope.__moonlabModulePromise instanceof Promise) {
    modulePromise = globalScope.__moonlabModulePromise as Promise<MoonlabModule>;
  }

  if (!modulePromise) {
    modulePromise = loadModule(options);
    globalScope.__moonlabModulePromise = modulePromise;
  }

  moduleInstance = await modulePromise;
  globalScope.__moonlabModuleInstance = moduleInstance;
  return moduleInstance;
}

/**
 * Check if the module is loaded
 */
export function isLoaded(): boolean {
  return moduleInstance !== null;
}

/**
 * Reset the module (for testing purposes)
 */
export function resetModule(): void {
  moduleInstance = null;
  modulePromise = null;
  delete globalScope.__moonlabModuleInstance;
  delete globalScope.__moonlabModulePromise;
}

/**
 * Load the WASM module
 */
async function loadModule(options?: LoadOptions): Promise<MoonlabModule> {
  const isNode =
    typeof process !== 'undefined' &&
    process.versions != null &&
    process.versions.node != null;
  const isDeno = isDenoRuntime();

  if (isNode || isDeno) {
    return loadServerModule(options);
  } else {
    return loadBrowserModule(options);
  }
}

/**
 * Load module in Node.js/Deno environment
 */
async function loadServerModule(options?: LoadOptions): Promise<MoonlabModule> {
  const candidates = await serverModuleCandidates(options);

  let lastError: unknown = null;
  for (const candidate of candidates) {
    try {
      const imported = await import(/* @vite-ignore */ candidate) as Record<string, unknown>;
      let MoonlabModuleFactory = getMoonlabFactory(imported);
      if (!MoonlabModuleFactory && isDenoRuntime()) {
        MoonlabModuleFactory = await loadDenoCommonJsFactory(candidate);
      }

      if (typeof MoonlabModuleFactory !== 'function') {
        throw new Error(`Module factory missing in ${candidate}`);
      }

      const module = await MoonlabModuleFactory(createServerModuleConfig(options, candidate));
      await (module as MoonlabModule).ready;
      return module as MoonlabModule;
    } catch (error) {
      lastError = error;
    }
  }

  const tried = candidates.join(', ');
  throw new Error(
    `Failed to load Moonlab WASM module in server runtime. Tried: ${tried}. Last error: ${lastError instanceof Error ? lastError.message : String(lastError)}`
  );
}

async function serverModuleCandidates(options?: LoadOptions): Promise<string[]> {
  const path = await import('node:path');
  const url = await import('node:url');
  const moduleApi = await import('node:module');
  const candidates: string[] = [];

  const addCandidate = (candidate: string): void => {
    const href = /^(file|https?):/i.test(candidate)
      ? candidate
      : url.pathToFileURL(path.resolve(candidate)).href;
    if (!candidates.includes(href)) {
      candidates.push(href);
    }
  };

  if (options?.jsPath) {
    addCandidate(options.jsPath);
  }

  if (typeof __dirname === 'string') {
    addCandidate(path.resolve(__dirname, 'moonlab.js'));
    addCandidate(path.resolve(__dirname, '../emscripten/build/moonlab.js'));
  }

  const cwd =
    typeof process !== 'undefined' && typeof process.cwd === 'function'
      ? process.cwd()
      : '.';
  const requireFromCwd = moduleApi.createRequire(
    url.pathToFileURL(path.resolve(cwd, 'moonlab-consumer.js')).href
  );
  try {
    const resolvedEntry = requireFromCwd.resolve('@moonlab/quantum-core');
    const distDir = path.dirname(resolvedEntry);
    addCandidate(path.join(distDir, 'moonlab.js'));
    addCandidate(path.resolve(distDir, '../emscripten/build/moonlab.js'));
  } catch {
    // Package self-resolution is unavailable in some source-tree test layouts.
  }

  addCandidate(path.resolve(cwd, 'dist/moonlab.js'));
  addCandidate(path.resolve(cwd, 'emscripten/build/moonlab.js'));

  return candidates;
}

function createServerModuleConfig(options: LoadOptions | undefined, candidate: string): Record<string, unknown> {
  return {
    locateFile: (file: string, scriptDirectory: string) => {
      if (file.endsWith('.wasm') && options?.wasmPath) {
        return options.wasmPath;
      }
      if (scriptDirectory) {
        return scriptDirectory + file;
      }
      return new URL(file, candidate).href;
    },
  };
}

function isDenoRuntime(): boolean {
  return typeof globalThis !== 'undefined' && 'Deno' in globalThis;
}

function getMoonlabFactory(imported: Record<string, unknown>): ((config?: unknown) => Promise<unknown>) | null {
  const factory =
    (imported.default as ((config?: unknown) => Promise<unknown>) | undefined) ??
    (imported.MoonlabModule as ((config?: unknown) => Promise<unknown>) | undefined) ??
    (typeof imported === 'function' ? (imported as unknown as (config?: unknown) => Promise<unknown>) : undefined);

  return typeof factory === 'function' ? factory : null;
}

async function loadDenoCommonJsFactory(candidate: string): Promise<((config?: unknown) => Promise<unknown>) | null> {
  const url = await import('node:url');
  const path = await import('node:path');
  const fs = await import('node:fs/promises');
  const moduleApi = await import('node:module');
  const processModule = await import('node:process');

  const modulePath = candidate.startsWith('file:')
    ? url.fileURLToPath(candidate)
    : candidate;

  const source = await fs.readFile(modulePath, 'utf8');
  const moduleShim: { exports: unknown } = { exports: {} };
  const exportsShim = moduleShim.exports;
  const require = moduleApi.createRequire(url.pathToFileURL(modulePath).href);

  const globalRecord = globalThis as Record<string, unknown>;
  const previousProcess = globalRecord.process;
  globalRecord.process = (processModule.default ?? processModule) as unknown;
  try {
    const evaluator = new Function(
      'module',
      'exports',
      'require',
      '__filename',
      '__dirname',
      source
    );
    evaluator(moduleShim, exportsShim, require, modulePath, path.dirname(modulePath));
  } finally {
    if (typeof previousProcess === 'undefined') {
      delete globalRecord.process;
    } else {
      globalRecord.process = previousProcess;
    }
  }

  if (typeof moduleShim.exports === 'function') {
    return moduleShim.exports as (config?: unknown) => Promise<unknown>;
  }

  if (moduleShim.exports && typeof moduleShim.exports === 'object') {
    const fromDefault = (moduleShim.exports as Record<string, unknown>).default;
    if (typeof fromDefault === 'function') {
      return fromDefault as (config?: unknown) => Promise<unknown>;
    }
  }

  return null;
}

/**
 * Load module in browser environment
 */
async function loadBrowserModule(options?: LoadOptions): Promise<MoonlabModule> {
  try {
    // If a previous instance exists (e.g., after HMR), reuse it to avoid double init
    if (globalScope.__moonlabModuleInstance && typeof globalScope.__moonlabModuleInstance === 'object') {
      return globalScope.__moonlabModuleInstance as MoonlabModule;
    }
    if (globalScope.__moonlabModulePromise instanceof Promise) {
      const cached = await globalScope.__moonlabModulePromise;
      globalScope.__moonlabModuleInstance = cached;
      return cached as MoonlabModule;
    }

    let MoonlabModuleFactory: (config?: unknown) => Promise<unknown>;

    // Check if already loaded globally (e.g., via script tag)
    if (typeof window !== 'undefined' && (window as unknown as Record<string, unknown>).MoonlabModule) {
      MoonlabModuleFactory = (window as unknown as Record<string, unknown>).MoonlabModule as typeof MoonlabModuleFactory;
    } else if (options?.jsPath) {
      // Custom path provided - use dynamic script loading
      MoonlabModuleFactory = await loadScript(options.jsPath);
    } else {
      // In browser, use script loading from public path (avoids Vite transform issues)
      const baseUrl = getBaseUrl();
      const jsUrl = `${baseUrl}/moonlab.js`;
      MoonlabModuleFactory = await loadScript(jsUrl);
    }

    // Initialize the module with configuration
    const moduleConfig: Record<string, unknown> = {};

    // Configure WASM file location
    moduleConfig.locateFile = (file: string, scriptDirectory: string) => {
      if (file.endsWith('.wasm')) {
        if (options?.wasmPath) {
          return options.wasmPath;
        }
        // Try relative to script directory first
        if (scriptDirectory) {
          return scriptDirectory + file;
        }
        // Fallback to base URL
        return getBaseUrl() + '/' + file;
      }
      return scriptDirectory + file;
    };

    if (typeof MoonlabModuleFactory !== 'function') {
      throw new Error('MoonlabModuleFactory is not available');
    }

    const loadingPromise = MoonlabModuleFactory(moduleConfig) as Promise<MoonlabModule>;
    globalScope.__moonlabModulePromise = loadingPromise;

    const module = await loadingPromise;
    await (module as MoonlabModule).ready;

    globalScope.__moonlabModuleInstance = module;

    return module as MoonlabModule;
  } catch (error) {
    throw new Error(
      `Failed to load Moonlab WASM module in browser: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Get the base URL for loading module files
 */
function getBaseUrl(): string {
  // Try to determine the base URL from the current script
  if (typeof document !== 'undefined') {
    const scripts = document.getElementsByTagName('script');
    for (let i = scripts.length - 1; i >= 0; i--) {
      const src = scripts[i].src;
      if (src.includes('moonlab')) {
        return src.substring(0, src.lastIndexOf('/'));
      }
    }
  }

  // Fallback to current origin
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }

  return '.';
}

/**
 * Load a script dynamically and return its default export
 */
async function loadScript(url: string): Promise<(config?: unknown) => Promise<unknown>> {
  // Check if MoonlabModule is already defined globally
  if (typeof window !== 'undefined' && (window as unknown as Record<string, unknown>).MoonlabModule) {
    return (window as unknown as Record<string, unknown>).MoonlabModule as (config?: unknown) => Promise<unknown>;
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.async = true;

    script.onload = () => {
      if ((window as unknown as Record<string, unknown>).MoonlabModule) {
        resolve((window as unknown as Record<string, unknown>).MoonlabModule as (config?: unknown) => Promise<unknown>);
      } else {
        reject(new Error('MoonlabModule not found after script load'));
      }
    };

    script.onerror = () => {
      reject(new Error(`Failed to load script: ${url}`));
    };

    document.head.appendChild(script);
  });
}

/**
 * Preload the WASM module (call early to reduce latency)
 */
export function preload(options?: LoadOptions): Promise<MoonlabModule> {
  return getModule(options);
}
