/**
 * WASM Module Loader
 *
 * Handles loading the Moonlab WASM module in both browser and Node.js environments.
 */

import type { MoonlabModule } from './memory';

// Singleton module instance
let modulePromise: Promise<MoonlabModule> | null = null;
let moduleInstance: MoonlabModule | null = null;

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
  if (moduleInstance) {
    return moduleInstance;
  }

  if (!modulePromise) {
    modulePromise = loadModule(options);
  }

  moduleInstance = await modulePromise;
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
}

/**
 * Load the WASM module
 */
async function loadModule(options?: LoadOptions): Promise<MoonlabModule> {
  const isNode =
    typeof process !== 'undefined' &&
    process.versions != null &&
    process.versions.node != null;

  if (isNode) {
    return loadNodeModule(options);
  } else {
    return loadBrowserModule(options);
  }
}

/**
 * Load module in Node.js environment
 */
async function loadNodeModule(_options?: LoadOptions): Promise<MoonlabModule> {
  // In Node.js, we need to use require or dynamic import
  // The path will be resolved relative to the package

  try {
    // Try to load the compiled WASM module
    // This assumes the module is built and available at dist/moonlab.js
    const modulePath = new URL('../dist/moonlab.js', import.meta.url).pathname;

    // Dynamic import for ES modules
    const MoonlabModuleFactory = (await import(modulePath)).default;

    const module = await MoonlabModuleFactory();
    await module.ready;

    return module as MoonlabModule;
  } catch (error) {
    throw new Error(
      `Failed to load Moonlab WASM module in Node.js: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Load module in browser environment
 */
async function loadBrowserModule(options?: LoadOptions): Promise<MoonlabModule> {
  try {
    let MoonlabModuleFactory: (config?: unknown) => Promise<unknown>;

    // Check if already loaded globally (e.g., via script tag)
    if (typeof window !== 'undefined' && (window as unknown as Record<string, unknown>).MoonlabModule) {
      MoonlabModuleFactory = (window as unknown as Record<string, unknown>).MoonlabModule as typeof MoonlabModuleFactory;
    } else if (options?.jsPath) {
      // Custom path provided - use dynamic script loading
      MoonlabModuleFactory = await loadScript(options.jsPath);
    } else {
      // Try ES module import (works with bundlers like Vite/webpack)
      try {
        // @ts-expect-error - Dynamic import of WASM glue code
        const wasmModule = await import('../dist/moonlab.js');
        MoonlabModuleFactory = wasmModule.default || wasmModule;
      } catch {
        // Fallback to script loading from common locations
        const baseUrl = getBaseUrl();
        const jsUrl = `${baseUrl}/moonlab.js`;
        MoonlabModuleFactory = await loadScript(jsUrl);
      }
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

    const module = await MoonlabModuleFactory(moduleConfig);
    await (module as MoonlabModule).ready;

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
