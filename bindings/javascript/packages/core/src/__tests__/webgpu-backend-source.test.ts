import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// Source guards for the browser WebGPU backend (gpu_webgpu.c). Both failure
// modes below are silent at runtime: every native dispatch catches its error,
// returns 0 and the gate runs on the CPU fallback, so results stay correct
// while gpu_is_native_accelerated() stays 0.
const source = readFileSync(
  fileURLToPath(new URL('../../../../../../src/optimization/gpu/backends/gpu_webgpu.c', import.meta.url)),
  'utf8'
);

describe('gpu_webgpu.c browser dispatch', () => {
  it('only calls EM_JS helpers (not EM_ASYNC_JS functions) by name from JS', () => {
    // Emscripten emits EM_ASYNC_JS(name) as __asyncjs__name behind
    // Asyncify.handleAsync, so the C name is undefined inside other JS bodies
    // (ReferenceError), and nesting handleAsync would break Asyncify anyway.
    const asyncNames = new Set([...source.matchAll(/EM_ASYNC_JS\(\s*\w+\s*,\s*(\w+)/g)].map((m) => m[1]));
    const syncNames = new Set([...source.matchAll(/EM_JS\(\s*\w+\s*,\s*(\w+)/g)].map((m) => m[1]));
    const calledFromJs = new Set([...source.matchAll(/await\s+(moonlab_webgpu_\w+)\s*\(/g)].map((m) => m[1]));
    expect(calledFromJs.size).toBeGreaterThan(0);
    for (const name of calledFromJs) {
      expect(asyncNames.has(name), `${name} is EM_ASYNC_JS but is called from JS`).toBe(false);
      expect(syncNames.has(name), `${name} is not an EM_JS helper`).toBe(true);
    }
  });

  it('uses no WGSL reserved words as identifiers', () => {
    // A reserved word fails the whole shader module, so every pipeline built
    // from it is invalid. Subset of the WGSL reserved-word list.
    const reserved = new Set(['target', 'module', 'filter', 'precision', 'meta', 'sample', 'layout', 'handle',
      'resource', 'common', 'attribute', 'partition', 'interface', 'namespace', 'new', 'null', 'self', 'super',
      'this', 'using', 'unless', 'yield', 'static', 'public', 'private', 'protected', 'export', 'import',
      'macro', 'typedef', 'union', 'virtual', 'volatile', 'where', 'with', 'writeonly', 'readonly']);
    const shaders = [...source.matchAll(/const shaderCode = `([\s\S]*?)`;/g)].map((m) => m[1]);
    expect(shaders.length).toBeGreaterThan(0);
    for (const wgsl of shaders) {
      const identifiers = [
        ...[...wgsl.matchAll(/^\s*(\w+)\s*:/gm)].map((m) => m[1]),
        ...[...wgsl.matchAll(/\b(?:let|var|const|fn|struct)\s+(\w+)/g)].map((m) => m[1]),
        ...[...wgsl.matchAll(/\.(\w+)/g)].map((m) => m[1])
      ];
      const hits = identifiers.filter((name) => reserved.has(name));
      expect(hits, `reserved WGSL words used as identifiers: ${hits.join(', ')}`).toEqual([]);
    }
  });
});
