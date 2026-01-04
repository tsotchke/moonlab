import { defineConfig } from 'vitest/config';

/**
 * Integration test configuration
 *
 * These tests require the WASM module to be built.
 * Run the C library build and Emscripten WASM build first.
 */
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.integration.test.ts'],
    testTimeout: 30000,
    hookTimeout: 30000,
  },
});
