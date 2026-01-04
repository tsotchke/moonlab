import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    // Unit tests (no WASM required)
    include: ['src/**/*.test.ts', 'src/**/*.spec.ts'],
    // Integration tests require WASM - run separately with: pnpm test:integration
    exclude: ['src/**/*.integration.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.ts'],
      exclude: [
        'src/**/*.test.ts',
        'src/**/*.spec.ts',
        'src/**/*.integration.test.ts',
        'src/**/__mocks__/**',
        'emscripten/**',
      ],
    },
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});
