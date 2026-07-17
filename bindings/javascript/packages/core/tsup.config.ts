import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  clean: true,
  esbuildOptions(options, context) {
    options.define ??= {};
    options.define.__MOONLAB_MODULE_URL__ =
      context.format === 'esm' ? 'import.meta.url' : 'undefined';
  },
});
