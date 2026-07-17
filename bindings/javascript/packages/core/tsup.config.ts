import { defineConfig } from 'tsup';

export default defineConfig({
  // control-plane.ts also builds standalone so the documented
  // `@moonlab/quantum-core/control-plane` subpath (see the header
  // comment in src/control-plane.ts) is a real, buildable entry --
  // package.json's "./control-plane" export condition points at
  // dist/control-plane.{js,mjs,d.ts}, generated from this same entry.
  entry: ['src/index.ts', 'src/control-plane.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  clean: true,
  esbuildOptions(options, context) {
    options.define ??= {};
    options.define.__MOONLAB_MODULE_URL__ =
      context.format === 'esm' ? 'import.meta.url' : 'undefined';
  },
});
