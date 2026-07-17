import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { copyFileSync, existsSync, mkdirSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Plugin to copy the moonlab.js / moonlab.wasm glue+binary pair into the
// public directory before every build/dev run.
//
// public/moonlab.{js,wasm} are themselves *generated* build artifacts --
// they are committed only so the demo has something to serve in a
// checkout that hasn't run the Emscripten toolchain. Whenever this
// plugin can find a freshly built pair, it must overwrite both files
// unconditionally so the committed copies never drift out of sync with
// whatever @moonlab/quantum-core actually built.
//
// moonlab.js and moonlab.wasm are produced together, by the same build
// step, in either packages/core/dist/ (packages/core's own `build:wasm`
// script) or packages/core/emscripten/build/ (a raw emcmake/cmake run).
// Picking wasmSrc and jsSrc from *different* candidate directories
// independently -- as this plugin used to -- can pair a stale glue file
// from one location with a newer wasm binary from the other if only one
// half of a pair survived a partial clean, silently reproducing exactly
// the "glue mismatched with wasm" problem this plugin exists to
// prevent. Always take both files from the same directory.
function copyWasmPlugin() {
  return {
    name: 'copy-wasm',
    buildStart() {
      const distDir = resolve(__dirname, '../packages/core/dist');
      const buildDir = resolve(__dirname, '../packages/core/emscripten/build');
      const hasPair = (dir: string) =>
        existsSync(resolve(dir, 'moonlab.wasm')) && existsSync(resolve(dir, 'moonlab.js'));
      const srcDir = hasPair(distDir) ? distDir : hasPair(buildDir) ? buildDir : null;
      const publicDir = resolve(__dirname, 'public');

      if (!existsSync(publicDir)) {
        mkdirSync(publicDir, { recursive: true });
      }

      // Neither location has a complete, freshly built pair (e.g. this
      // checkout hasn't run the WASM build yet) -- leave the committed
      // public/moonlab.{js,wasm} pair as-is rather than copying only
      // one half of a mismatched pair.
      if (srcDir) {
        copyFileSync(resolve(srcDir, 'moonlab.wasm'), resolve(publicDir, 'moonlab.wasm'));
        copyFileSync(resolve(srcDir, 'moonlab.js'), resolve(publicDir, 'moonlab.js'));
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), copyWasmPlugin()],
  base: './',
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Keep WASM files as separate assets
        assetFileNames: (assetInfo) => {
          if (assetInfo.name?.endsWith('.wasm')) {
            return '[name][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        },
      },
    },
  },
  optimizeDeps: {
    exclude: ['@moonlab/quantum-core'],
  },
  server: {
    port: 3000,
    open: true,
    headers: {
      // Required for SharedArrayBuffer (WASM threads)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  assetsInclude: ['**/*.wasm'],
});
