import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { copyFileSync, existsSync, mkdirSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Plugin to copy WASM files to public directory
function copyWasmPlugin() {
  return {
    name: 'copy-wasm',
    buildStart() {
      const distWasm = resolve(__dirname, '../packages/core/dist/moonlab.wasm');
      const distJs = resolve(__dirname, '../packages/core/dist/moonlab.js');
      const buildWasm = resolve(__dirname, '../packages/core/emscripten/build/moonlab.wasm');
      const buildJs = resolve(__dirname, '../packages/core/emscripten/build/moonlab.js');
      const wasmSrc = existsSync(distWasm) ? distWasm : buildWasm;
      const jsSrc = existsSync(distJs) ? distJs : buildJs;
      const publicDir = resolve(__dirname, 'public');

      if (!existsSync(publicDir)) {
        mkdirSync(publicDir, { recursive: true });
      }

      if (existsSync(wasmSrc)) {
        copyFileSync(wasmSrc, resolve(publicDir, 'moonlab.wasm'));
      }
      if (existsSync(jsSrc)) {
        copyFileSync(jsSrc, resolve(publicDir, 'moonlab.js'));
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
