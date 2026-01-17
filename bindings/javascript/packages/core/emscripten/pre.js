/**
 * Moonlab WASM Pre-initialization Script
 *
 * This runs before the WASM module is loaded.
 * IMPORTANT: onRuntimeInitialized MUST be set here, not in post.js,
 * because Emscripten marks it as "consumed" once it's called.
 */

// Store ready promise resolver
Module['_readyResolve'] = null;
Module['_readyReject'] = null;

// Create ready promise
Module['ready'] = new Promise(function(resolve, reject) {
  Module['_readyResolve'] = resolve;
  Module['_readyReject'] = reject;
});

// Track initialization state
Module['_initialized'] = false;

// Store callbacks
Module['_onReady'] = function() {
  Module['_initialized'] = true;
  if (Module['_readyResolve']) {
    Module['_readyResolve'](Module);
  }
};

// Error handling
Module['_onError'] = function(error) {
  console.error('[Moonlab WASM] Initialization error:', error);
  if (Module['_readyReject']) {
    Module['_readyReject'](error);
  }
};

// Set up the runtime initialization callback HERE (before module loads)
// This MUST be in pre.js, not post.js, because Emscripten consumes the
// callback once it's invoked and rejects any later assignment attempts.
Module['onRuntimeInitialized'] = function() {
  Module['_onReady']();
};
