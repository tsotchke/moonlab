/**
 * Moonlab WASM Pre-initialization Script
 *
 * This runs before the WASM module is loaded.
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
