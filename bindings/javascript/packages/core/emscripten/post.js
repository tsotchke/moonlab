/**
 * Moonlab WASM Post-initialization Script
 *
 * This runs after the WASM module is loaded.
 */

// Call ready callback
if (Module['onRuntimeInitialized']) {
  var originalOnInit = Module['onRuntimeInitialized'];
  Module['onRuntimeInitialized'] = function() {
    originalOnInit();
    Module['_onReady']();
  };
} else {
  Module['onRuntimeInitialized'] = Module['_onReady'];
}

/**
 * Generate cryptographically secure random number [0, 1)
 * Falls back to Math.random if crypto API unavailable
 */
Module['getSecureRandom'] = function() {
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    var arr = new Uint32Array(1);
    crypto.getRandomValues(arr);
    return arr[0] / 0xFFFFFFFF;
  } else {
    return Math.random();
  }
};

/**
 * Generate secure random bytes
 * @param {number} count - Number of bytes to generate
 * @returns {Uint8Array} Random bytes
 */
Module['getSecureRandomBytes'] = function(count) {
  var bytes = new Uint8Array(count);
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(bytes);
  } else {
    for (var i = 0; i < count; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  return bytes;
};

/**
 * Read complex array from WASM memory
 * @param {number} ptr - Pointer to complex array in WASM memory
 * @param {number} length - Number of complex numbers
 * @returns {Array<{real: number, imag: number}>} Array of complex numbers
 */
Module['readComplexArray'] = function(ptr, length) {
  var result = [];
  var offset = ptr / 8; // Float64 is 8 bytes
  var heap = Module['HEAPF64'];

  for (var i = 0; i < length; i++) {
    result.push({
      real: heap[offset + i * 2],
      imag: heap[offset + i * 2 + 1]
    });
  }
  return result;
};

/**
 * Write complex array to WASM memory
 * @param {number} ptr - Pointer to complex array in WASM memory
 * @param {Array<{real: number, imag: number}>} data - Complex numbers to write
 */
Module['writeComplexArray'] = function(ptr, data) {
  var offset = ptr / 8;
  var heap = Module['HEAPF64'];

  for (var i = 0; i < data.length; i++) {
    heap[offset + i * 2] = data[i].real;
    heap[offset + i * 2 + 1] = data[i].imag;
  }
};

/**
 * Read Float64 array from WASM memory
 * @param {number} ptr - Pointer in WASM memory
 * @param {number} length - Number of doubles
 * @returns {Float64Array} Copy of the data
 */
Module['readFloat64Array'] = function(ptr, length) {
  var offset = ptr / 8;
  return new Float64Array(Module['HEAPF64'].buffer, offset * 8, length).slice();
};

/**
 * Allocate memory for complex array
 * @param {number} length - Number of complex numbers
 * @returns {number} Pointer to allocated memory
 */
Module['allocComplexArray'] = function(length) {
  return Module['_malloc'](length * 16); // 2 doubles per complex
};

/**
 * Allocate memory for float64 array
 * @param {number} length - Number of doubles
 * @returns {number} Pointer to allocated memory
 */
Module['allocFloat64Array'] = function(length) {
  return Module['_malloc'](length * 8);
};

/**
 * Version info
 */
Module['version'] = {
  core: '1.0.0',
  wasm: true
};
