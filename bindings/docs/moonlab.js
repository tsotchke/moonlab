// This code implements the `-sMODULARIZE` settings by taking the generated
// JS program code (INNER_JS_CODE) and wrapping it in a factory function.

// Single threaded MINIMAL_RUNTIME programs do not need access to
// document.currentScript, so a simple export declaration is enough.
var MoonlabModule = (() => {
  // When MODULARIZE this JS may be executed later,
  // after document.currentScript is gone, so we save it.
  // In EXPORT_ES6 mode we can just use 'import.meta.url'.
  var _scriptName = globalThis.document?.currentScript?.src;
  return async function(moduleArg = {}) {
    var moduleRtn;

// include: shell.js
// include: minimum_runtime_check.js
(function() {
  // "30.0.0" -> 300000
  function humanReadableVersionToPacked(str) {
    str = str.split('-')[0]; // Remove any trailing part from e.g. "12.53.3-alpha"
    var vers = str.split('.').slice(0, 3);
    while(vers.length < 3) vers.push('00');
    vers = vers.map((n, i, arr) => n.padStart(2, '0'));
    return vers.join('');
  }
  // 300000 -> "30.0.0"
  var packedVersionToHumanReadable = n => [n / 10000 | 0, (n / 100 | 0) % 100, n % 100].join('.');

  var TARGET_NOT_SUPPORTED = 2147483647;

  // Note: We use a typeof check here instead of optional chaining using
  // globalThis because older browsers might not have globalThis defined.
  var currentNodeVersion = typeof process !== 'undefined' && process.versions?.node ? humanReadableVersionToPacked(process.versions.node) : TARGET_NOT_SUPPORTED;
  if (currentNodeVersion < 160000) {
    throw new Error(`This emscripten-generated code requires node v${ packedVersionToHumanReadable(160000) } (detected v${packedVersionToHumanReadable(currentNodeVersion)})`);
  }

  var userAgent = typeof navigator !== 'undefined' && navigator.userAgent;
  if (!userAgent) {
    return;
  }

  var currentSafariVersion = userAgent.includes("Safari/") && !userAgent.includes("Chrome/") && userAgent.match(/Version\/(\d+\.?\d*\.?\d*)/) ? humanReadableVersionToPacked(userAgent.match(/Version\/(\d+\.?\d*\.?\d*)/)[1]) : TARGET_NOT_SUPPORTED;
  if (currentSafariVersion < 150000) {
    throw new Error(`This emscripten-generated code requires Safari v${ packedVersionToHumanReadable(150000) } (detected v${currentSafariVersion})`);
  }

  var currentFirefoxVersion = userAgent.match(/Firefox\/(\d+(?:\.\d+)?)/) ? parseFloat(userAgent.match(/Firefox\/(\d+(?:\.\d+)?)/)[1]) : TARGET_NOT_SUPPORTED;
  if (currentFirefoxVersion < 79) {
    throw new Error(`This emscripten-generated code requires Firefox v79 (detected v${currentFirefoxVersion})`);
  }

  var currentChromeVersion = userAgent.match(/Chrome\/(\d+(?:\.\d+)?)/) ? parseFloat(userAgent.match(/Chrome\/(\d+(?:\.\d+)?)/)[1]) : TARGET_NOT_SUPPORTED;
  if (currentChromeVersion < 85) {
    throw new Error(`This emscripten-generated code requires Chrome v85 (detected v${currentChromeVersion})`);
  }
})();

// end include: minimum_runtime_check.js
// The Module object: Our interface to the outside world. We import
// and export values on it. There are various ways Module can be used:
// 1. Not defined. We create it here
// 2. A function parameter, function(moduleArg) => Promise<Module>
// 3. pre-run appended it, var Module = {}; ..generated code..
// 4. External script tag defines var Module.
// We need to check if Module already exists (e.g. case 3 above).
// Substitution will be replaced with actual code on later stage of the build,
// this way Closure Compiler will not mangle it (e.g. case 4. above).
// Note that if you want to run closure, and also to use Module
// after the generated code, you will need to define   var Module = {};
// before the code. Then that object will be used in the code, and you
// can continue to use Module afterwards as well.
var Module = moduleArg;

// Determine the runtime environment we are in. You can customize this by
// setting the ENVIRONMENT setting at compile time (see settings.js).

// Attempt to auto-detect the environment
var ENVIRONMENT_IS_WEB = !!globalThis.window;
var ENVIRONMENT_IS_WORKER = !!globalThis.WorkerGlobalScope;
// N.b. Electron.js environment is simultaneously a NODE-environment, but
// also a web environment.
var ENVIRONMENT_IS_NODE = globalThis.process?.versions?.node && globalThis.process?.type != 'renderer';
var ENVIRONMENT_IS_SHELL = !ENVIRONMENT_IS_WEB && !ENVIRONMENT_IS_NODE && !ENVIRONMENT_IS_WORKER;

// --pre-jses are emitted after the Module integration code, so that they can
// refer to Module (if they choose; they can also define Module)
// include: /home/cos/projects/moonlab/bindings/javascript/packages/core/emscripten/pre.js
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
// end include: /home/cos/projects/moonlab/bindings/javascript/packages/core/emscripten/pre.js


var arguments_ = [];
var thisProgram = './this.program';
var quit_ = (status, toThrow) => {
  throw toThrow;
};

if (typeof __filename != 'undefined') { // Node
  _scriptName = __filename;
} else
if (ENVIRONMENT_IS_WORKER) {
  _scriptName = self.location.href;
}

// `/` should be present at the end if `scriptDirectory` is not empty
var scriptDirectory = '';
function locateFile(path) {
  if (Module['locateFile']) {
    return Module['locateFile'](path, scriptDirectory);
  }
  return scriptDirectory + path;
}

// Hooks that are implemented differently in different runtime environments.
var readAsync, readBinary;

if (ENVIRONMENT_IS_NODE) {
  const isNode = globalThis.process?.versions?.node && globalThis.process?.type != 'renderer';
  if (!isNode) throw new Error('not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)');

  // These modules will usually be used on Node.js. Load them eagerly to avoid
  // the complexity of lazy-loading.
  var fs = require('fs');

  scriptDirectory = __dirname + '/';

// include: node_shell_read.js
readBinary = (filename) => {
  // We need to re-wrap `file://` strings to URLs.
  filename = isFileURI(filename) ? new URL(filename) : filename;
  var ret = fs.readFileSync(filename);
  assert(Buffer.isBuffer(ret));
  return ret;
};

readAsync = async (filename, binary = true) => {
  // See the comment in the `readBinary` function.
  filename = isFileURI(filename) ? new URL(filename) : filename;
  var ret = fs.readFileSync(filename, binary ? undefined : 'utf8');
  assert(binary ? Buffer.isBuffer(ret) : typeof ret == 'string');
  return ret;
};
// end include: node_shell_read.js
  if (process.argv.length > 1) {
    thisProgram = process.argv[1].replace(/\\/g, '/');
  }

  arguments_ = process.argv.slice(2);

  quit_ = (status, toThrow) => {
    process.exitCode = status;
    throw toThrow;
  };

} else
if (ENVIRONMENT_IS_SHELL) {

} else

// Note that this includes Node.js workers when relevant (pthreads is enabled).
// Node.js workers are detected as a combination of ENVIRONMENT_IS_WORKER and
// ENVIRONMENT_IS_NODE.
if (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
  try {
    scriptDirectory = new URL('.', _scriptName).href; // includes trailing slash
  } catch {
    // Must be a `blob:` or `data:` URL (e.g. `blob:http://site.com/etc/etc`), we cannot
    // infer anything from them.
  }

  if (!(globalThis.window || globalThis.WorkerGlobalScope)) throw new Error('not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)');

  {
// include: web_or_worker_shell_read.js
if (ENVIRONMENT_IS_WORKER) {
    readBinary = (url) => {
      var xhr = new XMLHttpRequest();
      xhr.open('GET', url, false);
      xhr.responseType = 'arraybuffer';
      xhr.send(null);
      return new Uint8Array(/** @type{!ArrayBuffer} */(xhr.response));
    };
  }

  readAsync = async (url) => {
    assert(!isFileURI(url), "readAsync does not work with file:// URLs");
    var response = await fetch(url, { credentials: 'same-origin' });
    if (response.ok) {
      return response.arrayBuffer();
    }
    throw new Error(response.status + ' : ' + response.url);
  };
// end include: web_or_worker_shell_read.js
  }
} else
{
  throw new Error('environment detection error');
}

var out = console.log.bind(console);
var err = console.error.bind(console);

var IDBFS = 'IDBFS is no longer included by default; build with -lidbfs.js';
var PROXYFS = 'PROXYFS is no longer included by default; build with -lproxyfs.js';
var WORKERFS = 'WORKERFS is no longer included by default; build with -lworkerfs.js';
var FETCHFS = 'FETCHFS is no longer included by default; build with -lfetchfs.js';
var ICASEFS = 'ICASEFS is no longer included by default; build with -licasefs.js';
var JSFILEFS = 'JSFILEFS is no longer included by default; build with -ljsfilefs.js';
var OPFS = 'OPFS is no longer included by default; build with -lopfs.js';

var NODEFS = 'NODEFS is no longer included by default; build with -lnodefs.js';

// perform assertions in shell.js after we set up out() and err(), as otherwise
// if an assertion fails it cannot print the message

assert(!ENVIRONMENT_IS_SHELL, 'shell environment detected but not enabled at build time.  Add `shell` to `-sENVIRONMENT` to enable.');

// end include: shell.js

// include: preamble.js
// === Preamble library stuff ===

// Documentation for the public APIs defined in this file must be updated in:
//    site/source/docs/api_reference/preamble.js.rst
// A prebuilt local version of the documentation is available at:
//    site/build/text/docs/api_reference/preamble.js.txt
// You can also build docs locally as HTML or other formats in site/
// An online HTML version (which may be of a different version of Emscripten)
//    is up at http://kripken.github.io/emscripten-site/docs/api_reference/preamble.js.html

var wasmBinary;

if (!globalThis.WebAssembly) {
  err('no native wasm support detected');
}

// Wasm globals

//========================================
// Runtime essentials
//========================================

// whether we are quitting the application. no code should run after this.
// set in exit() and abort()
var ABORT = false;

// set by exit() and abort().  Passed to 'onExit' handler.
// NOTE: This is also used as the process return code in shell environments
// but only when noExitRuntime is false.
var EXITSTATUS;

// In STRICT mode, we only define assert() when ASSERTIONS is set.  i.e. we
// don't define it at all in release modes.  This matches the behaviour of
// MINIMAL_RUNTIME.
// TODO(sbc): Make this the default even without STRICT enabled.
/** @type {function(*, string=)} */
function assert(condition, text) {
  if (!condition) {
    abort('Assertion failed' + (text ? ': ' + text : ''));
  }
}

// We used to include malloc/free by default in the past. Show a helpful error in
// builds with assertions.

/**
 * Indicates whether filename is delivered via file protocol (as opposed to http/https)
 * @noinline
 */
var isFileURI = (filename) => filename.startsWith('file://');

// include: runtime_common.js
// include: runtime_stack_check.js
// Initializes the stack cookie. Called at the startup of main and at the startup of each thread in pthreads mode.
function writeStackCookie() {
  var max = _emscripten_stack_get_end();
  assert((max & 3) == 0);
  // If the stack ends at address zero we write our cookies 4 bytes into the
  // stack.  This prevents interference with SAFE_HEAP and ASAN which also
  // monitor writes to address zero.
  if (max == 0) {
    max += 4;
  }
  // The stack grow downwards towards _emscripten_stack_get_end.
  // We write cookies to the final two words in the stack and detect if they are
  // ever overwritten.
  HEAPU32[((max)>>2)] = 0x02135467;
  HEAPU32[(((max)+(4))>>2)] = 0x89BACDFE;
  // Also test the global address 0 for integrity.
  HEAPU32[((0)>>2)] = 1668509029;
}

function checkStackCookie() {
  if (ABORT) return;
  var max = _emscripten_stack_get_end();
  // See writeStackCookie().
  if (max == 0) {
    max += 4;
  }
  var cookie1 = HEAPU32[((max)>>2)];
  var cookie2 = HEAPU32[(((max)+(4))>>2)];
  if (cookie1 != 0x02135467 || cookie2 != 0x89BACDFE) {
    abort(`Stack overflow! Stack cookie has been overwritten at ${ptrToString(max)}, expected hex dwords 0x89BACDFE and 0x2135467, but received ${ptrToString(cookie2)} ${ptrToString(cookie1)}`);
  }
  // Also test the global address 0 for integrity.
  if (HEAPU32[((0)>>2)] != 0x63736d65 /* 'emsc' */) {
    abort('Runtime error: The application has corrupted its heap memory area (address zero)!');
  }
}
// end include: runtime_stack_check.js
// include: runtime_exceptions.js
// end include: runtime_exceptions.js
// include: runtime_debug.js
var runtimeDebug = true; // Switch to false at runtime to disable logging at the right times

// Used by XXXXX_DEBUG settings to output debug messages.
function dbg(...args) {
  if (!runtimeDebug && typeof runtimeDebug != 'undefined') return;
  // TODO(sbc): Make this configurable somehow.  Its not always convenient for
  // logging to show up as warnings.
  console.warn(...args);
}

// Endianness check
(() => {
  var h16 = new Int16Array(1);
  var h8 = new Int8Array(h16.buffer);
  h16[0] = 0x6373;
  if (h8[0] !== 0x73 || h8[1] !== 0x63) abort('Runtime error: expected the system to be little-endian! (Run with -sSUPPORT_BIG_ENDIAN to bypass)');
})();

function consumedModuleProp(prop) {
  if (!Object.getOwnPropertyDescriptor(Module, prop)) {
    Object.defineProperty(Module, prop, {
      configurable: true,
      set() {
        abort(`Attempt to set \`Module.${prop}\` after it has already been processed.  This can happen, for example, when code is injected via '--post-js' rather than '--pre-js'`);

      }
    });
  }
}

function makeInvalidEarlyAccess(name) {
  return () => assert(false, `call to '${name}' via reference taken before Wasm module initialization`);

}

function ignoredModuleProp(prop) {
  if (Object.getOwnPropertyDescriptor(Module, prop)) {
    abort(`\`Module.${prop}\` was supplied but \`${prop}\` not included in INCOMING_MODULE_JS_API`);
  }
}

// forcing the filesystem exports a few things by default
function isExportedByForceFilesystem(name) {
  return name === 'FS_createPath' ||
         name === 'FS_createDataFile' ||
         name === 'FS_createPreloadedFile' ||
         name === 'FS_preloadFile' ||
         name === 'FS_unlink' ||
         name === 'addRunDependency' ||
         // The old FS has some functionality that WasmFS lacks.
         name === 'FS_createLazyFile' ||
         name === 'FS_createDevice' ||
         name === 'removeRunDependency';
}

function missingLibrarySymbol(sym) {

  // Any symbol that is not included from the JS library is also (by definition)
  // not exported on the Module object.
  unexportedRuntimeSymbol(sym);
}

function unexportedRuntimeSymbol(sym) {
  if (!Object.getOwnPropertyDescriptor(Module, sym)) {
    Object.defineProperty(Module, sym, {
      configurable: true,
      get() {
        var msg = `'${sym}' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the Emscripten FAQ)`;
        if (isExportedByForceFilesystem(sym)) {
          msg += '. Alternatively, forcing filesystem support (-sFORCE_FILESYSTEM) can export this for you';
        }
        abort(msg);
      },
    });
  }
}

// end include: runtime_debug.js
var readyPromiseResolve, readyPromiseReject;

// Memory management
var
/** @type {!Int8Array} */
  HEAP8,
/** @type {!Uint8Array} */
  HEAPU8,
/** @type {!Int16Array} */
  HEAP16,
/** @type {!Uint16Array} */
  HEAPU16,
/** @type {!Int32Array} */
  HEAP32,
/** @type {!Uint32Array} */
  HEAPU32,
/** @type {!Float32Array} */
  HEAPF32,
/** @type {!Float64Array} */
  HEAPF64;

// BigInt64Array type is not correctly defined in closure
var
/** not-@type {!BigInt64Array} */
  HEAP64,
/* BigUint64Array type is not correctly defined in closure
/** not-@type {!BigUint64Array} */
  HEAPU64;

var runtimeInitialized = false;



function updateMemoryViews() {
  var b = wasmMemory.buffer;
  Module['HEAP8'] = HEAP8 = new Int8Array(b);
  HEAP16 = new Int16Array(b);
  Module['HEAPU8'] = HEAPU8 = new Uint8Array(b);
  HEAPU16 = new Uint16Array(b);
  Module['HEAP32'] = HEAP32 = new Int32Array(b);
  Module['HEAPU32'] = HEAPU32 = new Uint32Array(b);
  HEAPF32 = new Float32Array(b);
  Module['HEAPF64'] = HEAPF64 = new Float64Array(b);
  HEAP64 = new BigInt64Array(b);
  HEAPU64 = new BigUint64Array(b);
}

// include: memoryprofiler.js
// end include: memoryprofiler.js
// end include: runtime_common.js
assert(globalThis.Int32Array && globalThis.Float64Array && Int32Array.prototype.subarray && Int32Array.prototype.set,
       'JS engine does not provide full typed array support');

function preRun() {
  if (Module['preRun']) {
    if (typeof Module['preRun'] == 'function') Module['preRun'] = [Module['preRun']];
    while (Module['preRun'].length) {
      addOnPreRun(Module['preRun'].shift());
    }
  }
  consumedModuleProp('preRun');
  // Begin ATPRERUNS hooks
  callRuntimeCallbacks(onPreRuns);
  // End ATPRERUNS hooks
}

function initRuntime() {
  assert(!runtimeInitialized);
  runtimeInitialized = true;

  checkStackCookie();

  // No ATINITS hooks

  wasmExports['__wasm_call_ctors']();

  // No ATPOSTCTORS hooks
}

function postRun() {
  checkStackCookie();
   // PThreads reuse the runtime from the main thread.

  if (Module['postRun']) {
    if (typeof Module['postRun'] == 'function') Module['postRun'] = [Module['postRun']];
    while (Module['postRun'].length) {
      addOnPostRun(Module['postRun'].shift());
    }
  }
  consumedModuleProp('postRun');

  // Begin ATPOSTRUNS hooks
  callRuntimeCallbacks(onPostRuns);
  // End ATPOSTRUNS hooks
}

/** @param {string|number=} what */
function abort(what) {
  Module['onAbort']?.(what);

  what = 'Aborted(' + what + ')';
  // TODO(sbc): Should we remove printing and leave it up to whoever
  // catches the exception?
  err(what);

  ABORT = true;

  // Use a wasm runtime error, because a JS error might be seen as a foreign
  // exception, which means we'd run destructors on it. We need the error to
  // simply make the program stop.
  // FIXME This approach does not work in Wasm EH because it currently does not assume
  // all RuntimeErrors are from traps; it decides whether a RuntimeError is from
  // a trap or not based on a hidden field within the object. So at the moment
  // we don't have a way of throwing a wasm trap from JS. TODO Make a JS API that
  // allows this in the wasm spec.

  // Suppress closure compiler warning here. Closure compiler's builtin extern
  // definition for WebAssembly.RuntimeError claims it takes no arguments even
  // though it can.
  // TODO(https://github.com/google/closure-compiler/pull/3913): Remove if/when upstream closure gets fixed.
  /** @suppress {checkTypes} */
  var e = new WebAssembly.RuntimeError(what);

  readyPromiseReject?.(e);
  // Throw the error whether or not MODULARIZE is set because abort is used
  // in code paths apart from instantiation where an exception is expected
  // to be thrown when abort is called.
  throw e;
}

// show errors on likely calls to FS when it was not included
var FS = {
  error() {
    abort('Filesystem support (FS) was not included. The problem is that you are using files from JS, but files were not used from C/C++, so filesystem support was not auto-included. You can force-include filesystem support with -sFORCE_FILESYSTEM');
  },
  init() { FS.error() },
  createDataFile() { FS.error() },
  createPreloadedFile() { FS.error() },
  createLazyFile() { FS.error() },
  open() { FS.error() },
  mkdev() { FS.error() },
  registerDevice() { FS.error() },
  analyzePath() { FS.error() },

  ErrnoError() { FS.error() },
};


function createExportWrapper(name, nargs) {
  return (...args) => {
    assert(runtimeInitialized, `native function \`${name}\` called before runtime initialization`);
    var f = wasmExports[name];
    assert(f, `exported native function \`${name}\` not found`);
    // Only assert for too many arguments. Too few can be valid since the missing arguments will be zero filled.
    assert(args.length <= nargs, `native function \`${name}\` called with ${args.length} args but expects ${nargs}`);
    return f(...args);
  };
}

var wasmBinaryFile;

function findWasmBinary() {
  return locateFile('moonlab.wasm');
}

function getBinarySync(file) {
  if (file == wasmBinaryFile && wasmBinary) {
    return new Uint8Array(wasmBinary);
  }
  if (readBinary) {
    return readBinary(file);
  }
  // Throwing a plain string here, even though it not normally advisable since
  // this gets turning into an `abort` in instantiateArrayBuffer.
  throw 'both async and sync fetching of the wasm failed';
}

async function getWasmBinary(binaryFile) {
  // If we don't have the binary yet, load it asynchronously using readAsync.
  if (!wasmBinary) {
    // Fetch the binary using readAsync
    try {
      var response = await readAsync(binaryFile);
      return new Uint8Array(response);
    } catch {
      // Fall back to getBinarySync below;
    }
  }

  // Otherwise, getBinarySync should be able to get it synchronously
  return getBinarySync(binaryFile);
}

async function instantiateArrayBuffer(binaryFile, imports) {
  try {
    var binary = await getWasmBinary(binaryFile);
    var instance = await WebAssembly.instantiate(binary, imports);
    return instance;
  } catch (reason) {
    err(`failed to asynchronously prepare wasm: ${reason}`);

    // Warn on some common problems.
    if (isFileURI(binaryFile)) {
      err(`warning: Loading from a file URI (${binaryFile}) is not supported in most browsers. See https://emscripten.org/docs/getting_started/FAQ.html#how-do-i-run-a-local-webserver-for-testing-why-does-my-program-stall-in-downloading-or-preparing`);
    }
    abort(reason);
  }
}

async function instantiateAsync(binary, binaryFile, imports) {
  if (!binary
      // Avoid instantiateStreaming() on Node.js environment for now, as while
      // Node.js v18.1.0 implements it, it does not have a full fetch()
      // implementation yet.
      //
      // Reference:
      //   https://github.com/emscripten-core/emscripten/pull/16917
      && !ENVIRONMENT_IS_NODE
     ) {
    try {
      var response = fetch(binaryFile, { credentials: 'same-origin' });
      var instantiationResult = await WebAssembly.instantiateStreaming(response, imports);
      return instantiationResult;
    } catch (reason) {
      // We expect the most common failure cause to be a bad MIME type for the binary,
      // in which case falling back to ArrayBuffer instantiation should work.
      err(`wasm streaming compile failed: ${reason}`);
      err('falling back to ArrayBuffer instantiation');
      // fall back of instantiateArrayBuffer below
    };
  }
  return instantiateArrayBuffer(binaryFile, imports);
}

function getWasmImports() {
  // prepare imports
  var imports = {
    'env': wasmImports,
    'wasi_snapshot_preview1': wasmImports,
  };
  return imports;
}

// Create the wasm instance.
// Receives the wasm imports, returns the exports.
async function createWasm() {
  // Load the wasm module and create an instance of using native support in the JS engine.
  // handle a generated wasm instance, receiving its exports and
  // performing other necessary setup
  /** @param {WebAssembly.Module=} module*/
  function receiveInstance(instance, module) {
    wasmExports = instance.exports;

    assignWasmExports(wasmExports);

    updateMemoryViews();

    return wasmExports;
  }

  // Prefer streaming instantiation if available.
  // Async compilation can be confusing when an error on the page overwrites Module
  // (for example, if the order of elements is wrong, and the one defining Module is
  // later), so we save Module and check it later.
  var trueModule = Module;
  function receiveInstantiationResult(result) {
    // 'result' is a ResultObject object which has both the module and instance.
    // receiveInstance() will swap in the exports (to Module.asm) so they can be called
    assert(Module === trueModule, 'the Module object should not be replaced during async compilation - perhaps the order of HTML elements is wrong?');
    trueModule = null;
    // TODO: Due to Closure regression https://github.com/google/closure-compiler/issues/3193, the above line no longer optimizes out down to the following line.
    // When the regression is fixed, can restore the above PTHREADS-enabled path.
    return receiveInstance(result['instance']);
  }

  var info = getWasmImports();

  // User shell pages can write their own Module.instantiateWasm = function(imports, successCallback) callback
  // to manually instantiate the Wasm module themselves. This allows pages to
  // run the instantiation parallel to any other async startup actions they are
  // performing.
  // Also pthreads and wasm workers initialize the wasm instance through this
  // path.
  if (Module['instantiateWasm']) {
    return new Promise((resolve, reject) => {
      try {
        Module['instantiateWasm'](info, (inst, mod) => {
          resolve(receiveInstance(inst, mod));
        });
      } catch(e) {
        err(`Module.instantiateWasm callback failed with error: ${e}`);
        reject(e);
      }
    });
  }

  wasmBinaryFile ??= findWasmBinary();
  var result = await instantiateAsync(wasmBinary, wasmBinaryFile, info);
  var exports = receiveInstantiationResult(result);
  return exports;
}

// end include: preamble.js

// Begin JS library code


  class ExitStatus {
      name = 'ExitStatus';
      constructor(status) {
        this.message = `Program terminated with exit(${status})`;
        this.status = status;
      }
    }

  var callRuntimeCallbacks = (callbacks) => {
      while (callbacks.length > 0) {
        // Pass the module as the first argument.
        callbacks.shift()(Module);
      }
    };
  var onPostRuns = [];
  var addOnPostRun = (cb) => onPostRuns.push(cb);

  var onPreRuns = [];
  var addOnPreRun = (cb) => onPreRuns.push(cb);


  
    /**
     * @param {number} ptr
     * @param {string} type
     */
  function getValue(ptr, type = 'i8') {
    if (type.endsWith('*')) type = '*';
    switch (type) {
      case 'i1': return HEAP8[ptr];
      case 'i8': return HEAP8[ptr];
      case 'i16': return HEAP16[((ptr)>>1)];
      case 'i32': return HEAP32[((ptr)>>2)];
      case 'i64': return HEAP64[((ptr)>>3)];
      case 'float': return HEAPF32[((ptr)>>2)];
      case 'double': return HEAPF64[((ptr)>>3)];
      case '*': return HEAPU32[((ptr)>>2)];
      default: abort(`invalid type for getValue: ${type}`);
    }
  }

  var noExitRuntime = true;

  var ptrToString = (ptr) => {
      assert(typeof ptr === 'number', `ptrToString expects a number, got ${typeof ptr}`);
      // Convert to 32-bit unsigned value
      ptr >>>= 0;
      return '0x' + ptr.toString(16).padStart(8, '0');
    };

  
    /**
     * @param {number} ptr
     * @param {number} value
     * @param {string} type
     */
  function setValue(ptr, value, type = 'i8') {
    if (type.endsWith('*')) type = '*';
    switch (type) {
      case 'i1': HEAP8[ptr] = value; break;
      case 'i8': HEAP8[ptr] = value; break;
      case 'i16': HEAP16[((ptr)>>1)] = value; break;
      case 'i32': HEAP32[((ptr)>>2)] = value; break;
      case 'i64': HEAP64[((ptr)>>3)] = BigInt(value); break;
      case 'float': HEAPF32[((ptr)>>2)] = value; break;
      case 'double': HEAPF64[((ptr)>>3)] = value; break;
      case '*': HEAPU32[((ptr)>>2)] = value; break;
      default: abort(`invalid type for setValue: ${type}`);
    }
  }

  var stackRestore = (val) => __emscripten_stack_restore(val);

  var stackSave = () => _emscripten_stack_get_current();

  var warnOnce = (text) => {
      warnOnce.shown ||= {};
      if (!warnOnce.shown[text]) {
        warnOnce.shown[text] = 1;
        if (ENVIRONMENT_IS_NODE) text = 'warning: ' + text;
        err(text);
      }
    };

  

  var __abort_js = () =>
      abort('native code called abort()');

  var getHeapMax = () =>
      // Stay one Wasm page short of 4GB: while e.g. Chrome is able to allocate
      // full 4GB Wasm memories, the size will wrap back to 0 bytes in Wasm side
      // for any code that deals with heap sizes, which would require special
      // casing all heap size related code to treat 0 specially.
      2147483648;
  
  var alignMemory = (size, alignment) => {
      assert(alignment, "alignment argument is required");
      return Math.ceil(size / alignment) * alignment;
    };
  
  var growMemory = (size) => {
      var oldHeapSize = wasmMemory.buffer.byteLength;
      var pages = ((size - oldHeapSize + 65535) / 65536) | 0;
      try {
        // round size grow request up to wasm page size (fixed 64KB per spec)
        wasmMemory.grow(pages); // .grow() takes a delta compared to the previous size
        updateMemoryViews();
        return 1 /*success*/;
      } catch(e) {
        err(`growMemory: Attempted to grow heap from ${oldHeapSize} bytes to ${size} bytes, but got error: ${e}`);
      }
      // implicit 0 return to save code size (caller will cast "undefined" into 0
      // anyhow)
    };
  var _emscripten_resize_heap = (requestedSize) => {
      var oldSize = HEAPU8.length;
      // With CAN_ADDRESS_2GB or MEMORY64, pointers are already unsigned.
      requestedSize >>>= 0;
      // With multithreaded builds, races can happen (another thread might increase the size
      // in between), so return a failure, and let the caller retry.
      assert(requestedSize > oldSize);
  
      // Memory resize rules:
      // 1.  Always increase heap size to at least the requested size, rounded up
      //     to next page multiple.
      // 2a. If MEMORY_GROWTH_LINEAR_STEP == -1, excessively resize the heap
      //     geometrically: increase the heap size according to
      //     MEMORY_GROWTH_GEOMETRIC_STEP factor (default +20%), At most
      //     overreserve by MEMORY_GROWTH_GEOMETRIC_CAP bytes (default 96MB).
      // 2b. If MEMORY_GROWTH_LINEAR_STEP != -1, excessively resize the heap
      //     linearly: increase the heap size by at least
      //     MEMORY_GROWTH_LINEAR_STEP bytes.
      // 3.  Max size for the heap is capped at 2048MB-WASM_PAGE_SIZE, or by
      //     MAXIMUM_MEMORY, or by ASAN limit, depending on which is smallest
      // 4.  If we were unable to allocate as much memory, it may be due to
      //     over-eager decision to excessively reserve due to (3) above.
      //     Hence if an allocation fails, cut down on the amount of excess
      //     growth, in an attempt to succeed to perform a smaller allocation.
  
      // A limit is set for how much we can grow. We should not exceed that
      // (the wasm binary specifies it, so if we tried, we'd fail anyhow).
      var maxHeapSize = getHeapMax();
      if (requestedSize > maxHeapSize) {
        err(`Cannot enlarge memory, requested ${requestedSize} bytes, but the limit is ${maxHeapSize} bytes!`);
        return false;
      }
  
      // Loop through potential heap size increases. If we attempt a too eager
      // reservation that fails, cut down on the attempted size and reserve a
      // smaller bump instead. (max 3 times, chosen somewhat arbitrarily)
      for (var cutDown = 1; cutDown <= 4; cutDown *= 2) {
        var overGrownHeapSize = oldSize * (1 + 0.2 / cutDown); // ensure geometric growth
        // but limit overreserving (default to capping at +96MB overgrowth at most)
        overGrownHeapSize = Math.min(overGrownHeapSize, requestedSize + 100663296 );
  
        var newSize = Math.min(maxHeapSize, alignMemory(Math.max(requestedSize, overGrownHeapSize), 65536));
  
        var replacement = growMemory(newSize);
        if (replacement) {
  
          return true;
        }
      }
      err(`Failed to grow the heap from ${oldSize} bytes to ${newSize} bytes, not enough memory!`);
      return false;
    };

  var ENV = {
  };
  
  var getExecutableName = () => thisProgram || './this.program';
  var getEnvStrings = () => {
      if (!getEnvStrings.strings) {
        // Default values.
        // Browser language detection #8751
        var lang = (globalThis.navigator?.language ?? 'C').replace('-', '_') + '.UTF-8';
        var env = {
          'USER': 'web_user',
          'LOGNAME': 'web_user',
          'PATH': '/',
          'PWD': '/',
          'HOME': '/home/web_user',
          'LANG': lang,
          '_': getExecutableName()
        };
        // Apply the user-provided values, if any.
        for (var x in ENV) {
          // x is a key in ENV; if ENV[x] is undefined, that means it was
          // explicitly set to be so. We allow user code to do that to
          // force variables with default values to remain unset.
          if (ENV[x] === undefined) delete env[x];
          else env[x] = ENV[x];
        }
        var strings = [];
        for (var x in env) {
          strings.push(`${x}=${env[x]}`);
        }
        getEnvStrings.strings = strings;
      }
      return getEnvStrings.strings;
    };
  
  var stringToUTF8Array = (str, heap, outIdx, maxBytesToWrite) => {
      assert(typeof str === 'string', `stringToUTF8Array expects a string (got ${typeof str})`);
      // Parameter maxBytesToWrite is not optional. Negative values, 0, null,
      // undefined and false each don't write out any bytes.
      if (!(maxBytesToWrite > 0))
        return 0;
  
      var startIdx = outIdx;
      var endIdx = outIdx + maxBytesToWrite - 1; // -1 for string null terminator.
      for (var i = 0; i < str.length; ++i) {
        // For UTF8 byte structure, see http://en.wikipedia.org/wiki/UTF-8#Description
        // and https://www.ietf.org/rfc/rfc2279.txt
        // and https://tools.ietf.org/html/rfc3629
        var u = str.codePointAt(i);
        if (u <= 0x7F) {
          if (outIdx >= endIdx) break;
          heap[outIdx++] = u;
        } else if (u <= 0x7FF) {
          if (outIdx + 1 >= endIdx) break;
          heap[outIdx++] = 0xC0 | (u >> 6);
          heap[outIdx++] = 0x80 | (u & 63);
        } else if (u <= 0xFFFF) {
          if (outIdx + 2 >= endIdx) break;
          heap[outIdx++] = 0xE0 | (u >> 12);
          heap[outIdx++] = 0x80 | ((u >> 6) & 63);
          heap[outIdx++] = 0x80 | (u & 63);
        } else {
          if (outIdx + 3 >= endIdx) break;
          if (u > 0x10FFFF) warnOnce('Invalid Unicode code point ' + ptrToString(u) + ' encountered when serializing a JS string to a UTF-8 string in wasm memory! (Valid unicode code points should be in range 0-0x10FFFF).');
          heap[outIdx++] = 0xF0 | (u >> 18);
          heap[outIdx++] = 0x80 | ((u >> 12) & 63);
          heap[outIdx++] = 0x80 | ((u >> 6) & 63);
          heap[outIdx++] = 0x80 | (u & 63);
          // Gotcha: if codePoint is over 0xFFFF, it is represented as a surrogate pair in UTF-16.
          // We need to manually skip over the second code unit for correct iteration.
          i++;
        }
      }
      // Null-terminate the pointer to the buffer.
      heap[outIdx] = 0;
      return outIdx - startIdx;
    };
  var stringToUTF8 = (str, outPtr, maxBytesToWrite) => {
      assert(typeof maxBytesToWrite == 'number', 'stringToUTF8(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!');
      return stringToUTF8Array(str, HEAPU8, outPtr, maxBytesToWrite);
    };
  var _environ_get = (__environ, environ_buf) => {
      var bufSize = 0;
      var envp = 0;
      for (var string of getEnvStrings()) {
        var ptr = environ_buf + bufSize;
        HEAPU32[(((__environ)+(envp))>>2)] = ptr;
        bufSize += stringToUTF8(string, ptr, Infinity) + 1;
        envp += 4;
      }
      return 0;
    };

  
  var lengthBytesUTF8 = (str) => {
      var len = 0;
      for (var i = 0; i < str.length; ++i) {
        // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded code
        // unit, not a Unicode code point of the character! So decode
        // UTF16->UTF32->UTF8.
        // See http://unicode.org/faq/utf_bom.html#utf16-3
        var c = str.charCodeAt(i); // possibly a lead surrogate
        if (c <= 0x7F) {
          len++;
        } else if (c <= 0x7FF) {
          len += 2;
        } else if (c >= 0xD800 && c <= 0xDFFF) {
          len += 4; ++i;
        } else {
          len += 3;
        }
      }
      return len;
    };
  var _environ_sizes_get = (penviron_count, penviron_buf_size) => {
      var strings = getEnvStrings();
      HEAPU32[((penviron_count)>>2)] = strings.length;
      var bufSize = 0;
      for (var string of strings) {
        bufSize += lengthBytesUTF8(string) + 1;
      }
      HEAPU32[((penviron_buf_size)>>2)] = bufSize;
      return 0;
    };

  var UTF8Decoder = globalThis.TextDecoder && new TextDecoder();
  
  var findStringEnd = (heapOrArray, idx, maxBytesToRead, ignoreNul) => {
      var maxIdx = idx + maxBytesToRead;
      if (ignoreNul) return maxIdx;
      // TextDecoder needs to know the byte length in advance, it doesn't stop on
      // null terminator by itself.
      // As a tiny code save trick, compare idx against maxIdx using a negation,
      // so that maxBytesToRead=undefined/NaN means Infinity.
      while (heapOrArray[idx] && !(idx >= maxIdx)) ++idx;
      return idx;
    };
  
  
    /**
     * Given a pointer 'idx' to a null-terminated UTF8-encoded string in the given
     * array that contains uint8 values, returns a copy of that string as a
     * Javascript String object.
     * heapOrArray is either a regular array, or a JavaScript typed array view.
     * @param {number=} idx
     * @param {number=} maxBytesToRead
     * @param {boolean=} ignoreNul - If true, the function will not stop on a NUL character.
     * @return {string}
     */
  var UTF8ArrayToString = (heapOrArray, idx = 0, maxBytesToRead, ignoreNul) => {
  
      var endPtr = findStringEnd(heapOrArray, idx, maxBytesToRead, ignoreNul);
  
      // When using conditional TextDecoder, skip it for short strings as the overhead of the native call is not worth it.
      if (endPtr - idx > 16 && heapOrArray.buffer && UTF8Decoder) {
        return UTF8Decoder.decode(heapOrArray.subarray(idx, endPtr));
      }
      var str = '';
      while (idx < endPtr) {
        // For UTF8 byte structure, see:
        // http://en.wikipedia.org/wiki/UTF-8#Description
        // https://www.ietf.org/rfc/rfc2279.txt
        // https://tools.ietf.org/html/rfc3629
        var u0 = heapOrArray[idx++];
        if (!(u0 & 0x80)) { str += String.fromCharCode(u0); continue; }
        var u1 = heapOrArray[idx++] & 63;
        if ((u0 & 0xE0) == 0xC0) { str += String.fromCharCode(((u0 & 31) << 6) | u1); continue; }
        var u2 = heapOrArray[idx++] & 63;
        if ((u0 & 0xF0) == 0xE0) {
          u0 = ((u0 & 15) << 12) | (u1 << 6) | u2;
        } else {
          if ((u0 & 0xF8) != 0xF0) warnOnce('Invalid UTF-8 leading byte ' + ptrToString(u0) + ' encountered when deserializing a UTF-8 string in wasm memory to a JS string!');
          u0 = ((u0 & 7) << 18) | (u1 << 12) | (u2 << 6) | (heapOrArray[idx++] & 63);
        }
  
        if (u0 < 0x10000) {
          str += String.fromCharCode(u0);
        } else {
          var ch = u0 - 0x10000;
          str += String.fromCharCode(0xD800 | (ch >> 10), 0xDC00 | (ch & 0x3FF));
        }
      }
      return str;
    };
  
    /**
     * Given a pointer 'ptr' to a null-terminated UTF8-encoded string in the
     * emscripten HEAP, returns a copy of that string as a Javascript String object.
     *
     * @param {number} ptr
     * @param {number=} maxBytesToRead - An optional length that specifies the
     *   maximum number of bytes to read. You can omit this parameter to scan the
     *   string until the first 0 byte. If maxBytesToRead is passed, and the string
     *   at [ptr, ptr+maxBytesToReadr[ contains a null byte in the middle, then the
     *   string will cut short at that byte index.
     * @param {boolean=} ignoreNul - If true, the function will not stop on a NUL character.
     * @return {string}
     */
  var UTF8ToString = (ptr, maxBytesToRead, ignoreNul) => {
      assert(typeof ptr == 'number', `UTF8ToString expects a number (got ${typeof ptr})`);
      return ptr ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead, ignoreNul) : '';
    };
  var SYSCALLS = {
  varargs:undefined,
  getStr(ptr) {
        var ret = UTF8ToString(ptr);
        return ret;
      },
  };
  var _fd_close = (fd) => {
      abort('fd_close called without SYSCALLS_REQUIRE_FILESYSTEM');
    };

  var INT53_MAX = 9007199254740992;
  
  var INT53_MIN = -9007199254740992;
  var bigintToI53Checked = (num) => (num < INT53_MIN || num > INT53_MAX) ? NaN : Number(num);
  function _fd_seek(fd, offset, whence, newOffset) {
    offset = bigintToI53Checked(offset);
  
  
      return 70;
    ;
  }

  var printCharBuffers = [null,[],[]];
  
  var printChar = (stream, curr) => {
      var buffer = printCharBuffers[stream];
      assert(buffer);
      if (curr === 0 || curr === 10) {
        (stream === 1 ? out : err)(UTF8ArrayToString(buffer));
        buffer.length = 0;
      } else {
        buffer.push(curr);
      }
    };
  
  var flush_NO_FILESYSTEM = () => {
      // flush anything remaining in the buffers during shutdown
      _fflush(0);
      if (printCharBuffers[1].length) printChar(1, 10);
      if (printCharBuffers[2].length) printChar(2, 10);
    };
  
  
  var _fd_write = (fd, iov, iovcnt, pnum) => {
      // hack to support printf in SYSCALLS_REQUIRE_FILESYSTEM=0
      var num = 0;
      for (var i = 0; i < iovcnt; i++) {
        var ptr = HEAPU32[((iov)>>2)];
        var len = HEAPU32[(((iov)+(4))>>2)];
        iov += 8;
        for (var j = 0; j < len; j++) {
          printChar(fd, HEAPU8[ptr+j]);
        }
        num += len;
      }
      HEAPU32[((pnum)>>2)] = num;
      return 0;
    };

  var getCFunc = (ident) => {
      var func = Module['_' + ident]; // closure exported function
      assert(func, 'Cannot call unknown function ' + ident + ', make sure it is exported');
      return func;
    };
  
  var writeArrayToMemory = (array, buffer) => {
      assert(array.length >= 0, 'writeArrayToMemory array must have a length (should be an array or typed array)')
      HEAP8.set(array, buffer);
    };
  
  
  
  var stackAlloc = (sz) => __emscripten_stack_alloc(sz);
  var stringToUTF8OnStack = (str) => {
      var size = lengthBytesUTF8(str) + 1;
      var ret = stackAlloc(size);
      stringToUTF8(str, ret, size);
      return ret;
    };
  
  
  
  
  
    /**
     * @param {string|null=} returnType
     * @param {Array=} argTypes
     * @param {Array=} args
     * @param {Object=} opts
     */
  var ccall = (ident, returnType, argTypes, args, opts) => {
      // For fast lookup of conversion functions
      var toC = {
        'string': (str) => {
          var ret = 0;
          if (str !== null && str !== undefined && str !== 0) { // null string
            ret = stringToUTF8OnStack(str);
          }
          return ret;
        },
        'array': (arr) => {
          var ret = stackAlloc(arr.length);
          writeArrayToMemory(arr, ret);
          return ret;
        }
      };
  
      function convertReturnValue(ret) {
        if (returnType === 'string') {
          return UTF8ToString(ret);
        }
        if (returnType === 'boolean') return Boolean(ret);
        return ret;
      }
  
      var func = getCFunc(ident);
      var cArgs = [];
      var stack = 0;
      assert(returnType !== 'array', 'Return type should not be "array".');
      if (args) {
        for (var i = 0; i < args.length; i++) {
          var converter = toC[argTypes[i]];
          if (converter) {
            if (stack === 0) stack = stackSave();
            cArgs[i] = converter(args[i]);
          } else {
            cArgs[i] = args[i];
          }
        }
      }
      var ret = func(...cArgs);
      function onDone(ret) {
        if (stack !== 0) stackRestore(stack);
        return convertReturnValue(ret);
      }
  
      ret = onDone(ret);
      return ret;
    };

  
    /**
     * @param {string=} returnType
     * @param {Array=} argTypes
     * @param {Object=} opts
     */
  var cwrap = (ident, returnType, argTypes, opts) => {
      return (...args) => ccall(ident, returnType, argTypes, args, opts);
    };




// End JS library code

// include: postlibrary.js
// This file is included after the automatically-generated JS library code
// but before the wasm module is created.

{

  // Begin ATMODULES hooks
  if (Module['noExitRuntime']) noExitRuntime = Module['noExitRuntime'];
if (Module['print']) out = Module['print'];
if (Module['printErr']) err = Module['printErr'];
if (Module['wasmBinary']) wasmBinary = Module['wasmBinary'];

Module['FS_createDataFile'] = FS.createDataFile;
Module['FS_createPreloadedFile'] = FS.createPreloadedFile;

  // End ATMODULES hooks

  checkIncomingModuleAPI();

  if (Module['arguments']) arguments_ = Module['arguments'];
  if (Module['thisProgram']) thisProgram = Module['thisProgram'];

  // Assertions on removed incoming Module JS APIs.
  assert(typeof Module['memoryInitializerPrefixURL'] == 'undefined', 'Module.memoryInitializerPrefixURL option was removed, use Module.locateFile instead');
  assert(typeof Module['pthreadMainPrefixURL'] == 'undefined', 'Module.pthreadMainPrefixURL option was removed, use Module.locateFile instead');
  assert(typeof Module['cdInitializerPrefixURL'] == 'undefined', 'Module.cdInitializerPrefixURL option was removed, use Module.locateFile instead');
  assert(typeof Module['filePackagePrefixURL'] == 'undefined', 'Module.filePackagePrefixURL option was removed, use Module.locateFile instead');
  assert(typeof Module['read'] == 'undefined', 'Module.read option was removed');
  assert(typeof Module['readAsync'] == 'undefined', 'Module.readAsync option was removed (modify readAsync in JS)');
  assert(typeof Module['readBinary'] == 'undefined', 'Module.readBinary option was removed (modify readBinary in JS)');
  assert(typeof Module['setWindowTitle'] == 'undefined', 'Module.setWindowTitle option was removed (modify emscripten_set_window_title in JS)');
  assert(typeof Module['TOTAL_MEMORY'] == 'undefined', 'Module.TOTAL_MEMORY has been renamed Module.INITIAL_MEMORY');
  assert(typeof Module['ENVIRONMENT'] == 'undefined', 'Module.ENVIRONMENT has been deprecated. To force the environment, use the ENVIRONMENT compile-time option (for example, -sENVIRONMENT=web or -sENVIRONMENT=node)');
  assert(typeof Module['STACK_SIZE'] == 'undefined', 'STACK_SIZE can no longer be set at runtime.  Use -sSTACK_SIZE at link time')
  // If memory is defined in wasm, the user can't provide it, or set INITIAL_MEMORY
  assert(typeof Module['wasmMemory'] == 'undefined', 'Use of `wasmMemory` detected.  Use -sIMPORTED_MEMORY to define wasmMemory externally');
  assert(typeof Module['INITIAL_MEMORY'] == 'undefined', 'Detected runtime INITIAL_MEMORY setting.  Use -sIMPORTED_MEMORY to define wasmMemory dynamically');

  if (Module['preInit']) {
    if (typeof Module['preInit'] == 'function') Module['preInit'] = [Module['preInit']];
    while (Module['preInit'].length > 0) {
      Module['preInit'].shift()();
    }
  }
  consumedModuleProp('preInit');
}

// Begin runtime exports
  Module['ccall'] = ccall;
  Module['cwrap'] = cwrap;
  Module['setValue'] = setValue;
  Module['getValue'] = getValue;
  Module['UTF8ToString'] = UTF8ToString;
  Module['stringToUTF8'] = stringToUTF8;
  var missingLibrarySymbols = [
  'writeI53ToI64',
  'writeI53ToI64Clamped',
  'writeI53ToI64Signaling',
  'writeI53ToU64Clamped',
  'writeI53ToU64Signaling',
  'readI53FromI64',
  'readI53FromU64',
  'convertI32PairToI53',
  'convertI32PairToI53Checked',
  'convertU32PairToI53',
  'getTempRet0',
  'setTempRet0',
  'createNamedFunction',
  'zeroMemory',
  'exitJS',
  'withStackSave',
  'strError',
  'inetPton4',
  'inetNtop4',
  'inetPton6',
  'inetNtop6',
  'readSockaddr',
  'writeSockaddr',
  'readEmAsmArgs',
  'jstoi_q',
  'autoResumeAudioContext',
  'getDynCaller',
  'dynCall',
  'handleException',
  'keepRuntimeAlive',
  'runtimeKeepalivePush',
  'runtimeKeepalivePop',
  'callUserCallback',
  'maybeExit',
  'asyncLoad',
  'asmjsMangle',
  'mmapAlloc',
  'HandleAllocator',
  'getUniqueRunDependency',
  'addRunDependency',
  'removeRunDependency',
  'addOnInit',
  'addOnPostCtor',
  'addOnPreMain',
  'addOnExit',
  'STACK_SIZE',
  'STACK_ALIGN',
  'POINTER_SIZE',
  'ASSERTIONS',
  'convertJsFunctionToWasm',
  'getEmptyTableSlot',
  'updateTableMap',
  'getFunctionAddress',
  'addFunction',
  'removeFunction',
  'intArrayFromString',
  'intArrayToString',
  'AsciiToString',
  'stringToAscii',
  'UTF16ToString',
  'stringToUTF16',
  'lengthBytesUTF16',
  'UTF32ToString',
  'stringToUTF32',
  'lengthBytesUTF32',
  'stringToNewUTF8',
  'registerKeyEventCallback',
  'maybeCStringToJsString',
  'findEventTarget',
  'getBoundingClientRect',
  'fillMouseEventData',
  'registerMouseEventCallback',
  'registerWheelEventCallback',
  'registerUiEventCallback',
  'registerFocusEventCallback',
  'fillDeviceOrientationEventData',
  'registerDeviceOrientationEventCallback',
  'fillDeviceMotionEventData',
  'registerDeviceMotionEventCallback',
  'screenOrientation',
  'fillOrientationChangeEventData',
  'registerOrientationChangeEventCallback',
  'fillFullscreenChangeEventData',
  'registerFullscreenChangeEventCallback',
  'JSEvents_requestFullscreen',
  'JSEvents_resizeCanvasForFullscreen',
  'registerRestoreOldStyle',
  'hideEverythingExceptGivenElement',
  'restoreHiddenElements',
  'setLetterbox',
  'softFullscreenResizeWebGLRenderTarget',
  'doRequestFullscreen',
  'fillPointerlockChangeEventData',
  'registerPointerlockChangeEventCallback',
  'registerPointerlockErrorEventCallback',
  'requestPointerLock',
  'fillVisibilityChangeEventData',
  'registerVisibilityChangeEventCallback',
  'registerTouchEventCallback',
  'fillGamepadEventData',
  'registerGamepadEventCallback',
  'registerBeforeUnloadEventCallback',
  'fillBatteryEventData',
  'registerBatteryEventCallback',
  'setCanvasElementSize',
  'getCanvasElementSize',
  'jsStackTrace',
  'getCallstack',
  'convertPCtoSourceLocation',
  'checkWasiClock',
  'wasiRightsToMuslOFlags',
  'wasiOFlagsToMuslOFlags',
  'initRandomFill',
  'randomFill',
  'safeSetTimeout',
  'setImmediateWrapped',
  'safeRequestAnimationFrame',
  'clearImmediateWrapped',
  'registerPostMainLoop',
  'registerPreMainLoop',
  'getPromise',
  'makePromise',
  'idsToPromises',
  'makePromiseCallback',
  'ExceptionInfo',
  'findMatchingCatch',
  'Browser_asyncPrepareDataCounter',
  'isLeapYear',
  'ydayFromDate',
  'arraySum',
  'addDays',
  'getSocketFromFD',
  'getSocketAddress',
  'heapObjectForWebGLType',
  'toTypedArrayIndex',
  'webgl_enable_ANGLE_instanced_arrays',
  'webgl_enable_OES_vertex_array_object',
  'webgl_enable_WEBGL_draw_buffers',
  'webgl_enable_WEBGL_multi_draw',
  'webgl_enable_EXT_polygon_offset_clamp',
  'webgl_enable_EXT_clip_control',
  'webgl_enable_WEBGL_polygon_mode',
  'emscriptenWebGLGet',
  'computeUnpackAlignedImageSize',
  'colorChannelsInGlTextureFormat',
  'emscriptenWebGLGetTexPixelData',
  'emscriptenWebGLGetUniform',
  'webglGetUniformLocation',
  'webglPrepareUniformLocationsBeforeFirstUse',
  'webglGetLeftBracePos',
  'emscriptenWebGLGetVertexAttrib',
  '__glGetActiveAttribOrUniform',
  'writeGLArray',
  'registerWebGlEventCallback',
  'runAndAbortIfError',
  'ALLOC_NORMAL',
  'ALLOC_STACK',
  'allocate',
  'writeStringToMemory',
  'writeAsciiToMemory',
  'allocateUTF8',
  'allocateUTF8OnStack',
  'demangle',
  'stackTrace',
  'getNativeTypeSize',
];
missingLibrarySymbols.forEach(missingLibrarySymbol)

  var unexportedSymbols = [
  'run',
  'out',
  'err',
  'callMain',
  'abort',
  'wasmExports',
  'HEAPF32',
  'HEAP16',
  'HEAPU16',
  'HEAP64',
  'HEAPU64',
  'writeStackCookie',
  'checkStackCookie',
  'INT53_MAX',
  'INT53_MIN',
  'bigintToI53Checked',
  'stackSave',
  'stackRestore',
  'stackAlloc',
  'ptrToString',
  'getHeapMax',
  'growMemory',
  'ENV',
  'ERRNO_CODES',
  'DNS',
  'Protocols',
  'Sockets',
  'timers',
  'warnOnce',
  'readEmAsmArgsArray',
  'getExecutableName',
  'alignMemory',
  'wasmTable',
  'wasmMemory',
  'noExitRuntime',
  'addOnPreRun',
  'addOnPostRun',
  'freeTableIndexes',
  'functionsInTableMap',
  'PATH',
  'PATH_FS',
  'UTF8Decoder',
  'UTF8ArrayToString',
  'stringToUTF8Array',
  'lengthBytesUTF8',
  'UTF16Decoder',
  'stringToUTF8OnStack',
  'writeArrayToMemory',
  'JSEvents',
  'specialHTMLTargets',
  'findCanvasEventTarget',
  'currentFullscreenStrategy',
  'restoreOldWindowedStyle',
  'UNWIND_CACHE',
  'ExitStatus',
  'getEnvStrings',
  'flush_NO_FILESYSTEM',
  'emSetImmediate',
  'emClearImmediate_deps',
  'emClearImmediate',
  'promiseMap',
  'uncaughtExceptionCount',
  'exceptionLast',
  'exceptionCaught',
  'Browser',
  'requestFullscreen',
  'requestFullScreen',
  'setCanvasSize',
  'getUserMedia',
  'createContext',
  'getPreloadedImageData__data',
  'wget',
  'MONTH_DAYS_REGULAR',
  'MONTH_DAYS_LEAP',
  'MONTH_DAYS_REGULAR_CUMULATIVE',
  'MONTH_DAYS_LEAP_CUMULATIVE',
  'SYSCALLS',
  'tempFixedLengthArray',
  'miniTempWebGLFloatBuffers',
  'miniTempWebGLIntBuffers',
  'GL',
  'AL',
  'GLUT',
  'EGL',
  'GLEW',
  'IDBStore',
  'SDL',
  'SDL_gfx',
  'print',
  'printErr',
  'jstoi_s',
];
unexportedSymbols.forEach(unexportedRuntimeSymbol);

  // End runtime exports
  // Begin JS library exports
  // End JS library exports

// end include: postlibrary.js

function checkIncomingModuleAPI() {
  ignoredModuleProp('fetchSettings');
}

// Imports from the Wasm binary.
var _quantum_state_init = Module['_quantum_state_init'] = makeInvalidEarlyAccess('_quantum_state_init');
var _free = Module['_free'] = makeInvalidEarlyAccess('_free');
var _quantum_state_free = Module['_quantum_state_free'] = makeInvalidEarlyAccess('_quantum_state_free');
var _quantum_state_normalize = Module['_quantum_state_normalize'] = makeInvalidEarlyAccess('_quantum_state_normalize');
var _quantum_state_clone = Module['_quantum_state_clone'] = makeInvalidEarlyAccess('_quantum_state_clone');
var _quantum_state_reset = Module['_quantum_state_reset'] = makeInvalidEarlyAccess('_quantum_state_reset');
var _quantum_state_entropy = Module['_quantum_state_entropy'] = makeInvalidEarlyAccess('_quantum_state_entropy');
var _quantum_state_purity = Module['_quantum_state_purity'] = makeInvalidEarlyAccess('_quantum_state_purity');
var _quantum_state_fidelity = Module['_quantum_state_fidelity'] = makeInvalidEarlyAccess('_quantum_state_fidelity');
var _quantum_state_get_probability = Module['_quantum_state_get_probability'] = makeInvalidEarlyAccess('_quantum_state_get_probability');
var _gate_pauli_x = Module['_gate_pauli_x'] = makeInvalidEarlyAccess('_gate_pauli_x');
var _gate_pauli_y = Module['_gate_pauli_y'] = makeInvalidEarlyAccess('_gate_pauli_y');
var _gate_pauli_z = Module['_gate_pauli_z'] = makeInvalidEarlyAccess('_gate_pauli_z');
var _gate_hadamard = Module['_gate_hadamard'] = makeInvalidEarlyAccess('_gate_hadamard');
var _gate_s = Module['_gate_s'] = makeInvalidEarlyAccess('_gate_s');
var _gate_s_dagger = Module['_gate_s_dagger'] = makeInvalidEarlyAccess('_gate_s_dagger');
var _gate_t = Module['_gate_t'] = makeInvalidEarlyAccess('_gate_t');
var _gate_t_dagger = Module['_gate_t_dagger'] = makeInvalidEarlyAccess('_gate_t_dagger');
var _gate_phase = Module['_gate_phase'] = makeInvalidEarlyAccess('_gate_phase');
var _gate_rx = Module['_gate_rx'] = makeInvalidEarlyAccess('_gate_rx');
var _gate_ry = Module['_gate_ry'] = makeInvalidEarlyAccess('_gate_ry');
var _gate_rz = Module['_gate_rz'] = makeInvalidEarlyAccess('_gate_rz');
var _gate_u3 = Module['_gate_u3'] = makeInvalidEarlyAccess('_gate_u3');
var _gate_cnot = Module['_gate_cnot'] = makeInvalidEarlyAccess('_gate_cnot');
var _gate_cz = Module['_gate_cz'] = makeInvalidEarlyAccess('_gate_cz');
var _gate_cy = Module['_gate_cy'] = makeInvalidEarlyAccess('_gate_cy');
var _gate_swap = Module['_gate_swap'] = makeInvalidEarlyAccess('_gate_swap');
var _gate_cphase = Module['_gate_cphase'] = makeInvalidEarlyAccess('_gate_cphase');
var _gate_crx = Module['_gate_crx'] = makeInvalidEarlyAccess('_gate_crx');
var _gate_cry = Module['_gate_cry'] = makeInvalidEarlyAccess('_gate_cry');
var _gate_crz = Module['_gate_crz'] = makeInvalidEarlyAccess('_gate_crz');
var _gate_toffoli = Module['_gate_toffoli'] = makeInvalidEarlyAccess('_gate_toffoli');
var _gate_fredkin = Module['_gate_fredkin'] = makeInvalidEarlyAccess('_gate_fredkin');
var _gate_qft = Module['_gate_qft'] = makeInvalidEarlyAccess('_gate_qft');
var _gate_iqft = Module['_gate_iqft'] = makeInvalidEarlyAccess('_gate_iqft');
var _measurement_probability_one = Module['_measurement_probability_one'] = makeInvalidEarlyAccess('_measurement_probability_one');
var _measurement_probability_zero = Module['_measurement_probability_zero'] = makeInvalidEarlyAccess('_measurement_probability_zero');
var _measurement_all_probabilities = Module['_measurement_all_probabilities'] = makeInvalidEarlyAccess('_measurement_all_probabilities');
var _measurement_probability_distribution = Module['_measurement_probability_distribution'] = makeInvalidEarlyAccess('_measurement_probability_distribution');
var _measurement_single_qubit = Module['_measurement_single_qubit'] = makeInvalidEarlyAccess('_measurement_single_qubit');
var _measurement_all_qubits = Module['_measurement_all_qubits'] = makeInvalidEarlyAccess('_measurement_all_qubits');
var _measurement_expectation_z = Module['_measurement_expectation_z'] = makeInvalidEarlyAccess('_measurement_expectation_z');
var _measurement_expectation_x = Module['_measurement_expectation_x'] = makeInvalidEarlyAccess('_measurement_expectation_x');
var _measurement_expectation_y = Module['_measurement_expectation_y'] = makeInvalidEarlyAccess('_measurement_expectation_y');
var _measurement_correlation_zz = Module['_measurement_correlation_zz'] = makeInvalidEarlyAccess('_measurement_correlation_zz');
var _malloc = Module['_malloc'] = makeInvalidEarlyAccess('_malloc');
var _grover_oracle = Module['_grover_oracle'] = makeInvalidEarlyAccess('_grover_oracle');
var _grover_diffusion = Module['_grover_diffusion'] = makeInvalidEarlyAccess('_grover_diffusion');
var _grover_iteration = Module['_grover_iteration'] = makeInvalidEarlyAccess('_grover_iteration');
var _grover_optimal_iterations = Module['_grover_optimal_iterations'] = makeInvalidEarlyAccess('_grover_optimal_iterations');
var _grover_search = Module['_grover_search'] = makeInvalidEarlyAccess('_grover_search');
var _pauli_hamiltonian_create = Module['_pauli_hamiltonian_create'] = makeInvalidEarlyAccess('_pauli_hamiltonian_create');
var _pauli_hamiltonian_free = Module['_pauli_hamiltonian_free'] = makeInvalidEarlyAccess('_pauli_hamiltonian_free');
var _pauli_hamiltonian_add_term = Module['_pauli_hamiltonian_add_term'] = makeInvalidEarlyAccess('_pauli_hamiltonian_add_term');
var _vqe_create_h2_hamiltonian = Module['_vqe_create_h2_hamiltonian'] = makeInvalidEarlyAccess('_vqe_create_h2_hamiltonian');
var _vqe_create_hardware_efficient_ansatz = Module['_vqe_create_hardware_efficient_ansatz'] = makeInvalidEarlyAccess('_vqe_create_hardware_efficient_ansatz');
var _vqe_ansatz_free = Module['_vqe_ansatz_free'] = makeInvalidEarlyAccess('_vqe_ansatz_free');
var _vqe_apply_ansatz = Module['_vqe_apply_ansatz'] = makeInvalidEarlyAccess('_vqe_apply_ansatz');
var _vqe_optimizer_create = Module['_vqe_optimizer_create'] = makeInvalidEarlyAccess('_vqe_optimizer_create');
var _vqe_optimizer_free = Module['_vqe_optimizer_free'] = makeInvalidEarlyAccess('_vqe_optimizer_free');
var _vqe_solver_create = Module['_vqe_solver_create'] = makeInvalidEarlyAccess('_vqe_solver_create');
var _vqe_solver_free = Module['_vqe_solver_free'] = makeInvalidEarlyAccess('_vqe_solver_free');
var _vqe_compute_energy = Module['_vqe_compute_energy'] = makeInvalidEarlyAccess('_vqe_compute_energy');
var _vqe_solve = Module['_vqe_solve'] = makeInvalidEarlyAccess('_vqe_solve');
var _vqe_hartree_to_kcalmol = Module['_vqe_hartree_to_kcalmol'] = makeInvalidEarlyAccess('_vqe_hartree_to_kcalmol');
var _ising_model_create = Module['_ising_model_create'] = makeInvalidEarlyAccess('_ising_model_create');
var _ising_model_free = Module['_ising_model_free'] = makeInvalidEarlyAccess('_ising_model_free');
var _ising_model_set_coupling = Module['_ising_model_set_coupling'] = makeInvalidEarlyAccess('_ising_model_set_coupling');
var _ising_model_set_field = Module['_ising_model_set_field'] = makeInvalidEarlyAccess('_ising_model_set_field');
var _ising_model_evaluate = Module['_ising_model_evaluate'] = makeInvalidEarlyAccess('_ising_model_evaluate');
var _qaoa_solver_create = Module['_qaoa_solver_create'] = makeInvalidEarlyAccess('_qaoa_solver_create');
var _qaoa_solver_free = Module['_qaoa_solver_free'] = makeInvalidEarlyAccess('_qaoa_solver_free');
var _qaoa_apply_circuit = Module['_qaoa_apply_circuit'] = makeInvalidEarlyAccess('_qaoa_apply_circuit');
var _qaoa_compute_expectation = Module['_qaoa_compute_expectation'] = makeInvalidEarlyAccess('_qaoa_compute_expectation');
var _qaoa_solve = Module['_qaoa_solve'] = makeInvalidEarlyAccess('_qaoa_solve');
var _create_bell_state_phi_plus = Module['_create_bell_state_phi_plus'] = makeInvalidEarlyAccess('_create_bell_state_phi_plus');
var _create_bell_state_phi_minus = Module['_create_bell_state_phi_minus'] = makeInvalidEarlyAccess('_create_bell_state_phi_minus');
var _create_bell_state_psi_plus = Module['_create_bell_state_psi_plus'] = makeInvalidEarlyAccess('_create_bell_state_psi_plus');
var _create_bell_state_psi_minus = Module['_create_bell_state_psi_minus'] = makeInvalidEarlyAccess('_create_bell_state_psi_minus');
var _create_bell_state = Module['_create_bell_state'] = makeInvalidEarlyAccess('_create_bell_state');
var _calculate_chsh_parameter = Module['_calculate_chsh_parameter'] = makeInvalidEarlyAccess('_calculate_chsh_parameter');
var _bell_test_chsh = Module['_bell_test_chsh'] = makeInvalidEarlyAccess('_bell_test_chsh');
var _fflush = makeInvalidEarlyAccess('_fflush');
var _bell_get_optimal_settings = Module['_bell_get_optimal_settings'] = makeInvalidEarlyAccess('_bell_get_optimal_settings');
var _emscripten_stack_get_end = makeInvalidEarlyAccess('_emscripten_stack_get_end');
var _emscripten_stack_get_base = makeInvalidEarlyAccess('_emscripten_stack_get_base');
var _emscripten_stack_init = makeInvalidEarlyAccess('_emscripten_stack_init');
var _emscripten_stack_get_free = makeInvalidEarlyAccess('_emscripten_stack_get_free');
var __emscripten_stack_restore = makeInvalidEarlyAccess('__emscripten_stack_restore');
var __emscripten_stack_alloc = makeInvalidEarlyAccess('__emscripten_stack_alloc');
var _emscripten_stack_get_current = makeInvalidEarlyAccess('_emscripten_stack_get_current');
var memory = makeInvalidEarlyAccess('memory');
var __indirect_function_table = makeInvalidEarlyAccess('__indirect_function_table');
var wasmMemory = makeInvalidEarlyAccess('wasmMemory');

function assignWasmExports(wasmExports) {
  assert(typeof wasmExports['quantum_state_init'] != 'undefined', 'missing Wasm export: quantum_state_init');
  assert(typeof wasmExports['free'] != 'undefined', 'missing Wasm export: free');
  assert(typeof wasmExports['quantum_state_free'] != 'undefined', 'missing Wasm export: quantum_state_free');
  assert(typeof wasmExports['quantum_state_normalize'] != 'undefined', 'missing Wasm export: quantum_state_normalize');
  assert(typeof wasmExports['quantum_state_clone'] != 'undefined', 'missing Wasm export: quantum_state_clone');
  assert(typeof wasmExports['quantum_state_reset'] != 'undefined', 'missing Wasm export: quantum_state_reset');
  assert(typeof wasmExports['quantum_state_entropy'] != 'undefined', 'missing Wasm export: quantum_state_entropy');
  assert(typeof wasmExports['quantum_state_purity'] != 'undefined', 'missing Wasm export: quantum_state_purity');
  assert(typeof wasmExports['quantum_state_fidelity'] != 'undefined', 'missing Wasm export: quantum_state_fidelity');
  assert(typeof wasmExports['quantum_state_get_probability'] != 'undefined', 'missing Wasm export: quantum_state_get_probability');
  assert(typeof wasmExports['gate_pauli_x'] != 'undefined', 'missing Wasm export: gate_pauli_x');
  assert(typeof wasmExports['gate_pauli_y'] != 'undefined', 'missing Wasm export: gate_pauli_y');
  assert(typeof wasmExports['gate_pauli_z'] != 'undefined', 'missing Wasm export: gate_pauli_z');
  assert(typeof wasmExports['gate_hadamard'] != 'undefined', 'missing Wasm export: gate_hadamard');
  assert(typeof wasmExports['gate_s'] != 'undefined', 'missing Wasm export: gate_s');
  assert(typeof wasmExports['gate_s_dagger'] != 'undefined', 'missing Wasm export: gate_s_dagger');
  assert(typeof wasmExports['gate_t'] != 'undefined', 'missing Wasm export: gate_t');
  assert(typeof wasmExports['gate_t_dagger'] != 'undefined', 'missing Wasm export: gate_t_dagger');
  assert(typeof wasmExports['gate_phase'] != 'undefined', 'missing Wasm export: gate_phase');
  assert(typeof wasmExports['gate_rx'] != 'undefined', 'missing Wasm export: gate_rx');
  assert(typeof wasmExports['gate_ry'] != 'undefined', 'missing Wasm export: gate_ry');
  assert(typeof wasmExports['gate_rz'] != 'undefined', 'missing Wasm export: gate_rz');
  assert(typeof wasmExports['gate_u3'] != 'undefined', 'missing Wasm export: gate_u3');
  assert(typeof wasmExports['gate_cnot'] != 'undefined', 'missing Wasm export: gate_cnot');
  assert(typeof wasmExports['gate_cz'] != 'undefined', 'missing Wasm export: gate_cz');
  assert(typeof wasmExports['gate_cy'] != 'undefined', 'missing Wasm export: gate_cy');
  assert(typeof wasmExports['gate_swap'] != 'undefined', 'missing Wasm export: gate_swap');
  assert(typeof wasmExports['gate_cphase'] != 'undefined', 'missing Wasm export: gate_cphase');
  assert(typeof wasmExports['gate_crx'] != 'undefined', 'missing Wasm export: gate_crx');
  assert(typeof wasmExports['gate_cry'] != 'undefined', 'missing Wasm export: gate_cry');
  assert(typeof wasmExports['gate_crz'] != 'undefined', 'missing Wasm export: gate_crz');
  assert(typeof wasmExports['gate_toffoli'] != 'undefined', 'missing Wasm export: gate_toffoli');
  assert(typeof wasmExports['gate_fredkin'] != 'undefined', 'missing Wasm export: gate_fredkin');
  assert(typeof wasmExports['gate_qft'] != 'undefined', 'missing Wasm export: gate_qft');
  assert(typeof wasmExports['gate_iqft'] != 'undefined', 'missing Wasm export: gate_iqft');
  assert(typeof wasmExports['measurement_probability_one'] != 'undefined', 'missing Wasm export: measurement_probability_one');
  assert(typeof wasmExports['measurement_probability_zero'] != 'undefined', 'missing Wasm export: measurement_probability_zero');
  assert(typeof wasmExports['measurement_all_probabilities'] != 'undefined', 'missing Wasm export: measurement_all_probabilities');
  assert(typeof wasmExports['measurement_probability_distribution'] != 'undefined', 'missing Wasm export: measurement_probability_distribution');
  assert(typeof wasmExports['measurement_single_qubit'] != 'undefined', 'missing Wasm export: measurement_single_qubit');
  assert(typeof wasmExports['measurement_all_qubits'] != 'undefined', 'missing Wasm export: measurement_all_qubits');
  assert(typeof wasmExports['measurement_expectation_z'] != 'undefined', 'missing Wasm export: measurement_expectation_z');
  assert(typeof wasmExports['measurement_expectation_x'] != 'undefined', 'missing Wasm export: measurement_expectation_x');
  assert(typeof wasmExports['measurement_expectation_y'] != 'undefined', 'missing Wasm export: measurement_expectation_y');
  assert(typeof wasmExports['measurement_correlation_zz'] != 'undefined', 'missing Wasm export: measurement_correlation_zz');
  assert(typeof wasmExports['malloc'] != 'undefined', 'missing Wasm export: malloc');
  assert(typeof wasmExports['grover_oracle'] != 'undefined', 'missing Wasm export: grover_oracle');
  assert(typeof wasmExports['grover_diffusion'] != 'undefined', 'missing Wasm export: grover_diffusion');
  assert(typeof wasmExports['grover_iteration'] != 'undefined', 'missing Wasm export: grover_iteration');
  assert(typeof wasmExports['grover_optimal_iterations'] != 'undefined', 'missing Wasm export: grover_optimal_iterations');
  assert(typeof wasmExports['grover_search'] != 'undefined', 'missing Wasm export: grover_search');
  assert(typeof wasmExports['pauli_hamiltonian_create'] != 'undefined', 'missing Wasm export: pauli_hamiltonian_create');
  assert(typeof wasmExports['pauli_hamiltonian_free'] != 'undefined', 'missing Wasm export: pauli_hamiltonian_free');
  assert(typeof wasmExports['pauli_hamiltonian_add_term'] != 'undefined', 'missing Wasm export: pauli_hamiltonian_add_term');
  assert(typeof wasmExports['vqe_create_h2_hamiltonian'] != 'undefined', 'missing Wasm export: vqe_create_h2_hamiltonian');
  assert(typeof wasmExports['vqe_create_hardware_efficient_ansatz'] != 'undefined', 'missing Wasm export: vqe_create_hardware_efficient_ansatz');
  assert(typeof wasmExports['vqe_ansatz_free'] != 'undefined', 'missing Wasm export: vqe_ansatz_free');
  assert(typeof wasmExports['vqe_apply_ansatz'] != 'undefined', 'missing Wasm export: vqe_apply_ansatz');
  assert(typeof wasmExports['vqe_optimizer_create'] != 'undefined', 'missing Wasm export: vqe_optimizer_create');
  assert(typeof wasmExports['vqe_optimizer_free'] != 'undefined', 'missing Wasm export: vqe_optimizer_free');
  assert(typeof wasmExports['vqe_solver_create'] != 'undefined', 'missing Wasm export: vqe_solver_create');
  assert(typeof wasmExports['vqe_solver_free'] != 'undefined', 'missing Wasm export: vqe_solver_free');
  assert(typeof wasmExports['vqe_compute_energy'] != 'undefined', 'missing Wasm export: vqe_compute_energy');
  assert(typeof wasmExports['vqe_solve'] != 'undefined', 'missing Wasm export: vqe_solve');
  assert(typeof wasmExports['vqe_hartree_to_kcalmol'] != 'undefined', 'missing Wasm export: vqe_hartree_to_kcalmol');
  assert(typeof wasmExports['ising_model_create'] != 'undefined', 'missing Wasm export: ising_model_create');
  assert(typeof wasmExports['ising_model_free'] != 'undefined', 'missing Wasm export: ising_model_free');
  assert(typeof wasmExports['ising_model_set_coupling'] != 'undefined', 'missing Wasm export: ising_model_set_coupling');
  assert(typeof wasmExports['ising_model_set_field'] != 'undefined', 'missing Wasm export: ising_model_set_field');
  assert(typeof wasmExports['ising_model_evaluate'] != 'undefined', 'missing Wasm export: ising_model_evaluate');
  assert(typeof wasmExports['qaoa_solver_create'] != 'undefined', 'missing Wasm export: qaoa_solver_create');
  assert(typeof wasmExports['qaoa_solver_free'] != 'undefined', 'missing Wasm export: qaoa_solver_free');
  assert(typeof wasmExports['qaoa_apply_circuit'] != 'undefined', 'missing Wasm export: qaoa_apply_circuit');
  assert(typeof wasmExports['qaoa_compute_expectation'] != 'undefined', 'missing Wasm export: qaoa_compute_expectation');
  assert(typeof wasmExports['qaoa_solve'] != 'undefined', 'missing Wasm export: qaoa_solve');
  assert(typeof wasmExports['create_bell_state_phi_plus'] != 'undefined', 'missing Wasm export: create_bell_state_phi_plus');
  assert(typeof wasmExports['create_bell_state_phi_minus'] != 'undefined', 'missing Wasm export: create_bell_state_phi_minus');
  assert(typeof wasmExports['create_bell_state_psi_plus'] != 'undefined', 'missing Wasm export: create_bell_state_psi_plus');
  assert(typeof wasmExports['create_bell_state_psi_minus'] != 'undefined', 'missing Wasm export: create_bell_state_psi_minus');
  assert(typeof wasmExports['create_bell_state'] != 'undefined', 'missing Wasm export: create_bell_state');
  assert(typeof wasmExports['calculate_chsh_parameter'] != 'undefined', 'missing Wasm export: calculate_chsh_parameter');
  assert(typeof wasmExports['bell_test_chsh'] != 'undefined', 'missing Wasm export: bell_test_chsh');
  assert(typeof wasmExports['fflush'] != 'undefined', 'missing Wasm export: fflush');
  assert(typeof wasmExports['bell_get_optimal_settings'] != 'undefined', 'missing Wasm export: bell_get_optimal_settings');
  assert(typeof wasmExports['emscripten_stack_get_end'] != 'undefined', 'missing Wasm export: emscripten_stack_get_end');
  assert(typeof wasmExports['emscripten_stack_get_base'] != 'undefined', 'missing Wasm export: emscripten_stack_get_base');
  assert(typeof wasmExports['emscripten_stack_init'] != 'undefined', 'missing Wasm export: emscripten_stack_init');
  assert(typeof wasmExports['emscripten_stack_get_free'] != 'undefined', 'missing Wasm export: emscripten_stack_get_free');
  assert(typeof wasmExports['_emscripten_stack_restore'] != 'undefined', 'missing Wasm export: _emscripten_stack_restore');
  assert(typeof wasmExports['_emscripten_stack_alloc'] != 'undefined', 'missing Wasm export: _emscripten_stack_alloc');
  assert(typeof wasmExports['emscripten_stack_get_current'] != 'undefined', 'missing Wasm export: emscripten_stack_get_current');
  assert(typeof wasmExports['memory'] != 'undefined', 'missing Wasm export: memory');
  assert(typeof wasmExports['__indirect_function_table'] != 'undefined', 'missing Wasm export: __indirect_function_table');
  _quantum_state_init = Module['_quantum_state_init'] = createExportWrapper('quantum_state_init', 2);
  _free = Module['_free'] = createExportWrapper('free', 1);
  _quantum_state_free = Module['_quantum_state_free'] = createExportWrapper('quantum_state_free', 1);
  _quantum_state_normalize = Module['_quantum_state_normalize'] = createExportWrapper('quantum_state_normalize', 1);
  _quantum_state_clone = Module['_quantum_state_clone'] = createExportWrapper('quantum_state_clone', 2);
  _quantum_state_reset = Module['_quantum_state_reset'] = createExportWrapper('quantum_state_reset', 1);
  _quantum_state_entropy = Module['_quantum_state_entropy'] = createExportWrapper('quantum_state_entropy', 1);
  _quantum_state_purity = Module['_quantum_state_purity'] = createExportWrapper('quantum_state_purity', 1);
  _quantum_state_fidelity = Module['_quantum_state_fidelity'] = createExportWrapper('quantum_state_fidelity', 2);
  _quantum_state_get_probability = Module['_quantum_state_get_probability'] = createExportWrapper('quantum_state_get_probability', 2);
  _gate_pauli_x = Module['_gate_pauli_x'] = createExportWrapper('gate_pauli_x', 2);
  _gate_pauli_y = Module['_gate_pauli_y'] = createExportWrapper('gate_pauli_y', 2);
  _gate_pauli_z = Module['_gate_pauli_z'] = createExportWrapper('gate_pauli_z', 2);
  _gate_hadamard = Module['_gate_hadamard'] = createExportWrapper('gate_hadamard', 2);
  _gate_s = Module['_gate_s'] = createExportWrapper('gate_s', 2);
  _gate_s_dagger = Module['_gate_s_dagger'] = createExportWrapper('gate_s_dagger', 2);
  _gate_t = Module['_gate_t'] = createExportWrapper('gate_t', 2);
  _gate_t_dagger = Module['_gate_t_dagger'] = createExportWrapper('gate_t_dagger', 2);
  _gate_phase = Module['_gate_phase'] = createExportWrapper('gate_phase', 3);
  _gate_rx = Module['_gate_rx'] = createExportWrapper('gate_rx', 3);
  _gate_ry = Module['_gate_ry'] = createExportWrapper('gate_ry', 3);
  _gate_rz = Module['_gate_rz'] = createExportWrapper('gate_rz', 3);
  _gate_u3 = Module['_gate_u3'] = createExportWrapper('gate_u3', 5);
  _gate_cnot = Module['_gate_cnot'] = createExportWrapper('gate_cnot', 3);
  _gate_cz = Module['_gate_cz'] = createExportWrapper('gate_cz', 3);
  _gate_cy = Module['_gate_cy'] = createExportWrapper('gate_cy', 3);
  _gate_swap = Module['_gate_swap'] = createExportWrapper('gate_swap', 3);
  _gate_cphase = Module['_gate_cphase'] = createExportWrapper('gate_cphase', 4);
  _gate_crx = Module['_gate_crx'] = createExportWrapper('gate_crx', 4);
  _gate_cry = Module['_gate_cry'] = createExportWrapper('gate_cry', 4);
  _gate_crz = Module['_gate_crz'] = createExportWrapper('gate_crz', 4);
  _gate_toffoli = Module['_gate_toffoli'] = createExportWrapper('gate_toffoli', 4);
  _gate_fredkin = Module['_gate_fredkin'] = createExportWrapper('gate_fredkin', 4);
  _gate_qft = Module['_gate_qft'] = createExportWrapper('gate_qft', 3);
  _gate_iqft = Module['_gate_iqft'] = createExportWrapper('gate_iqft', 3);
  _measurement_probability_one = Module['_measurement_probability_one'] = createExportWrapper('measurement_probability_one', 2);
  _measurement_probability_zero = Module['_measurement_probability_zero'] = createExportWrapper('measurement_probability_zero', 2);
  _measurement_all_probabilities = Module['_measurement_all_probabilities'] = createExportWrapper('measurement_all_probabilities', 2);
  _measurement_probability_distribution = Module['_measurement_probability_distribution'] = createExportWrapper('measurement_probability_distribution', 2);
  _measurement_single_qubit = Module['_measurement_single_qubit'] = createExportWrapper('measurement_single_qubit', 3);
  _measurement_all_qubits = Module['_measurement_all_qubits'] = createExportWrapper('measurement_all_qubits', 2);
  _measurement_expectation_z = Module['_measurement_expectation_z'] = createExportWrapper('measurement_expectation_z', 2);
  _measurement_expectation_x = Module['_measurement_expectation_x'] = createExportWrapper('measurement_expectation_x', 2);
  _measurement_expectation_y = Module['_measurement_expectation_y'] = createExportWrapper('measurement_expectation_y', 2);
  _measurement_correlation_zz = Module['_measurement_correlation_zz'] = createExportWrapper('measurement_correlation_zz', 3);
  _malloc = Module['_malloc'] = createExportWrapper('malloc', 1);
  _grover_oracle = Module['_grover_oracle'] = createExportWrapper('grover_oracle', 2);
  _grover_diffusion = Module['_grover_diffusion'] = createExportWrapper('grover_diffusion', 1);
  _grover_iteration = Module['_grover_iteration'] = createExportWrapper('grover_iteration', 2);
  _grover_optimal_iterations = Module['_grover_optimal_iterations'] = createExportWrapper('grover_optimal_iterations', 1);
  _grover_search = Module['_grover_search'] = createExportWrapper('grover_search', 4);
  _pauli_hamiltonian_create = Module['_pauli_hamiltonian_create'] = createExportWrapper('pauli_hamiltonian_create', 2);
  _pauli_hamiltonian_free = Module['_pauli_hamiltonian_free'] = createExportWrapper('pauli_hamiltonian_free', 1);
  _pauli_hamiltonian_add_term = Module['_pauli_hamiltonian_add_term'] = createExportWrapper('pauli_hamiltonian_add_term', 4);
  _vqe_create_h2_hamiltonian = Module['_vqe_create_h2_hamiltonian'] = createExportWrapper('vqe_create_h2_hamiltonian', 1);
  _vqe_create_hardware_efficient_ansatz = Module['_vqe_create_hardware_efficient_ansatz'] = createExportWrapper('vqe_create_hardware_efficient_ansatz', 2);
  _vqe_ansatz_free = Module['_vqe_ansatz_free'] = createExportWrapper('vqe_ansatz_free', 1);
  _vqe_apply_ansatz = Module['_vqe_apply_ansatz'] = createExportWrapper('vqe_apply_ansatz', 2);
  _vqe_optimizer_create = Module['_vqe_optimizer_create'] = createExportWrapper('vqe_optimizer_create', 1);
  _vqe_optimizer_free = Module['_vqe_optimizer_free'] = createExportWrapper('vqe_optimizer_free', 1);
  _vqe_solver_create = Module['_vqe_solver_create'] = createExportWrapper('vqe_solver_create', 4);
  _vqe_solver_free = Module['_vqe_solver_free'] = createExportWrapper('vqe_solver_free', 1);
  _vqe_compute_energy = Module['_vqe_compute_energy'] = createExportWrapper('vqe_compute_energy', 2);
  _vqe_solve = Module['_vqe_solve'] = createExportWrapper('vqe_solve', 2);
  _vqe_hartree_to_kcalmol = Module['_vqe_hartree_to_kcalmol'] = createExportWrapper('vqe_hartree_to_kcalmol', 1);
  _ising_model_create = Module['_ising_model_create'] = createExportWrapper('ising_model_create', 1);
  _ising_model_free = Module['_ising_model_free'] = createExportWrapper('ising_model_free', 1);
  _ising_model_set_coupling = Module['_ising_model_set_coupling'] = createExportWrapper('ising_model_set_coupling', 4);
  _ising_model_set_field = Module['_ising_model_set_field'] = createExportWrapper('ising_model_set_field', 3);
  _ising_model_evaluate = Module['_ising_model_evaluate'] = createExportWrapper('ising_model_evaluate', 2);
  _qaoa_solver_create = Module['_qaoa_solver_create'] = createExportWrapper('qaoa_solver_create', 3);
  _qaoa_solver_free = Module['_qaoa_solver_free'] = createExportWrapper('qaoa_solver_free', 1);
  _qaoa_apply_circuit = Module['_qaoa_apply_circuit'] = createExportWrapper('qaoa_apply_circuit', 5);
  _qaoa_compute_expectation = Module['_qaoa_compute_expectation'] = createExportWrapper('qaoa_compute_expectation', 3);
  _qaoa_solve = Module['_qaoa_solve'] = createExportWrapper('qaoa_solve', 2);
  _create_bell_state_phi_plus = Module['_create_bell_state_phi_plus'] = createExportWrapper('create_bell_state_phi_plus', 3);
  _create_bell_state_phi_minus = Module['_create_bell_state_phi_minus'] = createExportWrapper('create_bell_state_phi_minus', 3);
  _create_bell_state_psi_plus = Module['_create_bell_state_psi_plus'] = createExportWrapper('create_bell_state_psi_plus', 3);
  _create_bell_state_psi_minus = Module['_create_bell_state_psi_minus'] = createExportWrapper('create_bell_state_psi_minus', 3);
  _create_bell_state = Module['_create_bell_state'] = createExportWrapper('create_bell_state', 4);
  _calculate_chsh_parameter = Module['_calculate_chsh_parameter'] = createExportWrapper('calculate_chsh_parameter', 1);
  _bell_test_chsh = Module['_bell_test_chsh'] = createExportWrapper('bell_test_chsh', 7);
  _fflush = createExportWrapper('fflush', 1);
  _bell_get_optimal_settings = Module['_bell_get_optimal_settings'] = createExportWrapper('bell_get_optimal_settings', 1);
  _emscripten_stack_get_end = wasmExports['emscripten_stack_get_end'];
  _emscripten_stack_get_base = wasmExports['emscripten_stack_get_base'];
  _emscripten_stack_init = wasmExports['emscripten_stack_init'];
  _emscripten_stack_get_free = wasmExports['emscripten_stack_get_free'];
  __emscripten_stack_restore = wasmExports['_emscripten_stack_restore'];
  __emscripten_stack_alloc = wasmExports['_emscripten_stack_alloc'];
  _emscripten_stack_get_current = wasmExports['emscripten_stack_get_current'];
  memory = wasmMemory = wasmExports['memory'];
  __indirect_function_table = wasmExports['__indirect_function_table'];
}

var wasmImports = {
  /** @export */
  _abort_js: __abort_js,
  /** @export */
  emscripten_resize_heap: _emscripten_resize_heap,
  /** @export */
  environ_get: _environ_get,
  /** @export */
  environ_sizes_get: _environ_sizes_get,
  /** @export */
  fd_close: _fd_close,
  /** @export */
  fd_seek: _fd_seek,
  /** @export */
  fd_write: _fd_write
};


// include: postamble.js
// === Auto-generated postamble setup entry stuff ===

var calledRun;

function stackCheckInit() {
  // This is normally called automatically during __wasm_call_ctors but need to
  // get these values before even running any of the ctors so we call it redundantly
  // here.
  _emscripten_stack_init();
  // TODO(sbc): Move writeStackCookie to native to to avoid this.
  writeStackCookie();
}

function run() {

  stackCheckInit();

  preRun();

  function doRun() {
    // run may have just been called through dependencies being fulfilled just in this very frame,
    // or while the async setStatus time below was happening
    assert(!calledRun);
    calledRun = true;
    Module['calledRun'] = true;

    if (ABORT) return;

    initRuntime();

    readyPromiseResolve?.(Module);
    Module['onRuntimeInitialized']?.();
    consumedModuleProp('onRuntimeInitialized');

    assert(!Module['_main'], 'compiled without a main, but one is present. if you added it from JS, use Module["onRuntimeInitialized"]');

    postRun();
  }

  if (Module['setStatus']) {
    Module['setStatus']('Running...');
    setTimeout(() => {
      setTimeout(() => Module['setStatus'](''), 1);
      doRun();
    }, 1);
  } else
  {
    doRun();
  }
  checkStackCookie();
}

function checkUnflushedContent() {
  // Compiler settings do not allow exiting the runtime, so flushing
  // the streams is not possible. but in ASSERTIONS mode we check
  // if there was something to flush, and if so tell the user they
  // should request that the runtime be exitable.
  // Normally we would not even include flush() at all, but in ASSERTIONS
  // builds we do so just for this check, and here we see if there is any
  // content to flush, that is, we check if there would have been
  // something a non-ASSERTIONS build would have not seen.
  // How we flush the streams depends on whether we are in SYSCALLS_REQUIRE_FILESYSTEM=0
  // mode (which has its own special function for this; otherwise, all
  // the code is inside libc)
  var oldOut = out;
  var oldErr = err;
  var has = false;
  out = err = (x) => {
    has = true;
  }
  try { // it doesn't matter if it fails
    flush_NO_FILESYSTEM();
  } catch(e) {}
  out = oldOut;
  err = oldErr;
  if (has) {
    warnOnce('stdio streams had content in them that was not flushed. you should set EXIT_RUNTIME to 1 (see the Emscripten FAQ), or make sure to emit a newline when you printf etc.');
    warnOnce('(this may also be due to not including full filesystem support - try building with -sFORCE_FILESYSTEM)');
  }
}

var wasmExports;

// In modularize mode the generated code is within a factory function so we
// can use await here (since it's not top-level-await).
wasmExports = await (createWasm());

run();

// end include: postamble.js

// include: /home/cos/projects/moonlab/bindings/javascript/packages/core/emscripten/post.js
/**
 * Moonlab WASM Post-initialization Script
 *
 * This runs after the WASM module is loaded.
 * NOTE: onRuntimeInitialized is set up in pre.js, NOT here.
 * Emscripten marks it as "consumed" once called, so any assignment
 * here would fail with "Attempt to set Module.onRuntimeInitialized
 * after it has already been processed."
 */

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
  core: '0.1.1',
  wasm: true
};
// end include: /home/cos/projects/moonlab/bindings/javascript/packages/core/emscripten/post.js

// include: postamble_modularize.js
// In MODULARIZE mode we wrap the generated code in a factory function
// and return either the Module itself, or a promise of the module.
//
// We assign to the `moduleRtn` global here and configure closure to see
// this as an extern so it won't get minified.

if (runtimeInitialized)  {
  moduleRtn = Module;
} else {
  // Set up the promise that indicates the Module is initialized
  moduleRtn = new Promise((resolve, reject) => {
    readyPromiseResolve = resolve;
    readyPromiseReject = reject;
  });
}

// Assertion for attempting to access module properties on the incoming
// moduleArg.  In the past we used this object as the prototype of the module
// and assigned properties to it, but now we return a distinct object.  This
// keeps the instance private until it is ready (i.e the promise has been
// resolved).
for (const prop of Object.keys(Module)) {
  if (!(prop in moduleArg)) {
    Object.defineProperty(moduleArg, prop, {
      configurable: true,
      get() {
        abort(`Access to module property ('${prop}') is no longer possible via the module constructor argument; Instead, use the result of the module constructor.`)
      }
    });
  }
}
// end include: postamble_modularize.js



    return moduleRtn;
  };
})();

// Export using a UMD style export, or ES6 exports if selected
if (typeof exports === 'object' && typeof module === 'object') {
  module.exports = MoonlabModule;
  // This default export looks redundant, but it allows TS to import this
  // commonjs style module.
  module.exports.default = MoonlabModule;
} else if (typeof define === 'function' && define['amd'])
  define([], () => MoonlabModule);

