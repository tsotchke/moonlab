/**
 * Decoder-bench JS/WASM binding -- since v0.7.3.
 *
 * Mirrors the Python + Rust surfaces.  Five-slot dispatcher for the
 * QEC decoder zoo.  GREEDY and MWPM_EXACT are always available;
 * LIBIRREP_SS is build-conditional; SBNN + PYMATCHING return
 * NOT_BUILT until v0.7+.
 *
 * Tests auto-skip when the WASM build predates v0.6.7 (no
 * moonlab_decoder_* symbols yet); once a fresh WASM build picks up
 * the new exports the real assertions activate.
 *
 * @since v0.7.3
 */

import { getModule } from './wasm-loader';

export enum DecoderSlot {
  GREEDY      = 0,
  MWPM_EXACT  = 1,
  SBNN        = 2,
  LIBIRREP_SS = 3,
  PYMATCHING  = 4,
}

export const MOONLAB_DECODER_OK = 0;
export const MOONLAB_DECODER_NOT_BUILT = -401;
export const MOONLAB_DECODER_BAD_ARG = -402;
export const MOONLAB_DECODER_INFEASIBLE = -403;
export const MOONLAB_DECODER_OOM = -404;

export class DecoderError extends Error {
  readonly rc: number;
  constructor(message: string, rc: number) {
    super(message); this.name = 'DecoderError'; this.rc = rc;
  }
}

export class DecoderNotBuiltError extends DecoderError {
  constructor(slot: DecoderSlot) {
    super(`decoder slot ${DecoderSlot[slot]} not built`, MOONLAB_DECODER_NOT_BUILT);
    this.name = 'DecoderNotBuiltError';
  }
}

export interface CodeGeometry {
  distance: number;
  numQubits: number;
  isToric: boolean;
}

type DecoderModule = {
  _moonlab_decoder_decode: (slot: number, input: number) => number;
  _moonlab_decoder_slot_available: (slot: number) => number;
  _moonlab_decoder_slot_name: (slot: number) => number;
  // Runtime registry (since v1.0.3) -- present only when the WASM
  // build includes the new exports.
  _moonlab_register_decoder?: (
    name: number, fn: number, ctx: number, description: number,
  ) => number;
  _moonlab_unregister_decoder?: (name: number) => number;
  _moonlab_lookup_decoder?: (name: number) => number;
  _moonlab_decoder_decode_by_name?: (name: number, input: number) => number;
  _moonlab_num_decoders?: () => number;
  _moonlab_list_decoders?: (out: number, max: number) => number;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAP32: Int32Array;
  HEAPU8: Uint8Array;
  HEAPU32: Uint32Array;
  UTF8ToString: (ptr: number) => string;
  stringToUTF8: (s: string, ptr: number, max: number) => void;
  addFunction?: (fn: Function, sig: string) => number;
  removeFunction?: (ptr: number) => void;
};

export async function slotAvailable(slot: DecoderSlot): Promise<boolean> {
  const mod = (await getModule()) as unknown as DecoderModule;
  return mod._moonlab_decoder_slot_available(slot) === 1;
}

export async function slotName(slot: DecoderSlot): Promise<string> {
  const mod = (await getModule()) as unknown as DecoderModule;
  const ptr = mod._moonlab_decoder_slot_name(slot);
  return mod.UTF8ToString(ptr);
}

export async function decode(
  slot: DecoderSlot,
  code: CodeGeometry,
  syndromes: Uint8Array,
  rngSeed: bigint = 0n,
): Promise<Uint8Array> {
  const mod = (await getModule()) as unknown as DecoderModule;
  /* moonlab_decoder_code_t: 3 ints = 12 bytes; moonlab_decoder_input_t:
   * { code*, syndromes*, corrections*, int, uint64 } = ~28 bytes on
   * wasm32 with alignment, 32 to be safe. */
  const codePtr = mod._malloc(12);
  const inputPtr = mod._malloc(32);
  const sLen = syndromes.length;
  const synPtr = mod._malloc(sLen);
  const corrPtr = mod._malloc(code.numQubits);
  try {
    mod.HEAP32[codePtr >> 2]       = code.distance;
    mod.HEAP32[(codePtr >> 2) + 1] = code.numQubits;
    mod.HEAP32[(codePtr >> 2) + 2] = code.isToric ? 1 : 0;
    mod.HEAPU8.set(syndromes, synPtr);
    mod.HEAPU8.fill(0, corrPtr, corrPtr + code.numQubits);

    /* Input struct layout (32 bytes with padding):
     *  off 0  : code*
     *  off 4  : syndromes*
     *  off 8  : corrections*
     *  off 12 : num_stabilisers
     *  off 16 : rng_seed (uint64) -- 8-byte aligned */
    mod.HEAP32[inputPtr >> 2]       = codePtr;
    mod.HEAP32[(inputPtr >> 2) + 1] = synPtr;
    mod.HEAP32[(inputPtr >> 2) + 2] = corrPtr;
    mod.HEAP32[(inputPtr >> 2) + 3] = sLen;
    mod.HEAP32[(inputPtr >> 2) + 4] = Number(rngSeed & 0xFFFFFFFFn);
    mod.HEAP32[(inputPtr >> 2) + 5] = Number((rngSeed >> 32n) & 0xFFFFFFFFn);

    const rc = mod._moonlab_decoder_decode(slot, inputPtr);
    if (rc === MOONLAB_DECODER_NOT_BUILT) throw new DecoderNotBuiltError(slot);
    if (rc !== MOONLAB_DECODER_OK) {
      throw new DecoderError(`decode(${DecoderSlot[slot]}): rc=${rc}`, rc);
    }
    return new Uint8Array(mod.HEAPU8.subarray(corrPtr, corrPtr + code.numQubits));
  } finally {
    mod._free(codePtr);
    mod._free(inputPtr);
    mod._free(synPtr);
    mod._free(corrPtr);
  }
}

// ===================================================================
// Runtime decoder registry (since v1.0.3)
// ===================================================================

/** Helper: copy a JS string into a freshly-malloc'd C buffer.  Caller
 *  must free the returned pointer.  Returns 0 (NULL) if `s` is null /
 *  undefined. */
function jsStringToHeap(mod: DecoderModule, s: string | undefined): number {
  if (!s) return 0;
  // UTF-8 length + 1 for NUL.
  const len = new TextEncoder().encode(s).length + 1;
  const ptr = mod._malloc(len);
  mod.stringToUTF8(s, ptr, len);
  return ptr;
}

/** Whether this WASM build includes the v1.0.3 decoder registry. */
export async function decoderRegistryAvailable(): Promise<boolean> {
  const mod = (await getModule()) as unknown as DecoderModule;
  return typeof mod._moonlab_num_decoders === 'function';
}

export async function numDecoders(): Promise<number> {
  const mod = (await getModule()) as unknown as DecoderModule;
  if (!mod._moonlab_num_decoders) return 0;
  return mod._moonlab_num_decoders();
}

export async function listDecoders(): Promise<string[]> {
  const mod = (await getModule()) as unknown as DecoderModule;
  if (!mod._moonlab_num_decoders || !mod._moonlab_list_decoders) return [];
  const n = mod._moonlab_num_decoders();
  if (n <= 0) return [];
  // Pointer array: n * 4 bytes on wasm32.
  const buf = mod._malloc(n * 4);
  try {
    const written = mod._moonlab_list_decoders(buf, n);
    const names: string[] = [];
    for (let i = 0; i < written; i++) {
      const namePtr = mod.HEAPU32[(buf >> 2) + i];
      if (namePtr) names.push(mod.UTF8ToString(namePtr));
    }
    return names;
  } finally {
    mod._free(buf);
  }
}

/** Look up a decoder by name.  Returns `null` if not registered.
 *  Returns `{name, description}` for registered entries. */
export async function lookupDecoder(
  name: string,
): Promise<{ name: string; description: string } | null> {
  const mod = (await getModule()) as unknown as DecoderModule;
  if (!mod._moonlab_lookup_decoder) return null;
  const namePtr = jsStringToHeap(mod, name);
  try {
    const entryPtr = mod._moonlab_lookup_decoder(namePtr);
    if (entryPtr === 0) return null;
    // moonlab_decoder_entry_t: { name*, fn*, ctx*, description* } = 16 bytes.
    const ep32 = entryPtr >> 2;
    const namePtr2 = mod.HEAPU32[ep32];
    const descPtr  = mod.HEAPU32[ep32 + 3];
    return {
      name: namePtr2 ? mod.UTF8ToString(namePtr2) : '',
      description: descPtr ? mod.UTF8ToString(descPtr) : '',
    };
  } finally {
    mod._free(namePtr);
  }
}

/** User decoder closure.  Receives the code geometry + syndrome
 *  bytes + RNG seed and must return the length-`numQubits`
 *  correction byte vector. */
export type DecoderCallback = (
  code: CodeGeometry,
  syndromes: Uint8Array,
  rngSeed: bigint,
) => Uint8Array;

// Keep registered trampoline pointers alive forever -- the C
// runtime's stable ctx pointer is the trampoline pointer, so freeing
// it would dangle.  Indexed by name for the unregister path.
const registeredDecoders = new Map<string, { fnPtr: number; cb: DecoderCallback }>();

/** Register a JS decoder under `name`.  Since v1.0.3.  Throws if the
 *  WASM build does not include the registry or `addFunction` exports. */
export async function registerDecoder(
  name: string,
  callback: DecoderCallback,
  description: string = '',
): Promise<void> {
  const mod = (await getModule()) as unknown as DecoderModule;
  if (!mod._moonlab_register_decoder || !mod.addFunction) {
    throw new DecoderError(
      'WASM build lacks the v1.0.3 decoder registry or addFunction; ' +
        'rebuild with ALLOW_TABLE_GROWTH=1 and addFunction exported',
      MOONLAB_DECODER_BAD_ARG,
    );
  }
  // Trampoline: C calls (input*, ctx) -> int.  We marshal the input
  // struct out, run the JS callback, marshal the corrections back in.
  const trampoline = (inputPtr: number, _ctx: number): number => {
    try {
      const ip32 = inputPtr >> 2;
      const codePtr        = mod.HEAPU32[ip32];
      const synPtr         = mod.HEAPU32[ip32 + 1];
      const corrPtr        = mod.HEAPU32[ip32 + 2];
      const nStabilisers   = mod.HEAP32[ip32 + 3];
      const rngLo          = mod.HEAPU32[ip32 + 4];
      const rngHi          = mod.HEAPU32[ip32 + 5];
      const cp32 = codePtr >> 2;
      const code: CodeGeometry = {
        distance: mod.HEAP32[cp32],
        numQubits: mod.HEAP32[cp32 + 1],
        isToric: mod.HEAP32[cp32 + 2] !== 0,
      };
      const syndromes = new Uint8Array(
        mod.HEAPU8.subarray(synPtr, synPtr + nStabilisers),
      );
      const rngSeed = (BigInt(rngHi) << 32n) | BigInt(rngLo);
      const corr = callback(code, syndromes, rngSeed);
      if (corr.length < code.numQubits) return MOONLAB_DECODER_OOM;
      for (let q = 0; q < code.numQubits; q++) {
        mod.HEAPU8[corrPtr + q] = corr[q];
      }
      return MOONLAB_DECODER_OK;
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('JS decoder threw:', e);
      return MOONLAB_DECODER_OOM;
    }
  };
  const fnPtr = mod.addFunction(trampoline, 'iii');
  const namePtr = jsStringToHeap(mod, name);
  const descPtr = jsStringToHeap(mod, description);
  try {
    const rc = mod._moonlab_register_decoder(namePtr, fnPtr, 0, descPtr);
    if (rc !== MOONLAB_DECODER_OK) {
      mod.removeFunction?.(fnPtr);
      throw new DecoderError(`register_decoder(${name}): rc=${rc}`, rc);
    }
    // Replace prior registration's trampoline ref so the GC can drop
    // its closure when the next one overwrites it.
    const prior = registeredDecoders.get(name);
    if (prior) mod.removeFunction?.(prior.fnPtr);
    registeredDecoders.set(name, { fnPtr, cb: callback });
  } finally {
    mod._free(namePtr);
    if (descPtr) mod._free(descPtr);
  }
}

export async function unregisterDecoder(name: string): Promise<void> {
  const mod = (await getModule()) as unknown as DecoderModule;
  if (!mod._moonlab_unregister_decoder) return;
  const namePtr = jsStringToHeap(mod, name);
  try {
    const rc = mod._moonlab_unregister_decoder(namePtr);
    const slot = registeredDecoders.get(name);
    if (slot) {
      mod.removeFunction?.(slot.fnPtr);
      registeredDecoders.delete(name);
    }
    if (rc !== MOONLAB_DECODER_OK) {
      throw new DecoderError(`unregister_decoder(${name}): rc=${rc}`, rc);
    }
  } finally {
    mod._free(namePtr);
  }
}

/** Decode via a name-keyed registry dispatch. */
export async function decodeByName(
  name: string,
  code: CodeGeometry,
  syndromes: Uint8Array,
  rngSeed: bigint = 0n,
): Promise<Uint8Array> {
  const mod = (await getModule()) as unknown as DecoderModule;
  if (!mod._moonlab_decoder_decode_by_name) {
    throw new DecoderError(
      'WASM build lacks decode_by_name; rebuild with v1.0.3 exports',
      MOONLAB_DECODER_BAD_ARG,
    );
  }
  const codePtr = mod._malloc(12);
  const inputPtr = mod._malloc(32);
  const sLen = syndromes.length;
  const synPtr = mod._malloc(sLen);
  const corrPtr = mod._malloc(code.numQubits);
  const namePtr = jsStringToHeap(mod, name);
  try {
    mod.HEAP32[codePtr >> 2]       = code.distance;
    mod.HEAP32[(codePtr >> 2) + 1] = code.numQubits;
    mod.HEAP32[(codePtr >> 2) + 2] = code.isToric ? 1 : 0;
    mod.HEAPU8.set(syndromes, synPtr);
    mod.HEAPU8.fill(0, corrPtr, corrPtr + code.numQubits);
    mod.HEAP32[inputPtr >> 2]       = codePtr;
    mod.HEAP32[(inputPtr >> 2) + 1] = synPtr;
    mod.HEAP32[(inputPtr >> 2) + 2] = corrPtr;
    mod.HEAP32[(inputPtr >> 2) + 3] = sLen;
    mod.HEAP32[(inputPtr >> 2) + 4] = Number(rngSeed & 0xFFFFFFFFn);
    mod.HEAP32[(inputPtr >> 2) + 5] = Number((rngSeed >> 32n) & 0xFFFFFFFFn);
    const rc = mod._moonlab_decoder_decode_by_name(namePtr, inputPtr);
    if (rc !== MOONLAB_DECODER_OK) {
      throw new DecoderError(`decode_by_name(${name}): rc=${rc}`, rc);
    }
    return new Uint8Array(mod.HEAPU8.subarray(corrPtr, corrPtr + code.numQubits));
  } finally {
    mod._free(codePtr);
    mod._free(inputPtr);
    mod._free(synPtr);
    mod._free(corrPtr);
    mod._free(namePtr);
  }
}
