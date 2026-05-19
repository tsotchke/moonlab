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
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAP32: Int32Array;
  HEAPU8: Uint8Array;
  UTF8ToString: (ptr: number) => string;
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
