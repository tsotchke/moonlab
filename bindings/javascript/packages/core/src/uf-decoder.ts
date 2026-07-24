/**
 * Union-find QEC decoder (C-side since v1.2.0; JS binding since
 * v1.2.0).
 *
 * Wraps `src/qec/uf_decoder.{c,h}`: Delfosse-Nickerson union-find over
 * a detector error model in edge-list form.  Clusters grow around lit
 * detectors until every cluster has even parity or touches the
 * boundary, then the spanning forest is peeled to give a correction.
 * Consumes the detector-major layout `pauliFrame.sampleDetectors`
 * emits, so the sampler and the decoder compose directly.
 *
 * @example
 * ```typescript
 * import { UfDecoder, UF_BOUNDARY } from '@moonlab/quantum-core';
 *
 * // Repetition code: D0 -- boundary (flips the observable),
 * // D0 -- D1, D1 -- boundary.
 * const decoder = await UfDecoder.create(2, 1, [
 *   { a: 0, b: UF_BOUNDARY, observables: 1 },
 *   { a: 0, b: 1 },
 *   { a: 1, b: UF_BOUNDARY },
 * ]);
 * const det = new Uint8Array([0, 1, 1, 0,   // D0 over 4 shots
 *                             0, 0, 1, 1]); // D1 over 4 shots
 * const obs = decoder.decodeBatch(det, 4);  // -> [0, 1, 0, 0]
 * decoder.dispose();
 * ```
 *
 * @since v1.2.0
 */

import { getModule } from './wasm-loader';

/** Sentinel destination marking an edge that runs to the boundary
 *  (`MOONLAB_UF_BOUNDARY = UINT32_MAX`). */
export const UF_BOUNDARY = 0xffffffff;

/** One error mechanism of the detector error model. */
export interface UfEdge {
  /** First detector index. */
  a: number;
  /** Second detector index, or {@link UF_BOUNDARY}. */
  b: number;
  /** Edge weight; larger means less likely.  Growth is quantised from
   *  these, so relative size is what matters.  Omit on every edge for
   *  unweighted growth. */
  weight?: number;
  /** Bitmask of the logical observables this edge flips (up to 64;
   *  pass a bigint for masks above 2^53). */
  observables?: number | bigint;
}

type UfDecoderModule = {
  _moonlab_uf_decoder_new: (
    numDetectors: number, numObservables: number,
    edgeAPtr: number, edgeBPtr: number, edgeWeightPtr: number,
    edgeObsPtr: number, numEdges: number,
  ) => number;
  _moonlab_uf_decoder_free: (d: number) => void;
  _moonlab_uf_decode_batch: (
    d: number, detPtr: number, numShots: number,
    numThreads: number, obsOutPtr: number,
  ) => number;
  _moonlab_uf_decoder_num_edges: (d: number) => number;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPU8: Uint8Array;
  HEAPU32: Uint32Array;
  HEAPF64: Float64Array;
};

/**
 * Union-find decoder over a fixed detector error model.  Build once,
 * decode many batches.
 */
export class UfDecoder {
  private ptr: number;
  private mod: UfDecoderModule;
  private readonly nDetectors: number;
  private readonly nObservables: number;
  private freed = false;

  private constructor(
    ptr: number, mod: UfDecoderModule, numDetectors: number, numObservables: number,
  ) {
    this.ptr = ptr;
    this.mod = mod;
    this.nDetectors = numDetectors;
    this.nObservables = numObservables;
  }

  /**
   * Build a decoder from a detector error model in edge-list form
   * (`moonlab_uf_decoder_new`).
   *
   * @param numDetectors   number of detector nodes.
   * @param numObservables number of logical observables (<= 64).
   * @param edges          error mechanisms; see {@link UfEdge}.
   */
  static async create(
    numDetectors: number,
    numObservables: number,
    edges: readonly UfEdge[],
  ): Promise<UfDecoder> {
    const mod = (await getModule()) as unknown as UfDecoderModule;
    const n = edges.length;
    if (n === 0) throw new Error('UfDecoder requires at least one edge');

    const weighted = edges.some((e) => e.weight !== undefined);
    const edgeAPtr = mod._malloc(n * 4);
    const edgeBPtr = mod._malloc(n * 4);
    const edgeObsPtr = mod._malloc(n * 8);
    const edgeWeightPtr = weighted ? mod._malloc(n * 8) : 0;
    try {
      for (let i = 0; i < n; i++) {
        mod.HEAPU32[(edgeAPtr >> 2) + i] = edges[i].a >>> 0;
        mod.HEAPU32[(edgeBPtr >> 2) + i] = edges[i].b >>> 0;
        const obs = BigInt(edges[i].observables ?? 0);
        mod.HEAPU32[(edgeObsPtr >> 2) + 2 * i] = Number(obs & 0xffffffffn);
        mod.HEAPU32[(edgeObsPtr >> 2) + 2 * i + 1] = Number((obs >> 32n) & 0xffffffffn);
        if (weighted) {
          mod.HEAPF64[(edgeWeightPtr >> 3) + i] = edges[i].weight ?? 1.0;
        }
      }
      const ptr = mod._moonlab_uf_decoder_new(
        numDetectors, numObservables,
        edgeAPtr, edgeBPtr, edgeWeightPtr, edgeObsPtr, n,
      );
      if (!ptr) {
        throw new Error('moonlab_uf_decoder_new returned NULL (invalid model?)');
      }
      return new UfDecoder(ptr, mod, numDetectors, numObservables);
    } finally {
      if (edgeWeightPtr) mod._free(edgeWeightPtr);
      mod._free(edgeObsPtr);
      mod._free(edgeBPtr);
      mod._free(edgeAPtr);
    }
  }

  /** Number of detector nodes the decoder was built with. */
  get numDetectors(): number { return this.nDetectors; }

  /** Number of logical observables the decoder was built with. */
  get numObservables(): number { return this.nObservables; }

  /** Number of edges the decoder was built with. */
  get numEdges(): number {
    if (this.freed) throw new Error('UfDecoder disposed');
    return this.mod._moonlab_uf_decoder_num_edges(this.ptr);
  }

  /**
   * Decode a batch of shots (`moonlab_uf_decode_batch`).
   *
   * @param detectorData detector-major bytes: detector `i` of shot `s`
   *   at `detectorData[i * numShots + s]` -- the layout
   *   `pauliFrame.sampleDetectors` writes.  Must hold
   *   `numDetectors * numShots` bytes.
   * @param numShots     number of shots in the batch.
   * @param numThreads   <= 0 selects all cores (single-threaded on
   *   wasm32 where OpenMP is unavailable).
   * @returns observable-major bytes: observable `o` of shot `s` at
   *   `out[o * numShots + s]`, `numObservables * numShots` total.
   */
  decodeBatch(
    detectorData: Uint8Array, numShots: number, numThreads = 1,
  ): Uint8Array {
    if (this.freed) throw new Error('UfDecoder disposed');
    const expected = this.nDetectors * numShots;
    if (detectorData.length !== expected) {
      throw new Error(
        `detectorData length ${detectorData.length} != numDetectors * numShots = ${expected}`,
      );
    }
    const mod = this.mod;
    const detPtr = mod._malloc(Math.max(expected, 1));
    const outLen = this.nObservables * numShots;
    const outPtr = mod._malloc(Math.max(outLen, 1));
    try {
      mod.HEAPU8.set(detectorData, detPtr);
      const rc = mod._moonlab_uf_decode_batch(
        this.ptr, detPtr, numShots, numThreads, outPtr,
      );
      if (rc !== numShots) {
        throw new Error(`moonlab_uf_decode_batch failed (rc=${rc})`);
      }
      return mod.HEAPU8.slice(outPtr, outPtr + outLen);
    } finally {
      mod._free(outPtr);
      mod._free(detPtr);
    }
  }

  /** Release the underlying C allocation. */
  dispose(): void {
    if (this.freed) return;
    this.mod._moonlab_uf_decoder_free(this.ptr);
    this.freed = true;
  }
}
