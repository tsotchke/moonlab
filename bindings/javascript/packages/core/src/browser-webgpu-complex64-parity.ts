/**
 * Browser-safe entry for the reduced WebGPU complex64 parity contract.
 *
 * Keep this entry independent of the package root: the root preserves the
 * documented Node-only control-plane exports, which import TCP/TLS modules.
 */

export * from './webgpu-complex64-parity';
export { canonicalJson } from './canonical-json';
