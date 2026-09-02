# Detailed development log: `fix/browser-parity-entry`

- **Branch:** `fix/browser-parity-entry`
- **Started:** 2026-09-01
- **Base:** `feature/exomoonlab-tui`

## 2026-09-01 — Keep the parity harness browser-safe

### Prompt

ULG's GitHub Pages service-asset staging failed because MoonLab's browser
WebGPU parity harness imported the broad package entry, whose documented
Node-only control-plane exports resolve `net`, `tls`, `fs`, and `crypto`.

### Response and strategy

Preserve the established Node API and give the parity harness a narrow browser
entry. Extract the canonical JSON helper into a dependency-free module, export
the parity contract plus that helper from the new entry, and point the real
browser harness at the generated entry. Verify both the focused contracts and
the built browser module graph before using it to regenerate ULG assets.

### Results

- Focused parity and ULG artifact unit tests passed `34/34`.
- The full JavaScript core unit suite passed `133/133`; its integration suite
  passed `232/232` runnable tests with four declared control-plane skips.
- TypeScript/CJS/ESM/DTS builds passed and emitted the dedicated entry.
- Its complete static module closure contains no Node built-in import.
- The owned Chrome smoke acquired a WebGPU device and covered all five
  declared reduced operations; the handoff summary passed while retaining
  `fullPhysicsValidation=false`.
- The Release CMake build passed, followed by CTest with `184/184` runnable
  tests passing and the two declared libirrep skips.
