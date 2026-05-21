# Extension Surfaces

This document is the integration guide for the four runtime registries
that moonlab v1.0.3 exposes for private overlays, sibling libraries
(QGTL, libirrep, SbNN), and customer applications.

The C ABI is the contract.  All four surfaces are reachable from
Python, Rust, and JavaScript / WebAssembly through the in-tree
bindings.  No language-specific reservations -- if you can call a C
function pointer, you can extend moonlab.

For a working executable that uses every surface in one program, see
`examples/extensions/open_core_overlay_demo.c` (built and tested by
default under `example_open_core_overlay_demo`).

## Surface 1: Execution backends

`moonlab_register_backend(const moonlab_backend_t *)` adds a new
execution backend that any job pinned to its name dispatches to.
Use cases: live hardware (IBM Quantum, Rigetti, IonQ), GPU clusters,
alternative simulators, deterministic mocks for testing.

### C

```c
#include "src/distributed/scheduler.h"

static int my_backend_execute(const moonlab_job_t *job,
                              moonlab_job_results_t *out,
                              void *ctx)
{
    /* read circuit from job; write outcomes to out->outcomes;
       set out->{num_qubits, total_shots, num_workers_used,
                worker_seconds}.  Return MOONLAB_SCHED_OK on
       success, negative on failure. */
}

moonlab_backend_t b = {
    .name        = "my-backend",
    .execute     = my_backend_execute,
    .ctx         = &my_session,
    .description = "Connects to my proprietary cluster",
};
moonlab_register_backend(&b);
```

The struct is copied into the registry; pointer storage may be
freed after the call.

### Python

```python
# Backends are still C-only in Python; register from a host
# extension module if you need to plug in from Python.  The Python
# binding consumes registered backends via Job.set_backend(name).
from moonlab.scheduler import Job
from moonlab.qgtl import GateType

j = Job(num_qubits=2)
j.add_gate(GateType.H, 0).add_gate(GateType.CNOT, 1, 0)
j.set_num_shots(1024).set_backend("my-backend")
r = j.execute()
```

### Status codes

- `MOONLAB_SCHED_OK` (0)
- `MOONLAB_SCHED_BACKEND_NOT_FOUND` (-506)
- `MOONLAB_SCHED_BACKEND_BUSY` (-507)

## Surface 2: Vendor-noise profile registry

`moonlab_register_vendor_noise_profile(name, profile)` decouples
calibration data from backend registration.  Live-calibration
scrapers push today's IBM Falcon / Rigetti Aspen / IonQ Forte
snapshot under a name; backends look profiles up by name at
execute time, so updates take effect in place without re-registering
the backend.

### C

```c
#include "src/applications/vendor_noise_backend.h"

moonlab_vendor_noise_profile_t today = {
    .p_gate_1q   = 0.0011,
    .p_gate_2q   = 0.0095,
    .p_readout   = 0.0162,
    .description = "IBM Falcon r5.11 (2026-05-20 scraper output)",
};
moonlab_register_vendor_noise_backend_with_profile(
    "ibm-falcon-2026-05-20-snapshot", &today);
```

To update in place from a daily scraper, call
`moonlab_register_vendor_noise_profile()` again with the same
`name`.  The registry replaces the profile and the next
`scheduler_run` against the matching backend picks up the new
numbers.

### Python

```python
from moonlab.scheduler import (
    VendorNoiseProfile,
    register_vendor_noise_profile,
    lookup_vendor_noise_profile,
)

register_vendor_noise_profile(
    "ibm-falcon-2026-05-20-snapshot",
    VendorNoiseProfile(
        p_gate_1q=0.0011, p_gate_2q=0.0095, p_readout=0.0162,
        description="IBM Falcon r5.11 (live snapshot)",
    ),
)

# Read back:
prof = lookup_vendor_noise_profile("ibm-falcon-2026-05-20-snapshot")
assert prof.p_gate_2q == 0.0095
```

### Rust

```rust
use moonlab::scheduler::{
    register_vendor_noise_profile, lookup_vendor_noise_profile,
    VendorNoiseProfile,
};

let prof = VendorNoiseProfile {
    p_gate_1q: 0.0011, p_gate_2q: 0.0095, p_readout: 0.0162,
    description: "IBM Falcon r5.11 (live snapshot)".to_string(),
};
register_vendor_noise_profile("ibm-falcon-2026-05-20-snapshot", &prof)?;
```

### JavaScript

```typescript
import {
    registerVendorNoiseProfile,
    lookupVendorNoiseProfile,
} from '@moonlab/quantum-core';

await registerVendorNoiseProfile('ibm-falcon-2026-05-20-snapshot', {
    pGate1q: 0.0011, pGate2q: 0.0095, pReadout: 0.0162,
    description: 'IBM Falcon r5.11 (live snapshot)',
});
```

## Surface 3: Decoder runtime registry

`moonlab_register_decoder(name, fn, ctx, description)` plugs custom
QEC decoders into the same dispatcher as the five built-ins
(`greedy`, `mwpm_exact`, `sbnn`, `libirrep_single_shot`,
`pymatching`).  The enum surface
(`moonlab_decoder_decode(MOONLAB_DECODER_GREEDY, ...)`) routes
through the registry too, so re-registering `"greedy"` with a
proprietary BP-OSD takes over both dispatch paths.

### C

```c
#include "src/applications/decoder_bench.h"

static int my_decoder(const moonlab_decoder_input_t *in, void *ctx)
{
    /* read in->syndromes, write in->corrections.  Return
       MOONLAB_DECODER_OK on success. */
}

moonlab_register_decoder("my-bp-osd", my_decoder, &my_state,
    "Proprietary BP-OSD decoder for surface codes");

/* Dispatch by name: */
moonlab_decoder_input_t input = { /* ... */ };
moonlab_decoder_decode_by_name("my-bp-osd", &input);
```

### Python

```python
from moonlab.decoder import register_decoder, decode_by_name

def my_decoder(distance, num_qubits, is_toric, syndromes):
    """Return a length-num_qubits correction byte vector."""
    return [0] * num_qubits  # placeholder

register_decoder("my-bp-osd", my_decoder,
                 description="Proprietary BP-OSD")

corr = decode_by_name("my-bp-osd",
                     distance=5, num_qubits=50, is_toric=True,
                     syndromes=[0] * 25)
```

### Rust

```rust
use moonlab::decoder::{register_decoder, decode_by_name, CodeGeometry};

register_decoder("my-bp-osd", "Proprietary BP-OSD",
    |geom, _syndromes, _seed| {
        Ok(vec![0u8; geom.num_qubits as usize])
    })?;

let corr = decode_by_name("my-bp-osd",
    &CodeGeometry { distance: 5, num_qubits: 50, is_toric: true },
    &vec![0u8; 25], 0)?;
```

### JavaScript

```typescript
import { registerDecoder, decodeByName } from '@moonlab/quantum-core';

await registerDecoder('my-bp-osd', (code, _syndromes, _seed) => {
    return new Uint8Array(code.numQubits);  // placeholder
}, 'Proprietary BP-OSD');

const corr = await decodeByName('my-bp-osd', {
    distance: 5, numQubits: 50, isToric: true,
}, new Uint8Array(25));
```

### Status codes

- `MOONLAB_DECODER_OK` (0)
- `MOONLAB_DECODER_NOT_BUILT` (-401) -- build flag missing
- `MOONLAB_DECODER_BAD_ARG` (-402) -- NULL or unknown name
- `MOONLAB_DECODER_INFEASIBLE` (-403) -- odd-parity syndrome
- `MOONLAB_DECODER_OOM` (-404) -- allocation failure / runtime error

## Surface 4: Scheduler completion hook

`moonlab_scheduler_set_completion_hook(fn, ctx)` installs a callback
that fires after every successful `scheduler_run`.  Failed runs
(`MOONLAB_SCHED_BACKEND_NOT_FOUND`, backend execute errors) do not
fire the hook.  Single-slot.  Synchronous on the caller thread.

The hook is the primary integration point for commercial
deployments: billing meters, audit logs, customer dashboards,
alerting, quota enforcement.

### C

```c
#include "src/distributed/scheduler.h"

typedef struct { int n_runs; double total_cost; } billing_t;

static void billing_hook(const moonlab_job_t          *job,
                         const moonlab_job_results_t  *out,
                         const char                   *backend_name,
                         void                         *ctx)
{
    billing_t *b = (billing_t *)ctx;
    b->n_runs++;
    /* Bill at $0.01/kshot on simulator, $1.00/kshot on overlay,
       $0.50/kshot on ibm-falcon-... */
    b->total_cost += /* ... */ ;
}

billing_t state = {0};
moonlab_scheduler_set_completion_hook(billing_hook, &state);

/* ... run jobs ... */

moonlab_scheduler_set_completion_hook(NULL, NULL);  /* detach */
```

### Python

```python
from moonlab.scheduler import set_completion_hook, clear_completion_hook

def billing(num_qubits, total_shots, backend):
    # Record in a customer ledger or push to a metering service.
    log_run(qubits=num_qubits, shots=total_shots, backend=backend)

set_completion_hook(billing)
# ... run jobs ...
clear_completion_hook()
```

### Rust

```rust
use moonlab::scheduler::{set_completion_hook, clear_completion_hook};

set_completion_hook(|info| {
    log_run(info.num_qubits, info.total_shots,
            info.backend.as_deref().unwrap_or("(none)"));
})?;
// ... run jobs ...
clear_completion_hook()?;
```

### JavaScript

```typescript
import { setCompletionHook, clearCompletionHook } from '@moonlab/quantum-core';

await setCompletionHook(info => {
    metricsServer.recordRun(info.numQubits, info.totalShots,
                            info.backendName ?? '(none)');
});
// ... run jobs ...
await clearCompletionHook();
```

## Open-core rules

This list mirrors `COMMUNITY_EDITION.md` and is reproduced here so
overlay authors don't have to cross-reference.

1. **The community edition must stay useful without a commercial
   license.**  Every surface above is fully functional on the
   public C ABI with the built-in implementations.

2. **Public extension points should be stable, documented, and
   tested.**  All four registries are exercised by ctest +
   pytest + cargo test + vitest (vitest auto-skips pending WASM
   rebuild).

3. **Optional commercial or sibling-library behavior must fail
   with explicit status codes when unavailable.**  Decoder slots
   that need libirrep / SbNN return
   `MOONLAB_DECODER_NOT_BUILT` when the link-time flag is off,
   never silently degrade.

4. **Public examples must not require private credentials or
   customer data.**  The overlay demo uses an in-memory
   deterministic backend and toy pricing for the billing hook.

5. **Real-hardware claims require provider-path evidence, not
   simulator or emulator output.**  The vendor-noise emulators
   are explicitly labeled `-emu`; the bare `ibm-falcon` /
   `rigetti-aspen` / `ionq-forte` names are legacy aliases.  QGTL
   Commercial is the path to live submission.

6. **Billing, audit, dashboards, and enterprise auth attach
   through scheduler hooks and private deployment code; the
   public runtime remains clean.**  No proprietary code lives in
   this repository.  The public-CI hygiene grep enforces this on
   every PR.

## Lifetime and threading

- All four registries are mutex-protected and thread-safe.
- Registry entries copy the name and description strings; caller
  storage may be freed after registration.
- Backend / decoder `ctx` pointers are stored as-is and dereferenced
  inside the user's `fn`.  The overlay owns the ctx lifetime.
- The completion hook is single-slot and fires on the caller thread.
  It must not block (a long-running hook stalls the scheduler).
- Re-registering an existing name replaces the prior entry in place.
  Backends and decoders registered against the old `ctx` are
  silently re-routed; transients in-flight at the swap moment
  continue on the old implementation until they return.
- Unregistering a decoder does not invalidate the registered
  function pointer in any racing scheduler call.  In Rust this
  manifests as leak-by-design: the slot stays alive for program
  lifetime so the C ABI never dereferences freed memory.

## See also

- `examples/extensions/open_core_overlay_demo.c` -- end-to-end
  reference exercising all four surfaces.
- `bindings/python/tests/test_decoder.py` and
  `test_scheduler.py` -- runnable Python parity tests.
- `bindings/rust/moonlab/src/decoder.rs` and `scheduler.rs`
  (`#[cfg(test)] mod tests`) -- runnable Rust parity tests.
- `bindings/javascript/packages/core/src/__tests__/registry.integration.test.ts`
  -- runnable JS / WASM parity tests (auto-skip until WASM rebuild).
- `COMMUNITY_EDITION.md` -- the public / commercial product
  boundary at the project level.
- `docs/STABLE_ABI.md` -- the stable-ABI policy that governs how
  these surfaces evolve.
