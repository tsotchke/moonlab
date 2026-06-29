# Archived Moonlab Documentation: Moonlab Community Edition

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Moonlab Community Edition

Moonlab Community Edition is the public MIT-licensed simulator, scheduler,
control-plane, cryptography, and multi-language runtime in this repository. It
is intended to remain a serious free product for local simulation, research
workflows, education, public integrations, and commercial embedding.

## Product Boundary

Moonlab Community Edition includes:

- Dense state-vector, tensor-network, Clifford, MPDO/noise, topology,
  chemistry/VQE, quantum-algorithm, QRNG, and post-quantum-cryptography
  capabilities.
- Stable ABI symbols documented in `docs/STABLE_ABI.md`.
- Local scheduler and control-plane primitives documented in
  `docs/CONTROL_PLANE.md`.
- Public open-core extension surfaces:
  `moonlab_register_backend`, `moonlab_register_vendor_noise_profile`,
  `moonlab_register_decoder`, and
  `moonlab_scheduler_set_completion_hook`.
- Python, Rust, and JavaScript bindings for the public runtime surface.
- Public tests, examples, benchmarks, and documentation.

Commercial Moonlab products may add:

- Hosted Moonlab Cloud execution, managed queues, quotas, account controls,
  billing, audit logs, and customer dashboards.
- Private live-hardware, GPU-cluster, or customer-prem scheduler backends.
- Private calibration scrapers, provider credentials, vendor-noise feeds, and
  enterprise deployment automation.
- Certified builds, signed packages, Docker/Helm artifacts, support branches,
  and SLA-backed releases.
- Customer-specific integrations and internal operational tooling.

Commercial additions should consume the public extension surfaces rather than
forking the public runtime. Do not commit customer secrets, billing
implementations, private provider credentials, internal deployment scripts, or
customer-specific overlays to this repository.

## Open-Core Rules

1. The community edition must stay useful without a commercial license.
2. Public extension points should be stable, documented, and tested.
3. Optional commercial or sibling-library behavior must fail with explicit
   status codes when unavailable.
4. Public examples must not require private credentials or customer data.
5. Real-hardware claims require provider-path evidence, not simulator or
   emulator output.
6. Billing, audit, dashboards, and enterprise auth attach through scheduler
   hooks and private deployment code; the public runtime remains clean.

## Relationship To QGTL

Moonlab is the simulator, scheduler, control-plane, and binding substrate.
QGTL is the quantum-geometric, topological, learning, and hardware-orchestration
layer.

The public boundary is:

- Moonlab exposes extension surfaces and stable ABI contracts.
- QGTL consumes Moonlab optionally and can register gate-model provider
  backends through Moonlab's scheduler.
- Private commercial overlays provide provider sessions, credentials, billing,
  audit, dashboards, and customer deployment logic.
- D-Wave/annealer workflows remain outside the Moonlab gate-model scheduler
  until Moonlab grows an explicit annealer job kind.

## Release Gate

A public release candidate should pass:

- Default native CTest coverage for the supported platforms.
- Binding tests for every public language surface included in the release.
- Stable ABI review for any symbol, status-code, or wire-protocol change.
- Public hygiene checks that reject private/proprietary markers and secrets.
- Evidence-gated documentation review for performance, provider, distributed,
  or commercial-readiness claims.

```
