# Moonlab error codes

This file is preserved for backward-compatible links.  The canonical
error-codes reference now lives at
[`docs/reference/error-codes.md`](reference/error-codes.md), which
documents:

- the `moonlab_status_t` registry (`src/utils/moonlab_status.h`),
- the `qs_error_t` legacy enum used by `src/quantum/`,
- every per-module `*_error_t` enum with the header that declares it
  (CA-MPS, CA-PEPS, TN state/gates/measure, tensor, contract,
  svd_compress, Clifford, partition, dist_gate, collective,
  mpi_bridge, gpu, eshkol, MPDO),
- the `MOONLAB_ESHKOL_OK` naming exception and the convention for
  adding new error enums.

See `docs/reference/error-codes.md`.
