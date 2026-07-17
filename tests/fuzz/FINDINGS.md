# Fuzzing findings

Coverage-guided fuzz lane over libquantumsim's untrusted-input surfaces,
run under AddressSanitizer + UndefinedBehaviorSanitizer. HEAD at time of
run: `e92fea5` (ABI 0.5.0). This lane does not own `src/`; the items
below are handed off to the owning subsystems, not fixed here.

## 1. Resource amplification / memory-exhaustion DoS -- control plane

- **Severity**: medium (remote unauthenticated DoS in the default build).
- **Owning subsystem**: `src/control/control_plane.c` (`handle_one_request`)
  plus `src/applications/moonlab_qgtl_backend.c` (`moonlab_qgtl_execute`).
- **Surfaced by**: `control_plane_protocol_fuzz` libFuzzer soak (RSS
  out-of-memory at ~3.1 GB after ~40k executions, no ASan/UBSan error).
- **Reproducer** (checked in, quarantined):
  `corpora/control_plane_protocol_fuzz/crashes-pending/oom_qubit_amplification.bin`
  -- the 29 wire bytes `CIRCUIT 18\nNUM_QUBITS 30\nH 0\n`.

A single well-formed `CIRCUIT` (or `SHOTS`) frame, comfortably under the
4 MB body cap (`MOONLAB_CONTROL_MAX_BODY_BYTES`), can declare up to a
32-qubit circuit. `handle_one_request` deserializes it and calls
`moonlab_qgtl_execute`, which allocates a dense `2^num_qubits` state
vector: `2^30 * 16 B = 16 GB` for the reproducer, up to `2^32 * 16 B =
64 GB` at the deserializer's `num_qubits <= 32` ceiling. The public build
installs no admission hook, so nothing caps the qubit count between
parsing the verb header and running the circuit. One small request thus
drives an unbounded server-side allocation.

This is *not* a memory-safety violation -- no OOB access, no
use-after-free, no UB -- so it does not abort under ASan; it exhausts
memory. It is excluded from the replay PASS gate (quarantined) so the
gate can flip to required-pass the moment it is fixed.

**Suggested fix (owning lane)**: enforce a `num_qubits` / state-dimension
ceiling before `moonlab_qgtl_execute`. The natural seams are (a) a hard
cap in `handle_one_request` after `moonlab_qgtl_circuit_deserialize`
returns `num_qubits`, before `execute`, returning `ERR
MOONLAB_CONTROL_BAD_ARG`; or (b) plumbing the parsed `num_qubits` into the
admission-hook contract (currently it receives `-1` there) so overlays can
gate on it.

## 2. `strlen`-on-zero-length footgun -- circuit deserializer

- **Severity**: low (not reachable from untrusted input today).
- **Owning subsystem**: `src/applications/moonlab_qgtl_backend.c`
  (`moonlab_qgtl_circuit_deserialize`).

`moonlab_qgtl_circuit_deserialize(buf, buf_size, ...)` treats
`buf_size == 0` as "the buffer is NUL-terminated, `strlen` it". A caller
that passes `size == 0` with a non-terminated buffer reads off the end
(ASan heap-buffer-overflow in `strlen`). Every in-tree caller
NUL-terminates before calling (the control plane writes
`body[body_bytes] = '\0'`; `moonlab_qgtl_circuit_load` writes
`buf[got] = '\0'`) and never passes a bare `size == 0` with untrusted
content, so this is a latent contract footgun rather than a live bug. The
`circuit_deserialize_fuzz` harness NUL-terminates its input to mirror how
production reaches the function, so it fuzzes both the length-bounded and
the `strlen` paths safely.

**Suggested hardening (owning lane)**: reject `buf_size == 0` outright, or
scan with an explicit `memchr` bound instead of `strlen`, so the API is
not a footgun for a future binding that forwards a raw pointer + length.

## Clean surfaces

No memory-safety defects (heap-overflow, use-after-free, UB) were found on
the seed corpora for any surface:

| trace name                    | surface                                             |
|-------------------------------|-----------------------------------------------------|
| `control_plane_protocol_fuzz` | control-plane wire protocol (AUTH/CIRCUIT/SHOTS/...) |
| `circuit_deserialize_fuzz`    | moonlab-circuit v1 text deserializer                |
| `config_parse_fuzz`           | qsim config JSON + enum string decoders             |
| `mlkem_decode_fuzz`           | ML-KEM 512/768/1024 decaps/encaps/keygen + CTR_DRBG |
| `entropy_input_fuzz`          | entropy conditioner + SP 800-90B health tests       |
| `abi_boundary_fuzz`           | opaque-handle ABI (CA-MPS / QRNG / VQE gradient)    |

All six replay clean under ASan+UBSan; `fuzz_corpus_clean` (umbrella)
PASSes.

## Environment note (not a library finding)

On macOS/arm64, LeakSanitizer reports only one-time Objective-C / dyld /
libc initialization allocations (`load_images`, `initializeNonMetaClass`,
`libSystem_initializer`, `vsnprintf` float-format scratch) -- system
false positives, not leaks in quantumsim or the harnesses. `run_fuzz.sh`
therefore sets `detect_leaks=0` on Darwin and leaves it enabled elsewhere
(the Linux CI job is the leak oracle).
