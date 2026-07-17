# Moonlab concurrency lane -- ThreadSanitizer findings

Data-race / deadlock hunt across the heavily-threaded subsystems, using
ThreadSanitizer + targeted multi-threaded harnesses. All harnesses are
checked in under `tests/concurrency/`; `scripts/run_tsan.sh` builds and runs
them and emits `scripts/icc_traces/moonlab_tsan.jsonl`.

**Host used:** macOS arm64, Homebrew clang 21.1.7 (TSan available on-device).
**Library under test:** `libquantumsim.a` built with `-fsanitize=thread -g -O1`,
`QSIM_ENABLE_OPENMP` OFF (clean pthread signal) and ON (OpenMP regions), Metal
off, TLS off, WERROR off -- the documented sanitizer-safe configuration.

**Isolation technique.** The shared numeric core lazily initialises several
process-global singletons on first touch (see F1/F2). Those first-touch races
fire in *every* multi-threaded harness and would mask each subsystem's own
races, so each subsystem harness performs a single-threaded **warm-up** that
resolves the globals before the concurrent phase. The lazy-init races
themselves are reproduced in isolation by `conc_core_init` (nothing shared at
the harness level -> any race is inside the library).

## Summary

| # | Subsystem | File:line | Kind | Severity | Verdict | Harness |
|---|-----------|-----------|------|----------|---------|---------|
| F1 | Core config singleton | `config.c:127,130,148,136` + `state.c:68` | data race (lazy init) | High | TRUE | `conc_core_init` |
| F2 | SIMD dispatch vtable | `simd_ops.c:91,120,121` vs `:762` | data race (lazy init) | High | TRUE | `conc_core_init` |
| F3 | Control-plane server config fields | `control_plane.c:1296/1464`, `1297/1465`, `1262/1473`, `1250(1509)/1483` | data race | Medium | TRUE (misuse-adjacent; one contradicts docs) | `conc_control_plane adversarial` |
| F4 | Control-plane log-flag cache | `control_plane.c:74-78` via `:267` | data race (lazy init) | Low | TRUE but benign | `conc_control_plane` (pre-warm) |
| F5 | Audit ring buffer destroy | `audit_buffer.c:87/:76` (also `:116,:139,:153,:167`) | deadlock / use-of-destroyed-mutex | High | TRUE | `conc_audit_buffer destroy` |

**Verified race-free** (clean under TSan): entropy pool (steady + lifecycle
toggle), scheduler backend registry + completion hook, control-plane request
path (counters / token bucket / worker-registry reap-drain / metrics buffer /
tenant plumbing) once the core is warmed, Clifford + measurement RNG paths, and
the OpenMP numeric core (intra-gate threading, parallel-Grover entropy
isolation, scheduler shot fan-out) once un-annotated-libomp frames are
excluded. See "Clean results" below.

---

## F1 -- Unsynchronised lazy init of the global `qsim_config_t` (HIGH, true race)

**File:** `src/utils/config.c`

```
 49  static qsim_config_t* g_config = NULL;
 50  static int g_initialized = 0;
...
126  qsim_config_t* qsim_config_global(void) {
127      if (!g_initialized) {          // <-- unsynchronised READ of g_initialized
128          qsim_config_init();
130      return g_config;               // <-- unsynchronised READ of g_config
133  int qsim_config_init(void) {
134      if (g_initialized) return 0;
136      g_config = calloc(1, sizeof(qsim_config_t));   // <-- WRITE g_config
139      set_defaults(g_config);                        // <-- WRITEs into the struct
146      qsim_config_from_env(g_config);
148      g_initialized = 1;                             // <-- WRITE g_initialized (no barrier)
```

**Both accesses (TSan):**
- Read `g_initialized` `config.c:127` (thread A) vs write `config.c:148` (thread B).
- Read `g_config` `config.c:130` vs write `config.c:136`.
- Write the 248-byte struct `config.c:139` (`set_defaults` / `qsim_config_from_env`)
  vs read a field `state.c:68` (`cfg->algorithm.max_measurements` in
  `quantum_state_init`).

`g_initialized`/`g_config` are plain globals; no mutex, no `pthread_once`, no
atomics. When several threads first touch the simulator concurrently they race
on the flag, the pointer, and the struct body. Consequences: (1) two threads
both `calloc` `g_config` -> leak; (2) a thread can observe `g_initialized==1`
before the struct writes are visible (no release/acquire) -> torn read of
config fields; (3) `state->max_measurements` read from a half-initialised
struct.

**Why it matters:** this fires for ANY concurrent first use of the core --
the scheduler's own OpenMP shot fan-out (`moonlab_scheduler_run` with
`num_workers>1`), parallel Grover, and the control plane all reach it via
`quantum_state_init`. It is a real product bug, not a harness artifact.

**Repro:** `conc_core_init` -- N pthreads each build+execute their OWN circuit
from a cold process (nothing shared at the harness level).
```
./build-tsan-conc/conc_core_init      # TSAN_OPTIONS=halt_on_error=0
```
Reliably prints 4 distinct `SUMMARY: data race` sites in `qsim_config_global`
/ `qsim_config_init` / `state.c`.

**Fix direction (owner's call, not applied here):** guard init with
`pthread_once`, or make `g_initialized` an acquire/release atomic and publish
`g_config` only after the struct is fully written.

## F2 -- `volatile`-flag lazy init of the SIMD dispatch vtable (HIGH, true race)

**File:** `src/optimization/simd_ops.c`

```
 87  static simd_backend_vtable_t g_simd_vtable;   /* all fn pointers */
 88  static volatile int g_simd_vtable_ready = 0;
 90  static void simd_dispatch_init_once(void) {
 91      if (g_simd_vtable_ready) return;          // <-- READ flag
...
120      g_simd_vtable = vt;                       // <-- WRITE the whole vtable
121      g_simd_vtable_ready = 1;                  // <-- WRITE flag
...
761  void simd_complex_swap(...) {
762      simd_dispatch_init_once();
763      if (g_simd_vtable.complex_swap) { ... }   // <-- READ g_simd_vtable.<fn>
```

**Both accesses (TSan):** read `g_simd_vtable.complex_swap` at `simd_ops.c:762`
(reported as `:761`/`:770`/`:771` across builds) vs write `g_simd_vtable` at
`simd_ops.c:120` on another thread.

`volatile int` is NOT a synchronisation primitive in the C memory model -- it
provides neither atomicity nor happens-before. Concurrent first callers race
on both the flag and the vtable struct. Consequence: a thread can see
`g_simd_vtable_ready==1` while the vtable-struct store is not yet visible ->
call through a NULL / torn function pointer (crash) in the hot gate path
(`simd_complex_swap`, `simd_hadamard_pair`, `simd_normalize`, ... all use the
same `simd_dispatch_init_once`).

**Repro:** same `conc_core_init` run as F1 (the CNOT/measurement path reaches
`simd_complex_swap`).

**Fix direction:** `pthread_once`, or an acquire/release atomic flag with the
vtable published after its stores.

## F3 -- Control-plane server config read without synchronisation vs setters (MEDIUM, true race)

**File:** `src/control/control_plane.c`

The accept loop and per-connection worker read server config fields with no
lock while the public setters mutate them:

| Field | Read (accept loop / worker) | Write (setter) | Notes |
|-------|-----------------------------|----------------|-------|
| `admission_hook` | `:1296` | `:1464` (`set_admission_hook`) | **doc says "Thread-safe"** (header line ~428) -- contradiction |
| `admission_hook_ctx` | `:1297` | `:1465` | same setter |
| `max_concurrent` | `:1262` (also `:1264,:1272,:1295`) | `:1473` (`set_max_concurrent`) | plain int, no lock either side |
| `rate_rps` | `:1509` (via `rl_take_token`, call `:1250`) | `:1483` (`set_rate_limit`, under `rl_lock`) | the READ is lock-free; writer holds `rl_lock` -> still a race |
| `request_timeout_secs` | `:1294` | `:1451` (`set_request_timeout`) | same class (not separately surfaced this run) |

**Both accesses (TSan, adversarial run):**
- `admission_hook`: read `control_plane.c:1296` vs write `control_plane.c:1464`.
- `admission_hook_ctx`: read `:1297` vs write `:1465`.
- `max_concurrent`: read `:1262` vs write `:1473`.
- `rate_rps`: read (inlined from `rl_take_token`, `:1250`) vs write `:1483`
  (writer holds mutex `M0`=`rl_lock`; reader holds nothing).

The header/impl positions these as "set before `run()`", which makes most of
them documented-misuse when toggled mid-serve. **The exception is the
admission hook**, whose public doc comment explicitly promises the setter is
"Thread-safe" while the implementation performs a plain, unsynchronised
pointer + ctx store that races the acceptor's read -> a torn function-pointer /
ctx read is possible. That is a true bug: the advertised thread-safety is not
provided.

**Clean counterpart:** in steady mode (config set once before `run()`, the
supported usage) the control-plane request path is **race-free** -- see Clean
results. So F3 is specifically about mutating config while serving.

**Repro:**
```
./build-tsan-conc/conc_control_plane adversarial     # toggler mutates config mid-serve
```
Surfaces the four `SUMMARY: data race in moonlab_control_server_run` sites
above. `conc_control_plane steady` is clean.

**Fix direction:** publish `admission_hook`+`admission_hook_ctx` as an atomic
pair (or under a small lock / seqlock) since the header promises thread-safety;
document the other fields as set-before-run or guard them likewise.

## F4 -- Control-plane env-cached log flags (LOW, true race but benign)

**File:** `src/control/control_plane.c`

```
 73  static int log_enabled(void) {
 74      static int cached = -1;
 75      if (cached < 0) {
 76          const char *v = getenv("MOONLAB_CONTROL_LOG");
 77          cached = (v && *v && *v != '0') ? 1 : 0;   // racy lazy write
```
`log_format_json()` (lines 86-95) has the identical pattern. Read/write of the
function-local `static int cached` at `control_plane.c:74-77`, reached from
`handle_one_request` `control_plane.c:267` on every worker thread.

**Both accesses (TSan):** read `cached` (`control_plane.c:267` frame,
`log_enabled`) vs write `cached` (same) on two worker threads during the
first-touch window.

Benign in practice: the cached value is an idempotent, int-sized read of an
env var; every thread computes the identical value, and a torn 0/1 int read
does not change behaviour. Still a data race per the standard; folded into the
"pre-warm before the storm" so it does not mask F3. Reported for completeness.

## F5 -- Audit ring buffer: destroy() vs in-flight push/pop deadlocks (HIGH, true bug)

**File:** `src/utils/audit_buffer.c`

The buffer advertises destroy-race safety via an atomic `state` word:
> "No race window where a push tries to lock a destroyed mutex." (`audit_buffer.c` header comment)

That claim is **false**. `push` / `pop` / `len` / `drops` / `reset` do:
```
 86      if (atomic_load(&buf->state) != LIVE) return 0;   // pre-check
 87      pthread_mutex_lock(&buf->lock);                   // <-- can lock a DESTROYED mutex
```
while `destroy` does:
```
 73      pthread_mutex_lock(&buf->lock);
 74      atomic_store(&buf->state, DEAD);
 75      pthread_mutex_unlock(&buf->lock);
 76      pthread_mutex_destroy(&buf->lock);                // <-- destroys the mutex
```
A push that sampled `state==LIVE` at `:86` and then reaches `pthread_mutex_lock`
at `:87` **after** `destroy` ran `:76` locks a destroyed mutex -- undefined
behaviour. On macOS it wedges permanently: multiple pusher/popper threads
block forever in `_pthread_mutex_firstfit_lock_wait`. The lock return value is
also ignored, so on a platform where it returns `EINVAL` instead of blocking,
the thread would proceed into the critical section **without the lock** and
then unlock a destroyed mutex (unsynchronised access + UB).

**Both sites (confirmed via `sample` on the wedged process):**
- Blocked: `moonlab_audit_buffer_push` -> `pthread_mutex_lock` `audit_buffer.c:87`
  (also `moonlab_audit_buffer_drops` `:153`; same for `pop :116`, `len :139`,
  `reset :167`).
- Destroyer: `moonlab_audit_buffer_destroy` -> `pthread_mutex_destroy` `audit_buffer.c:76`.

**Impact:** the audit buffer is wired into the control-plane / scheduler
completion-hook (billing / SOC2 audit) path. A shutdown that destroys the ring
while a completion hook is mid-push hangs the pusher (and anything joining it).

**Repro:** `conc_audit_buffer destroy` -- pushers/poppers run, a destroyer
fires mid-flight; a built-in watchdog turns the resulting hang into a
deterministic `DEADLOCK: ...` message + `exit 7` (so CI does not stall). The
multi-producer/multi-consumer path WITHOUT concurrent destroy
(`conc_audit_buffer mpmc`) is clean.
```
./build-tsan-conc/conc_audit_buffer destroy   # prints DEADLOCK, exit 7
./build-tsan-conc/conc_audit_buffer mpmc      # clean
```

**Fix direction:** the state machine cannot make `pthread_mutex_destroy` safe
against a concurrent `pthread_mutex_lock`. Options: require quiescence before
destroy (the header's other sentence -- "caller must not push/pop afterward" --
already implies this, so drop the stronger "no race window" claim), or move to
a reference-counted / epoch scheme, or an RW/`try`-based drain that never
destroys the mutex while a lock is pending.

---

## Clean results (verified race-free under TSan)

These harnesses run their subsystem hard and report **zero** data races
(core lazy-init warmed first where relevant):

- **Control-plane request path** (`conc_control_plane steady`): many clients
  mixing HEALTH / METRICS / CIRCUIT / SHOTS / AUTH+tenant against one server
  with rate limit, `max_concurrent`, and an admission hook set BEFORE `run()`.
  The atomic metric counters, the per-IP token bucket under `rl_lock`, the
  bounded live-worker registry reap/drain path (v1.1.0), the admission +
  completion hook fan-out, the stack-local Prometheus metrics buffer, and the
  `__thread` tenant plumbing are all clean. (The steady-vs-adversarial split
  isolates F3 to mid-serve config mutation.)
- **Entropy pool** (`conc_entropy_pool steady` and `toggle`): background
  pre-generation pthread vs 8 consumers (mixed cache-hit/cache-miss sizes) +
  3 monitors, plus a lifecycle thread stopping/restarting the worker. The
  `pool_mutex` / `health_mutex` discipline holds; stats counters and ring
  cursors are clean. (Latent note: `stats.background_active` /
  `background_running` are written outside `pool_mutex` in start/stop; TSan did
  not surface it in the toggle run -- benign lifecycle flags, flagged here as a
  code observation only.)
- **Scheduler registry + completion hook** (`conc_scheduler`): 8 threads run
  jobs concurrently while a churn thread register/unregister a backend and
  toggles the completion hook. `g_backend_lock` and the atomic
  `g_count_completion_hook_fires` are clean.
- **Clifford + measurement RNG** (`conc_clifford_measurement`): 8+8 threads,
  each with its OWN tableau/state and OWN `rng_state`. No hidden global RNG --
  clean, confirming these paths carry randomness by caller-owned argument only.
- **OpenMP numeric core** (`conc_grover_gates`, OpenMP-ON library): the
  intra-gate block loop on an 18-qubit state (crosses `QS_BLOCK_THRESHOLD_DIM`
  = 2^18), `grover_parallel_random_batch` (VERIFIES the audit's aliased-
  `user_data` fix -- each worker gets its own splitmix stream, no race on the
  entropy contexts), and the scheduler shot fan-out (disjoint outcome slices).
  Raw TSan reports ~13-34 races, but **every one runs on libomp worker threads**
  (`__kmp_invoke_microtask`, created by `__kmp_create_worker`) -- the classic
  un-annotated-libomp false positive where TSan cannot see the OpenMP barrier's
  happens-before between successive parallel regions. All vanish under
  `tests/concurrency/tsan.supp` (`called_from_lib:libomp.dylib`); none has both
  accesses in moonlab code. Definitive on-device confirmation needs an
  Archer-instrumented libomp -- provided by the `openmp-archer` job in
  `.github/workflows/tsan.yml`.

## What could not be tested here

- **OpenMP intra-region certainty on this host:** the Homebrew libomp is not
  Archer-instrumented, so the OpenMP verdict relies on library-scoped
  suppression + the reasoning that successive parallel regions are
  barrier-separated. The Linux `openmp-archer` CI job removes that caveat.
- **TLS / mTLS control-plane transport:** built with `QSIM_ENABLE_TLS=OFF`
  (no OpenSSL dependency in the sanitizer build), so the `SSL_*` handshake
  paths and `g_count_tls_failed` were not exercised.
- **MPI scheduler fan-out** (`moonlab_scheduler_run_mpi`): compiled out
  (`HAS_MPI` unset); only the in-process OpenMP fan-out was tested.
- **GPU backends** (Metal/CUDA): off in the sanitizer build.

## Helgrind fallback

Where TSan is unavailable (e.g. a GCC-only host, or a platform without the
`clang_rt.tsan` runtime), Valgrind's **Helgrind** covers the same pthread
findings F1, F3, F4, F5 (it does not need instrumentation and detects
lock-order / destroyed-mutex / unsynchronised-access errors). Build the
harnesses normally (no `-fsanitize=thread`) against a plain `libquantumsim.a`
and run, e.g.:

```
valgrind --tool=helgrind --history-level=full ./conc_core_init
valgrind --tool=helgrind ./conc_audit_buffer destroy      # flags the destroyed-mutex lock
valgrind --tool=helgrind ./conc_control_plane adversarial
```

Helgrind will report `F5` as "pthread_mutex_lock: mutex is invalid" /
"destroyed mutex" rather than a hang, and `F1`/`F2` as unsynchronised
read/write of the global. It is slower than TSan and noisier on the OpenMP
paths (same libomp issue -- pair with `--fair-sched=yes` and a libomp
suppression). TSan remains the primary tool; Helgrind is the portable
fallback.

## Running the lane

```
./scripts/run_tsan.sh          # builds both TSan libs + harnesses, runs all, emits JSONL
# or, against prebuilt libs, via the self-contained CMake project:
cmake -S tests/concurrency -B build-tsan-conc \
      -DMOONLAB_TSAN_LIB=$PWD/build-tsan/libquantumsim.a \
      -DMOONLAB_TSAN_LIB_OMP=$PWD/build-tsan-omp/libquantumsim.a
cmake --build build-tsan-conc -j2
ctest --test-dir build-tsan-conc --output-on-failure
```

Trace: `scripts/icc_traces/moonlab_tsan.jsonl` (`kind:"moonlab_tsan"`,
umbrella event `name:"tsan_clean"` with `value` PASS/FAIL + `races` count).
