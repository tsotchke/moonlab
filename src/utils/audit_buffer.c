/**
 * @file  audit_buffer.c
 * @brief Mutex-guarded bounded ring buffer impl (see audit_buffer.h).
 *
 * Cursors and count protected by a single mutex.  Push/pop hold
 * the lock long enough to memcpy the payload (record_size bytes,
 * typically ~64 bytes).  At a few-hundred-per-second push rate
 * the lock contention is invisible compared to the durable-sink
 * latency the buffer is sized to absorb.
 *
 * Overflow policy: drop OLDEST.  When push() finds count ==
 * capacity, it overwrites tail and advances tail; drops counter
 * increments.  Newest records are preserved because they're the
 * most actionable for SREs and the most expensive to recompute.
 *
 * Destroy-race handling: an unsynchronised ``pthread_mutex_destroy``
 * can never be made safe against a concurrent ``pthread_mutex_lock`` by
 * a state flag alone -- a pusher that samples state==LIVE and is then
 * preempted before ``pthread_mutex_lock`` can lock a mutex destroy()
 * already destroyed (UB; on macOS it wedges forever).  Instead every
 * operation registers itself in the ``in_flight`` counter and re-checks
 * the state before locking; destroy() publishes DEAD, then spins until
 * ``in_flight`` drains to zero, and only then destroys the mutex:
 *   - thread A enters push, registers in_flight, sees state==LIVE, locks
 *   - thread B calls destroy, publishes DEAD, waits for in_flight==0
 *   - thread A finishes, unlocks, decrements in_flight
 *   - thread B observes in_flight==0 and destroys the mutex
 *   - thread C enters push, sees state==DEAD, never registers, returns 0
 *   - a pusher preempted after the state pre-check but before it
 *     registers re-checks state==DEAD after registering and bails
 *     WITHOUT locking, so it never touches the (possibly destroyed) mutex
 * The counter uses default (sequentially-consistent) atomics, which give
 * the single total order the drain argument relies on.
 */

#include "audit_buffer.h"

#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
/* SwitchToThread() is declared in <windows.h>. Include it explicitly rather
 * than relying on transitive inclusion via the pthread compat shim: MinGW's
 * winpthreads does not always pull it in (observed undeclared under UCRT64
 * gcc 16), which -Werror turns into a build failure. */
#include <windows.h>
#else
#include <sched.h>
#endif

/* Yield the CPU during the destroy drain so a registered-but-descheduled
 * operation can run to completion instead of being starved by a busy spin. */
static inline void audit_yield(void)
{
#if defined(_WIN32) || defined(_WIN64)
    SwitchToThread();
#else
    sched_yield();
#endif
}

/* Register an in-flight operation and confirm the buffer is still LIVE.
 * Returns 1 if the caller may proceed to take the lock (and MUST pair the
 * success with audit_leave()); returns 0 if the buffer is not LIVE, in which
 * case no in_flight reference is held and the caller must NOT touch the lock.
 *
 * The fetch_add is ordered BEFORE the gating state load: any operation that
 * observes LIVE here has already incremented in_flight, so a concurrent
 * destroy() that has published DEAD is forced to wait for this operation in
 * its drain loop -- it cannot destroy the mutex underneath us. */
static int audit_enter(moonlab_audit_buffer_t *buf)
{
    /* Cheap pre-check: skip the RMW entirely for a never-init'd or already
     * dead buffer (e.g. an overlay pushing before it built one). */
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) return 0;
    atomic_fetch_add(&buf->in_flight, 1u);
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) {
        atomic_fetch_sub(&buf->in_flight, 1u);
        return 0;
    }
    return 1;
}

static void audit_leave(moonlab_audit_buffer_t *buf)
{
    atomic_fetch_sub(&buf->in_flight, 1u);
}

void moonlab_audit_buffer_init(moonlab_audit_buffer_t *buf,
                               void                   *slots,
                               size_t                  record_size,
                               size_t                  capacity)
{
    if (!buf) return;
    /* Re-init on a live buffer: destroy the previous mutex first.
     * State machine treats a DEAD buffer as a candidate for re-init
     * too (caller called destroy() and now wants to reuse). */
    const int prev_state = atomic_load(&buf->state);
    if (prev_state == MOONLAB_AUDIT_STATE_LIVE) {
        pthread_mutex_destroy(&buf->lock);
    }
    if (!slots || record_size == 0 || capacity == 0) {
        memset(buf, 0, sizeof(*buf));
        atomic_store(&buf->state, MOONLAB_AUDIT_STATE_UNINIT);
        return;
    }
    buf->slots       = (uint8_t *)slots;
    buf->record_size = record_size;
    buf->capacity    = capacity;
    buf->head        = 0;
    buf->tail        = 0;
    buf->count       = 0;
    buf->drops       = 0;
    atomic_store(&buf->in_flight, 0u);
    pthread_mutex_init(&buf->lock, NULL);
    atomic_store(&buf->state, MOONLAB_AUDIT_STATE_LIVE);
}

void moonlab_audit_buffer_destroy(moonlab_audit_buffer_t *buf)
{
    if (!buf) return;
    const int prev_state = atomic_load(&buf->state);
    if (prev_state != MOONLAB_AUDIT_STATE_LIVE) {
        /* Never-init or already destroyed -- no mutex to clean. */
        return;
    }
    /* Publish DEAD first: after this store every new operation sees DEAD
     * at its pre-check and never registers in in_flight or touches the
     * lock.  Then wait for any operation that already registered (and may
     * hold or be about to take the lock) to drain to zero.  Only then is
     * it safe to destroy the mutex -- no lock can be pending against it.
     * The in_flight set is bounded (only ops that registered before the
     * DEAD publish contribute), so this drains promptly. */
    atomic_store(&buf->state, MOONLAB_AUDIT_STATE_DEAD);
    while (atomic_load(&buf->in_flight) != 0u) {
        audit_yield();
    }
    pthread_mutex_destroy(&buf->lock);
}

int moonlab_audit_buffer_push(moonlab_audit_buffer_t *buf,
                              const void             *record)
{
    if (!buf || !record) return 0;
    /* Register in-flight + confirm LIVE before touching the mutex.  If the
     * buffer is not LIVE (never-init'd or being destroyed) we bail without
     * locking; see "destroy-race handling" in the header. */
    if (!audit_enter(buf)) return 0;
    pthread_mutex_lock(&buf->lock);

    int dropped = 0;
    if (buf->count == buf->capacity) {
        buf->tail = (buf->tail + 1) % buf->capacity;
        buf->drops += 1;
        dropped = 1;
    } else {
        buf->count += 1;
    }
    uint8_t *dst = buf->slots + buf->head * buf->record_size;
    memcpy(dst, record, buf->record_size);
    buf->head = (buf->head + 1) % buf->capacity;

    pthread_mutex_unlock(&buf->lock);
    audit_leave(buf);
    return dropped ? 0 : 1;
}

int moonlab_audit_buffer_pop(moonlab_audit_buffer_t *buf, void *out)
{
    if (!buf || !out) return 0;
    if (!audit_enter(buf)) return 0;
    pthread_mutex_lock(&buf->lock);

    if (buf->count == 0) {
        pthread_mutex_unlock(&buf->lock);
        audit_leave(buf);
        return 0;
    }
    const uint8_t *src = buf->slots + buf->tail * buf->record_size;
    memcpy(out, src, buf->record_size);
    buf->tail   = (buf->tail + 1) % buf->capacity;
    buf->count -= 1;

    pthread_mutex_unlock(&buf->lock);
    audit_leave(buf);
    return 1;
}

size_t moonlab_audit_buffer_len(moonlab_audit_buffer_t *buf)
{
    if (!buf) return 0;
    if (!audit_enter(buf)) return 0;
    pthread_mutex_lock(&buf->lock);
    const size_t n = buf->count;
    pthread_mutex_unlock(&buf->lock);
    audit_leave(buf);
    return n;
}

uint64_t moonlab_audit_buffer_drops(moonlab_audit_buffer_t *buf)
{
    if (!buf) return 0;
    if (!audit_enter(buf)) return 0;
    pthread_mutex_lock(&buf->lock);
    const uint64_t d = buf->drops;
    pthread_mutex_unlock(&buf->lock);
    audit_leave(buf);
    return d;
}

void moonlab_audit_buffer_reset_drops(moonlab_audit_buffer_t *buf)
{
    if (!buf) return;
    if (!audit_enter(buf)) return;
    pthread_mutex_lock(&buf->lock);
    buf->drops = 0;
    pthread_mutex_unlock(&buf->lock);
    audit_leave(buf);
}
