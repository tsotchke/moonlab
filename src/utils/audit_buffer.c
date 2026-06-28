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
 * Destroy-race handling: push/pop both load the atomic ``state``
 * before taking the lock, and re-check after acquiring it.
 * destroy() flips state to DEAD UNDER the lock and only then
 * destroys the mutex.  This means:
 *   - thread A enters push, sees state=LIVE, grabs the lock
 *   - thread B calls destroy, can't grab the lock yet
 *   - thread A finishes, releases the lock
 *   - thread B grabs the lock, sets state=DEAD, releases, destroys
 *   - thread C enters push, sees state=DEAD, returns 0
 * No race window where a push tries to lock a destroyed mutex.
 */

#include "audit_buffer.h"

#include <string.h>

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
    /* Lock to drain any in-flight push/pop, flip state to DEAD
     * UNDER the lock so a subsequent push/pop sees the right value
     * even if they raced state-load + lock-acquire.  Then unlock
     * and destroy. */
    pthread_mutex_lock(&buf->lock);
    atomic_store(&buf->state, MOONLAB_AUDIT_STATE_DEAD);
    pthread_mutex_unlock(&buf->lock);
    pthread_mutex_destroy(&buf->lock);
}

int moonlab_audit_buffer_push(moonlab_audit_buffer_t *buf,
                              const void             *record)
{
    if (!buf || !record) return 0;
    /* Fast bail without touching the mutex.  Avoids the lock cost
     * for callers that push to a never-init'd or destroyed buffer
     * (e.g. an admission overlay that hasn't built one yet). */
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) return 0;
    pthread_mutex_lock(&buf->lock);
    /* Re-check under the lock: destroy() flips state under the
     * same lock, so a state value sampled before the lock could
     * be stale.  See "destroy-race handling" in the header. */
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) {
        pthread_mutex_unlock(&buf->lock);
        return 0;
    }

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
    return dropped ? 0 : 1;
}

int moonlab_audit_buffer_pop(moonlab_audit_buffer_t *buf, void *out)
{
    if (!buf || !out) return 0;
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) return 0;
    pthread_mutex_lock(&buf->lock);
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) {
        pthread_mutex_unlock(&buf->lock);
        return 0;
    }

    if (buf->count == 0) {
        pthread_mutex_unlock(&buf->lock);
        return 0;
    }
    const uint8_t *src = buf->slots + buf->tail * buf->record_size;
    memcpy(out, src, buf->record_size);
    buf->tail   = (buf->tail + 1) % buf->capacity;
    buf->count -= 1;

    pthread_mutex_unlock(&buf->lock);
    return 1;
}

size_t moonlab_audit_buffer_len(moonlab_audit_buffer_t *buf)
{
    if (!buf) return 0;
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) return 0;
    pthread_mutex_lock(&buf->lock);
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) {
        pthread_mutex_unlock(&buf->lock);
        return 0;
    }
    const size_t n = buf->count;
    pthread_mutex_unlock(&buf->lock);
    return n;
}

uint64_t moonlab_audit_buffer_drops(moonlab_audit_buffer_t *buf)
{
    if (!buf) return 0;
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) return 0;
    pthread_mutex_lock(&buf->lock);
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) {
        pthread_mutex_unlock(&buf->lock);
        return 0;
    }
    const uint64_t d = buf->drops;
    pthread_mutex_unlock(&buf->lock);
    return d;
}

void moonlab_audit_buffer_reset_drops(moonlab_audit_buffer_t *buf)
{
    if (!buf) return;
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) return;
    pthread_mutex_lock(&buf->lock);
    if (atomic_load(&buf->state) != MOONLAB_AUDIT_STATE_LIVE) {
        pthread_mutex_unlock(&buf->lock);
        return;
    }
    buf->drops = 0;
    pthread_mutex_unlock(&buf->lock);
}
