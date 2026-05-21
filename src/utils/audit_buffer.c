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
 */

#include "audit_buffer.h"

#include <string.h>

void moonlab_audit_buffer_init(moonlab_audit_buffer_t *buf,
                               void                   *slots,
                               size_t                  record_size,
                               size_t                  capacity)
{
    if (!buf) return;
    /* If this is a re-init on a live buffer, destroy the existing
     * mutex first.  Without this, the second pthread_mutex_init
     * leaks the previous mutex's kernel state.  capacity == 0 means
     * the buffer is either fresh-allocated (zero-initialised by the
     * caller) or was previously destroy()'d -- either way, no
     * mutex to clean up. */
    if (buf->capacity != 0) {
        pthread_mutex_destroy(&buf->lock);
    }
    if (!slots || record_size == 0 || capacity == 0) {
        memset(buf, 0, sizeof(*buf));
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
}

void moonlab_audit_buffer_destroy(moonlab_audit_buffer_t *buf)
{
    if (!buf || buf->capacity == 0) return;
    pthread_mutex_destroy(&buf->lock);
    /* Mark unusable so future calls no-op. */
    buf->capacity = 0;
}

int moonlab_audit_buffer_push(moonlab_audit_buffer_t *buf,
                              const void             *record)
{
    if (!buf || !record || buf->capacity == 0) return 0;
    pthread_mutex_lock(&buf->lock);

    int dropped = 0;
    if (buf->count == buf->capacity) {
        /* Buffer full -- overwrite oldest.  tail advances; count
         * stays at capacity. */
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
    if (!buf || !out || buf->capacity == 0) return 0;
    pthread_mutex_lock(&buf->lock);

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
    if (!buf || buf->capacity == 0) return 0;
    pthread_mutex_lock(&buf->lock);
    const size_t n = buf->count;
    pthread_mutex_unlock(&buf->lock);
    return n;
}

uint64_t moonlab_audit_buffer_drops(moonlab_audit_buffer_t *buf)
{
    if (!buf || buf->capacity == 0) return 0;
    pthread_mutex_lock(&buf->lock);
    const uint64_t d = buf->drops;
    pthread_mutex_unlock(&buf->lock);
    return d;
}

void moonlab_audit_buffer_reset_drops(moonlab_audit_buffer_t *buf)
{
    if (!buf || buf->capacity == 0) return;
    pthread_mutex_lock(&buf->lock);
    buf->drops = 0;
    pthread_mutex_unlock(&buf->lock);
}
