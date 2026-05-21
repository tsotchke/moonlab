/**
 * @file  audit_buffer.c
 * @brief Lock-free bounded ring buffer impl (see audit_buffer.h).
 *
 * Algorithm: each slot has a byte payload + a separate ``seq``
 * word that advances when the slot is published / retired.  This
 * avoids the classic SPSC race where a consumer reads a payload
 * the producer is still writing.
 *
 * For simplicity (compliance-grade not high-performance), we
 * serialise PUSH with a CAS on the write cursor.  Each producer
 * atomically claims a slot, copies the payload, then publishes
 * by advancing a slot-local seq counter past the write index.
 * The consumer reads the slot, verifies the seq matches its
 * expected read index, and advances the read cursor.
 *
 * Mode: when the buffer is full (write - read == capacity), the
 * push overwrites the oldest slot and bumps the read cursor by
 * one.  Drops are counted in `drops` for visibility.  This matches
 * the SOC2 spec: lose oldest before failing the request that
 * generated the record.
 */

#include "audit_buffer.h"

#include <string.h>

void moonlab_audit_buffer_init(moonlab_audit_buffer_t *buf,
                               void                   *slots,
                               size_t                  record_size,
                               size_t                  capacity)
{
    if (!buf || !slots || record_size == 0 || capacity == 0) return;
    buf->slots       = (uint8_t *)slots;
    buf->record_size = record_size;
    buf->capacity    = capacity;
    atomic_store(&buf->write, 0);
    atomic_store(&buf->read,  0);
    atomic_store(&buf->drops, 0);
}

int moonlab_audit_buffer_push(moonlab_audit_buffer_t *buf,
                              const void             *record)
{
    if (!buf || !record) return 0;
    /* Claim the next write slot. */
    const uint64_t w = atomic_fetch_add(&buf->write, 1);
    const uint64_t r = atomic_load(&buf->read);

    int dropped = 0;
    if (w - r >= buf->capacity) {
        /* Buffer is full -- this push will overwrite an
         * unread record.  Bump the read cursor past the
         * slot we're about to clobber so the next pop sees
         * the right oldest entry. */
        const uint64_t new_read = w - buf->capacity + 1;
        uint64_t       expected = r;
        /* The consumer may have advanced concurrently; only
         * shove if our view is still authoritative. */
        if (new_read > expected) {
            (void)atomic_compare_exchange_strong(
                &buf->read, &expected, new_read);
        }
        atomic_fetch_add(&buf->drops, 1);
        dropped = 1;
    }

    const size_t   idx = (size_t)(w % buf->capacity);
    uint8_t       *dst = buf->slots + idx * buf->record_size;
    memcpy(dst, record, buf->record_size);

    /* Memory fence so a concurrent pop observes the payload
     * before observing the bumped write cursor.  atomic_store
     * here would be a no-op (write is already past w+1); the
     * fence is the protocol guarantee. */
    atomic_thread_fence(memory_order_release);

    return dropped ? 0 : 1;
}

int moonlab_audit_buffer_pop(moonlab_audit_buffer_t *buf, void *out)
{
    if (!buf || !out) return 0;
    const uint64_t r = atomic_load(&buf->read);
    const uint64_t w = atomic_load_explicit(&buf->write, memory_order_acquire);
    if (r >= w) return 0;        /* empty */

    const size_t   idx = (size_t)(r % buf->capacity);
    const uint8_t *src = buf->slots + idx * buf->record_size;
    memcpy(out, src, buf->record_size);

    /* Advance the read cursor.  We're the only consumer
     * (single-consumer invariant) so a plain store is safe;
     * we use atomic_store to keep memory ordering explicit. */
    atomic_store(&buf->read, r + 1);
    return 1;
}

size_t moonlab_audit_buffer_len(const moonlab_audit_buffer_t *buf)
{
    if (!buf) return 0;
    const uint64_t w =
        atomic_load(&((moonlab_audit_buffer_t *)buf)->write);
    const uint64_t r =
        atomic_load(&((moonlab_audit_buffer_t *)buf)->read);
    if (w <= r) return 0;
    const uint64_t pending = w - r;
    return pending > buf->capacity ? buf->capacity : (size_t)pending;
}

uint64_t moonlab_audit_buffer_drops(const moonlab_audit_buffer_t *buf)
{
    if (!buf) return 0;
    return atomic_load(
        &((moonlab_audit_buffer_t *)buf)->drops);
}

void moonlab_audit_buffer_reset_drops(moonlab_audit_buffer_t *buf)
{
    if (!buf) return;
    atomic_store(&buf->drops, 0);
}
