/**
 * @file  audit_buffer.h
 * @brief Bounded lock-free audit ring buffer (since v1.0.3).
 *
 * Compliance / SOC2 use case: an overlay installs a completion
 * hook that calls @ref moonlab_audit_buffer_push for every
 * successful run.  A background thread drains the buffer to
 * durable storage (disk, S3, Kafka, ...) on its own cadence.
 * If the durable sink stalls, the ring absorbs up to ``capacity``
 * records before dropping the oldest -- losing the oldest record
 * is preferable to losing the newest, and the overlay can
 * monitor @ref moonlab_audit_buffer_drops to know it happened.
 *
 * Mechanism only -- record shape is caller-defined.  Each slot
 * is a fixed-size byte blob (caller picks ``record_size`` at
 * init).  The buffer keeps NO interpretation of the bytes.
 *
 * Lock-free: producer uses fetch_add on a write index, copy
 * payload into slot, fence to publish.  Consumer reads from
 * the read index, copies out, fence to retire.  Single-producer
 * + single-consumer is wait-free; multi-producer + single-
 * consumer is lock-free with bounded retry on the write fence.
 */

#ifndef MOONLAB_AUDIT_BUFFER_H
#define MOONLAB_AUDIT_BUFFER_H

#include "../applications/moonlab_api.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque audit-buffer state.  Caller owns the struct
 *        storage and the ``slots`` block; init() does not malloc.
 */
typedef struct {
    /* Caller-owned storage.  slots is record_size * capacity bytes
     * contiguous, slot N starts at slots[N * record_size]. */
    uint8_t          *slots;
    size_t            record_size;
    size_t            capacity;

    /* Single producer cursor + single consumer cursor.  ``write``
     * is the next slot to be written; ``read`` is the next slot
     * to be drained.  read == write means empty; read + capacity
     * == write means full (next push will drop the oldest).
     * Both cursors are monotonic; we modulo by capacity to find
     * the slot index. */
    _Atomic uint64_t  write;
    _Atomic uint64_t  read;

    /* How many pushes have dropped a slot because the consumer
     * fell behind.  Exposed via peek_drops; reset by reset_drops. */
    _Atomic uint64_t  drops;
} moonlab_audit_buffer_t;

/**
 * @brief Initialise an audit buffer.  ``slots`` must point at
 *        at least ``record_size * capacity`` bytes the caller
 *        keeps alive for the lifetime of the buffer.
 *
 * @param[out] buf          Buffer to initialise.
 * @param[in]  slots        Caller-owned storage block.
 * @param[in]  record_size  Bytes per record.  Must be > 0.
 * @param[in]  capacity     Number of records.  Must be > 0; a
 *                          power of two is recommended for cheap
 *                          modulo but not required.
 */
MOONLAB_API void
moonlab_audit_buffer_init(moonlab_audit_buffer_t *buf,
                          void                   *slots,
                          size_t                  record_size,
                          size_t                  capacity);

/**
 * @brief Push a record.  Copies ``record_size`` bytes from
 *        ``record`` into the next slot.  If the buffer is full,
 *        the OLDEST record is overwritten and ``drops`` is
 *        incremented.  Thread-safe; multiple producers may push
 *        concurrently.
 *
 *        Returns 1 if the record was stored without dropping
 *        anything, 0 if a drop occurred (record still stored).
 */
MOONLAB_API int
moonlab_audit_buffer_push(moonlab_audit_buffer_t *buf,
                          const void             *record);

/**
 * @brief Pop the oldest pending record into ``out``.  Returns 1
 *        on success, 0 if the buffer is empty.  Thread-safe for
 *        a single consumer; multi-consumer is NOT supported (the
 *        common case is one drain thread per buffer).
 */
MOONLAB_API int
moonlab_audit_buffer_pop(moonlab_audit_buffer_t *buf, void *out);

/**
 * @brief Read the number of records currently pending (write -
 *        read).  Thread-safe but the result is a snapshot --
 *        concurrent push/pop may change it before the caller
 *        acts on the number.
 */
MOONLAB_API size_t
moonlab_audit_buffer_len(const moonlab_audit_buffer_t *buf);

/**
 * @brief Read the cumulative drop count.  Operators expose this
 *        as a Prometheus counter so they know when the consumer
 *        is falling behind faster than the buffer can absorb.
 */
MOONLAB_API uint64_t
moonlab_audit_buffer_drops(const moonlab_audit_buffer_t *buf);

/**
 * @brief Reset the drop counter to zero.  Used after the overlay
 *        ships an "I noticed and acted on the drops" event so the
 *        counter only reflects FRESH backpressure.
 */
MOONLAB_API void
moonlab_audit_buffer_reset_drops(moonlab_audit_buffer_t *buf);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_AUDIT_BUFFER_H */
