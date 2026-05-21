/**
 * @file  audit_buffer.h
 * @brief Bounded mutex-guarded audit ring buffer (since v1.0.3).
 *
 * Compliance / SOC2 use case: an overlay installs a completion
 * hook that calls @ref moonlab_audit_buffer_push for every
 * successful run.  A background thread drains the buffer to
 * durable storage (disk, S3, Kafka, ...) on its own cadence.
 * If the durable sink stalls, the ring absorbs up to ``capacity``
 * records before dropping the OLDEST -- losing the oldest record
 * is preferable to losing the newest, and the overlay can
 * monitor @ref moonlab_audit_buffer_drops to know it happened.
 *
 * Mechanism only -- record shape is caller-defined.  Each slot
 * is a fixed-size byte blob (caller picks ``record_size`` at
 * init).  The buffer keeps NO interpretation of the bytes.
 *
 * Concurrency model: multi-producer / multi-consumer with a
 * single mutex.  Push and pop both acquire the lock briefly to
 * copy payload and advance the cursors.  Audit records fire at
 * modest rates (one per scheduler completion -- order of
 * hundreds per second per replica at peak load), so lock
 * contention is negligible compared to the durable-sink
 * latency the buffer exists to absorb.  Lock-free was tried
 * (Vyukov MPSC) and discarded as overengineered for this rate.
 */

#ifndef MOONLAB_AUDIT_BUFFER_H
#define MOONLAB_AUDIT_BUFFER_H

#include "../applications/moonlab_api.h"

#include <pthread.h>
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
    /* Caller-owned storage. */
    uint8_t          *slots;        /* record_size * capacity bytes */
    size_t            record_size;  /* payload bytes per slot */
    size_t            capacity;     /* number of slots */

    /* head = next slot to write; tail = next slot to read.
     * head - tail (mod 2*capacity) gives the count of pending
     * records, with the special case head == tail meaning empty
     * (NOT full).  We track `count` explicitly to avoid the
     * ambiguity.  All three under `lock`. */
    size_t            head;
    size_t            tail;
    size_t            count;        /* pending records */
    uint64_t          drops;        /* cumulative oldest-dropped count */
    pthread_mutex_t   lock;
} moonlab_audit_buffer_t;

/**
 * @brief Initialise an audit buffer.
 *
 *        ``slots`` must point at at least
 *        ``record_size * capacity`` bytes the caller keeps alive
 *        for the lifetime of the buffer.  Init does not malloc.
 *
 * @param[out] buf          Buffer to initialise.
 * @param[in]  slots        Caller-owned storage block.
 * @param[in]  record_size  Bytes per record.  Must be > 0.
 * @param[in]  capacity     Number of records.  Must be > 0.
 */
MOONLAB_API void
moonlab_audit_buffer_init(moonlab_audit_buffer_t *buf,
                          void                   *slots,
                          size_t                  record_size,
                          size_t                  capacity);

/**
 * @brief Release the mutex; caller must not push/pop afterward.
 *        Safe to call on a zero-initialised (never-init'd) buffer.
 */
MOONLAB_API void
moonlab_audit_buffer_destroy(moonlab_audit_buffer_t *buf);

/**
 * @brief Push a record.  Copies ``record_size`` bytes from
 *        ``record`` into the next slot.  If the buffer is full,
 *        the OLDEST record is overwritten and ``drops`` is
 *        incremented.  Thread-safe.
 *
 *        Returns 1 if the record was stored without dropping
 *        anything, 0 if a drop occurred (record still stored).
 *        Returns 0 also if buf is unusable (init failed).
 */
MOONLAB_API int
moonlab_audit_buffer_push(moonlab_audit_buffer_t *buf,
                          const void             *record);

/**
 * @brief Pop the oldest pending record into ``out``.  Returns 1
 *        on success, 0 if the buffer is empty.  Thread-safe.
 */
MOONLAB_API int
moonlab_audit_buffer_pop(moonlab_audit_buffer_t *buf, void *out);

/**
 * @brief Number of records currently pending.  Snapshot value;
 *        a concurrent push/pop may change it immediately after.
 */
MOONLAB_API size_t
moonlab_audit_buffer_len(moonlab_audit_buffer_t *buf);

/**
 * @brief Cumulative drop count.  Operators expose this as a
 *        Prometheus counter so they know the consumer is falling
 *        behind faster than the buffer can absorb.
 */
MOONLAB_API uint64_t
moonlab_audit_buffer_drops(moonlab_audit_buffer_t *buf);

/**
 * @brief Reset the drop counter to zero.
 */
MOONLAB_API void
moonlab_audit_buffer_reset_drops(moonlab_audit_buffer_t *buf);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_AUDIT_BUFFER_H */
