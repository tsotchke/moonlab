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
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Internal state machine: makes destroy() safe against concurrent
 * push/pop.  Each operation registers itself in `in_flight` before
 * taking the lock and re-checks the state; destroy() publishes DEAD
 * and drains in_flight to zero before destroying the mutex.  An
 * operation that has not registered by the time DEAD is published sees
 * DEAD and bails without touching the mutex, and destroy() cannot
 * destroy the mutex while a registered operation still holds it. */
enum {
    MOONLAB_AUDIT_STATE_UNINIT = 0,
    MOONLAB_AUDIT_STATE_LIVE   = 1,
    MOONLAB_AUDIT_STATE_DEAD   = 2,
};

/**
 * @brief Opaque audit-buffer state.  Caller owns the struct
 *        storage and the ``slots`` block; init() does not malloc.
 */
typedef struct {
    /* Caller-owned storage. */
    uint8_t          *slots;        /* record_size * capacity bytes */
    size_t            record_size;  /* payload bytes per slot */
    size_t            capacity;     /* number of slots */

    /* Cursors + count + drops.  All under `lock`. */
    size_t            head;
    size_t            tail;
    size_t            count;
    uint64_t          drops;
    pthread_mutex_t   lock;

    /* MOONLAB_AUDIT_STATE_*.  Loaded atomically at the top of each
     * operation and re-checked after registering in-flight (below).
     * init() sets UNINIT->LIVE, destroy() sets LIVE->DEAD. */
    _Atomic int       state;

    /* In-flight push/pop/len/drops/reset operations that have passed
     * the LIVE gate and may hold or be about to take `lock`.  destroy()
     * publishes DEAD then drains this to zero BEFORE destroying the
     * mutex, so no operation can lock a destroyed mutex.  New callers
     * observe DEAD and never register.  See audit_buffer.c. */
    _Atomic unsigned  in_flight;
} moonlab_audit_buffer_t;

/**
 * @brief Initialise an audit buffer.
 *
 *        ``slots`` must point at at least
 *        ``record_size * capacity`` bytes the caller keeps alive
 *        for the lifetime of the buffer.  Init does not malloc.
 *
 *        ``buf`` MUST be zero-initialised the first time init() is
 *        called on it.  Stack-allocated structs use
 *        ``moonlab_audit_buffer_t b = {0};`` or memset() before
 *        passing in.  Failing to do so makes a re-init detect-and-
 *        destroy the (uninit'd) mutex -- undefined behavior.
 *
 *        Calling init() again on a live buffer (capacity != 0) is
 *        safe -- the previous mutex is destroyed first.
 *
 * @param[out] buf          Buffer to initialise.  Zero-init'd or
 *                          previously destroy()'d.
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
 * @brief Release the mutex.  Safe to call while other threads are
 *        still calling push/pop/len/drops/reset concurrently: destroy
 *        publishes the DEAD state, drains any operation that already
 *        passed the LIVE gate, and only then destroys the mutex.  A
 *        concurrent operation that has not yet registered sees DEAD and
 *        returns without touching the mutex.  After destroy returns the
 *        buffer is dead; further operations are safe no-ops.
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
