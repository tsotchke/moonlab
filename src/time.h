#ifndef QSIM_TIME_COMPAT_H
#define QSIM_TIME_COMPAT_H

#include_next <time.h>

#if defined(_WIN32) && !defined(__cplusplus)

#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

static inline int clock_gettime(int clock_id, struct timespec *ts) {
    (void)clock_id;
    return timespec_get(ts, TIME_UTC) == TIME_UTC ? 0 : -1;
}

#endif

#endif /* QSIM_TIME_COMPAT_H */
