#ifndef QSIM_TIME_COMPAT_H
#define QSIM_TIME_COMPAT_H

/* #include_next is an intentional GCC/Clang extension; gcc 16 flags it under
 * -Wpedantic, which -Werror promotes to an error. Silence just that one
 * diagnostic (see src/compat/complex.h for the full rationale). */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include_next <time.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#if defined(_WIN32) && !defined(__cplusplus) && !defined(__MINGW32__)

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

/* Win32 Sleep, forward-declared so this header does not drag in <windows.h>
 * (with its min/max macros) into every time.h consumer. */
#ifndef _WINDOWS_
__declspec(dllimport) void __stdcall Sleep(unsigned long);
#endif

/* POSIX nanosleep backed by Sleep. The call sites that use it (a brief yield
 * in the audit-buffer concurrency test) tolerate millisecond granularity;
 * sub-millisecond requests round up to a 1 ms yield. */
static inline int nanosleep(const struct timespec *req, struct timespec *rem) {
    (void)rem;
    if (!req) return -1;
    unsigned long ms =
        (unsigned long)((unsigned long long)req->tv_sec * 1000ULL +
                        ((unsigned long long)req->tv_nsec + 999999ULL) / 1000000ULL);
    Sleep(ms);
    return 0;
}

#endif

#endif /* QSIM_TIME_COMPAT_H */
