#ifndef QSIM_UNISTD_COMPAT_H
#define QSIM_UNISTD_COMPAT_H

#if defined(_WIN32) || defined(_WIN64)

#include <io.h>
#include <process.h>
#include <stddef.h>
#include <windows.h>

#define close _close
#define read _read
#define write _write
#define getpid _getpid
#define access _access

#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 1
#endif
#ifndef _SC_PHYS_PAGES
#define _SC_PHYS_PAGES 2
#endif
#ifndef _SC_PAGE_SIZE
#define _SC_PAGE_SIZE 3
#endif
#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE _SC_PAGE_SIZE
#endif

static inline int usleep(unsigned int usec) {
    Sleep((usec + 999U) / 1000U);
    return 0;
}

static inline unsigned int sleep(unsigned int seconds) {
    Sleep(seconds * 1000U);
    return 0;
}

static inline long sysconf(int name) {
    SYSTEM_INFO si;
    MEMORYSTATUSEX mem;
    switch (name) {
        case _SC_NPROCESSORS_ONLN:
            GetSystemInfo(&si);
            return (long)si.dwNumberOfProcessors;
        case _SC_PAGE_SIZE:
            GetSystemInfo(&si);
            return (long)si.dwPageSize;
        case _SC_PHYS_PAGES:
            mem.dwLength = sizeof(mem);
            if (GlobalMemoryStatusEx(&mem)) {
                GetSystemInfo(&si);
                return (long)(mem.ullTotalPhys / si.dwPageSize);
            }
            return -1;
        default:
            return -1;
    }
}

#else

#include_next <unistd.h>

#endif

#endif /* QSIM_UNISTD_COMPAT_H */
