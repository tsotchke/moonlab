#ifndef QSIM_PTHREAD_COMPAT_H
#define QSIM_PTHREAD_COMPAT_H

#if defined(_WIN32) || defined(_WIN64)

#include <stdlib.h>
#include <windows.h>

typedef struct {
    INIT_ONCE once;
    CRITICAL_SECTION cs;
} pthread_mutex_t;

typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;
typedef INIT_ONCE pthread_once_t;

#define PTHREAD_MUTEX_INITIALIZER { INIT_ONCE_STATIC_INIT, {0} }
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

static BOOL CALLBACK qsim_pthread_mutex_init_once(
    PINIT_ONCE once,
    PVOID parameter,
    PVOID *context
) {
    (void)once;
    (void)context;
    InitializeCriticalSection((CRITICAL_SECTION*)parameter);
    return TRUE;
}

static inline int qsim_pthread_mutex_ensure(pthread_mutex_t *mutex) {
    return InitOnceExecuteOnce(
        &mutex->once,
        qsim_pthread_mutex_init_once,
        &mutex->cs,
        NULL) ? 0 : 1;
}

static inline int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr) {
    (void)attr;
    if (!mutex) return 1;
    mutex->once = (INIT_ONCE)INIT_ONCE_STATIC_INIT;
    return qsim_pthread_mutex_ensure(mutex);
}

static inline int pthread_mutex_destroy(pthread_mutex_t *mutex) {
    if (!mutex) return 1;
    if (qsim_pthread_mutex_ensure(mutex) != 0) return 1;
    DeleteCriticalSection(&mutex->cs);
    return 0;
}

static inline int pthread_mutex_lock(pthread_mutex_t *mutex) {
    if (!mutex) return 1;
    if (qsim_pthread_mutex_ensure(mutex) != 0) return 1;
    EnterCriticalSection(&mutex->cs);
    return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    if (!mutex) return 1;
    LeaveCriticalSection(&mutex->cs);
    return 0;
}

static inline int pthread_cond_init(pthread_cond_t *cond, const void *attr) {
    (void)attr;
    if (!cond) return 1;
    InitializeConditionVariable(cond);
    return 0;
}

static inline int pthread_cond_destroy(pthread_cond_t *cond) {
    (void)cond;
    return 0;
}

static inline int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    if (!cond || !mutex) return 1;
    if (qsim_pthread_mutex_ensure(mutex) != 0) return 1;
    return SleepConditionVariableCS(cond, &mutex->cs, INFINITE) ? 0 : 1;
}

static inline int pthread_cond_signal(pthread_cond_t *cond) {
    if (!cond) return 1;
    WakeConditionVariable(cond);
    return 0;
}

static inline int pthread_cond_broadcast(pthread_cond_t *cond) {
    if (!cond) return 1;
    WakeAllConditionVariable(cond);
    return 0;
}

typedef struct {
    void *(*start)(void*);
    void *arg;
} qsim_pthread_start_t;

static DWORD WINAPI qsim_pthread_entry(LPVOID parameter) {
    qsim_pthread_start_t *start = (qsim_pthread_start_t*)parameter;
    void *(*fn)(void*) = start->start;
    void *arg = start->arg;
    free(start);
    (void)fn(arg);
    return 0;
}

static inline int pthread_create(
    pthread_t *thread,
    const void *attr,
    void *(*start_routine)(void*),
    void *arg
) {
    (void)attr;
    if (!thread || !start_routine) return 1;
    qsim_pthread_start_t *start = (qsim_pthread_start_t*)malloc(sizeof(*start));
    if (!start) return 1;
    start->start = start_routine;
    start->arg = arg;
    *thread = CreateThread(NULL, 0, qsim_pthread_entry, start, 0, NULL);
    if (!*thread) {
        free(start);
        return 1;
    }
    return 0;
}

static inline int pthread_join(pthread_t thread, void **value_ptr) {
    (void)value_ptr;
    if (!thread) return 1;
    DWORD wait = WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return wait == WAIT_OBJECT_0 ? 0 : 1;
}

static BOOL CALLBACK qsim_pthread_once_callback(
    PINIT_ONCE once,
    PVOID parameter,
    PVOID *context
) {
    (void)once;
    (void)context;
    void (*init_routine)(void) = (void (*)(void))parameter;
    init_routine();
    return TRUE;
}

static inline int pthread_once(pthread_once_t *once_control, void (*init_routine)(void)) {
    if (!once_control || !init_routine) return 1;
    return InitOnceExecuteOnce(
        once_control,
        qsim_pthread_once_callback,
        (PVOID)init_routine,
        NULL) ? 0 : 1;
}

#else

#include_next <pthread.h>

#endif

#endif /* QSIM_PTHREAD_COMPAT_H */
