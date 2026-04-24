/**
 * @file manifest.c
 * @brief Runtime capture + JSON emission for the reproducibility
 *        manifest.  See manifest.h for the full design rationale.
 */

#include "manifest.h"
#include "moonlab_build_info.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

/* ---------------------------------------------------------------- */
/* small helpers                                                     */
/* ---------------------------------------------------------------- */

static char* dup_str(const char* s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char* out = (char*)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static char* iso_utc_now(void) {
    time_t t = time(NULL);
    struct tm tm_utc;
#if defined(_WIN32) || defined(_WIN64)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[32];
    /* 2026-04-19T15:42:31Z */
    strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return dup_str(buf);
}

/* ---------------------------------------------------------------- */
/* host-info probes                                                  */
/* ---------------------------------------------------------------- */

static char* probe_hostname(void) {
    char buf[256];
#if defined(_WIN32) || defined(_WIN64)
    DWORD size = (DWORD)sizeof buf;
    if (GetComputerNameA(buf, &size)) {
        buf[sizeof buf - 1] = '\0';
        return dup_str(buf);
    }
#else
    if (gethostname(buf, sizeof buf) == 0) {
        buf[sizeof buf - 1] = '\0';
        return dup_str(buf);
    }
#endif
    return dup_str("unknown");
}

static char* probe_os_release(void) {
#if defined(_WIN32) || defined(_WIN64)
    return dup_str("Windows");
#else
    struct utsname u;
    if (uname(&u) == 0) {
        char buf[256];
        snprintf(buf, sizeof buf, "%s %s", u.sysname, u.release);
        return dup_str(buf);
    }
    return dup_str("unknown");
#endif
}

static int probe_cpu_count(void) {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
#endif
}

static char* probe_cpu_brand(void) {
#ifdef __APPLE__
    char buf[256];
    size_t len = sizeof buf;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, NULL, 0) == 0) {
        buf[sizeof buf - 1] = '\0';
        return dup_str(buf);
    }
    /* On Apple Silicon the brand string may be empty; fall back. */
    len = sizeof buf;
    if (sysctlbyname("hw.model", buf, &len, NULL, 0) == 0) {
        buf[sizeof buf - 1] = '\0';
        return dup_str(buf);
    }
    return dup_str("Apple CPU (unknown)");
#elif defined(__linux__)
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return dup_str("unknown");
    char line[512];
    char out[256] = "unknown";
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "model name", 10) == 0) {
            const char* colon = strchr(line, ':');
            if (colon) {
                colon++;
                while (*colon == ' ') colon++;
                size_t n = strlen(colon);
                while (n > 0 && (colon[n - 1] == '\n' || colon[n - 1] == '\r')) n--;
                if (n >= sizeof out) n = sizeof out - 1;
                memcpy(out, colon, n);
                out[n] = '\0';
                break;
            }
        }
    }
    fclose(f);
    return dup_str(out);
#else
    return dup_str("unknown");
#endif
}

static uint64_t probe_mem_total_bytes(void) {
#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) return (uint64_t)status.ullTotalPhys;
    return 0;
#elif defined(__APPLE__)
    uint64_t v = 0;
    size_t len = sizeof v;
    if (sysctlbyname("hw.memsize", &v, &len, NULL, 0) == 0) return v;
    return 0;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) return (uint64_t)si.totalram * (uint64_t)si.mem_unit;
    return 0;
#else
    return 0;
#endif
}

/* ---------------------------------------------------------------- */
/* capture / finish / release                                        */
/* ---------------------------------------------------------------- */

int moonlab_manifest_capture(moonlab_manifest_t* m,
                             const char* label,
                             uint64_t seed)
{
    if (!m) return -1;
    memset(m, 0, sizeof(*m));

    m->run_label = dup_str(label ? label : "(unnamed)");
    m->seed = seed;

    m->version          = MOONLAB_VERSION_STRING;
    m->git_sha          = MOONLAB_GIT_SHA;
    m->git_sha_short    = MOONLAB_GIT_SHA_SHORT;
    m->git_dirty        = MOONLAB_GIT_DIRTY;
    m->git_branch       = MOONLAB_GIT_BRANCH;
    m->build_timestamp  = MOONLAB_BUILD_TIMESTAMP;
    m->build_type       = MOONLAB_BUILD_TYPE;
    m->compiler_id      = MOONLAB_COMPILER_ID;
    m->compiler_version = MOONLAB_COMPILER_VERSION;
    m->platform_name    = MOONLAB_PLATFORM_NAME;
    m->arch_name        = MOONLAB_ARCH_NAME;
    m->enabled_features = MOONLAB_ENABLED_FEATURES;

    m->hostname         = probe_hostname();
    m->os_release       = probe_os_release();
    m->cpu_brand        = probe_cpu_brand();
    m->cpu_count        = probe_cpu_count();
    m->mem_total_bytes  = probe_mem_total_bytes();
    m->run_start_iso    = iso_utc_now();
    m->run_finish_iso   = dup_str("");
    m->run_elapsed_seconds = 0.0;

    if (!m->run_label || !m->hostname || !m->os_release ||
        !m->cpu_brand || !m->run_start_iso || !m->run_finish_iso) {
        moonlab_manifest_release(m);
        return -2;
    }
    return 0;
}

void moonlab_manifest_stamp_finish(moonlab_manifest_t* m) {
    if (!m) return;
    if (m->run_finish_iso && *m->run_finish_iso != '\0') return; /* idempotent */

    free(m->run_finish_iso);
    m->run_finish_iso = iso_utc_now();

    /* Elapsed via parsing back the two ISO strings would be brittle;
     * use clock_gettime at start/finish to get a monotonic delta.
     * We approximate by re-walking the start string; if that fails,
     * leave 0.  Callers who need nanosecond-grade elapsed should use
     * their own timer. */
    struct tm start_tm = {0}, finish_tm = {0};
#if !defined(_WIN32) && !defined(_WIN64)
    if (m->run_start_iso &&
        strptime(m->run_start_iso, "%Y-%m-%dT%H:%M:%SZ", &start_tm) &&
        m->run_finish_iso &&
        strptime(m->run_finish_iso, "%Y-%m-%dT%H:%M:%SZ", &finish_tm)) {
        time_t s = timegm(&start_tm);
        time_t f = timegm(&finish_tm);
        if (s > 0 && f >= s) m->run_elapsed_seconds = (double)(f - s);
    }
#else
    (void)start_tm;
    (void)finish_tm;
#endif
}

void moonlab_manifest_release(moonlab_manifest_t* m) {
    if (!m) return;
    free(m->run_label);
    free(m->hostname);
    free(m->os_release);
    free(m->cpu_brand);
    free(m->run_start_iso);
    free(m->run_finish_iso);
    /* Do not free the const char* build-info pointers. */
    memset(m, 0, sizeof(*m));
}

/* ---------------------------------------------------------------- */
/* JSON emission                                                     */
/* ---------------------------------------------------------------- */

static void emit_json_string(FILE* out, const char* s) {
    fputc('"', out);
    if (!s) { fputc('"', out); return; }
    for (const unsigned char* p = (const unsigned char*)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '"':  fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\b': fputs("\\b", out);  break;
            case '\f': fputs("\\f", out);  break;
            case '\n': fputs("\\n", out);  break;
            case '\r': fputs("\\r", out);  break;
            case '\t': fputs("\\t", out);  break;
            default:
                if (c < 0x20) fprintf(out, "\\u%04x", c);
                else fputc((int)c, out);
        }
    }
    fputc('"', out);
}

static void emit_common(const moonlab_manifest_t* m, FILE* out,
                        const char* sep, const char* ind)
{
    #define KV_STR(key, val) do { \
        fputs(ind, out); emit_json_string(out, key); fputs(": ", out); \
        emit_json_string(out, (val) ? (val) : ""); fputs(sep, out); \
    } while (0)
    #define KV_INT(key, val) do { \
        fputs(ind, out); emit_json_string(out, key); \
        fprintf(out, ": %lld%s", (long long)(val), sep); \
    } while (0)
    #define KV_U64(key, val) do { \
        fputs(ind, out); emit_json_string(out, key); \
        fprintf(out, ": %llu%s", (unsigned long long)(val), sep); \
    } while (0)
    #define KV_F64(key, val) do { \
        fputs(ind, out); emit_json_string(out, key); \
        fprintf(out, ": %.6f%s", (double)(val), sep); \
    } while (0)

    KV_STR("run_label",        m->run_label);
    KV_U64("seed",             m->seed);
    KV_STR("version",          m->version);
    KV_STR("git_sha",          m->git_sha);
    KV_STR("git_sha_short",    m->git_sha_short);
    KV_INT("git_dirty",        m->git_dirty);
    KV_STR("git_branch",       m->git_branch);
    KV_STR("build_timestamp",  m->build_timestamp);
    KV_STR("build_type",       m->build_type);
    KV_STR("compiler_id",      m->compiler_id);
    KV_STR("compiler_version", m->compiler_version);
    KV_STR("platform",         m->platform_name);
    KV_STR("arch",             m->arch_name);
    KV_STR("enabled_features", m->enabled_features);
    KV_STR("hostname",         m->hostname);
    KV_STR("os_release",       m->os_release);
    KV_STR("cpu_brand",        m->cpu_brand);
    KV_INT("cpu_count",        m->cpu_count);
    KV_U64("mem_total_bytes",  m->mem_total_bytes);
    KV_STR("run_start",        m->run_start_iso);
    KV_STR("run_finish",       m->run_finish_iso);
    KV_F64("run_elapsed_s",    m->run_elapsed_seconds);

    /* Metrics is the last field; no trailing separator. */
    fputs(ind, out); emit_json_string(out, "metrics"); fputs(": ", out);
    if (m->metrics_json && *m->metrics_json) {
        fputs(m->metrics_json, out);
    } else {
        fputs("null", out);
    }

    #undef KV_STR
    #undef KV_INT
    #undef KV_U64
    #undef KV_F64
}

void moonlab_manifest_write_json(const moonlab_manifest_t* m, FILE* out) {
    if (!m || !out) return;
    fputc('{', out);
    emit_common(m, out, ",", "");
    fputc('}', out);
}

void moonlab_manifest_write_json_pretty(const moonlab_manifest_t* m, FILE* out) {
    if (!m || !out) return;
    fputs("{\n", out);
    emit_common(m, out, ",\n", "  ");
    fputs("\n}\n", out);
}
