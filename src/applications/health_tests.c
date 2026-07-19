#include "health_tests.h"
#include "../utils/secure_memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>

/* SP 800-90B Section 4.4 lets the developer choose alpha for the consuming
 * application.  One 4 MiB conditioned Moonlab request can exercise fewer
 * than 2^29 raw-source samples through the certified Bell extractor.  Using
 * alpha=2^-60 per continuous-test decision leaves a union-bound false-reject
 * probability below 2^-31 for that request while retaining ample margin for
 * the stuck-source and biased-source failure modes these tests target. */
#define HEALTH_FALSE_POSITIVE_BITS 60.0

// ============================================================================
// CONFIGURATION CALCULATIONS
// ============================================================================

uint32_t health_calculate_rct_cutoff(double min_entropy) {
    // NIST SP 800-90B formula: C = ceil(1 + (-log2(alpha) / H)).
    if (min_entropy <= 0.0 || min_entropy > 8.0) {
        min_entropy = 4.0;
    }

    double cutoff = 1.0 + (HEALTH_FALSE_POSITIVE_BITS / min_entropy);
    return (uint32_t)ceil(cutoff);
}

/* Smallest APT cutoff C satisfying
 *
 *   Pr[1 + Binomial(trials, p) >= C] <= alpha.
 *
 * The leading one is the reference sample selected at the start of each APT
 * window.  Work in log space so alpha=2^-60 remains representable without
 * cancellation in 1-CDF or underflow in the extreme upper tail. */
static uint32_t health_binomial_apt_cutoff(
    uint32_t trials,
    double p,
    double log_alpha
) {
    double log_tail = -INFINITY;
    double log_pmf = (double)trials * log(p);
    const double log_odds_inverse = log1p(-p) - log(p);

    for (uint32_t k = trials;; --k) {
        if (isinf(log_tail) && log_tail < 0.0) {
            log_tail = log_pmf;
        } else {
            double hi = log_tail > log_pmf ? log_tail : log_pmf;
            double lo = log_tail > log_pmf ? log_pmf : log_tail;
            log_tail = hi + log1p(exp(lo - hi));
        }

        if (log_tail > log_alpha) {
            /* Tail(k) is too large; Tail(k+1) was the last acceptable upper
             * tail.  Convert the remaining-match threshold k+1 to the APT's
             * count including its fixed first sample. */
            return k + 2u;
        }
        if (k == 0u) break;

        log_pmf += log((double)k)
                 - log((double)(trials - k + 1u))
                 + log_odds_inverse;
    }

    return 1u;
}

uint32_t health_calculate_apt_cutoff(double min_entropy, uint32_t window_size) {
    // SP 800-90B APT: choose C from the exact binomial upper tail.
    if (min_entropy <= 0.0 || min_entropy > 8.0 || window_size == 0) {
        min_entropy = 4.0;
        window_size = 512;
    }

    double p = pow(2.0, -min_entropy);
    double log_alpha = -HEALTH_FALSE_POSITIVE_BITS * log(2.0);
    return health_binomial_apt_cutoff(window_size - 1u, p, log_alpha);
}

void health_get_recommended_config(double min_entropy, health_test_config_t *config) {
    if (!config) return;
    
    // Ensure min_entropy is in valid range
    if (min_entropy < 1.0) min_entropy = 4.0; // Conservative default
    if (min_entropy > 8.0) min_entropy = 8.0;
    
    config->min_entropy_estimate = min_entropy;
    config->rct_cutoff = health_calculate_rct_cutoff(min_entropy);
    config->apt_window_size = 512; // NIST recommended
    config->apt_cutoff = health_calculate_apt_cutoff(min_entropy, config->apt_window_size);
    config->startup_test_samples = 1024; // NIST minimum
}

int health_validate_config(const health_test_config_t *config) {
    if (!config) return 0;
    
    // Check reasonable bounds
    if (config->rct_cutoff < 2 || config->rct_cutoff > 1000) return 0;
    if (config->apt_cutoff < 10 || config->apt_cutoff > 10000) return 0;
    if (config->apt_window_size < 16 || config->apt_window_size > 65536) return 0;
    if (config->startup_test_samples < 100) return 0;
    if (config->min_entropy_estimate <= 0.0 || config->min_entropy_estimate > 8.0) return 0;
    
    return 1;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

health_error_t health_tests_init(health_test_ctx_t *ctx) {
    if (!ctx) return HEALTH_ERROR_INVALID_PARAM;
    
    memset(ctx, 0, sizeof(*ctx));
    
    // Set conservative defaults (H_min = 4 bits/sample)
    health_get_recommended_config(4.0, &ctx->config);
    
    // Allocate APT window buffer
    ctx->stats.apt_window_buffer = calloc(ctx->config.apt_window_size, sizeof(uint8_t));
    if (!ctx->stats.apt_window_buffer) {
        return HEALTH_ERROR_INVALID_PARAM;
    }
    
    ctx->stats.tests_enabled = 1;
    
    return HEALTH_SUCCESS;
}

health_error_t health_tests_init_custom(health_test_ctx_t *ctx, const health_test_config_t *config) {
    if (!ctx || !config) return HEALTH_ERROR_INVALID_PARAM;
    if (!health_validate_config(config)) return HEALTH_ERROR_INVALID_PARAM;
    
    memset(ctx, 0, sizeof(*ctx));
    ctx->config = *config;
    
    // Allocate APT window buffer
    ctx->stats.apt_window_buffer = calloc(ctx->config.apt_window_size, sizeof(uint8_t));
    if (!ctx->stats.apt_window_buffer) {
        return HEALTH_ERROR_INVALID_PARAM;
    }
    
    ctx->stats.tests_enabled = 1;
    
    return HEALTH_SUCCESS;
}

void health_tests_free(health_test_ctx_t *ctx) {
    if (!ctx) return;
    
    if (ctx->stats.apt_window_buffer) {
        // Secure erase before freeing
        secure_memzero(ctx->stats.apt_window_buffer, ctx->config.apt_window_size);
        free(ctx->stats.apt_window_buffer);
        ctx->stats.apt_window_buffer = NULL;
    }
    
    secure_memzero(ctx, sizeof(*ctx));
}

void health_tests_reset(health_test_ctx_t *ctx) {
    if (!ctx) return;
    
    // Save configuration and buffer
    health_test_config_t saved_config = ctx->config;
    uint8_t *saved_buffer = ctx->stats.apt_window_buffer;
    void (*saved_callback)(health_error_t, void*) = ctx->failure_callback;
    void *saved_user_data = ctx->callback_user_data;
    
    // Clear statistics
    memset(&ctx->stats, 0, sizeof(ctx->stats));
    
    // Restore
    ctx->config = saved_config;
    ctx->stats.apt_window_buffer = saved_buffer;
    ctx->failure_callback = saved_callback;
    ctx->callback_user_data = saved_user_data;
    ctx->stats.tests_enabled = 1;
    
    // Clear APT buffer
    if (ctx->stats.apt_window_buffer) {
        secure_memzero(ctx->stats.apt_window_buffer, ctx->config.apt_window_size);
    }
}

// ============================================================================
// INDIVIDUAL TESTS
// ============================================================================

health_error_t health_test_rct(health_test_ctx_t *ctx, uint8_t sample) {
    if (!ctx) return HEALTH_ERROR_INVALID_PARAM;
    if (!ctx->stats.tests_enabled) return HEALTH_SUCCESS;
    
    // First sample initializes the test
    if (ctx->stats.samples_tested == 0) {
        ctx->stats.rct_last_sample = sample;
        ctx->stats.rct_current_count = 1;
        return HEALTH_SUCCESS;
    }
    
    // Check if sample repeats
    if (sample == ctx->stats.rct_last_sample) {
        ctx->stats.rct_current_count++;
        
        // Test failure if count exceeds cutoff
        if (ctx->stats.rct_current_count >= ctx->config.rct_cutoff) {
            ctx->stats.rct_failures++;
            ctx->stats.total_failures++;
            
            // Invoke callback if set
            if (ctx->failure_callback) {
                ctx->failure_callback(HEALTH_ERROR_RCT_FAILURE, ctx->callback_user_data);
            }
            
            // Reset counter but keep last sample
            ctx->stats.rct_current_count = 1;
            
            return HEALTH_ERROR_RCT_FAILURE;
        }
    } else {
        // Different sample, reset counter
        ctx->stats.rct_last_sample = sample;
        ctx->stats.rct_current_count = 1;
    }
    
    return HEALTH_SUCCESS;
}

health_error_t health_test_apt(health_test_ctx_t *ctx, uint8_t sample) {
    if (!ctx) return HEALTH_ERROR_INVALID_PARAM;
    if (!ctx->stats.tests_enabled) return HEALTH_SUCCESS;
    if (!ctx->stats.apt_window_buffer) return HEALTH_ERROR_NOT_INITIALIZED;
    
    // First sample in window
    if (ctx->stats.apt_window_pos == 0) {
        ctx->stats.apt_first_sample = sample;
        ctx->stats.apt_current_count = 1;
        ctx->stats.apt_window_buffer[0] = sample;
        ctx->stats.apt_window_pos = 1;
        return HEALTH_SUCCESS;
    }
    
    // Correctly used contexts keep apt_window_pos in
    // [0, apt_window_size). Fail closed if state corruption or an unsafe
    // caller violates that invariant; silently continuing would turn the
    // continuous health test into an unreliable signal.
    if (ctx->stats.apt_window_pos >= ctx->config.apt_window_size) {
        return HEALTH_ERROR_INVALID_PARAM;
    }
    ctx->stats.apt_window_buffer[ctx->stats.apt_window_pos] = sample;
    
    // Count occurrences of first sample
    if (sample == ctx->stats.apt_first_sample) {
        ctx->stats.apt_current_count++;
    }
    
    ctx->stats.apt_window_pos++;
    
    // Window full - perform test
    if (ctx->stats.apt_window_pos >= ctx->config.apt_window_size) {
        // Check if count exceeds cutoff
        if (ctx->stats.apt_current_count >= ctx->config.apt_cutoff) {
            ctx->stats.apt_failures++;
            ctx->stats.total_failures++;
            
            // Invoke callback if set
            if (ctx->failure_callback) {
                ctx->failure_callback(HEALTH_ERROR_APT_FAILURE, ctx->callback_user_data);
            }
            
            // Reset window
            ctx->stats.apt_window_pos = 0;
            ctx->stats.apt_current_count = 0;
            
            return HEALTH_ERROR_APT_FAILURE;
        }
        
        // Test passed, reset window
        ctx->stats.apt_window_pos = 0;
        ctx->stats.apt_current_count = 0;
    }
    
    return HEALTH_SUCCESS;
}

// ============================================================================
// TEST EXECUTION
// ============================================================================

health_error_t health_tests_run(health_test_ctx_t *ctx, uint8_t sample) {
    if (!ctx) return HEALTH_ERROR_INVALID_PARAM;
    if (!ctx->stats.tests_enabled) return HEALTH_SUCCESS;
    
    ctx->stats.samples_tested++;
    
    // Run RCT
    health_error_t rct_result = health_test_rct(ctx, sample);
    if (rct_result != HEALTH_SUCCESS) {
        return rct_result;
    }
    
    // Run APT
    health_error_t apt_result = health_test_apt(ctx, sample);
    if (apt_result != HEALTH_SUCCESS) {
        return apt_result;
    }
    
    return HEALTH_SUCCESS;
}

health_error_t health_tests_run_batch(health_test_ctx_t *ctx, const uint8_t *samples, size_t num_samples) {
    if (!ctx || !samples) return HEALTH_ERROR_INVALID_PARAM;
    if (!ctx->stats.tests_enabled) return HEALTH_SUCCESS;
    
    for (size_t i = 0; i < num_samples; i++) {
        health_error_t result = health_tests_run(ctx, samples[i]);
        if (result != HEALTH_SUCCESS) {
            return result; // Fail on first error
        }
    }
    
    return HEALTH_SUCCESS;
}

health_error_t health_tests_startup(health_test_ctx_t *ctx, const uint8_t *samples, size_t num_samples) {
    if (!ctx || !samples) return HEALTH_ERROR_INVALID_PARAM;
    if (num_samples < ctx->config.startup_test_samples) return HEALTH_ERROR_INVALID_PARAM;
    
    // Reset tests for startup
    health_tests_reset(ctx);
    
    // Run tests on startup samples
    health_error_t result = health_tests_run_batch(ctx, samples, num_samples);
    
    if (result == HEALTH_SUCCESS) {
        ctx->stats.startup_complete = 1;
    } else {
        ctx->stats.startup_failures++;
        ctx->stats.startup_complete = 0;
    }
    
    return result;
}

// ============================================================================
// STATISTICS & MONITORING
// ============================================================================

health_test_stats_t health_tests_get_stats(const health_test_ctx_t *ctx) {
    health_test_stats_t empty = {0};
    if (!ctx) return empty;
    return ctx->stats;
}

int health_tests_startup_complete(const health_test_ctx_t *ctx) {
    if (!ctx) return 0;
    return ctx->stats.startup_complete;
}

void health_tests_set_callback(
    health_test_ctx_t *ctx,
    void (*callback)(health_error_t error, void *user_data),
    void *user_data
) {
    if (!ctx) return;
    ctx->failure_callback = callback;
    ctx->callback_user_data = user_data;
}

void health_tests_set_enabled(health_test_ctx_t *ctx, int enabled) {
    if (!ctx) return;
    ctx->stats.tests_enabled = enabled ? 1 : 0;
}

void health_tests_print_stats(const health_test_ctx_t *ctx) {
    if (!ctx) return;
    
    printf("=== Health Test Statistics ===\n");
    printf("Configuration:\n");
    printf("  Min-entropy:      %.2f bits/sample\n", ctx->config.min_entropy_estimate);
    printf("  RCT cutoff:       %u\n", ctx->config.rct_cutoff);
    printf("  APT cutoff:       %u\n", ctx->config.apt_cutoff);
    printf("  APT window:       %u samples\n", ctx->config.apt_window_size);
    printf("\n");
    printf("Status:\n");
    printf("  Tests enabled:    %s\n", ctx->stats.tests_enabled ? "Yes" : "No");
    printf("  Startup complete: %s\n", ctx->stats.startup_complete ? "Yes" : "No");
    printf("\n");
    printf("Statistics:\n");
    printf("  Samples tested:   %" PRIu64 "\n", ctx->stats.samples_tested);
    printf("  Total failures:   %" PRIu64 "\n", ctx->stats.total_failures);
    printf("  RCT failures:     %" PRIu64 "\n", ctx->stats.rct_failures);
    printf("  APT failures:     %" PRIu64 "\n", ctx->stats.apt_failures);
    printf("  Startup failures: %" PRIu64 "\n", ctx->stats.startup_failures);
    
    if (ctx->stats.samples_tested > 0) {
        double failure_rate = (double)ctx->stats.total_failures / ctx->stats.samples_tested * 100.0;
        printf("  Failure rate:     %.6f%%\n", failure_rate);
    }
    
    printf("\n");
    printf("Current RCT state:\n");
    printf("  Last sample:      0x%02X\n", ctx->stats.rct_last_sample);
    printf("  Current count:    %u / %u\n", ctx->stats.rct_current_count, ctx->config.rct_cutoff);
    printf("\n");
    printf("Current APT state:\n");
    printf("  First sample:     0x%02X\n", ctx->stats.apt_first_sample);
    printf("  Current count:    %u / %u\n", ctx->stats.apt_current_count, ctx->config.apt_cutoff);
    printf("  Window position:  %u / %u\n", ctx->stats.apt_window_pos, ctx->config.apt_window_size);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const char* health_error_string(health_error_t error) {
    switch (error) {
        case HEALTH_SUCCESS:
            return "Success";
        case HEALTH_ERROR_RCT_FAILURE:
            return "Repetition Count Test failure - entropy source may be stuck";
        case HEALTH_ERROR_APT_FAILURE:
            return "Adaptive Proportion Test failure - loss of entropy detected";
        case HEALTH_ERROR_STARTUP_FAILURE:
            return "Startup test failure - entropy source not functioning correctly";
        case HEALTH_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case HEALTH_ERROR_NOT_INITIALIZED:
            return "Health tests not initialized";
        default:
            return "Unknown error";
    }
}
