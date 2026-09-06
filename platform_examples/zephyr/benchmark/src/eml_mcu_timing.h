/*
 * MCU Timing Abstraction for emlearn benchmarks
 *
 * Provides a unified interface for timing measurements across different
 * ARM Cortex-M variants:
 * - DWT cycle counter (Cortex-M3+): cycle-accurate
 * - Systick (Cortex-M0/M0+): microsecond resolution via k_cyc_to_us_*
 *
 * Configuration via Kconfig:
 * - CONFIG_EMLEARN_TIMING_DWT: Use DWT cycle counter (default on M3+)
 * - CONFIG_EMLEARN_TIMING_SYSTICK: Use systick via Zephyr kernel
 */

#ifndef EML_MCU_TIMING_H
#define EML_MCU_TIMING_H

#include <zephyr/kernel.h>
#include <stdint.h>

#ifdef CONFIG_EMLEARN_TIMING_DWT
#include "eml_benchmark.h"
#endif

#ifdef CONFIG_EMLEARN_TIMING_POSIX
#include <sys/time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Timing mode enumeration */
typedef enum {
    EML_TIMING_SYSTICK,
    EML_TIMING_DWT,
    EML_TIMING_POSIX,
} eml_timing_mode_t;

/* Timing context - stores configuration for a timing session */
struct eml_timing_ctx {
    eml_timing_mode_t mode;
    uint32_t cpu_freq_hz;  /* For DWT cycle->time conversion */
};

/* Timing sample - raw timing data before conversion */
struct eml_timing_sample {
    uint32_t start;
    uint32_t end;
#ifdef CONFIG_EMLEARN_TIMING_POSIX
    uint64_t start_us;  /* POSIX: microseconds from gettimeofday */
    uint64_t end_us;
#endif
};

/**
 * Initialize timing context.
 * Detects available timing hardware and initializes it.
 *
 * @param ctx Timing context to initialize
 * @return 0 on success, negative on error
 */
static inline int eml_timing_init(struct eml_timing_ctx *ctx) {
    ctx->cpu_freq_hz = CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC;

#ifdef CONFIG_EMLEARN_TIMING_POSIX
    ctx->mode = EML_TIMING_POSIX;
    ctx->cpu_freq_hz = 1000000000;  /* Report as 1GHz for cycle display */
    return 0;
#endif

#ifdef CONFIG_EMLEARN_TIMING_DWT
#ifdef EML_HAVE_DWT
    if (eml_dwt_init() == 0) {
        ctx->mode = EML_TIMING_DWT;
        /* DWT counts CPU cycles, not system clock ticks.
         * Use dedicated config since SYS_CLOCK_HW_CYCLES_PER_SEC
         * may reflect RTC frequency (e.g., 32768 Hz on nRF52). */
        ctx->cpu_freq_hz = CONFIG_EMLEARN_DWT_CPU_FREQ_HZ;
        return 0;
    }
#endif
#endif

    /* Fall back to systick */
    ctx->mode = EML_TIMING_SYSTICK;
    return 0;
}

/**
 * Get timing mode name string.
 */
static inline const char *eml_timing_mode_str(eml_timing_mode_t mode) {
    switch (mode) {
        case EML_TIMING_DWT:     return "dwt";
        case EML_TIMING_SYSTICK: return "systick";
        case EML_TIMING_POSIX:   return "posix";
        default:                 return "unknown";
    }
}

#ifdef CONFIG_EMLEARN_TIMING_POSIX
static inline uint64_t eml_posix_gettime_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}
#endif

/**
 * Start timing measurement.
 */
static inline void eml_timing_start(const struct eml_timing_ctx *ctx,
                                    struct eml_timing_sample *sample) {
#ifdef CONFIG_EMLEARN_TIMING_POSIX
    if (ctx->mode == EML_TIMING_POSIX) {
        sample->start_us = eml_posix_gettime_us();
        return;
    }
#endif
#ifdef CONFIG_EMLEARN_TIMING_DWT
    if (ctx->mode == EML_TIMING_DWT) {
        sample->start = eml_dwt_cycles();
        return;
    }
#endif
    sample->start = k_cycle_get_32();
}

/**
 * Stop timing measurement.
 */
static inline void eml_timing_stop(const struct eml_timing_ctx *ctx,
                                   struct eml_timing_sample *sample) {
#ifdef CONFIG_EMLEARN_TIMING_POSIX
    if (ctx->mode == EML_TIMING_POSIX) {
        sample->end_us = eml_posix_gettime_us();
        return;
    }
#endif
#ifdef CONFIG_EMLEARN_TIMING_DWT
    if (ctx->mode == EML_TIMING_DWT) {
        sample->end = eml_dwt_cycles();
        return;
    }
#endif
    sample->end = k_cycle_get_32();
}

/**
 * Get elapsed time in CPU cycles.
 */
static inline uint32_t eml_timing_cycles(const struct eml_timing_ctx *ctx,
                                         const struct eml_timing_sample *sample) {
#ifdef CONFIG_EMLEARN_TIMING_POSIX
    if (ctx->mode == EML_TIMING_POSIX) {
        /* Return microseconds * 1000 as "cycles" for POSIX mode (pseudo-ns) */
        return (uint32_t)((sample->end_us - sample->start_us) * 1000);
    }
#endif
    (void)ctx;
    return sample->end - sample->start;
}

/**
 * Get elapsed time in microseconds.
 */
static inline uint32_t eml_timing_us(const struct eml_timing_ctx *ctx,
                                     const struct eml_timing_sample *sample) {
#ifdef CONFIG_EMLEARN_TIMING_POSIX
    if (ctx->mode == EML_TIMING_POSIX) {
        return (uint32_t)(sample->end_us - sample->start_us);
    }
#endif

    uint32_t cycles = sample->end - sample->start;

#ifdef CONFIG_EMLEARN_TIMING_DWT
    if (ctx->mode == EML_TIMING_DWT) {
        return eml_cycles_to_us(cycles, ctx->cpu_freq_hz);
    }
#endif

    return k_cyc_to_us_floor32(cycles);
}

/**
 * Get elapsed time in nanoseconds.
 */
static inline uint32_t eml_timing_ns(const struct eml_timing_ctx *ctx,
                                     const struct eml_timing_sample *sample) {
#ifdef CONFIG_EMLEARN_TIMING_POSIX
    if (ctx->mode == EML_TIMING_POSIX) {
        /* gettimeofday only has microsecond precision */
        return (uint32_t)((sample->end_us - sample->start_us) * 1000);
    }
#endif

    uint32_t cycles = sample->end - sample->start;

#ifdef CONFIG_EMLEARN_TIMING_DWT
    if (ctx->mode == EML_TIMING_DWT) {
        return eml_cycles_to_ns(cycles, ctx->cpu_freq_hz);
    }
#endif

    return k_cyc_to_ns_floor32(cycles);
}

#ifdef __cplusplus
}
#endif

#endif /* EML_MCU_TIMING_H */
