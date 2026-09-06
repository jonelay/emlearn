
#ifndef EML_BENCHMARK_H
#define EML_BENCHMARK_H
#ifdef __cplusplus
extern "C" {
#endif

#include "eml_common.h"

/*
 * DWT Cycle Counter for ARM Cortex-M3+
 *
 * Provides cycle-accurate timing using the Data Watchpoint and Trace (DWT) unit.
 * Available on Cortex-M3, M4, M7, M33, and other ARMv7-M/ARMv8-M cores.
 *
 * Usage:
 *   eml_dwt_init();  // Call once at startup
 *   uint32_t start = eml_dwt_cycles();
 *   // ... code to measure ...
 *   uint32_t end = eml_dwt_cycles();
 *   uint32_t elapsed_cycles = end - start;
 *   uint32_t elapsed_us = eml_cycles_to_us(elapsed_cycles, cpu_freq_hz);
 */

/* Check for ARM Cortex-M3+ architecture */
#if defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__) || \
    defined(__ARM_ARCH_8M_MAIN__) || defined(__ARM_ARCH_8_1M_MAIN__)
#define EML_HAVE_DWT 1
#endif

#ifdef EML_HAVE_DWT

/* DWT registers (ARMv7-M Architecture Reference Manual) */
#define EML_DWT_CTRL    (*(volatile uint32_t *)0xE0001000)
#define EML_DWT_CYCCNT  (*(volatile uint32_t *)0xE0001004)

/* CoreDebug DEMCR - Debug Exception and Monitor Control Register */
#define EML_DEMCR       (*(volatile uint32_t *)0xE000EDFC)

/* Bit definitions */
#define EML_DEMCR_TRCENA      (1UL << 24)  /* Enable DWT and ITM */
#define EML_DWT_CTRL_CYCCNTENA (1UL << 0)  /* Enable cycle counter */

/* Memory barriers - use inline functions to avoid collision with CMSIS macros */
static inline void eml_dsb(void) { __asm volatile ("dsb 0xF":::"memory"); }
static inline void eml_isb(void) { __asm volatile ("isb 0xF":::"memory"); }

/**
 * Initialize DWT cycle counter.
 * Must be called once before using eml_dwt_cycles().
 * Returns 0 on success, -1 if DWT is not available or locked.
 */
static inline int eml_dwt_init(void) {
    /* Enable trace (required for DWT) */
    EML_DEMCR |= EML_DEMCR_TRCENA;

    /* Reset cycle counter */
    EML_DWT_CYCCNT = 0;

    /* Enable cycle counter */
    EML_DWT_CTRL |= EML_DWT_CTRL_CYCCNTENA;

    /* Synchronization barriers - ensure peripheral writes complete */
    eml_dsb();
    eml_isb();

    /* Read-back verify: check enable bit was actually set.
     * May fail if debug registers are locked (TrustZone, security fuses). */
    if (!(EML_DWT_CTRL & EML_DWT_CTRL_CYCCNTENA)) {
        return -1;
    }

    /* Verify counter is incrementing (100 iterations for reliable detection) */
    volatile uint32_t start = EML_DWT_CYCCNT;
    for (volatile int i = 0; i < 100; i++) { }
    volatile uint32_t end = EML_DWT_CYCCNT;

    return (end > start) ? 0 : -1;
}

/**
 * Get current DWT cycle count.
 * Note: 32-bit counter wraps at ~53 seconds at 80MHz.
 */
static inline uint32_t eml_dwt_cycles(void) {
    return EML_DWT_CYCCNT;
}

/**
 * Convert cycles to microseconds.
 * @param cycles Number of CPU cycles
 * @param cpu_freq_hz CPU frequency in Hz (e.g., 64000000 for 64 MHz)
 */
static inline uint32_t eml_cycles_to_us(uint32_t cycles, uint32_t cpu_freq_hz) {
    return (uint32_t)((uint64_t)cycles * 1000000ULL / cpu_freq_hz);
}

/**
 * Convert cycles to nanoseconds.
 * @param cycles Number of CPU cycles
 * @param cpu_freq_hz CPU frequency in Hz
 */
static inline uint32_t eml_cycles_to_ns(uint32_t cycles, uint32_t cpu_freq_hz) {
    return (uint32_t)((uint64_t)cycles * 1000000000ULL / cpu_freq_hz);
}

/**
 * Maximum reasonable cycle count for a single measurement.
 * At 64 MHz, 32-bit counter wraps after ~67 seconds.
 * Set threshold to 10 seconds (640M cycles) as sanity check.
 * If a measurement exceeds this, it likely wrapped multiple times.
 */
#define EML_DWT_MAX_REASONABLE_CYCLES (64000000UL * 10)

/**
 * Calculate elapsed cycles with wrap-around handling.
 * C unsigned subtraction handles single wrap correctly:
 * if start=0xFFFFFF00, end=0x00000100, then end-start=0x200.
 *
 * @param start Cycle count at start
 * @param end Cycle count at end
 * @return Elapsed cycles (handles single 32-bit wrap)
 */
static inline uint32_t eml_dwt_elapsed(uint32_t start, uint32_t end) {
    return end - start;
}

/**
 * Check if elapsed cycles exceeds sanity threshold.
 * Use this to detect measurements that likely wrapped multiple times
 * or are otherwise invalid.
 *
 * @param elapsed Elapsed cycle count from eml_dwt_elapsed()
 * @return 1 if elapsed exceeds EML_DWT_MAX_REASONABLE_CYCLES, 0 otherwise
 */
static inline int eml_dwt_elapsed_invalid(uint32_t elapsed) {
    return elapsed > EML_DWT_MAX_REASONABLE_CYCLES;
}

#endif /* EML_HAVE_DWT */

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
// Unix-like system
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#define EML_HAVE_SYS_TIME 1
#endif


#ifdef EML_HAVE_SYS_TIME
#include <sys/time.h>
int64_t eml_benchmark_micros() {
    struct timeval spec;
    gettimeofday(&spec, NULL);
    //struct timespec spec; 
    //clock_gettime(CLOCK_MONOTONIC, &spec);
    const int64_t micros = (int64_t)(spec.tv_sec)*1000LL*1000LL + spec.tv_usec;
    return micros;
}
#endif

#ifdef _WIN32
#include <windows.h>
int64_t eml_benchmark_micros(void)
{
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    double sec = (double)t.QuadPart/(double)f.QuadPart;
    return (int64_t)(sec * 1000000LL);
}
#endif

#ifdef ARDUINO
int64_t eml_benchmark_micros() {
    return micros();
}
#endif

#ifdef __ZEPHYR__
#include <zephyr/kernel.h>

int64_t eml_benchmark_micros() {

    //int64_t cycles =
    int64_t ticks = k_uptime_ticks();
    int64_t micros = k_ticks_to_us_ceil64(ticks);
    return micros;
}
#endif

// https://en.wikipedia.org/wiki/Lehmer_random_number_generator#Parameters_in_common_use
static uint32_t
eml_lcg_parkmiller(uint32_t *state) {
    const uint32_t N = 0x7fffffff;
    const uint32_t G = 48271u;

    uint32_t div = *state / (N / G);
    uint32_t rem = *state % (N / G);

    uint32_t a = rem * G;
    uint32_t b = div * (N % G);

    return *state = (a > b) ? (a - b) : (a + (N - b));
}

EmlError
eml_benchmark_fill(float *values, int features) {
    uint32_t rng_state = 1;    

    for (int i=0; i<features; i++) {
        values[i] = (float)eml_lcg_parkmiller(&rng_state);
    }
    return EmlOk;
}


#ifdef __cplusplus
} // extern "C"
#endif
#endif // EML_BENCHMARK_H
