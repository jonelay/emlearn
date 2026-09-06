/*
 * emlearn benchmark for Zephyr RTOS
 *
 * Measures inference latency for tree ensemble models.
 * Reports timing statistics over multiple iterations.
 *
 * Output format (with CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT=y):
 *   BENCHMARK:START
 *   BENCHMARK:CONFIG board=<board> timing=<dwt|systick> freq=<hz>
 *   BENCHMARK:MODEL name=<name> features=<n>
 *   RESULT:INFERENCE min_ns=<ns> max_ns=<ns> avg_ns=<ns> min_cycles=<cyc> avg_cycles=<cyc> iterations=<n>
 *   BENCHMARK:END status=<ok|error>
 *
 * Micro-benchmarks (with CONFIG_EMLEARN_BENCHMARK_MICRO=y):
 *   RESULT:MICRO:<name> avg_cycles=<cyc> iterations=<n>
 *   Available micro-benchmarks: EXPF, SIGMOID, INT_OPS, FP_OPS, FP_DIV
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/barrier.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "eml_mcu_timing.h"
#include "eml_trees.h"
#include "eml_gbtrees.h"
#include "benchmark_model.h"
#include "benchmark_model_testdata.h"

/* Configuration from Kconfig */
#ifndef CONFIG_EMLEARN_BENCHMARK_ITERATIONS
#define CONFIG_EMLEARN_BENCHMARK_ITERATIONS 100
#endif

#ifndef CONFIG_EMLEARN_BENCHMARK_WARMUP
#define CONFIG_EMLEARN_BENCHMARK_WARMUP 50
#endif

/* Micro-benchmark iterations (fixed, not configurable) */
#define MICRO_ITERATIONS 100
#define MICRO_OPS_PER_ITERATION 100  /* 100 ops dilutes ~5 cycle loop overhead to <5% */

/* Statistics tracking (nanoseconds for precision) */
struct benchmark_stats {
    uint32_t min_ns;
    uint32_t max_ns;
    uint64_t total_ns;
    uint32_t min_cycles;
    uint32_t max_cycles;
    uint64_t total_cycles;
    int count;
};

static void stats_init(struct benchmark_stats *stats)
{
    stats->min_ns = UINT32_MAX;
    stats->max_ns = 0;
    stats->total_ns = 0;
    stats->min_cycles = UINT32_MAX;
    stats->max_cycles = 0;
    stats->total_cycles = 0;
    stats->count = 0;
}

static void stats_update(struct benchmark_stats *stats, uint32_t elapsed_ns, uint32_t elapsed_cycles)
{
    if (elapsed_ns < stats->min_ns) {
        stats->min_ns = elapsed_ns;
    }
    if (elapsed_ns > stats->max_ns) {
        stats->max_ns = elapsed_ns;
    }
    if (elapsed_cycles < stats->min_cycles) {
        stats->min_cycles = elapsed_cycles;
    }
    if (elapsed_cycles > stats->max_cycles) {
        stats->max_cycles = elapsed_cycles;
    }
    stats->total_ns += elapsed_ns;
    stats->total_cycles += elapsed_cycles;
    stats->count++;
}

static void stats_print_structured(const struct benchmark_stats *stats, const char *label)
{
    uint32_t avg_ns = stats->count > 0 ? (uint32_t)(stats->total_ns / stats->count) : 0;
    uint32_t avg_cycles = stats->count > 0 ? (uint32_t)(stats->total_cycles / stats->count) : 0;

#ifdef CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT
    printf("RESULT:%s min_ns=%u max_ns=%u avg_ns=%u min_cycles=%u avg_cycles=%u iterations=%d\n",
           label,
           stats->min_ns,
           stats->max_ns,
           avg_ns,
           stats->min_cycles,
           avg_cycles,
           stats->count);
#else
    printf("BENCHMARK %s: min=%u ns, max=%u ns, avg=%u ns, min_cycles=%u, iterations=%d\n",
           label,
           stats->min_ns,
           stats->max_ns,
           avg_ns,
           stats->min_cycles,
           stats->count);
#endif
}

static void stats_print_legacy(const struct benchmark_stats *stats, const char *label)
{
    uint32_t avg_ns = stats->count > 0 ? (uint32_t)(stats->total_ns / stats->count) : 0;

    printf("BENCHMARK %s: min=%u ns, max=%u ns, avg=%u ns, iterations=%d\n",
           label,
           stats->min_ns,
           stats->max_ns,
           avg_ns,
           stats->count);
}

/*
 * Marker functions for GDB breakpoint-controlled execution tracing.
 * These must be non-inlined so GDB can set breakpoints on them.
 * The nop prevents the compiler from optimizing them away entirely.
 */
void __attribute__((noinline, used)) benchmark_loop_start(void)
{
    __asm__ volatile("nop");
}

void __attribute__((noinline, used)) benchmark_loop_end(void)
{
    __asm__ volatile("nop");
}

/*
 * Phase marker functions for overhead isolation in trace analysis.
 * These mark transitions between boot, warmup, benchmark, and output phases.
 * Addresses can be extracted with nm/objdump to count instructions per phase.
 */
void __attribute__((noinline, used)) marker_boot_end(void)
{
    __asm__ volatile("nop");
}

void __attribute__((noinline, used)) marker_warmup_start(void)
{
    __asm__ volatile("nop");
}

void __attribute__((noinline, used)) marker_warmup_end(void)
{
    __asm__ volatile("nop");
}

void __attribute__((noinline, used)) marker_benchmark_start(void)
{
    __asm__ volatile("nop");
}

void __attribute__((noinline, used)) marker_benchmark_end(void)
{
    __asm__ volatile("nop");
}

/*
 * Micro-benchmark functions for validating timing assumptions.
 * Each measures a specific operation type to establish CPI baselines.
 */
#ifdef CONFIG_EMLEARN_BENCHMARK_MICRO

/* Prevent constant folding of test values */
static volatile float micro_test_val = -1.5f;
static volatile uint32_t micro_test_int = 12345;

static void run_micro_expf(struct eml_timing_ctx *ctx)
{
    struct eml_timing_sample sample;
    uint64_t total_cycles = 0;
    volatile float result = 0.0f;
    float x;

    /* Warm up */
    for (int i = 0; i < MICRO_ITERATIONS / 10; i++) {
        result = expf(micro_test_val + (float)i * 0.001f);
    }

    /* Benchmark */
    for (int i = 0; i < MICRO_ITERATIONS; i++) {
        x = micro_test_val + (float)i * 0.001f;

        eml_timing_start(ctx, &sample);
        for (int j = 0; j < MICRO_OPS_PER_ITERATION; j++) {
            result = expf(x);
            x = result * 0.1f + micro_test_val;
        }
        eml_timing_stop(ctx, &sample);

        total_cycles += eml_timing_cycles(ctx, &sample);
        __asm__ volatile("" : : "r"(result) : "memory");
    }

    uint32_t avg_cycles = (uint32_t)(total_cycles / (MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION));
    printf("RESULT:MICRO:EXPF avg_cycles=%u iterations=%d\n", avg_cycles, MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION);
}

static void run_micro_sigmoid(struct eml_timing_ctx *ctx)
{
    struct eml_timing_sample sample;
    uint64_t total_cycles = 0;
    volatile float result = 0.0f;
    float x;

    /* Warm up */
    for (int i = 0; i < MICRO_ITERATIONS / 10; i++) {
        x = micro_test_val + (float)i * 0.001f;
        result = 1.0f / (1.0f + expf(-x));
    }

    /* Benchmark: full sigmoid = 1/(1+exp(-x)) */
    for (int i = 0; i < MICRO_ITERATIONS; i++) {
        x = micro_test_val + (float)i * 0.001f;

        eml_timing_start(ctx, &sample);
        for (int j = 0; j < MICRO_OPS_PER_ITERATION; j++) {
            result = 1.0f / (1.0f + expf(-x));
            x = result - 0.5f + micro_test_val;
        }
        eml_timing_stop(ctx, &sample);

        total_cycles += eml_timing_cycles(ctx, &sample);
        __asm__ volatile("" : : "r"(result) : "memory");
    }

    uint32_t avg_cycles = (uint32_t)(total_cycles / (MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION));
    printf("RESULT:MICRO:SIGMOID avg_cycles=%u iterations=%d\n", avg_cycles, MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION);
}

static void run_micro_int_ops(struct eml_timing_ctx *ctx)
{
    struct eml_timing_sample sample;
    uint64_t total_cycles = 0;
    volatile uint32_t result = 0;
    uint32_t x;

    /* Benchmark: simple integer arithmetic (should be ~1 CPI) */
    for (int i = 0; i < MICRO_ITERATIONS; i++) {
        x = micro_test_int + i;

        eml_timing_start(ctx, &sample);
        for (int j = 0; j < MICRO_OPS_PER_ITERATION; j++) {
            result = x + (j * 3);
            x = result ^ 0x5555;
            result = x - j;
            x = result | (j << 2);
            result = x & 0xFFFF;
        }
        eml_timing_stop(ctx, &sample);

        total_cycles += eml_timing_cycles(ctx, &sample);
        __asm__ volatile("" : : "r"(result) : "memory");
    }

    /* 5 ops per inner loop iteration */
    uint32_t avg_cycles = (uint32_t)(total_cycles / (MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION * 5));
    printf("RESULT:MICRO:INT_OPS avg_cycles=%u iterations=%d\n", avg_cycles, MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION * 5);
}

static void run_micro_fp_ops(struct eml_timing_ctx *ctx)
{
    struct eml_timing_sample sample;
    uint64_t total_cycles = 0;
    volatile float result = 0.0f;
    float x;

    /* Benchmark: simple FP add/mul (should be ~1 CPI on FPU) */
    for (int i = 0; i < MICRO_ITERATIONS; i++) {
        x = (float)micro_test_int * 0.001f + (float)i;

        eml_timing_start(ctx, &sample);
        for (int j = 0; j < MICRO_OPS_PER_ITERATION; j++) {
            result = x + (float)j;
            x = result * 1.1f;
            result = x - 0.5f;
            x = result * 0.9f;
            result = x + 1.0f;
        }
        eml_timing_stop(ctx, &sample);

        total_cycles += eml_timing_cycles(ctx, &sample);
        __asm__ volatile("" : : "r"(result) : "memory");
    }

    /* 5 ops per inner loop iteration */
    uint32_t avg_cycles = (uint32_t)(total_cycles / (MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION * 5));
    printf("RESULT:MICRO:FP_OPS avg_cycles=%u iterations=%d\n", avg_cycles, MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION * 5);
}

static void run_micro_fp_div(struct eml_timing_ctx *ctx)
{
    struct eml_timing_sample sample;
    uint64_t total_cycles = 0;
    volatile float result = 0.0f;
    float x;

    /* Benchmark: FP division (known 14-cycle operation on Cortex-M4F) */
    for (int i = 0; i < MICRO_ITERATIONS; i++) {
        x = (float)micro_test_int + (float)(i + 1);

        eml_timing_start(ctx, &sample);
        for (int j = 0; j < MICRO_OPS_PER_ITERATION; j++) {
            result = x / (float)(j + 1);
            x = result + (float)micro_test_int;
        }
        eml_timing_stop(ctx, &sample);

        total_cycles += eml_timing_cycles(ctx, &sample);
        __asm__ volatile("" : : "r"(result) : "memory");
    }

    uint32_t avg_cycles = (uint32_t)(total_cycles / (MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION));
    printf("RESULT:MICRO:FP_DIV avg_cycles=%u iterations=%d\n", avg_cycles, MICRO_ITERATIONS * MICRO_OPS_PER_ITERATION);
}

static void run_micro_benchmarks(struct eml_timing_ctx *ctx)
{
#ifdef CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT
    printf("BENCHMARK:MICRO_START\n");
#else
    printf("\n=== Micro-benchmarks ===\n");
#endif

    run_micro_int_ops(ctx);
    run_micro_fp_ops(ctx);
    run_micro_fp_div(ctx);
    run_micro_expf(ctx);
    run_micro_sigmoid(ctx);

#ifdef CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT
    printf("BENCHMARK:MICRO_END\n");
#else
    printf("\n");
#endif
}

#endif /* CONFIG_EMLEARN_BENCHMARK_MICRO */

/* predict_proba output length. Inline GBT models require the exact class
 * count, so prefer BENCHMARK_MODEL_N_CLASSES from the generated test-data
 * header and only fall back to the buffer size when it is not provided. */
#ifdef BENCHMARK_MODEL_N_CLASSES
#define BENCHMARK_PROBA_LENGTH BENCHMARK_MODEL_N_CLASSES
#else
#define BENCHMARK_PROBA_LENGTH EMTREES_MAX_CLASSES
#endif

/* Maximum classes for predict_proba buffer */
#ifndef EMTREES_MAX_CLASSES
#define EMTREES_MAX_CLASSES 30
#endif

static int run_benchmark(void)
{
    struct eml_timing_ctx timing_ctx;
    struct eml_timing_sample sample;
    struct benchmark_stats stats;
#ifdef CONFIG_EMLEARN_BENCHMARK_PREDICT_PROBA
    float probas[EMTREES_MAX_CLASSES];
    int proba_err;
#else
    int prediction;
#endif
    int ret;

    /* Mark end of boot phase (for trace analysis) */
    marker_boot_end();

    /* Initialize timing */
    ret = eml_timing_init(&timing_ctx);
    if (ret < 0) {
        printf("ERROR: Failed to initialize timing\n");
        return ret;
    }

#ifdef CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT
    printf("BENCHMARK:START\n");
    printf("BENCHMARK:CONFIG board=%s timing=%s freq=%u\n",
           CONFIG_BOARD,
           eml_timing_mode_str(timing_ctx.mode),
           timing_ctx.cpu_freq_hz);
    printf("BENCHMARK:MODEL name=benchmark_model features=%d samples=%d\n",
           BENCHMARK_MODEL_N_FEATURES,
           BENCHMARK_MODEL_N_SAMPLES);
#else
    printf("\n=== emlearn Benchmark ===\n");
    printf("Board: %s\n", CONFIG_BOARD);
    printf("Timing: %s @ %u Hz\n", eml_timing_mode_str(timing_ctx.mode), timing_ctx.cpu_freq_hz);
    printf("Model: benchmark_model\n");
    printf("Features: %d\n", BENCHMARK_MODEL_N_FEATURES);
    printf("Test samples: %d\n", BENCHMARK_MODEL_N_SAMPLES);
    printf("Warmup iterations: %d\n", CONFIG_EMLEARN_BENCHMARK_WARMUP);
    printf("Benchmark iterations: %d\n", CONFIG_EMLEARN_BENCHMARK_ITERATIONS);
    printf("\n");
#endif

#ifdef CONFIG_EMLEARN_BENCHMARK_MICRO
    /* Run micro-benchmarks before main benchmark */
    run_micro_benchmarks(&timing_ctx);
#endif

    /* Mark start of warmup phase */
    marker_warmup_start();

    /* Warmup - run inference without timing to warm caches */
#ifndef CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT
    printf("Running warmup...\n");
#endif
    for (int i = 0; i < CONFIG_EMLEARN_BENCHMARK_WARMUP; i++) {
        int sample_idx = i % BENCHMARK_MODEL_N_SAMPLES;
#ifdef CONFIG_EMLEARN_BENCHMARK_PREDICT_PROBA
        proba_err = benchmark_model_predict_proba(benchmark_model_test_data[sample_idx],
                                                   BENCHMARK_MODEL_N_FEATURES,
                                                   probas, BENCHMARK_PROBA_LENGTH);
        if (proba_err != 0) {
            /* A failing predict_proba returns in a few cycles and would be
             * reported as a very fast inference. Fail the run instead. */
            printf("ERROR: predict_proba returned %d (out_length=%d)\n",
                   proba_err, BENCHMARK_PROBA_LENGTH);
            return -1;
        }
#else
        prediction = benchmark_model_predict(benchmark_model_test_data[sample_idx],
                                             BENCHMARK_MODEL_N_FEATURES);
        (void)prediction;  /* suppress unused warning */
#endif
    }

    /* Mark end of warmup phase */
    marker_warmup_end();

    /* Benchmark - time each inference */
#ifndef CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT
    printf("Running benchmark...\n");
#endif
    stats_init(&stats);

    /* Marker for GDB-controlled trace start */
    benchmark_loop_start();

    /* Mark start of benchmark phase (same as benchmark_loop_start for now) */
    marker_benchmark_start();

    for (int i = 0; i < CONFIG_EMLEARN_BENCHMARK_ITERATIONS; i++) {
        int sample_idx = i % BENCHMARK_MODEL_N_SAMPLES;
        const float *features = benchmark_model_test_data[sample_idx];

        barrier_dmem_fence_full();  /* Complete pending memory operations */
        eml_timing_start(&timing_ctx, &sample);
#ifdef CONFIG_EMLEARN_BENCHMARK_PREDICT_PROBA
        proba_err = benchmark_model_predict_proba(features, BENCHMARK_MODEL_N_FEATURES,
                                                   probas, BENCHMARK_PROBA_LENGTH);
#else
        prediction = benchmark_model_predict(features, BENCHMARK_MODEL_N_FEATURES);
#endif
        eml_timing_stop(&timing_ctx, &sample);
        barrier_dmem_fence_full();  /* Ensure timing read completes */

        uint32_t elapsed_ns = eml_timing_ns(&timing_ctx, &sample);
        uint32_t elapsed_cycles = eml_timing_cycles(&timing_ctx, &sample);
        stats_update(&stats, elapsed_ns, elapsed_cycles);

        /* Use result to prevent compiler from optimizing away inference.
         * The asm volatile with memory clobber ensures the compiler cannot
         * eliminate the benchmark call as dead code. */
#ifdef CONFIG_EMLEARN_BENCHMARK_PREDICT_PROBA
        __asm__ volatile("" : : "m"(probas) : "memory");
        (void)proba_err;
#else
        __asm__ volatile("" : : "r"(prediction) : "memory");
#endif
    }

    /* Mark end of benchmark phase */
    marker_benchmark_end();

    /* Marker for GDB-controlled trace end */
    benchmark_loop_end();

    /* Report results */
#ifdef CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT
    stats_print_structured(&stats, "INFERENCE");
    printf("BENCHMARK:END status=ok\n");
#else
    printf("\n");
    stats_print_legacy(&stats, "inference");
    printf("\n=== Benchmark Complete ===\n");
#endif

    return 0;
}

int main(void)
{
#ifndef CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT
    printf("\nemlearn Zephyr Benchmark\n");
    printf("Tick rate: %d Hz\n", CONFIG_SYS_CLOCK_TICKS_PER_SEC);
#endif

    int ret = run_benchmark();

#ifdef CONFIG_EMLEARN_BENCHMARK_STRUCTURED_OUTPUT
    if (ret != 0) {
        printf("BENCHMARK:END status=error code=%d\n", ret);
    }
#endif

    return ret;
}
