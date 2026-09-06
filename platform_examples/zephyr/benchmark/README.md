# emlearn Zephyr Benchmark

Benchmark inference latency for emlearn tree models on Zephyr RTOS.

## Quick Start

```bash
# Generate model (from src/ directory)
cd src && python train_model.py && cd ..

# Initialize workspace (first time only)
west init -l .
west update

# Build for native_sim (functional test, timing not representative)
west build --board native_sim -t run

# Build for real hardware
west build --board nrf52dk_nrf52832
west flash
```

## Timing Resolution

The benchmark uses `eml_benchmark_micros()` which relies on Zephyr's tick-based timing.
`prj.conf` sets `CONFIG_SYS_CLOCK_TICKS_PER_SEC=1000000` for microsecond resolution.

For cycle-accurate timing on Cortex-M3/M4/M7, the benchmark uses the DWT cycle counter
when `CONFIG_EMLEARN_TIMING_DWT=y` is set (default for Cortex-M3+ boards).

## Output Format

```
BENCHMARK inference: min=X us, max=Y us, avg=Z us, iterations=100
```

## Customizing the Model

```bash
python train_model.py --trees 20 --depth 8 --features 16 --name my_model
```

Then update `main.c` to include your model header.
