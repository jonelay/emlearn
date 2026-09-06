#!/usr/bin/env python3
"""
MCU Comparative Benchmark Runner
================================

Orchestrates benchmarks across Host/Renode/Hardware platforms with
comprehensive GBT vs RF comparison.

Usage:
    python run_all.py --benchmark all --quick --host-only
    python run_all.py --benchmark latency --quick --renode-only
    python run_all.py --benchmark latency sample_efficiency --quick
    python run_all.py --resume runs/2026-01-18_143052_sweep   # reuse run dir (re-runs configs)

Platforms:
    - host: CFFI-based host execution (fast validation)
    - native_sim: Zephyr native_sim (host-native integration)
    - renode_nrf52840: Renode nRF52840 emulator (cycle-accurate DWT timing)
    - nrf52dk_nrf52832: Nordic nRF52 DK hardware (ground truth)

Benchmarks:
    - sample_efficiency: Trees needed to reach accuracy thresholds
    - pareto_efficiency: Accuracy vs flash size Pareto frontiers
    - latency: Inference timing across platforms
    - regression_accuracy: MSE/R² on regression tasks
    - probability_calibration: Brier score and ECE
    - size_constrained: Best accuracy at flash budgets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.mcu_benchmark.common import (
    RunContext,
    print_summary,
    parse_duration,
    SCRIPT_DIR,
)
from examples.mcu_benchmark.model_configs import (
    get_platforms,
    count_sweep_configs,
    PLATFORMS,
)
from examples.mcu_benchmark.benchmarks import (
    BENCHMARK_NAMES,
    BENCHMARK_FUNCTIONS,
)


def run_all_benchmarks(
    benchmarks: list[str],
    platforms: list[str],
    task: str = 'both',
    datasets: list[str] | None = None,
    quick: bool = False,
    extended: bool = False,
    methods: list[str] | None = None,
    test_timeout: float | None = None,
    n_jobs: int = -1,
    run_ctx: RunContext | None = None,
    traversal: str = 'index',
) -> dict[str, pd.DataFrame]:
    """Run selected benchmarks and return results.

    Args:
        benchmarks: List of benchmark names to run.
        platforms: List of platforms to benchmark.
        task: Task type ('classification', 'regression', or 'both').
        datasets: Specific datasets to benchmark.
        quick: Use quick mode (reduced configs).
        extended: Use extended mode (1-100 trees) for sample efficiency.
        methods: Compilation methods to test ('inline', 'loadable').
        test_timeout: Max time per test in seconds (None for no limit).
        n_jobs: Number of parallel jobs.
        run_ctx: Run context for artifact management.

    Returns:
        Dictionary mapping benchmark name to results DataFrame.
    """
    if methods is None:
        methods = ['inline']
    results = {}

    for benchmark in benchmarks:
        if benchmark not in BENCHMARK_FUNCTIONS:
            print(f"Warning: Unknown benchmark '{benchmark}', skipping")
            continue

        print("\n" + "#" * 70)
        print(f"Running: {benchmark}")
        print("#" * 70)

        # Get benchmark function
        benchmark_fn = BENCHMARK_FUNCTIONS[benchmark]

        # Run benchmark - all functions accept **kwargs for unused params
        try:
            df = benchmark_fn(
                platforms=platforms,
                datasets=datasets,
                task=task,
                quick=quick,
                extended=extended,
                methods=methods,
                test_timeout=test_timeout,
                n_jobs=n_jobs,
                run_ctx=run_ctx,
                traversal=traversal,
            )
            results[benchmark] = df
        except Exception as e:
            print(f"Error running {benchmark}: {e}")
            results[benchmark] = pd.DataFrame()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='MCU Comparative Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation (host only)
    python run_all.py --benchmark all --quick --host-only

    # Specific benchmarks
    python run_all.py --benchmark latency sample_efficiency --quick

    # Full sweep with parallelism
    python run_all.py --benchmark all --n_jobs -1

    # Add results to an existing run directory (selected configs are re-run)
    python run_all.py --resume runs/2026-01-18_143052_sweep

    # Classification only
    python run_all.py --benchmark all --task classification --quick
        """
    )

    # Benchmark selection
    parser.add_argument('--benchmark', '-b', type=str, nargs='+',
                        default=['latency'],
                        choices=['all'] + BENCHMARK_NAMES,
                        help="Benchmarks to run. Use 'all' for complete sweep.")

    # Parallelization
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Parallel jobs (-1=all cores, 1=serial)')

    # Task selection
    parser.add_argument('--task', type=str,
                        choices=['classification', 'regression', 'both'],
                        default='both',
                        help='Task type to benchmark')

    # Dataset selection
    parser.add_argument('--datasets', type=str, nargs='+',
                        help='Specific datasets to benchmark (default: all for task)')

    # Platform selection
    parser.add_argument('--host-only', action='store_true',
                        help='Run host CFFI benchmarks only')
    parser.add_argument('--renode-only', action='store_true',
                        help='Run Renode emulator benchmarks only')
    parser.add_argument('--hardware-only', action='store_true',
                        help='Run hardware benchmarks only')
    parser.add_argument('--no-host', action='store_true',
                        help='Skip host benchmarks (run Renode/hardware only)')
    parser.add_argument('--validation-only', action='store_true',
                        help='Run validation track only (host, native_sim)')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Run benchmark track only (Renode, hardware)')
    parser.add_argument('--platforms', type=str, nargs='+',
                        choices=list(PLATFORMS.keys()),
                        help='Specific platforms to benchmark')

    # Quick mode
    parser.add_argument('--quick', action='store_true',
                        help='Use reduced sweep parameters for quick validation')

    # Extended mode (for sample efficiency analysis with 1-100 trees)
    parser.add_argument('--extended', action='store_true',
                        help='Extended n_estimators sweep (1-100) for sample efficiency analysis')

    # Methods selection (inline vs loadable)
    parser.add_argument('--methods', type=str, nargs='+',
                        choices=['inline', 'loadable'],
                        default=['inline'],
                        help="Compilation methods to test. 'loadable' uses int16 quantization.")

    # Tree traversal mode (for benchmarking pointer vs index traversal)
    parser.add_argument('--traversal', type=str,
                        choices=['index', 'pointer'],
                        default='index',
                        help="Tree traversal mode. 'pointer' may be faster on some architectures.")

    # Per-test timeout (skips slow tests without killing sweep)
    parser.add_argument('-t', '--test-timeout', type=str, default=None,
                        help='Max time per test (e.g., "5m", "300s", "300"). '
                             'Tests exceeding timeout are skipped.')

    # Legacy compatibility
    parser.add_argument('--iterations', type=int, default=100,
                        help='Host benchmark iterations (default: 100)')
    parser.add_argument('--timeout', type=float, default=120.0,
                        help='MCU benchmark timeout in seconds (default: 120)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    # Resume/output
    parser.add_argument('--resume', type=Path,
                        help='Write results into an existing run directory. '
                             'All selected configurations are re-run; per-config '
                             'resume is not implemented.')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR,
                        help='Base output directory for runs')
    parser.add_argument('--name', type=str, default='sweep',
                        help='Run name suffix for output directory')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to files')

    args = parser.parse_args()

    # Determine benchmarks to run
    if 'all' in args.benchmark:
        benchmarks = BENCHMARK_NAMES
    else:
        benchmarks = args.benchmark

    # Determine platforms
    if args.platforms:
        platforms = args.platforms
    elif args.validation_only:
        platforms = ['host', 'native_sim']
    elif args.benchmark_only:
        platforms = ['renode_nrf52840', 'nrf52dk_nrf52832']
    else:
        platforms = get_platforms(
            quick=args.quick,
            host_only=args.host_only,
            renode_only=args.renode_only,
            hardware_only=args.hardware_only,
            no_host=args.no_host,
        )

    # Parse test timeout
    test_timeout = parse_duration(args.test_timeout)

    # Create or resume run context
    if args.resume:
        run_ctx = RunContext.resume(args.resume)
        print(f"Resuming from: {run_ctx.run_dir}")
    elif not args.no_save:
        config = {
            'benchmarks': benchmarks,
            'platforms': platforms,
            'task': args.task,
            'datasets': args.datasets,
            'quick': args.quick,
            'extended': args.extended,
            'methods': args.methods,
            'traversal': args.traversal,
            'test_timeout': args.test_timeout,
            'n_jobs': args.n_jobs,
        }
        run_ctx = RunContext.create(
            base_dir=args.output_dir,
            config=config,
            name=args.name,
        )
        print(f"Run directory: {run_ctx.run_dir}")
    else:
        run_ctx = None

    # Print configuration
    print("\n" + "#" * 70)
    print("MCU Comparative Benchmark Suite")
    print("#" * 70)
    print(f"\nBenchmarks: {', '.join(benchmarks)}")
    print(f"Platforms: {', '.join(platforms)}")
    print(f"Task: {args.task}")
    print(f"Quick mode: {args.quick}")
    print(f"Extended mode: {args.extended}")
    print(f"Methods: {', '.join(args.methods)}")
    if args.traversal != 'index':
        print(f"Tree traversal: {args.traversal}")
    if test_timeout:
        print(f"Test timeout: {test_timeout}s")
    print(f"Parallelism: n_jobs={args.n_jobs}")

    if args.extended:
        counts = count_sweep_configs(task=args.task, extended=True)
        print(f"\nExtended mode config count: {counts['total']} per platform")
    elif args.quick:
        counts = count_sweep_configs(task=args.task, quick=True)
        print(f"\nQuick mode config count: {counts['total']} per platform")
    else:
        counts = count_sweep_configs(task=args.task, quick=False)
        print(f"\nFull mode config count: {counts['total']} per platform")

    # Run benchmarks
    results = run_all_benchmarks(
        benchmarks=benchmarks,
        platforms=platforms,
        task=args.task,
        datasets=args.datasets,
        quick=args.quick,
        extended=args.extended,
        methods=args.methods,
        test_timeout=test_timeout,
        n_jobs=args.n_jobs,
        run_ctx=run_ctx,
        traversal=args.traversal,
    )

    # Combine and save results
    if run_ctx is not None:
        all_dfs = [df for df in results.values() if len(df) > 0]
        if all_dfs:
            run_ctx.save_combined_results(all_dfs)

    # Print overall summary
    print("\n" + "#" * 70)
    print("Benchmark Complete")
    print("#" * 70)

    for benchmark, df in results.items():
        if len(df) > 0:
            print(f"\n{benchmark}: {len(df)} results")
        else:
            print(f"\n{benchmark}: No results")

    if run_ctx is not None:
        print(f"\nResults saved to: {run_ctx.run_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
