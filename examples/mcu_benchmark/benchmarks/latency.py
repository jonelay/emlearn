"""
Latency Benchmark for MCU Platforms
===================================

Measures inference latency across different platforms.

Key finding: GBT can be 10-50x faster on FPU targets due to hardware expf().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, r2_score

import emlearn

from ..common import RunContext, TimingResult, create_result_row
from ..datasets import (
    get_classification_datasets,
    get_regression_datasets,
    get_all_datasets,
    get_quick_datasets,
)
from ..mcu_runner import measure_mcu_timing, is_emulator_platform, estimate_timeout
from ..model_configs import (
    get_sweep_configs,
    get_platforms,
    prepare_data_for_method,
    get_method_dtype,
    PLATFORMS,
    QUICK_N_ESTIMATORS,
    QUICK_MAX_DEPTHS,
)

# Default random state
RANDOM_STATE = 42


def measure_host_timing(
    cmodel,
    X_test: np.ndarray,
    iterations: int = 100,
) -> TimingResult:
    """Measure timing on host platform using CFFI.

    Args:
        cmodel: Converted emlearn model.
        X_test: Test features.
        iterations: Number of timing iterations.

    Returns:
        TimingResult with timing measurements.
    """
    import time

    times_ns = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        _ = cmodel.predict(X_test[:1])
        end = time.perf_counter_ns()
        times_ns.append(end - start)

    min_ns = min(times_ns)
    avg_ns = sum(times_ns) / len(times_ns)

    return TimingResult(
        success=True,
        min_ns=min_ns,
        avg_ns=avg_ns,
        min_cycles=int(min_ns),  # Rough estimate (1 cycle per ns at 1GHz)
        avg_cycles=int(avg_ns),
        iterations=iterations,
        timing_mode='cffi',
    )


def run_latency_benchmark(
    platforms: list[str] | None = None,
    datasets: list[str] | None = None,
    task: str = 'classification',
    quick: bool = False,
    methods: list[str] | None = None,
    n_jobs: int = -1,
    run_ctx: RunContext | None = None,
    traversal: str = 'index',
    **kwargs,
) -> pd.DataFrame:
    """Run latency benchmark across platforms.

    Args:
        platforms: Platforms to benchmark.
        datasets: Specific datasets to benchmark.
        task: Task type ('classification' or 'regression').
        quick: Use reduced parameters.
        methods: Compilation methods to test ('inline', 'loadable').
        n_jobs: Number of parallel jobs.
        run_ctx: Run context for artifact management.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        DataFrame with latency results.
    """
    if methods is None:
        methods = ['inline']

    # Build extra_defines based on traversal mode
    extra_defines = []
    if traversal == 'pointer':
        extra_defines.append('EML_TREES_USE_POINTER_TRAVERSAL')

    print("\n" + "=" * 70)
    print("Latency Benchmark")
    print("=" * 70)
    print("Measuring inference time across platforms")
    print(f"Methods: {', '.join(methods)}")
    if traversal != 'index':
        print(f"Tree traversal: {traversal}")

    if platforms is None:
        # Include emulator platforms by default for latency benchmarks
        platforms = get_platforms(quick=quick)

    # Get datasets based on task type
    if quick:
        all_datasets = get_quick_datasets(task=task)
    elif task == 'classification':
        all_datasets = get_classification_datasets()
    elif task == 'regression':
        all_datasets = get_regression_datasets()
    else:  # 'both'
        all_datasets = get_all_datasets()

    if datasets:
        all_datasets = {k: v for k, v in all_datasets.items() if k in datasets}

    # Get model configs
    n_estimators = QUICK_N_ESTIMATORS if quick else [3, 10, 20, 40]
    max_depths = QUICK_MAX_DEPTHS if quick else [3, 5]

    all_results = []

    for ds_name, (X_train, X_test, y_train, y_test, task_type) in all_datasets.items():
        print(f"\nDataset: {ds_name}")
        print(f"  Features: {X_train.shape[1]}, Train: {len(X_train)}, Test: {len(X_test)}")

        for platform in platforms:
            platform_info = PLATFORMS.get(platform, {})
            print(f"\n  Platform: {platform} ({platform_info.get('description', '')})")

            for method in methods:
                if len(methods) > 1:
                    print(f"    Method: {method}")

                # Prepare data for this method (scaling for loadable)
                X_train_m, dtype = prepare_data_for_method(X_train, method)
                X_test_m, _ = prepare_data_for_method(X_test, method)

                for model_type in ['gbt', 'rf']:
                    # Select classifier or regressor based on dataset task type
                    if task_type == 'regression':
                        model_cls = GradientBoostingRegressor if model_type == 'gbt' else RandomForestRegressor
                    else:
                        model_cls = GradientBoostingClassifier if model_type == 'gbt' else RandomForestClassifier

                    for n_est in n_estimators:
                        for max_d in max_depths:
                            # Build config
                            config = {
                                'name': f"{model_type}_n{n_est}_d{max_d}_{method}",
                                'type': model_type,
                                'n_estimators': n_est,
                                'max_depth': max_d,
                                'n_features': X_train.shape[1],
                                'method': method,
                            }

                            # Build model kwargs
                            kwargs = {
                                'n_estimators': n_est,
                                'max_depth': max_d,
                                'random_state': RANDOM_STATE,
                            }
                            if model_type == 'gbt':
                                kwargs['learning_rate'] = 0.1

                            try:
                                # Train model on original float data
                                model = model_cls(**kwargs)
                                model.fit(X_train, y_train)

                                # Convert to emlearn with appropriate method/dtype
                                cmodel = emlearn.convert(model, method=method, dtype=dtype)

                                # Evaluate accuracy/score using method-prepared data
                                y_pred = cmodel.predict(X_test_m)
                                if task_type == 'regression':
                                    accuracy = r2_score(y_test, y_pred)
                                else:
                                    accuracy = accuracy_score(y_test, y_pred)

                                # Get flash size estimate (same for all benchmark modes)
                                try:
                                    code = cmodel.save(name='model')
                                    flash_bytes = len(code.encode('utf-8'))
                                except Exception:
                                    flash_bytes = 0

                                # Benchmark modes: predict always, predict_proba for classification
                                benchmark_modes = ['predict']
                                if task_type == 'classification':
                                    benchmark_modes.append('predict_proba')

                                for benchmark_mode in benchmark_modes:
                                    # Measure timing
                                    if platform == 'host':
                                        # Host timing only supports predict mode
                                        if benchmark_mode == 'predict_proba':
                                            continue
                                        timing = measure_host_timing(cmodel, X_test_m)
                                    elif is_emulator_platform(platform):
                                        timeout = estimate_timeout(cmodel)
                                        timing = measure_mcu_timing(
                                            cmodel, X_test_m, platform,
                                            timeout=timeout, verbose=False,
                                            extra_defines=extra_defines if extra_defines else None,
                                            benchmark_mode=benchmark_mode,
                                        )
                                    else:
                                        # Hardware platforms not yet supported in sweep
                                        timing = TimingResult(
                                            success=False,
                                            min_ns=0,
                                            avg_ns=0,
                                            min_cycles=0,
                                            avg_cycles=0,
                                            iterations=0,
                                            timing_mode='unknown',
                                            error=f'Hardware platform {platform} not supported in sweep',
                                        )

                                    if timing.success:
                                        row = create_result_row(
                                            model_config=config,
                                            platform=platform,
                                            timing=timing,
                                            accuracy=accuracy,
                                            flash_bytes=flash_bytes,
                                        )
                                        row['dataset'] = ds_name
                                        row['task_type'] = task_type
                                        row['traversal'] = traversal
                                        row['benchmark_mode'] = benchmark_mode
                                        all_results.append(row)

                                        # Save incrementally after each result
                                        if run_ctx is not None:
                                            df_incremental = pd.DataFrame(all_results)
                                            run_ctx.save_results('latency', df_incremental)

                                        mode_suffix = f" [{benchmark_mode}]" if benchmark_mode != 'predict' else ""
                                        print(f"      {config['name']}{mode_suffix}: {timing.avg_ns:.0f}ns, acc={accuracy:.3f}")
                                    else:
                                        # Log timing failure with error details
                                        error_msg = timing.error if timing.error else "Unknown error"
                                        mode_suffix = f" [{benchmark_mode}]" if benchmark_mode != 'predict' else ""
                                        print(f"      {config['name']}{mode_suffix}: FAILED - {error_msg}")

                            except Exception as e:
                                print(f"      {config['name']}: Error - {e}")

        # Save per-dataset CSV for granular recovery after each dataset completes
        if run_ctx is not None and len(all_results) > 0:
            dataset_results = [r for r in all_results if r.get('dataset') == ds_name]
            if dataset_results:
                df_dataset = pd.DataFrame(dataset_results)
                run_ctx.save_results(f'latency_{ds_name}', df_dataset)
                print(f"  Saved {len(dataset_results)} results for {ds_name}")

    df = pd.DataFrame(all_results)

    # Save results if run context provided
    if run_ctx is not None and len(df) > 0:
        run_ctx.save_results('latency', df)

    # Print summary
    if len(df) > 0:
        print_latency_summary(df)

    return df


def print_latency_summary(df: pd.DataFrame) -> None:
    """Print latency benchmark summary.

    Args:
        df: Benchmark results DataFrame.
    """
    print("\n" + "=" * 70)
    print("Latency Summary")
    print("=" * 70)

    for platform in df['platform'].unique():
        platform_df = df[df['platform'] == platform]
        print(f"\n{platform}:")

        for model_type in ['gbt', 'rf']:
            type_df = platform_df[platform_df['model_type'] == model_type]
            if len(type_df) == 0:
                continue

            print(f"  {model_type.upper()}:")
            print(f"    Avg latency: {type_df['avg_ns'].mean():.0f} ns")
            print(f"    Min latency: {type_df['min_ns'].min():.0f} ns")
            print(f"    Max latency: {type_df['avg_ns'].max():.0f} ns")

        # Compare GBT vs RF
        gbt_df = platform_df[platform_df['model_type'] == 'gbt']
        rf_df = platform_df[platform_df['model_type'] == 'rf']

        if len(gbt_df) > 0 and len(rf_df) > 0:
            gbt_avg = gbt_df['avg_ns'].mean()
            rf_avg = rf_df['avg_ns'].mean()
            ratio = rf_avg / gbt_avg if gbt_avg > 0 else 0
            print(f"\n  GBT vs RF: GBT is {ratio:.1f}x {'faster' if ratio > 1 else 'slower'}")
