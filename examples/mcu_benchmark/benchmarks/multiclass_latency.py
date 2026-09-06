"""
Multiclass Latency Benchmark
=============================

Measures how inference latency scales with number of classes for GBT models.

Binary GBT uses sigmoid (1 expf call), multiclass uses softmax (n_classes expf calls).
This benchmark verifies that cycle counts scale appropriately with n_classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import emlearn

from ..common import RunContext, TimingResult, create_result_row
from ..mcu_runner import measure_mcu_timing, is_emulator_platform, estimate_timeout
from ..model_configs import (
    get_multiclass_configs,
    get_platforms,
    make_benchmark_dataset,
    PLATFORMS,
    RANDOM_STATE,
)
from .latency import measure_host_timing


def run_multiclass_latency_benchmark(
    platforms: list[str] | None = None,
    quick: bool = False,
    base_config: dict | None = None,
    n_classes_list: list[int] | None = None,
    run_ctx: RunContext | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run multiclass latency benchmark across platforms.

    Trains GBT models with varying n_classes and measures inference timing
    to verify softmax overhead scales with class count.

    Args:
        platforms: Platforms to benchmark.
        quick: Use reduced parameters.
        base_config: Base GBT config (default: n=5, d=3, lr=0.1).
        n_classes_list: List of n_classes to test (default: [2, 3, 5, 10]).
        run_ctx: Run context for artifact management.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        DataFrame with multiclass latency results.
    """
    print("\n" + "=" * 70)
    print("Multiclass Latency Benchmark")
    print("=" * 70)
    print("Measuring softmax overhead scaling with n_classes")

    if platforms is None:
        platforms = get_platforms(quick=quick)

    configs = get_multiclass_configs(
        base_config=base_config,
        n_classes_list=n_classes_list,
    )

    print(f"Configs: {len(configs)} (n_classes: {[c['n_classes'] for c in configs]})")
    print(f"Platforms: {', '.join(platforms)}")

    all_results = []

    for config in configs:
        n_classes = config['n_classes']
        print(f"\n  n_classes={n_classes}: {config['name']}")

        # Generate synthetic dataset with the right number of classes
        X_train, X_test, y_train, y_test = make_benchmark_dataset(
            n_samples=500,
            n_features=10,
            n_classes=n_classes,
            random_state=RANDOM_STATE,
        )

        # Train GBT classifier
        clf = GradientBoostingClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config.get('learning_rate', 0.1),
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)

        # Convert to emlearn
        cmodel = emlearn.convert(clf, method='inline')
        accuracy = accuracy_score(y_test, cmodel.predict(X_test))

        # Get flash size
        try:
            code = cmodel.save(name='model')
            flash_bytes = len(code.encode('utf-8'))
        except Exception:
            flash_bytes = 0

        for platform in platforms:
            platform_info = PLATFORMS.get(platform, {})
            print(f"    Platform: {platform}", end=" ")

            try:
                if platform == 'host':
                    # Host CFFI timing only supports predict mode
                    benchmark_mode = 'predict'
                    timing = measure_host_timing(cmodel, X_test)
                elif is_emulator_platform(platform):
                    # Softmax cost is only exercised by predict_proba
                    benchmark_mode = 'predict_proba'
                    timeout = estimate_timeout(cmodel)
                    timing = measure_mcu_timing(
                        cmodel, X_test, platform,
                        timeout=timeout, verbose=False,
                        benchmark_mode=benchmark_mode,
                    )
                else:
                    timing = TimingResult(
                        success=False, min_ns=0, avg_ns=0,
                        min_cycles=0, avg_cycles=0, iterations=0,
                        timing_mode='unknown',
                        error=f'Hardware platform {platform} not supported',
                    )

                if timing.success:
                    row = create_result_row(
                        model_config=config,
                        platform=platform,
                        timing=timing,
                        accuracy=accuracy,
                        flash_bytes=flash_bytes,
                    )
                    row['dataset'] = f'synthetic_{n_classes}class'
                    row['task_type'] = 'classification'
                    row['n_classes'] = n_classes
                    row['benchmark_mode'] = benchmark_mode
                    all_results.append(row)
                    print(f"→ {timing.avg_cycles} avg_cycles, acc={accuracy:.3f}")
                else:
                    print(f"→ FAILED: {timing.error}")

            except Exception as e:
                print(f"→ Error: {e}")

    df = pd.DataFrame(all_results)

    if run_ctx is not None and len(df) > 0:
        run_ctx.save_results('multiclass_latency', df)

    # Print scaling summary
    if len(df) > 0:
        print("\n" + "=" * 70)
        print("Multiclass Scaling Summary")
        print("=" * 70)
        for platform in df['platform'].unique():
            pdf = df[df['platform'] == platform]
            print(f"\n  {platform}:")
            for _, row in pdf.iterrows():
                print(f"    n_classes={row['n_classes']}: "
                      f"avg_cycles={row['avg_cycles']}, "
                      f"avg_ns={row['avg_ns']:.0f}")

    return df
