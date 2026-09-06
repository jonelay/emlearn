"""
Regression Accuracy Benchmark for MCU Platforms
================================================

Compare MSE/R² on regression tasks.

Key finding: GBT excels on additive relationships due to its
sequential error correction approach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score

import emlearn

from ..common import RunContext, create_benchmark_result, measure_host_latency
from ..checkpointing import config_hash
from ..mcu_runner import measure_mcu_timing, is_emulator_platform
from ..process_manager import run_parallel
from ..datasets import get_regression_datasets
from ..model_configs import get_sweep_configs

# Default random state
RANDOM_STATE = 42


def _train_and_evaluate_regression(
    config: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    measure_latency: bool = True,
) -> dict | None:
    """Train regression model and evaluate.

    Args:
        config: Model configuration.
        X_train: Training features.
        X_test: Test features.
        y_train: Training targets.
        y_test: Test targets.
        measure_latency: Whether to measure inference latency.

    Returns:
        Result dictionary or None on failure.
    """
    model_type = config['type']
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    learning_rate = config.get('learning_rate')
    dataset = config.get('dataset', 'unknown')
    platform = config.get('platform', 'host')

    # Select model class
    model_cls = GradientBoostingRegressor if model_type == 'gbt' else RandomForestRegressor

    # Build model kwargs
    kwargs = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'random_state': RANDOM_STATE,
    }
    if learning_rate is not None:
        kwargs['learning_rate'] = learning_rate

    try:
        # Train model
        model = model_cls(**kwargs)
        model.fit(X_train, y_train)

        # Convert to emlearn (float features for accuracy)
        cmodel = emlearn.convert(model, method='inline', dtype='float')

        # Evaluate with sklearn model (emlearn predict for regression)
        y_pred_sklearn = model.predict(X_test)
        y_pred_emlearn = cmodel.predict(X_test)

        # Calculate metrics
        mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
        r2_sklearn = r2_score(y_test, y_pred_sklearn)

        mse_emlearn = mean_squared_error(y_test, y_pred_emlearn)
        r2_emlearn = r2_score(y_test, y_pred_emlearn)

        # Get flash size
        try:
            code = cmodel.save(name='model')
            flash_bytes = len(code.encode('utf-8'))
        except Exception:
            flash_bytes = 0

        # Measure latency (adaptive iterations for slow models)
        latency_ns = 0.0
        if measure_latency:
            if platform == 'host':
                timing = measure_host_latency(cmodel, X_test[:1])
                latency_ns = timing.avg_ns
            elif is_emulator_platform(platform):
                timing = measure_mcu_timing(cmodel, X_test, platform)
                if timing.success:
                    latency_ns = timing.avg_ns

        return create_benchmark_result(
            platform=platform,
            dataset=dataset,
            task='regression',
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            score=r2_emlearn,  # Use R² as primary score
            score_type='r2',
            flash_bytes=flash_bytes,
            latency_ns=latency_ns,
            config_hash=config_hash(config),
            # Regression-specific extra fields
            mse_sklearn=mse_sklearn,
            r2_sklearn=r2_sklearn,
            mse_emlearn=mse_emlearn,
            r2_emlearn=r2_emlearn,
        )

    except Exception as e:
        print(f"Error processing {config['name']}: {e}")
        return None


def run_regression_benchmark(
    platforms: list[str] | None = None,
    datasets: list[str] | None = None,
    quick: bool = False,
    n_jobs: int = -1,
    run_ctx: RunContext | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run regression accuracy benchmark.

    Args:
        platforms: Platforms to benchmark (host only for this benchmark).
        datasets: Specific datasets to benchmark.
        quick: Use reduced parameters.
        n_jobs: Number of parallel jobs.
        run_ctx: Run context for artifact management.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        DataFrame with regression accuracy results.
    """
    print("\n" + "=" * 70)
    print("Regression Accuracy Benchmark")
    print("=" * 70)
    print("Comparing MSE/R² on regression tasks")

    # This benchmark only supports host platform (CFFI)
    if platforms is None:
        platforms = ['host']
    if 'host' not in platforms:
        print("Warning: regression_accuracy only supports host platform, skipping")
        return pd.DataFrame()

    all_results = []

    reg_datasets = get_regression_datasets()

    if datasets:
        reg_datasets = {k: v for k, v in reg_datasets.items() if k in datasets}

    for ds_name, (X_train, X_test, y_train, y_test, _) in reg_datasets.items():
        print(f"\nDataset: {ds_name}")
        print(f"  Features: {X_train.shape[1]}, Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")

        # Get sweep configs
        configs = get_sweep_configs(task='regression', quick=quick, dataset=ds_name)

        # Process configs
        def process_config(cfg):
            return _train_and_evaluate_regression(cfg, X_train, X_test, y_train, y_test)

        results = run_parallel(configs, process_config, n_jobs=n_jobs, verbose=0)
        results = [r for r in results if r is not None]
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Save results if run context provided
    if run_ctx is not None and len(df) > 0:
        run_ctx.save_results('regression_accuracy', df)

    # Print summary
    if len(df) > 0:
        print_regression_summary(df)

    return df


def print_regression_summary(df: pd.DataFrame) -> None:
    """Print regression benchmark summary.

    Args:
        df: DataFrame with regression results.
    """
    print("\n" + "=" * 70)
    print("Regression Accuracy Summary")
    print("=" * 70)

    for ds in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds]

        print(f"\n{ds}:")
        print(f"  {'Model':<10} {'Best R²':>10} {'Avg R²':>10} {'Best MSE':>12}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

        for model_type in ['gbt', 'rf']:
            mt_df = ds_df[ds_df['model_type'] == model_type]
            if len(mt_df) == 0:
                continue

            best_r2 = mt_df['r2_emlearn'].max()
            avg_r2 = mt_df['r2_emlearn'].mean()
            best_mse = mt_df['mse_emlearn'].min()

            print(f"  {model_type.upper():<10} {best_r2:>10.4f} {avg_r2:>10.4f} {best_mse:>12.4f}")

        # Compare
        gbt_df = ds_df[ds_df['model_type'] == 'gbt']
        rf_df = ds_df[ds_df['model_type'] == 'rf']

        if len(gbt_df) > 0 and len(rf_df) > 0:
            gbt_best = gbt_df['r2_emlearn'].max()
            rf_best = rf_df['r2_emlearn'].max()
            winner = "GBT" if gbt_best > rf_best else "RF"
            diff = abs(gbt_best - rf_best)
            print(f"\n  Winner: {winner} (R² diff: {diff:.4f})")
