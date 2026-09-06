"""
Pareto Efficiency Benchmark for MCU Platforms
==============================================

Find accuracy vs flash size Pareto frontiers.

Key finding: GBT dominates at larger flash budgets (>4KB),
RF competitive at tiny budgets (<2KB).
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
from sklearn.metrics import accuracy_score, mean_squared_error

import emlearn

from ..common import RunContext, create_benchmark_result, measure_host_latency
from ..checkpointing import config_hash
from ..mcu_runner import measure_mcu_timing, is_emulator_platform
from ..process_manager import run_parallel
from ..datasets import get_classification_datasets, get_regression_datasets
from ..model_configs import get_sweep_configs

# Default random state
RANDOM_STATE = 42

# Model type mappings
CLASSIFIER_TYPES = {
    'gbt': GradientBoostingClassifier,
    'rf': RandomForestClassifier,
}

REGRESSOR_TYPES = {
    'gbt': GradientBoostingRegressor,
    'rf': RandomForestRegressor,
}


def _train_and_measure(
    config: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    measure_latency: bool = True,
) -> dict | None:
    """Train model and measure accuracy, flash size, and latency.

    Args:
        config: Model configuration.
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        measure_latency: Whether to measure inference latency.

    Returns:
        Result dictionary or None on failure.
    """
    task_type = config['task']
    model_type = config['type']
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    learning_rate = config.get('learning_rate')
    dataset = config.get('dataset', 'unknown')
    platform = config.get('platform', 'host')

    # Select model class
    if task_type == 'classification':
        model_cls = CLASSIFIER_TYPES[model_type]
    else:
        model_cls = REGRESSOR_TYPES[model_type]

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

        # Evaluate
        y_pred = cmodel.predict(X_test)

        if task_type == 'classification':
            score = accuracy_score(y_test, y_pred)
            score_type = 'accuracy'
        else:
            score = 1.0 - mean_squared_error(y_test, y_pred) / np.var(y_test)  # R²
            score_type = 'r2'

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
            task=task_type,
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            score=score,
            score_type=score_type,
            flash_bytes=flash_bytes,
            latency_ns=latency_ns,
            config_hash=config_hash(config),
        )

    except Exception as e:
        print(f"Error processing {config['name']}: {e}")
        return None


def find_pareto_frontier(df: pd.DataFrame, score_col: str = 'score') -> pd.DataFrame:
    """Find Pareto frontier points in score vs flash size space.

    A point is on the Pareto frontier if no other point has both
    higher score AND smaller flash size.

    Args:
        df: DataFrame with score and 'flash_bytes' columns.
        score_col: Name of the score column (default 'score').

    Returns:
        DataFrame with 'is_pareto' column added.
    """
    df = df.copy()
    df['is_pareto'] = False

    for idx, row in df.iterrows():
        # Check if any other point dominates this one
        dominated = False
        for other_idx, other_row in df.iterrows():
            if idx == other_idx:
                continue

            # Other point dominates if it has >= score AND <= flash
            # with at least one strict inequality
            if (other_row[score_col] >= row[score_col] and
                other_row['flash_bytes'] <= row['flash_bytes'] and
                (other_row[score_col] > row[score_col] or
                 other_row['flash_bytes'] < row['flash_bytes'])):
                dominated = True
                break

        df.loc[idx, 'is_pareto'] = not dominated

    return df


def run_pareto_benchmark(
    platforms: list[str] | None = None,
    datasets: list[str] | None = None,
    task: str = 'both',
    quick: bool = False,
    n_jobs: int = -1,
    run_ctx: RunContext | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run Pareto efficiency benchmark.

    Args:
        platforms: Platforms to benchmark (host only for this benchmark).
        datasets: Specific datasets to benchmark.
        task: Task type ('classification', 'regression', or 'both').
        quick: Use reduced parameters.
        n_jobs: Number of parallel jobs.
        run_ctx: Run context for artifact management.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        DataFrame with Pareto analysis results.
    """
    print("\n" + "=" * 70)
    print("Pareto Efficiency Benchmark")
    print("=" * 70)
    print("Finding accuracy vs flash size Pareto frontiers")

    # This benchmark only supports host platform (CFFI)
    if platforms is None:
        platforms = ['host']
    if 'host' not in platforms:
        print("Warning: pareto_efficiency only supports host platform, skipping")
        return pd.DataFrame()

    all_results = []

    # Classification
    if task in ('classification', 'both'):
        print("\n--- Classification ---")
        class_datasets = get_classification_datasets()

        if datasets:
            class_datasets = {k: v for k, v in class_datasets.items() if k in datasets}

        for ds_name, (X_train, X_test, y_train, y_test, _) in class_datasets.items():
            print(f"\nDataset: {ds_name}")

            # Get sweep configs
            configs = get_sweep_configs(task='classification', quick=quick, dataset=ds_name)

            # Process configs
            def process_config(cfg):
                return _train_and_measure(cfg, X_train, X_test, y_train, y_test)

            results = run_parallel(configs, process_config, n_jobs=n_jobs, verbose=0)
            results = [r for r in results if r is not None]
            all_results.extend(results)

    # Regression
    if task in ('regression', 'both'):
        print("\n--- Regression ---")
        reg_datasets = get_regression_datasets()

        if datasets:
            reg_datasets = {k: v for k, v in reg_datasets.items() if k in datasets}

        for ds_name, (X_train, X_test, y_train, y_test, _) in reg_datasets.items():
            print(f"\nDataset: {ds_name}")

            # Get sweep configs
            configs = get_sweep_configs(task='regression', quick=quick, dataset=ds_name)

            # Process configs
            def process_config(cfg):
                return _train_and_measure(cfg, X_train, X_test, y_train, y_test)

            results = run_parallel(configs, process_config, n_jobs=n_jobs, verbose=0)
            results = [r for r in results if r is not None]
            all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Find Pareto frontiers per dataset and model type
    if len(df) > 0:
        pareto_dfs = []
        for ds in df['dataset'].unique():
            for model_type in df['model_type'].unique():
                subset = df[(df['dataset'] == ds) & (df['model_type'] == model_type)]
                if len(subset) > 0:
                    subset_pareto = find_pareto_frontier(subset)
                    pareto_dfs.append(subset_pareto)

        df = pd.concat(pareto_dfs, ignore_index=True)

    # Save results if run context provided
    if run_ctx is not None and len(df) > 0:
        run_ctx.save_results('pareto_efficiency', df)

    # Print summary
    if len(df) > 0:
        print_pareto_summary(df)

    return df


def print_pareto_summary(df: pd.DataFrame) -> None:
    """Print Pareto efficiency summary.

    Args:
        df: DataFrame with Pareto analysis.
    """
    print("\n" + "=" * 70)
    print("Pareto Efficiency Summary")
    print("=" * 70)

    for ds in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds]

        print(f"\n{ds}:")
        print(f"  {'Model':<10} {'Pareto Pts':>12} {'Best Score':>10} {'Min Flash':>12}")
        print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*12}")

        for model_type in ['gbt', 'rf']:
            mt_df = ds_df[ds_df['model_type'] == model_type]
            if len(mt_df) == 0:
                continue

            pareto_count = mt_df['is_pareto'].sum()
            best_score = mt_df['score'].max()
            min_flash = mt_df['flash_bytes'].min()

            print(f"  {model_type.upper():<10} {pareto_count:>12} {best_score:>10.4f} {min_flash:>10}B")

        # Compare Pareto frontiers
        gbt_pareto = ds_df[(ds_df['model_type'] == 'gbt') & (ds_df['is_pareto'])]
        rf_pareto = ds_df[(ds_df['model_type'] == 'rf') & (ds_df['is_pareto'])]

        if len(gbt_pareto) > 0 and len(rf_pareto) > 0:
            gbt_best = gbt_pareto['score'].max()
            rf_best = rf_pareto['score'].max()
            winner = "GBT" if gbt_best > rf_best else "RF"
            print(f"\n  Best Pareto score: {winner}")
