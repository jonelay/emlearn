"""
Sample Efficiency Benchmark for MCU Platforms
==============================================

Measures how many trees are needed to reach target accuracy levels.

Key finding: GBT reaches target accuracy with fewer trees because each
tree specifically corrects errors from previous trees (boosting),
rather than independently voting (bagging).
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

from ..common import RunContext, create_benchmark_result, measure_host_latency, run_with_timeout
from ..checkpointing import config_hash
from ..mcu_runner import measure_mcu_timing, is_emulator_platform
from ..process_manager import run_parallel
from ..datasets import get_classification_datasets, get_regression_datasets, get_quick_datasets
from ..model_configs import (
    get_n_estimators_list,
    get_max_depths_list,
    get_learning_rates_list,
    RANDOM_STATE,
)

try:
    from joblib import delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# Model type mappings
CLASSIFIER_TYPES = {
    'gbt': GradientBoostingClassifier,
    'rf': RandomForestClassifier,
}

REGRESSOR_TYPES = {
    'gbt': GradientBoostingRegressor,
    'rf': RandomForestRegressor,
}

# Default random state
RANDOM_STATE = 42


def _train_and_evaluate(
    config: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    measure_latency: bool = True,
) -> dict | None:
    """Train model and evaluate performance.

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
            score = -mean_squared_error(y_test, y_pred)  # Negative for consistency
            score_type = 'neg_mse'

        # Get flash size estimate
        try:
            code = cmodel.save(name='model')
            flash_bytes = len(code.encode('utf-8'))  # Rough estimate
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


def run_sample_efficiency_benchmark(
    platforms: list[str] | None = None,
    datasets: list[str] | None = None,
    task: str = 'both',
    quick: bool = False,
    extended: bool = False,
    test_timeout: float | None = None,
    n_jobs: int = -1,
    run_ctx: RunContext | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run sample efficiency benchmark.

    Measures how many trees are needed to reach target accuracy thresholds.

    Args:
        platforms: Platforms to benchmark (host only for this benchmark).
        datasets: Specific datasets to benchmark.
        task: Task type ('classification', 'regression', or 'both').
        quick: Use reduced parameters for quick testing.
        extended: Use extended parameters (1-100 trees) for sample efficiency.
        test_timeout: Max time per test in seconds (None for no limit).
        n_jobs: Number of parallel jobs (-1 for all cores).
        run_ctx: Run context for artifact management.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        DataFrame with sample efficiency results.
    """
    print("\n" + "=" * 70)
    print("Sample Efficiency Benchmark")
    print("=" * 70)
    print("Finding n_estimators needed to reach target accuracy")

    # This benchmark only supports host platform (CFFI)
    if platforms is None:
        platforms = ['host']
    if 'host' not in platforms:
        print("Warning: sample_efficiency only supports host platform, skipping")
        return pd.DataFrame()

    # Get n_estimators range based on mode
    n_estimators_list = get_n_estimators_list(quick=quick, extended=extended)

    # Get depth and learning rate based on mode
    # For extended mode, use fixed values for fair comparison
    # For quick/full mode, use the first value from the list
    max_depths = get_max_depths_list(quick=quick, extended=extended)
    learning_rates = get_learning_rates_list(quick=quick, extended=extended)
    max_depth = max_depths[0] if len(max_depths) == 1 else 5  # Use first if single, else 5
    learning_rate = learning_rates[0] if learning_rates else 0.1

    if extended:
        print(f"Extended mode: n_estimators up to {max(n_estimators_list)}, "
              f"depth={max_depth}, lr={learning_rate}")

    if test_timeout:
        print(f"Test timeout: {test_timeout}s per config")

    all_results = []
    timeout_count = 0

    # Classification
    if task in ('classification', 'both'):
        print("\n--- Classification ---")
        if quick:
            class_datasets = get_quick_datasets(task='classification')
        else:
            class_datasets = get_classification_datasets()

        if datasets:
            class_datasets = {k: v for k, v in class_datasets.items() if k in datasets}

        for ds_name, (X_train, X_test, y_train, y_test, _) in class_datasets.items():
            print(f"\nDataset: {ds_name}")
            print(f"  Features: {X_train.shape[1]}, Train: {len(X_train)}, Test: {len(X_test)}")

            # Build configs for this dataset
            configs = []
            for n_est in n_estimators_list:
                # GBT config
                configs.append({
                    'name': f"gbt_n{n_est}_d{max_depth}",
                    'type': 'gbt',
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'task': 'classification',
                    'dataset': ds_name,
                    'platform': 'host',
                })
                # RF config
                configs.append({
                    'name': f"rf_n{n_est}_d{max_depth}",
                    'type': 'rf',
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'learning_rate': None,
                    'task': 'classification',
                    'dataset': ds_name,
                    'platform': 'host',
                })

            # Process configs with optional timeout
            def process_config(cfg):
                if test_timeout:
                    result, timed_out = run_with_timeout(
                        lambda: _train_and_evaluate(cfg, X_train, X_test, y_train, y_test),
                        test_timeout,
                    )
                    if timed_out:
                        # Return partial result with timeout marker
                        return create_benchmark_result(
                            platform=cfg.get('platform', 'host'),
                            dataset=cfg.get('dataset', 'unknown'),
                            task=cfg['task'],
                            model_type=cfg['type'],
                            n_estimators=cfg['n_estimators'],
                            max_depth=cfg['max_depth'],
                            learning_rate=cfg.get('learning_rate'),
                            score=float('nan'),
                            score_type='timeout',
                            flash_bytes=0,
                            latency_ns=0.0,
                            timeout=True,
                            config_hash=config_hash(cfg),
                        )
                    return result
                return _train_and_evaluate(cfg, X_train, X_test, y_train, y_test)

            results = run_parallel(configs, process_config, n_jobs=n_jobs, verbose=0)
            results = [r for r in results if r is not None]
            timeout_count += sum(1 for r in results if r.get('timeout', False))
            all_results.extend(results)

    # Regression
    if task in ('regression', 'both'):
        print("\n--- Regression ---")
        if quick:
            reg_datasets = get_quick_datasets(task='regression')
        else:
            reg_datasets = get_regression_datasets()

        if datasets:
            reg_datasets = {k: v for k, v in reg_datasets.items() if k in datasets}

        for ds_name, (X_train, X_test, y_train, y_test, _) in reg_datasets.items():
            print(f"\nDataset: {ds_name}")
            print(f"  Features: {X_train.shape[1]}, Train: {len(X_train)}, Test: {len(X_test)}")

            # Build configs for this dataset
            configs = []
            for n_est in n_estimators_list:
                # GBT config
                configs.append({
                    'name': f"gbt_n{n_est}_d{max_depth}",
                    'type': 'gbt',
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'task': 'regression',
                    'dataset': ds_name,
                    'platform': 'host',
                })
                # RF config
                configs.append({
                    'name': f"rf_n{n_est}_d{max_depth}",
                    'type': 'rf',
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'learning_rate': None,
                    'task': 'regression',
                    'dataset': ds_name,
                    'platform': 'host',
                })

            # Process configs with optional timeout
            def process_config(cfg):
                if test_timeout:
                    result, timed_out = run_with_timeout(
                        lambda: _train_and_evaluate(cfg, X_train, X_test, y_train, y_test),
                        test_timeout,
                    )
                    if timed_out:
                        # Return partial result with timeout marker
                        return create_benchmark_result(
                            platform=cfg.get('platform', 'host'),
                            dataset=cfg.get('dataset', 'unknown'),
                            task=cfg['task'],
                            model_type=cfg['type'],
                            n_estimators=cfg['n_estimators'],
                            max_depth=cfg['max_depth'],
                            learning_rate=cfg.get('learning_rate'),
                            score=float('nan'),
                            score_type='timeout',
                            flash_bytes=0,
                            latency_ns=0.0,
                            timeout=True,
                            config_hash=config_hash(cfg),
                        )
                    return result
                return _train_and_evaluate(cfg, X_train, X_test, y_train, y_test)

            results = run_parallel(configs, process_config, n_jobs=n_jobs, verbose=0)
            results = [r for r in results if r is not None]
            timeout_count += sum(1 for r in results if r.get('timeout', False))
            all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Report timeout stats
    if timeout_count > 0:
        print(f"\n  Tests timed out: {timeout_count}")

    # Save results if run context provided
    if run_ctx is not None and len(df) > 0:
        run_ctx.save_results('sample_efficiency', df)

    # Print summary
    if len(df) > 0:
        print_sample_efficiency_summary(df)

    return df


def find_threshold_n_estimators(
    df: pd.DataFrame,
    threshold_pct: float = 0.95,
) -> pd.DataFrame:
    """Find minimum n_estimators to reach threshold of best score.

    Args:
        df: Benchmark results DataFrame.
        threshold_pct: Percentage of best score to target (0.95 = 95%).

    Returns:
        DataFrame with threshold analysis.
    """
    threshold_results = []

    for ds in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds]

        for model_type in ['gbt', 'rf']:
            mt_df = ds_df[ds_df['model_type'] == model_type].sort_values('n_estimators')

            if len(mt_df) == 0:
                continue

            best_score = mt_df['score'].max()
            threshold_score = best_score * threshold_pct

            # Find first n_estimators that meets threshold
            meets_threshold = mt_df[mt_df['score'] >= threshold_score]

            if len(meets_threshold) > 0:
                first_meeting = meets_threshold.iloc[0]
                n_est_threshold = first_meeting['n_estimators']
                score_at_threshold = first_meeting['score']
            else:
                n_est_threshold = None
                score_at_threshold = None

            threshold_results.append({
                'dataset': ds,
                'model_type': model_type,
                'threshold_pct': threshold_pct,
                'best_score': best_score,
                'threshold_score': threshold_score,
                'n_estimators_at_threshold': n_est_threshold,
                'score_at_threshold': score_at_threshold,
            })

    return pd.DataFrame(threshold_results)


def print_sample_efficiency_summary(df: pd.DataFrame) -> None:
    """Print sample efficiency analysis summary.

    Args:
        df: Benchmark results DataFrame.
    """
    print("\n" + "=" * 70)
    print("Sample Efficiency Summary")
    print("=" * 70)

    # Threshold analysis
    threshold_95 = find_threshold_n_estimators(df, threshold_pct=0.95)

    print(f"\n  n_estimators to reach 95% of best score:")
    print(f"  {'Dataset':<20} {'GBT':>10} {'RF':>10} {'Winner':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    gbt_wins = 0
    rf_wins = 0

    for ds in threshold_95['dataset'].unique():
        gbt_row = threshold_95[(threshold_95['dataset'] == ds) & (threshold_95['model_type'] == 'gbt')]
        rf_row = threshold_95[(threshold_95['dataset'] == ds) & (threshold_95['model_type'] == 'rf')]

        gbt_n = gbt_row['n_estimators_at_threshold'].values[0] if len(gbt_row) > 0 else None
        rf_n = rf_row['n_estimators_at_threshold'].values[0] if len(rf_row) > 0 else None

        # NaN values from the DataFrame pass `is not None` but fail int()
        if gbt_n is not None and pd.isna(gbt_n):
            gbt_n = None
        if rf_n is not None and pd.isna(rf_n):
            rf_n = None

        gbt_str = str(int(gbt_n)) if gbt_n is not None else "N/A"
        rf_str = str(int(rf_n)) if rf_n is not None else "N/A"

        if gbt_n is not None and rf_n is not None:
            if gbt_n < rf_n:
                winner = "GBT"
                gbt_wins += 1
            elif rf_n < gbt_n:
                winner = "RF"
                rf_wins += 1
            else:
                winner = "TIE"
        else:
            winner = "-"

        print(f"  {ds:<20} {gbt_str:>10} {rf_str:>10} {winner:>10}")

    print(f"\n  Wins: GBT={gbt_wins}, RF={rf_wins}")
