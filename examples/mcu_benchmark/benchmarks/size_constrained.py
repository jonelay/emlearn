"""
Size Constrained Benchmark for MCU Platforms
=============================================

Find best accuracy at fixed flash budgets.
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

from ..common import RunContext, create_benchmark_result, measure_host_latency
from ..checkpointing import config_hash
from ..mcu_runner import measure_mcu_timing, is_emulator_platform
from ..process_manager import run_parallel
from ..datasets import get_classification_datasets, get_regression_datasets, get_quick_datasets
from ..model_configs import get_sweep_configs, FLASH_BUDGETS

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


def _train_and_measure_size(
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
            score = r2_score(y_test, y_pred)
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


def find_best_at_budget(
    df: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    """Find best model at a given flash budget.

    Args:
        df: DataFrame with benchmark results.
        budget: Flash budget in bytes.

    Returns:
        DataFrame with best models per dataset/model_type.
    """
    # Filter to models within budget
    if df.empty or 'flash_bytes' not in df.columns:
        return pd.DataFrame()
    within_budget = df[df['flash_bytes'] <= budget]

    if len(within_budget) == 0:
        return pd.DataFrame()

    best_models = []

    for ds in within_budget['dataset'].unique():
        ds_df = within_budget[within_budget['dataset'] == ds]

        for model_type in ds_df['model_type'].unique():
            mt_df = ds_df[ds_df['model_type'] == model_type]

            if len(mt_df) == 0:
                continue

            # Find best score within budget
            best_idx = mt_df['score'].idxmax()
            best_row = mt_df.loc[best_idx].to_dict()
            best_row['budget'] = budget
            best_models.append(best_row)

    return pd.DataFrame(best_models)


def run_size_constrained_benchmark(
    platforms: list[str] | None = None,
    datasets: list[str] | None = None,
    task: str = 'both',
    flash_budgets: list[int] | None = None,
    quick: bool = False,
    n_jobs: int = -1,
    run_ctx: RunContext | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run size constrained benchmark.

    Args:
        platforms: Platforms to benchmark (host only for this benchmark).
        datasets: Specific datasets to benchmark.
        task: Task type ('classification', 'regression', or 'both').
        flash_budgets: List of flash budgets to test.
        quick: Use reduced parameters.
        n_jobs: Number of parallel jobs.
        run_ctx: Run context for artifact management.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        DataFrame with size-constrained results.
    """
    print("\n" + "=" * 70)
    print("Size Constrained Benchmark")
    print("=" * 70)
    print("Finding best accuracy at fixed flash budgets")

    # This benchmark only supports host platform (CFFI)
    if platforms is None:
        platforms = ['host']
    if 'host' not in platforms:
        print("Warning: size_constrained only supports host platform, skipping")
        return pd.DataFrame()

    if flash_budgets is None:
        flash_budgets = FLASH_BUDGETS

    all_results = []

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

            # Get sweep configs
            configs = get_sweep_configs(task='classification', quick=quick, dataset=ds_name)

            # Process configs
            def process_config(cfg):
                return _train_and_measure_size(cfg, X_train, X_test, y_train, y_test)

            results = run_parallel(configs, process_config, n_jobs=n_jobs, verbose=0)
            results = [r for r in results if r is not None]
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

            # Get sweep configs
            configs = get_sweep_configs(task='regression', quick=quick, dataset=ds_name)

            # Process configs
            def process_config(cfg):
                return _train_and_measure_size(cfg, X_train, X_test, y_train, y_test)

            results = run_parallel(configs, process_config, n_jobs=n_jobs, verbose=0)
            results = [r for r in results if r is not None]
            all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Save raw results
    if run_ctx is not None and len(df) > 0:
        run_ctx.save_results('size_constrained_raw', df)

    # Find best at each budget
    budget_results = []
    for budget in flash_budgets:
        budget_df = find_best_at_budget(df, budget)
        if len(budget_df) > 0:
            budget_results.append(budget_df)

    if budget_results:
        budget_df = pd.concat(budget_results, ignore_index=True)

        # Save budget analysis
        if run_ctx is not None:
            run_ctx.save_results('size_constrained', budget_df)

        # Print summary
        print_size_constrained_summary(budget_df, flash_budgets)

        return budget_df

    return df


def print_size_constrained_summary(df: pd.DataFrame, budgets: list[int]) -> None:
    """Print size constrained benchmark summary.

    Args:
        df: DataFrame with budget analysis.
        budgets: List of flash budgets tested.
    """
    print("\n" + "=" * 70)
    print("Size Constrained Summary")
    print("=" * 70)

    for ds in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds]

        print(f"\n{ds}:")

        # Header
        budget_headers = " ".join([f"{b//1024}KB" for b in budgets])
        print(f"  {'Model':<10} {budget_headers}")
        print(f"  {'-'*10} " + " ".join(["-" * 4 for _ in budgets]))

        for model_type in ['gbt', 'rf']:
            mt_df = ds_df[ds_df['model_type'] == model_type]

            scores = []
            for budget in budgets:
                budget_row = mt_df[mt_df['budget'] == budget]
                if len(budget_row) > 0:
                    score = budget_row['score'].values[0]
                    scores.append(f"{score:.2f}")
                else:
                    scores.append("-")

            print(f"  {model_type.upper():<10} " + " ".join([f"{s:>4}" for s in scores]))

        # Compare at each budget
        print("\n  Winners by budget:")
        for budget in budgets:
            gbt_row = ds_df[(ds_df['model_type'] == 'gbt') & (ds_df['budget'] == budget)]
            rf_row = ds_df[(ds_df['model_type'] == 'rf') & (ds_df['budget'] == budget)]

            if len(gbt_row) > 0 and len(rf_row) > 0:
                gbt_score = gbt_row['score'].values[0]
                rf_score = rf_row['score'].values[0]
                winner = "GBT" if gbt_score > rf_score else "RF"
                diff = abs(gbt_score - rf_score)
                print(f"    {budget//1024}KB: {winner} (+{diff:.4f})")
            elif len(gbt_row) > 0:
                print(f"    {budget//1024}KB: GBT only")
            elif len(rf_row) > 0:
                print(f"    {budget//1024}KB: RF only")
            else:
                print(f"    {budget//1024}KB: No models fit")
