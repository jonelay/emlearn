"""
Probability Calibration Benchmark for MCU Platforms
====================================================

Measure probability calibration quality (Brier score, ECE).

Classification only - requires predict_proba support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, brier_score_loss

import emlearn

from ..common import RunContext, create_benchmark_result, measure_host_latency
from ..checkpointing import config_hash
from ..mcu_runner import measure_mcu_timing, is_emulator_platform
from ..process_manager import run_parallel
from ..datasets import get_classification_datasets
from ..model_configs import get_sweep_configs
from ..metrics import expected_calibration_error

# Default random state
RANDOM_STATE = 42

def _train_and_evaluate_calibration(
    config: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    measure_latency: bool = True,
) -> dict | None:
    """Train model and evaluate calibration.

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
    model_type = config['type']
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    learning_rate = config.get('learning_rate')
    dataset = config.get('dataset', 'unknown')
    platform = config.get('platform', 'host')

    model_cls = GradientBoostingClassifier if model_type == 'gbt' else RandomForestClassifier

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

        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # For binary classification, use positive class probability
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            y_prob_pos = y_prob[:, 1]
            brier = brier_score_loss(y_test, y_prob_pos)
        else:
            # Multi-class Brier score
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
            brier = np.mean((y_prob - y_test_bin) ** 2)

        # ECE
        ece = expected_calibration_error(y_test, y_prob)

        # Convert to emlearn for flash size and latency (float features for accuracy)
        cmodel = emlearn.convert(model, method='inline', dtype='float')

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
            task='classification',
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            score=accuracy,
            score_type='accuracy',
            flash_bytes=flash_bytes,
            latency_ns=latency_ns,
            config_hash=config_hash(config),
            # Calibration-specific extra fields
            brier_score=brier,
            ece=ece,
            n_classes=n_classes,
        )

    except Exception as e:
        print(f"Error processing {config['name']}: {e}")
        return None


def run_calibration_benchmark(
    platforms: list[str] | None = None,
    datasets: list[str] | None = None,
    quick: bool = False,
    n_jobs: int = -1,
    run_ctx: RunContext | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run probability calibration benchmark.

    Args:
        platforms: Platforms to benchmark (host only for this benchmark).
        datasets: Specific datasets to benchmark.
        quick: Use reduced parameters.
        n_jobs: Number of parallel jobs.
        run_ctx: Run context for artifact management.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        DataFrame with calibration results.
    """
    print("\n" + "=" * 70)
    print("Probability Calibration Benchmark")
    print("=" * 70)
    print("Measuring Brier score and ECE")

    # This benchmark only supports host platform (CFFI)
    if platforms is None:
        platforms = ['host']
    if 'host' not in platforms:
        print("Warning: probability_calibration only supports host platform, skipping")
        return pd.DataFrame()

    all_results = []

    class_datasets = get_classification_datasets()

    if datasets:
        class_datasets = {k: v for k, v in class_datasets.items() if k in datasets}

    for ds_name, (X_train, X_test, y_train, y_test, _) in class_datasets.items():
        n_classes = len(np.unique(y_train))
        print(f"\nDataset: {ds_name} ({n_classes} classes)")
        print(f"  Features: {X_train.shape[1]}, Train: {len(X_train)}, Test: {len(X_test)}")

        configs = get_sweep_configs(task='classification', quick=quick, dataset=ds_name)

        # Process configs
        def process_config(cfg):
            return _train_and_evaluate_calibration(cfg, X_train, X_test, y_train, y_test)

        results = run_parallel(configs, process_config, n_jobs=n_jobs, verbose=0)
        results = [r for r in results if r is not None]
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Save results if run context provided
    if run_ctx is not None and len(df) > 0:
        run_ctx.save_results('probability_calibration', df)

    # Print summary
    if len(df) > 0:
        print_calibration_summary(df)

    return df


def print_calibration_summary(df: pd.DataFrame) -> None:
    """Print calibration benchmark summary.

    Args:
        df: DataFrame with calibration results.
    """
    print("\n" + "=" * 70)
    print("Probability Calibration Summary")
    print("=" * 70)

    for ds in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds]

        print(f"\n{ds}:")
        print(f"  {'Model':<10} {'Best Brier':>12} {'Best ECE':>10} {'Avg ECE':>10}")
        print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

        for model_type in ['gbt', 'rf']:
            mt_df = ds_df[ds_df['model_type'] == model_type]
            if len(mt_df) == 0:
                continue

            best_brier = mt_df['brier_score'].min()
            best_ece = mt_df['ece'].min()
            avg_ece = mt_df['ece'].mean()

            label = {'gbt': 'GBT', 'rf': 'RF'}[model_type]
            print(f"  {label:<10} {best_brier:>12.4f} {best_ece:>10.4f} {avg_ece:>10.4f}")

        # Compare (lower is better for calibration)
        gbt_df = ds_df[ds_df['model_type'] == 'gbt']
        rf_df = ds_df[ds_df['model_type'] == 'rf']

        if len(gbt_df) > 0 and len(rf_df) > 0:
            gbt_best = gbt_df['brier_score'].min()
            rf_best = rf_df['brier_score'].min()
            winner = "GBT" if gbt_best < rf_best else "RF"
            print(f"\n  Best calibration (Brier): {winner}")
