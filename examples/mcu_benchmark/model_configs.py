"""
Model Configuration Matrix for MCU Benchmarks
==============================================

Defines model configurations for GBT vs RF comparison across different sizes.
Supports comprehensive sweep matrices for benchmarking.

Also provides shared functions for model training and C header generation
to avoid code duplication across benchmark scripts.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

import emlearn

# =============================================================================
# Sweep Parameters
# =============================================================================

# Full sweep parameters (aligned with latency benchmark configs)
FULL_N_ESTIMATORS = [3, 10, 20, 40]
FULL_MAX_DEPTHS = [3, 5]
FULL_LEARNING_RATES = [0.1, 0.2, 0.5]  # GBT only

# Quick sweep parameters (for testing)
QUICK_N_ESTIMATORS = [2, 8]
QUICK_MAX_DEPTHS = [2, 4]
QUICK_LEARNING_RATES = [0.1]

# Extended sweep parameters for sample efficiency analysis
# Used with --extended flag for comprehensive n_estimators curves up to 100 trees
EXTENDED_N_ESTIMATORS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
EXTENDED_MAX_DEPTHS = [3, 5]  # Fixed for fair comparison
EXTENDED_LEARNING_RATES = [0.1]  # Fixed for fair comparison

# Compilation methods with dtype mappings
# 'inline' uses float features directly
# 'loadable' uses int16 features with scaling (smaller flash, slight accuracy loss)
METHODS = {
    'inline': {'dtype': 'float', 'scale': 1.0},
    'loadable': {'dtype': 'int16_t', 'scale': 1000.0},
}

# Default methods for sweeps
DEFAULT_METHODS = ['inline']


def prepare_data_for_method(
    X: NDArray,
    method: str,
    scale_factor: float | None = None,
) -> tuple[NDArray, str]:
    """Prepare feature data for a given compilation method.

    The 'loadable' method uses int16 features, so we scale and convert.
    The 'inline' method uses float features directly.

    Args:
        X: Input feature array (float).
        method: Compilation method ('inline' or 'loadable').
        scale_factor: Override scaling factor (default from METHODS).

    Returns:
        Tuple of (X_prepared, dtype_str) where dtype_str is 'float' or 'int16_t'.
    """
    method_info = METHODS.get(method, METHODS['inline'])
    dtype = method_info['dtype']
    scale = scale_factor if scale_factor is not None else method_info['scale']

    if method == 'loadable':
        # Scale features to int16 range (-32768 to 32767)
        X_scaled = (X * scale).astype(np.int16)
        return X_scaled, dtype
    else:
        # Inline method uses float features
        return X.astype(np.float32), dtype


def get_method_dtype(method: str) -> str:
    """Get the dtype string for a compilation method.

    Args:
        method: Compilation method ('inline' or 'loadable').

    Returns:
        Dtype string ('float' or 'int16_t').
    """
    return METHODS.get(method, METHODS['inline'])['dtype']

# Default number of features for benchmark models
DEFAULT_N_FEATURES = 10

# Default random state for reproducibility
RANDOM_STATE = 42

# Flash budget levels for comparison (bytes)
FLASH_BUDGETS = [2048, 4096, 8192, 16384, 32768, 65536]  # 2KB, 4KB, 8KB, 16KB, 32KB, 64KB

# Multiclass configurations for softmax scaling tests
# Binary uses sigmoid (1 expf call), multiclass uses softmax (n_classes expf calls)
MULTICLASS_CONFIGS = [
    {'n_classes': 2, 'expected_expf_calls': 1},   # Binary uses sigmoid
    {'n_classes': 3, 'expected_expf_calls': 3},   # Softmax
    {'n_classes': 5, 'expected_expf_calls': 5},
    {'n_classes': 10, 'expected_expf_calls': 10},
]


# =============================================================================
# Platform Configurations
# =============================================================================

PLATFORMS = {
    'host': {
        'name': 'host',
        'type': 'host',
        'description': 'Host machine (CFFI)',
        'has_fpu': True,
    },
    'native_sim': {
        'name': 'native_sim',
        'type': 'native',
        'description': 'Native sim (host FPU)',
        'has_fpu': True,
    },
    'renode_nrf52840': {
        'name': 'renode_nrf52840',
        'type': 'renode',
        'description': 'Renode nRF52840 (Cortex-M4F, DWT timing)',
        'has_fpu': True,
    },
    'nrf52dk_nrf52832': {
        'name': 'nrf52dk_nrf52832',
        'type': 'hardware',
        'description': 'Nordic nRF52 DK (Cortex-M4F, FPU)',
        'has_fpu': True,
    },
}

# Platform groups for filtering
HOST_PLATFORMS = ['host']
RENODE_PLATFORMS = ['renode_nrf52840']
NATIVE_PLATFORMS = ['native_sim']
HARDWARE_PLATFORMS = ['nrf52dk_nrf52832']
EMULATED_PLATFORMS = RENODE_PLATFORMS + NATIVE_PLATFORMS

# Quick mode platforms (no hardware)
QUICK_PLATFORMS = ['host', 'native_sim', 'renode_nrf52840']

# Full mode platforms (includes hardware)
FULL_PLATFORMS = ['host', 'native_sim', 'renode_nrf52840', 'nrf52dk_nrf52832']


# =============================================================================
# Legacy Configs (for backward compatibility)
# =============================================================================

MODEL_CONFIGS = [
    {'name': 'gbt_small', 'type': 'gbt', 'n_estimators': 5, 'max_depth': 3},
    {'name': 'rf_small', 'type': 'rf', 'n_estimators': 5, 'max_depth': 3},
    {'name': 'gbt_medium', 'type': 'gbt', 'n_estimators': 10, 'max_depth': 5},
    {'name': 'rf_medium', 'type': 'rf', 'n_estimators': 10, 'max_depth': 5},
    {'name': 'gbt_large', 'type': 'gbt', 'n_estimators': 20, 'max_depth': 5},
    {'name': 'rf_large', 'type': 'rf', 'n_estimators': 20, 'max_depth': 5},
]

QUICK_MODEL_CONFIGS = [
    {'name': 'gbt_small', 'type': 'gbt', 'n_estimators': 5, 'max_depth': 3},
    {'name': 'rf_small', 'type': 'rf', 'n_estimators': 5, 'max_depth': 3},
    {'name': 'gbt_medium', 'type': 'gbt', 'n_estimators': 10, 'max_depth': 5},
    {'name': 'rf_medium', 'type': 'rf', 'n_estimators': 10, 'max_depth': 5},
]


# =============================================================================
# Sweep Configuration Generation
# =============================================================================


def get_n_estimators_list(
    quick: bool = False,
    extended: bool = False,
) -> list[int]:
    """Get n_estimators list based on sweep mode.

    Args:
        quick: Use reduced parameters for quick testing.
        extended: Use extended parameters (1-100) for sample efficiency analysis.

    Returns:
        List of n_estimators values to sweep.
    """
    if extended:
        return EXTENDED_N_ESTIMATORS
    elif quick:
        return QUICK_N_ESTIMATORS
    return FULL_N_ESTIMATORS


def get_max_depths_list(
    quick: bool = False,
    extended: bool = False,
) -> list[int]:
    """Get max_depth list based on sweep mode.

    Args:
        quick: Use reduced parameters for quick testing.
        extended: Use extended parameters (fixed depths for fair comparison).

    Returns:
        List of max_depth values to sweep.
    """
    if extended:
        return EXTENDED_MAX_DEPTHS
    elif quick:
        return QUICK_MAX_DEPTHS
    return FULL_MAX_DEPTHS


def get_learning_rates_list(
    quick: bool = False,
    extended: bool = False,
) -> list[float]:
    """Get learning_rate list based on sweep mode.

    Args:
        quick: Use reduced parameters for quick testing.
        extended: Use extended parameters (fixed rate for fair comparison).

    Returns:
        List of learning_rate values to sweep.
    """
    if extended:
        return EXTENDED_LEARNING_RATES
    elif quick:
        return QUICK_LEARNING_RATES
    return FULL_LEARNING_RATES


def get_sweep_configs(
    task: Literal['classification', 'regression'] = 'classification',
    quick: bool = False,
    extended: bool = False,
    method: str = 'inline',
    dataset: str | None = None,
    platform: str = 'host',
) -> list[dict]:
    """Generate full sweep configuration matrix.

    Args:
        task: Task type ('classification' or 'regression').
        quick: Use reduced parameters for quick testing.
        extended: Use extended parameters (1-100 trees) for sample efficiency.
        method: Compilation method ('inline' or 'loadable').
        dataset: Specific dataset name (optional, added to all configs).
        platform: Platform identifier (added to all configs).

    Returns:
        List of model configuration dictionaries.
    """
    n_estimators_list = get_n_estimators_list(quick=quick, extended=extended)
    max_depths_list = get_max_depths_list(quick=quick, extended=extended)
    learning_rates_list = get_learning_rates_list(quick=quick, extended=extended)

    configs = []

    # GBT configurations (with learning_rate)
    for n_est, depth, lr in product(n_estimators_list, max_depths_list, learning_rates_list):
        config = {
            'name': f"gbt_n{n_est}_d{depth}_lr{int(lr * 10)}",
            'type': 'gbt',
            'n_estimators': n_est,
            'max_depth': depth,
            'learning_rate': lr,
            'method': method,
            'task': task,
            'platform': platform,
        }
        if dataset:
            config['dataset'] = dataset
        configs.append(config)

    # RF configurations (no learning_rate)
    for n_est, depth in product(n_estimators_list, max_depths_list):
        config = {
            'name': f"rf_n{n_est}_d{depth}",
            'type': 'rf',
            'n_estimators': n_est,
            'max_depth': depth,
            'learning_rate': None,
            'method': method,
            'task': task,
            'platform': platform,
        }
        if dataset:
            config['dataset'] = dataset
        configs.append(config)

    return configs


def get_sweep_configs_for_dataset(
    dataset: str,
    task: Literal['classification', 'regression'],
    quick: bool = False,
    extended: bool = False,
    method: str = 'inline',
) -> list[dict]:
    """Generate sweep configs for a specific dataset.

    Args:
        dataset: Dataset name.
        task: Task type.
        quick: Use reduced parameters.
        extended: Use extended parameters (1-100 trees).
        method: Compilation method.

    Returns:
        List of configs with dataset field set.
    """
    return get_sweep_configs(
        task=task, quick=quick, extended=extended, method=method, dataset=dataset
    )


def get_all_sweep_configs(
    task: Literal['classification', 'regression', 'both'] = 'both',
    datasets: list[str] | None = None,
    quick: bool = False,
    extended: bool = False,
    method: str = 'inline',
) -> list[dict]:
    """Generate sweep configs for multiple datasets.

    Args:
        task: Task type ('classification', 'regression', or 'both').
        datasets: List of dataset names. If None, uses defaults.
        quick: Use reduced parameters.
        extended: Use extended parameters (1-100 trees).
        method: Compilation method.

    Returns:
        List of all configs across datasets.
    """
    from .datasets import (
        CLASSIFICATION_DATASETS,
        REGRESSION_DATASETS,
        QUICK_CLASSIFICATION_DATASETS,
        QUICK_REGRESSION_DATASETS,
    )

    all_configs = []

    if task in ('classification', 'both'):
        if datasets is None:
            ds_list = QUICK_CLASSIFICATION_DATASETS if quick else CLASSIFICATION_DATASETS
        else:
            ds_list = [d for d in datasets if d in CLASSIFICATION_DATASETS]

        for dataset in ds_list:
            configs = get_sweep_configs_for_dataset(
                dataset=dataset,
                task='classification',
                quick=quick,
                extended=extended,
                method=method,
            )
            all_configs.extend(configs)

    if task in ('regression', 'both'):
        if datasets is None:
            ds_list = QUICK_REGRESSION_DATASETS if quick else REGRESSION_DATASETS
        else:
            ds_list = [d for d in datasets if d in REGRESSION_DATASETS]

        for dataset in ds_list:
            configs = get_sweep_configs_for_dataset(
                dataset=dataset,
                task='regression',
                quick=quick,
                extended=extended,
                method=method,
            )
            all_configs.extend(configs)

    return all_configs


# =============================================================================
# Helper Functions
# =============================================================================


def get_model_configs(quick: bool = False) -> list[dict]:
    """Get model configurations for benchmark.

    Args:
        quick: If True, return reduced set for quick testing.

    Returns:
        List of model configuration dictionaries.
    """
    configs = QUICK_MODEL_CONFIGS.copy() if quick else MODEL_CONFIGS.copy()

    # Add default n_features to all configs
    for config in configs:
        if 'n_features' not in config:
            config['n_features'] = DEFAULT_N_FEATURES

    return configs


def get_platforms(
    quick: bool = False,
    host_only: bool = False,
    renode_only: bool = False,
    hardware_only: bool = False,
    no_host: bool = False,
) -> list[str]:
    """Get platforms for benchmark.

    Args:
        quick: If True, return platforms for quick testing (no hardware).
        host_only: Only return host platform.
        renode_only: Only return Renode emulator platforms.
        hardware_only: Only return hardware platforms.
        no_host: Exclude host platform from results.

    Returns:
        List of platform names.
    """
    if host_only:
        return HOST_PLATFORMS
    if renode_only:
        return RENODE_PLATFORMS
    if hardware_only:
        return HARDWARE_PLATFORMS

    platforms = QUICK_PLATFORMS if quick else FULL_PLATFORMS

    if no_host:
        platforms = [p for p in platforms if p not in HOST_PLATFORMS]

    return platforms


def get_platform_type(platform: str) -> str:
    """Get platform type from platform name.

    Args:
        platform: Platform identifier.

    Returns:
        Platform type: 'host', 'renode', 'native', or 'hardware'.
    """
    return PLATFORMS.get(platform, {}).get('type', 'unknown')


def has_fpu(platform: str) -> bool:
    """Check if platform has FPU.

    Args:
        platform: Platform name.

    Returns:
        True if platform has FPU.
    """
    return PLATFORMS.get(platform, {}).get('has_fpu', False)


def get_model_id(config: dict) -> str:
    """Generate model identifier string.

    Args:
        config: Model configuration dictionary.

    Returns:
        Model identifier like 'gbt_n10_d5' or 'gbt_n10_d5_lr1'.
    """
    base = f"{config['type']}_n{config['n_estimators']}_d{config['max_depth']}"
    if config.get('learning_rate') is not None:
        base += f"_lr{int(config['learning_rate'] * 10)}"
    return base


def get_model_description(config: dict) -> str:
    """Generate human-readable model description.

    Args:
        config: Model configuration dictionary.

    Returns:
        Description like 'GBT (10 trees, depth 5, lr=0.1)'.
    """
    type_name = 'GBT' if config['type'] == 'gbt' else 'RF'
    desc = f"{type_name} ({config['n_estimators']} trees, depth {config['max_depth']}"
    if config.get('learning_rate') is not None:
        desc += f", lr={config['learning_rate']}"
    if config.get('n_classes') is not None and config['n_classes'] > 2:
        desc += f", {config['n_classes']} classes"
    desc += ")"
    return desc


def get_multiclass_configs(
    base_config: dict | None = None,
    n_classes_list: list[int] | None = None,
) -> list[dict]:
    """Generate model configs for multiclass softmax scaling tests.

    Creates GBT configs with varying n_classes to measure softmax overhead
    scaling. Binary (n_classes=2) uses sigmoid (1 expf), multiclass uses
    softmax (n_classes expf calls).

    Args:
        base_config: Base model config to extend. If None, uses small GBT.
        n_classes_list: List of n_classes values. If None, uses [2, 3, 5, 10].

    Returns:
        List of model configs with n_classes field set.
    """
    if base_config is None:
        base_config = {
            'type': 'gbt',
            'n_estimators': 5,
            'max_depth': 3,
            'learning_rate': 0.1,
        }

    if n_classes_list is None:
        n_classes_list = [mc['n_classes'] for mc in MULTICLASS_CONFIGS]

    configs = []
    for n_classes in n_classes_list:
        config = base_config.copy()
        config['n_classes'] = n_classes
        config['name'] = f"{get_model_id(base_config)}_c{n_classes}"
        configs.append(config)

    return configs


def count_sweep_configs(
    task: str = 'both',
    quick: bool = False,
    extended: bool = False,
) -> dict:
    """Count configurations in sweep matrix.

    Args:
        task: Task type.
        quick: Use quick parameters.
        extended: Use extended parameters (1-100 trees).

    Returns:
        Dictionary with counts by model type and total.
    """
    n_est = len(get_n_estimators_list(quick=quick, extended=extended))
    depths = len(get_max_depths_list(quick=quick, extended=extended))
    lrs = len(get_learning_rates_list(quick=quick, extended=extended))

    gbt_per_dataset = n_est * depths * lrs
    rf_per_dataset = n_est * depths

    from .datasets import (
        CLASSIFICATION_DATASETS,
        REGRESSION_DATASETS,
        QUICK_CLASSIFICATION_DATASETS,
        QUICK_REGRESSION_DATASETS,
    )

    if quick:
        n_class_ds = len(QUICK_CLASSIFICATION_DATASETS)
        n_reg_ds = len(QUICK_REGRESSION_DATASETS)
    else:
        n_class_ds = len(CLASSIFICATION_DATASETS)
        n_reg_ds = len(REGRESSION_DATASETS)

    if task == 'classification':
        n_datasets = n_class_ds
    elif task == 'regression':
        n_datasets = n_reg_ds
    else:
        n_datasets = n_class_ds + n_reg_ds

    return {
        'gbt_per_dataset': gbt_per_dataset,
        'rf_per_dataset': rf_per_dataset,
        'total_per_dataset': gbt_per_dataset + rf_per_dataset,
        'n_datasets': n_datasets,
        'total': (gbt_per_dataset + rf_per_dataset) * n_datasets,
    }


# =============================================================================
# Shared Model Training Functions
# =============================================================================


def train_model(
    config: dict,
    X: NDArray,
    y: NDArray,
    random_state: int = RANDOM_STATE,
    n_classes: int | None = None,
) -> Any:
    """Train a model from configuration.

    Shared function to avoid duplicate training code across benchmark scripts.

    Args:
        config: Model configuration with 'type', 'n_estimators', 'max_depth',
               and optionally 'learning_rate' for GBT and 'n_classes' for multiclass.
        X: Feature matrix for training.
        y: Target labels/values for training.
        random_state: Random seed for reproducibility.
        n_classes: Optional override for number of classes (for multiclass tests).
                   If provided, y is remapped to have n_classes unique values.

    Returns:
        Trained sklearn estimator (GradientBoostingClassifier or RandomForestClassifier).
    """
    model_type = config['type']
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']

    # Handle n_classes from config or parameter
    target_n_classes = n_classes or config.get('n_classes')
    if target_n_classes is not None and target_n_classes > 2:
        # Remap y to have target_n_classes unique values
        y = y % target_n_classes

    if model_type == 'gbt':
        learning_rate = config.get('learning_rate', 0.1)
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    elif model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    clf.fit(X, y)
    return clf


def make_benchmark_dataset(
    n_samples: int = 100,
    n_features: int = DEFAULT_N_FEATURES,
    n_classes: int = 2,
    random_state: int = RANDOM_STATE,
    test_size: float = 0.2,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Create a synthetic dataset for benchmarking.

    Shared function for consistent dataset generation across benchmarks.

    Args:
        n_samples: Total number of samples to generate.
        n_features: Number of features.
        n_classes: Number of classes (for classification).
        random_state: Random seed for reproducibility.
        test_size: Fraction of data for test set.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_classes=n_classes,
        random_state=random_state,
    )

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# =============================================================================
# Shared Header Generation Functions
# =============================================================================


def generate_model_header(cmodel: Any, name: str = 'benchmark_model') -> str:
    """Generate C header for emlearn model.

    Shared function to ensure consistent model header generation.

    Args:
        cmodel: Converted emlearn model (from emlearn.convert()).
        name: Model name for C identifiers.

    Returns:
        C header source code as string.
    """
    return cmodel.save(name=name)


def generate_testdata_header(
    X: NDArray,
    n_samples: int = 10,
    name: str = 'benchmark_model',
    dtype: str = 'float32',
    n_classes: int | None = None,
) -> str:
    """Generate C header with test data array.

    Shared function to ensure consistent test data header generation.
    Uses float32 by default for compatibility with emlearn C API.

    Args:
        X: Feature matrix (samples × features).
        n_samples: Number of samples to include in header.
        name: Name prefix for C identifiers.
        dtype: Data type ('float32' or 'int16').
        n_classes: Number of classes for classifiers. When given, emits
            ``<NAME>_N_CLASSES`` so the benchmark app can size the
            predict_proba output buffer exactly.

    Returns:
        C header source code as string.
    """
    n_samples = min(n_samples, len(X))
    n_features = X.shape[1]

    # Normalize name for C identifiers
    upper_name = name.upper()

    lines = [
        f"/* Auto-generated test data for {name} */",
        f"#ifndef {upper_name}_TESTDATA_H",
        f"#define {upper_name}_TESTDATA_H",
        "",
        f"#define {upper_name}_N_SAMPLES {n_samples}",
        f"#define {upper_name}_N_FEATURES {n_features}",
    ]
    if n_classes:
        lines.append(f"#define {upper_name}_N_CLASSES {n_classes}")
    lines.append("")

    if dtype == 'float32':
        c_type = 'float'
        format_spec = '.6f'
    elif dtype == 'int16':
        c_type = 'int16_t'
        format_spec = 'd'
        lines.insert(3, "#include <stdint.h>")
        lines.insert(4, "")
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Use volatile to prevent compiler from evaluating predictions at compile time
    # (with -O3 and const data, the compiler can constant-fold entire model inference)
    lines.append(
        f"static volatile const {c_type} {name}_test_data[{upper_name}_N_SAMPLES][{upper_name}_N_FEATURES] = {{"
    )

    for i in range(n_samples):
        row = X[i]
        if dtype == 'int16':
            # Scale and convert to int16 (assumes normalized data)
            row = (row * 1000).astype(np.int16)
        values = ", ".join(f"{v:{format_spec}}" for v in row)
        comma = "," if i < n_samples - 1 else ""
        lines.append(f"    {{ {values} }}{comma}")

    lines.extend([
        "};",
        "",
        f"#endif /* {upper_name}_TESTDATA_H */",
        "",
    ])

    return "\n".join(lines)


def write_benchmark_headers(
    cmodel: Any,
    X: NDArray,
    output_dir: Path,
    name: str = 'benchmark_model',
    n_test_samples: int = 10,
) -> tuple[Path, Path]:
    """Write both model and test data headers to directory.

    Convenience function that generates and writes both headers.

    Args:
        cmodel: Converted emlearn model.
        X: Feature matrix for test data.
        output_dir: Directory to write headers to.
        name: Name prefix for C identifiers.
        n_test_samples: Number of test samples in header.

    Returns:
        Tuple of (model_header_path, testdata_header_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_header = output_dir / f'{name}.h'
    model_header.write_text(generate_model_header(cmodel, name))

    testdata_header = output_dir / f'{name}_testdata.h'
    testdata_header.write_text(generate_testdata_header(
        X, n_test_samples, name, n_classes=getattr(cmodel, 'n_classes', 0) or None,
    ))

    return model_header, testdata_header
