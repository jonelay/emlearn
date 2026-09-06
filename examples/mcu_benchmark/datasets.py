"""
Dataset Definitions for MCU Benchmarks
======================================

Provides classification and regression datasets for benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import (
    make_classification,
    load_iris,
    load_wine,
    load_digits,
    load_breast_cancer,
    fetch_california_housing,
    load_diabetes,
)

# Default random state for reproducibility
RANDOM_STATE = 42

# Data directory for caching external datasets
DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class DatasetInfo:
    """Metadata about a dataset.

    Attributes:
        name: Dataset identifier.
        task_type: Either 'classification' or 'regression'.
        n_samples: Number of samples.
        n_features: Number of features.
        n_classes: Number of classes (classification only).
        size_bytes: Estimated size in bytes (n_samples * n_features * 2 for int16).
    """

    name: str
    task_type: str
    n_samples: int
    n_features: int
    n_classes: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        if self.size_bytes == 0:
            # Estimate size assuming int16 features
            self.size_bytes = self.n_samples * self.n_features * 2


# =============================================================================
# Synthetic Datasets
# =============================================================================


def make_embedded_classification(
    n_samples: int = 500,
    n_features: int = 15,
    n_informative: int = 8,
    n_classes: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset for embedded ML.

    Realistic feature count for microcontroller applications.

    Args:
        n_samples: Number of samples.
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_classes: Number of classes.
        random_state: Random seed.

    Returns:
        Tuple of (X, y) arrays.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state,
    )
    X = X.astype('float32')

    return X, y


def make_additive_regression(
    n_samples: int = 1000,
    n_features: int = 5,
    noise: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset with additive structure.

    y = sin(x0) + x1^2 + x2 + noise

    GradientBoosting should excel on this type of data because it
    learns additive corrections sequentially.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features (first 3 are used for target).
        noise: Standard deviation of Gaussian noise.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) arrays.
    """
    rng = np.random.RandomState(random_state)

    X = rng.uniform(-2, 2, size=(n_samples, n_features)).astype('float32')

    # Additive structure: y = sin(x0) + x1^2 + x2
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + X[:, 2]
    y += rng.normal(0, noise, size=n_samples)
    y = y.astype('float32')

    return X, y


# =============================================================================
# Real Dataset Loaders
# =============================================================================


def load_sonar_xy(data_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load Sonar dataset as X, y arrays.

    Args:
        data_dir: Directory to cache dataset.

    Returns:
        Tuple of (X, y) where X is features and y is encoded labels.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    from emlearn.examples.datasets.sonar import load_sonar_dataset
    data = load_sonar_dataset(str(data_dir))
    feature_cols = [c for c in data.columns if c.startswith('b.')]

    X = data[feature_cols].values.astype('float32')
    y = LabelEncoder().fit_transform(data['label'].astype(str))

    return X, y


def load_wine_xy() -> tuple[np.ndarray, np.ndarray]:
    """Load Wine dataset as X, y arrays."""
    wine = load_wine()
    return wine.data.astype('float32'), wine.target


def load_digits_xy() -> tuple[np.ndarray, np.ndarray]:
    """Load Digits dataset as X, y arrays."""
    digits = load_digits()
    return digits.data.astype('float32'), digits.target


def load_california_xy(
    max_samples: int = 2000,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load California Housing dataset as X, y arrays (regression).

    Args:
        max_samples: Maximum samples to return (subsampled for speed).
        random_state: Random seed for subsampling.

    Returns:
        Tuple of (X, y) arrays.
    """
    cal = fetch_california_housing()
    X, y = cal.data.astype('float32'), cal.target.astype('float32')

    # Subsample for faster benchmarking
    if len(X) > max_samples:
        indices = np.random.RandomState(random_state).choice(
            len(X), size=max_samples, replace=False
        )
        X, y = X[indices], y[indices]

    return X, y


def load_diabetes_xy() -> tuple[np.ndarray, np.ndarray]:
    """Load Diabetes dataset as X, y arrays (regression)."""
    diab = load_diabetes()
    return diab.data.astype('float32'), diab.target.astype('float32')


def load_iris_xy() -> tuple[np.ndarray, np.ndarray]:
    """Load Iris dataset as X, y arrays."""
    iris = load_iris()
    return iris.data.astype('float32'), iris.target


def load_breast_cancer_xy() -> tuple[np.ndarray, np.ndarray]:
    """Load Breast Cancer dataset as X, y arrays."""
    bc = load_breast_cancer()
    return bc.data.astype('float32'), bc.target


# =============================================================================
# Dataset Collections
# =============================================================================


def get_classification_datasets(
    data_dir: str | Path | None = None,
    test_size: float = 0.33,
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load all classification benchmark datasets.

    Args:
        data_dir: Directory for caching datasets.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for train/test split.

    Returns:
        Dictionary mapping name -> (X_train, X_test, y_train, y_test, task_type).
    """
    if data_dir is None:
        data_dir = DATA_DIR

    datasets = {}

    # Synthetic: embedded classification
    X, y = make_embedded_classification(random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    datasets['embedded_synth'] = (X_train, X_test, y_train, y_test, 'classification')

    # Real: Sonar (60 features, binary)
    X, y = load_sonar_xy(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    datasets['sonar'] = (X_train, X_test, y_train, y_test, 'classification')

    # Real: Wine (13 features, 3-class)
    X, y = load_wine_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    datasets['wine'] = (X_train, X_test, y_train, y_test, 'classification')

    # Real: Iris (4 features, 3-class) - classic baseline
    X, y = load_iris_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    datasets['iris'] = (X_train, X_test, y_train, y_test, 'classification')

    # Real: Breast Cancer (30 features, binary)
    X, y = load_breast_cancer_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    datasets['breast_cancer'] = (X_train, X_test, y_train, y_test, 'classification')

    # Real: Digits (64 features, 10-class)
    X, y = load_digits_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    datasets['digits'] = (X_train, X_test, y_train, y_test, 'classification')

    return datasets


def get_regression_datasets(
    test_size: float = 0.33,
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load all regression benchmark datasets.

    Args:
        test_size: Fraction of data to use for testing.
        random_state: Random seed for train/test split.

    Returns:
        Dictionary mapping name -> (X_train, X_test, y_train, y_test, task_type).
    """
    datasets = {}

    # Synthetic: additive regression (GBT advantage expected)
    X, y = make_additive_regression(random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    datasets['additive_synth'] = (X_train, X_test, y_train, y_test, 'regression')

    # Real: California Housing
    X, y = load_california_xy(random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    datasets['california'] = (X_train, X_test, y_train, y_test, 'regression')

    # Real: Diabetes
    X, y = load_diabetes_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    datasets['diabetes'] = (X_train, X_test, y_train, y_test, 'regression')

    return datasets


def get_all_datasets(
    data_dir: str | Path | None = None,
    test_size: float = 0.33,
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load all benchmark datasets (classification + regression).

    Args:
        data_dir: Directory for caching datasets.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for train/test split.

    Returns:
        Dictionary mapping name -> (X_train, X_test, y_train, y_test, task_type).
    """
    datasets = get_classification_datasets(data_dir, test_size, random_state)
    datasets.update(get_regression_datasets(test_size, random_state))
    return datasets


def get_dataset_info(name: str) -> DatasetInfo:
    """Return dataset metadata.

    Args:
        name: Dataset name.

    Returns:
        DatasetInfo with metadata about the dataset.

    Raises:
        ValueError: If dataset name is unknown.
    """
    info_map = {
        'embedded_synth': DatasetInfo(
            name='embedded_synth',
            task_type='classification',
            n_samples=500,
            n_features=15,
            n_classes=3,
        ),
        'sonar': DatasetInfo(
            name='sonar',
            task_type='classification',
            n_samples=208,
            n_features=60,
            n_classes=2,
        ),
        'wine': DatasetInfo(
            name='wine',
            task_type='classification',
            n_samples=178,
            n_features=13,
            n_classes=3,
        ),
        'iris': DatasetInfo(
            name='iris',
            task_type='classification',
            n_samples=150,
            n_features=4,
            n_classes=3,
        ),
        'breast_cancer': DatasetInfo(
            name='breast_cancer',
            task_type='classification',
            n_samples=569,
            n_features=30,
            n_classes=2,
        ),
        'digits': DatasetInfo(
            name='digits',
            task_type='classification',
            n_samples=1797,
            n_features=64,
            n_classes=10,
        ),
        'additive_synth': DatasetInfo(
            name='additive_synth',
            task_type='regression',
            n_samples=1000,
            n_features=5,
        ),
        'california': DatasetInfo(
            name='california',
            task_type='regression',
            n_samples=2000,
            n_features=8,
        ),
        'diabetes': DatasetInfo(
            name='diabetes',
            task_type='regression',
            n_samples=442,
            n_features=10,
        ),
    }

    if name not in info_map:
        raise ValueError(f"Unknown dataset: {name}. Known: {list(info_map.keys())}")

    return info_map[name]


def get_quick_datasets(
    task: str = 'both',
    data_dir: str | Path | None = None,
    test_size: float = 0.33,
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Get reduced dataset set for quick testing.

    Args:
        task: 'classification', 'regression', or 'both'.
        data_dir: Directory for caching datasets.
        test_size: Fraction for testing.
        random_state: Random seed.

    Returns:
        Dictionary of datasets.
    """
    datasets = {}

    if task in ('classification', 'both'):
        # Only embedded_synth and digits for quick mode
        X, y = make_embedded_classification(random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        datasets['embedded_synth'] = (X_train, X_test, y_train, y_test, 'classification')

        X, y = load_digits_xy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        datasets['digits'] = (X_train, X_test, y_train, y_test, 'classification')

    if task in ('regression', 'both'):
        # Only additive_synth for quick mode
        X, y = make_additive_regression(random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        datasets['additive_synth'] = (X_train, X_test, y_train, y_test, 'regression')

    return datasets


# Classification dataset names
CLASSIFICATION_DATASETS = ['embedded_synth', 'sonar', 'wine', 'iris', 'breast_cancer', 'digits']

# Regression dataset names
REGRESSION_DATASETS = ['additive_synth', 'california', 'diabetes']

# Quick mode datasets
QUICK_CLASSIFICATION_DATASETS = ['embedded_synth', 'digits']
QUICK_REGRESSION_DATASETS = ['additive_synth']
