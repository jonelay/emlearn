"""Tests for benchmark datasets module."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add examples to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'examples'))

from mcu_benchmark.datasets import (
    make_embedded_classification,
    make_additive_regression,
    get_classification_datasets,
    get_regression_datasets,
    load_wine_xy,
    load_digits_xy,
    load_diabetes_xy,
)


class TestSyntheticDatasets:
    def test_embedded_classification_shape(self):
        X, y = make_embedded_classification(n_samples=100, n_features=15)
        assert X.shape == (100, 15)
        assert y.shape == (100,)

    def test_embedded_classification_default_classes(self):
        X, y = make_embedded_classification(n_samples=100)
        assert len(np.unique(y)) == 3  # default n_classes

    def test_embedded_classification_float32(self):
        X, y = make_embedded_classification()
        assert X.dtype == np.float32

    def test_additive_regression_structure(self):
        """Verify y = sin(x0) + x1^2 + x2 + noise structure."""
        X, y = make_additive_regression(n_samples=100, noise=0.0)
        expected = np.sin(X[:, 0]) + X[:, 1] ** 2 + X[:, 2]
        np.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_additive_regression_float32(self):
        X, y = make_additive_regression()
        assert X.dtype == np.float32


class TestRealDatasets:
    def test_wine_shape(self):
        X, y = load_wine_xy()
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.dtype == np.float32

    def test_digits_shape(self):
        X, y = load_digits_xy()
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

    def test_diabetes_shape(self):
        X, y = load_diabetes_xy()
        assert X.ndim == 2
        assert y.ndim == 1


class TestDatasetGetters:
    def test_get_classification_datasets(self):
        datasets = get_classification_datasets()
        assert 'embedded_synth' in datasets
        assert 'wine' in datasets
        for name, (X_train, X_test, y_train, y_test, task_type) in datasets.items():
            assert X_train.ndim == 2
            assert y_train.ndim == 1
            assert X_train.shape[0] == y_train.shape[0]
            assert task_type == 'classification'

    def test_get_regression_datasets(self):
        datasets = get_regression_datasets()
        assert 'additive_synth' in datasets
        for name, (X_train, X_test, y_train, y_test, task_type) in datasets.items():
            assert X_train.ndim == 2
            assert y_train.ndim == 1
            assert task_type == 'regression'
