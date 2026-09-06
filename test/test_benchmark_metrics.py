"""Tests for benchmark utilities used by the GBT MCU benchmarks.

Covers metrics and dataset helpers from examples/mcu_benchmark, plus
predict_proba API contract tests. Moved out of the core GBT tests
because they depend on benchmark infrastructure.
"""

import numpy
import numpy.testing
import pytest
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier

import emlearn


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error() function."""

    @pytest.fixture
    def ece_function(self):
        """Import ECE function from mcu_benchmark metrics."""
        import sys
        sys.path.insert(0, 'examples')
        from mcu_benchmark.metrics import expected_calibration_error
        return expected_calibration_error

    def test_perfect_calibration_binary(self, ece_function):
        """Perfectly calibrated predictions should have ECE near 0.

        Perfect calibration: confidence matches empirical accuracy.
        E.g., 80% confidence (y_prob=0.8) predictions are correct 80% of time.
        """
        # 10 samples at 0.8 confidence predicting class 1: 8 correct, 2 wrong
        # 10 samples at 0.6 confidence predicting class 1: 6 correct, 4 wrong
        y_true = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0,  # 0.8 bin: 8/10 correct
                              1, 1, 1, 1, 1, 1, 0, 0, 0, 0]) # 0.6 bin: 6/10 correct
        y_prob = numpy.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                              0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        ece = ece_function(y_true, y_prob, n_bins=5)
        assert ece < 0.05, f"Perfect calibration should have ECE near 0, got {ece}"

    def test_overconfident_predictions(self, ece_function):
        """Overconfident predictions should have higher ECE."""
        # All samples at 0.9 confidence but only 50% accuracy
        y_true = numpy.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = numpy.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        ece = ece_function(y_true, y_prob, n_bins=5)
        # Confidence is 0.9, accuracy is 0.5, so ECE should be around 0.4
        assert ece > 0.3, f"Overconfident predictions should have high ECE, got {ece}"

    def test_multiclass_ece(self, ece_function):
        """Test ECE with multiclass probability matrix."""
        y_true = numpy.array([0, 1, 2, 0, 1, 2])
        # Confidently correct predictions
        y_prob = numpy.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        ece = ece_function(y_true, y_prob, n_bins=5)
        # All correct with 0.8 confidence gives some calibration error
        assert ece <= 0.21, f"ECE should be reasonable, got {ece}"

    def test_ece_range(self, ece_function):
        """ECE should always be between 0 and 1."""
        rng = numpy.random.RandomState(42)
        for _ in range(10):
            n = rng.randint(10, 100)
            y_true = rng.randint(0, 2, size=n)
            y_prob = rng.rand(n)
            ece = ece_function(y_true, y_prob)
            assert 0 <= ece <= 1, f"ECE should be in [0,1], got {ece}"


class TestBrierScore:
    """Tests for brier_score (using sklearn.metrics.brier_score_loss)."""

    @pytest.fixture
    def brier_function(self):
        """Import Brier score function from sklearn."""
        from sklearn.metrics import brier_score_loss
        return brier_score_loss

    def test_perfect_predictions(self, brier_function):
        """Perfect predictions should have Brier score of 0."""
        y_true = numpy.array([0, 0, 1, 1])
        y_prob = numpy.array([0.0, 0.0, 1.0, 1.0])
        score = brier_function(y_true, y_prob)
        numpy.testing.assert_allclose(score, 0.0, atol=1e-10)

    def test_worst_predictions(self, brier_function):
        """Completely wrong predictions should have Brier score of 1."""
        y_true = numpy.array([0, 0, 1, 1])
        y_prob = numpy.array([1.0, 1.0, 0.0, 0.0])
        score = brier_function(y_true, y_prob)
        numpy.testing.assert_allclose(score, 1.0, atol=1e-10)

    def test_uncertain_predictions(self, brier_function):
        """Uncertain (0.5) predictions should have Brier score of 0.25."""
        y_true = numpy.array([0, 1])
        y_prob = numpy.array([0.5, 0.5])
        score = brier_function(y_true, y_prob)
        numpy.testing.assert_allclose(score, 0.25, atol=1e-10)

    def test_brier_range(self, brier_function):
        """Brier score should always be between 0 and 1."""
        rng = numpy.random.RandomState(42)
        for _ in range(10):
            n = rng.randint(10, 100)
            y_true = rng.randint(0, 2, size=n)
            y_prob = rng.rand(n)
            score = brier_function(y_true, y_prob)
            assert 0 <= score <= 1, f"Brier score should be in [0,1], got {score}"


class TestMakeAdditiveRegression:
    """Tests for make_additive_regression() function."""

    @pytest.fixture
    def make_additive_function(self):
        """Import make_additive_regression from mcu_benchmark datasets."""
        import sys
        sys.path.insert(0, 'examples')
        from mcu_benchmark.datasets import make_additive_regression
        return make_additive_regression

    def test_output_shape(self, make_additive_function):
        """Output should have correct shapes."""
        X, y = make_additive_function(n_samples=100, n_features=5)
        assert X.shape == (100, 5)
        assert y.shape == (100,)

    def test_output_dtype(self, make_additive_function):
        """Output should be float32."""
        X, y = make_additive_function()
        assert X.dtype == numpy.float32
        assert y.dtype == numpy.float32

    def test_reproducibility(self, make_additive_function):
        """Same random_state should produce identical results."""
        X1, y1 = make_additive_function(random_state=42)
        X2, y2 = make_additive_function(random_state=42)
        numpy.testing.assert_array_equal(X1, X2)
        numpy.testing.assert_array_equal(y1, y2)

    def test_different_seeds(self, make_additive_function):
        """Different random_state should produce different results."""
        X1, y1 = make_additive_function(random_state=42)
        X2, y2 = make_additive_function(random_state=123)
        assert not numpy.array_equal(X1, X2)
        assert not numpy.array_equal(y1, y2)

    def test_additive_structure(self, make_additive_function):
        """Target should follow additive structure: y = sin(x0) + x1^2 + x2 + noise."""
        X, y = make_additive_function(n_samples=1000, noise=0.0, random_state=42)

        # With zero noise, y should exactly equal sin(x0) + x1^2 + x2
        y_expected = numpy.sin(X[:, 0]) + X[:, 1] ** 2 + X[:, 2]
        numpy.testing.assert_allclose(y, y_expected, rtol=1e-5)

    def test_feature_range(self, make_additive_function):
        """Features should be in [-2, 2] range."""
        X, y = make_additive_function(n_samples=1000)
        assert X.min() >= -2.0
        assert X.max() <= 2.0


# ============================================================================
# predict_proba Shape and Normalization Tests


class TestPredictProbaContract:
    """Tests for the predict_proba API contract on host.

    These tests verify output shape, normalization and range for the
    standard sigmoid/softmax implementations.
    """

    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    def test_proba_rows_sum_to_one_standard(self, n_classes):
        """predict_proba rows should sum to 1.0 with standard sigmoid."""
        X, y = datasets.make_classification(
            n_classes=n_classes,
            n_informative=max(n_classes, 2),
            n_samples=max(n_classes * 20, 100),
            n_features=max(n_classes, 10),
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )

        clf = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        clf.fit(X, y)
        cmodel = emlearn.convert(clf, method='inline')

        proba = cmodel.predict_proba(X[:20])
        row_sums = proba.sum(axis=1)

        numpy.testing.assert_allclose(
            row_sums, 1.0, rtol=1e-5,
            err_msg=f"Probability rows should sum to 1.0, got sums: {row_sums}"
        )

    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    def test_proba_values_valid_range_standard(self, n_classes):
        """predict_proba values should be in [0, 1] with standard sigmoid."""
        X, y = datasets.make_classification(
            n_classes=n_classes,
            n_informative=max(n_classes, 2),
            n_samples=max(n_classes * 20, 100),
            n_features=max(n_classes, 10),
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )

        clf = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        clf.fit(X, y)
        cmodel = emlearn.convert(clf, method='inline')

        proba = cmodel.predict_proba(X[:20])

        assert proba.min() >= 0.0, f"Min probability should be >= 0, got {proba.min()}"
        assert proba.max() <= 1.0, f"Max probability should be <= 1, got {proba.max()}"

    def test_brier_score_baseline(self):
        """Establish baseline Brier score for standard sigmoid."""
        from sklearn.metrics import brier_score_loss

        X, y = datasets.make_classification(
            n_samples=500, n_features=10, random_state=42
        )

        clf = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X, y)
        cmodel = emlearn.convert(clf, method='inline')

        proba = cmodel.predict_proba(X)[:, 1]  # Probability of class 1
        brier = brier_score_loss(y, proba)

        # Document baseline - actual Brier score depends on dataset/model
        # This is just a sanity check that calibration is reasonable
        assert brier < 0.25, f"Brier score {brier:.3f} indicates poor calibration"
