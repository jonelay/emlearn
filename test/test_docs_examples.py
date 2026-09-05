"""
Test that code examples from documentation actually work.

Validates Python examples from docs/tree_based_models.rst
to prevent documentation drift.
"""

import numpy
import pytest
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import emlearn


class TestDocsGradientBoosting:
    """Tests for Gradient Boosting documentation examples."""

    def test_binary_classification_example(self, tmp_path):
        """Validates tree_based_models.rst lines 91-100.

        Example:
            from sklearn.ensemble import GradientBoostingClassifier
            import emlearn

            clf = GradientBoostingClassifier(n_estimators=10, max_depth=3)
            clf.fit(X_train, y_train)

            cmodel = emlearn.convert(clf, method='inline')
            cmodel.save(file='gbc_model.h')
        """
        X_train, y_train = datasets.make_classification(
            n_samples=100, n_features=10, random_state=42
        )

        clf = GradientBoostingClassifier(n_estimators=10, max_depth=3)
        clf.fit(X_train, y_train)

        cmodel = emlearn.convert(clf, method='inline')
        file_path = tmp_path / 'gbc_model.h'
        cmodel.save(file=str(file_path))

        # Verify file created with expected content
        assert file_path.exists()
        content = file_path.read_text()
        assert 'gbc_model_predict' in content
        assert 'gbc_model_predict_proba' in content

    def test_multiclass_classification_example(self, tmp_path):
        """Validates tree_based_models.rst lines 118-126.

        Example:
            from sklearn.ensemble import GradientBoostingClassifier

            clf = GradientBoostingClassifier(n_estimators=10, max_depth=3)
            clf.fit(X_train, y_train)  # y_train has >2 classes

            cmodel = emlearn.convert(clf, method='inline')
            cmodel.save(file='gbc_multiclass.h')
        """
        X_train, y_train = datasets.make_classification(
            n_samples=150, n_features=10, n_classes=3,
            n_informative=5, n_redundant=0, random_state=42
        )

        clf = GradientBoostingClassifier(n_estimators=10, max_depth=3)
        clf.fit(X_train, y_train)

        cmodel = emlearn.convert(clf, method='inline')
        file_path = tmp_path / 'gbc_multiclass.h'
        cmodel.save(file=str(file_path))

        assert file_path.exists()
        content = file_path.read_text()
        assert 'gbc_multiclass_predict' in content

    def test_regression_example(self, tmp_path):
        """Validates tree_based_models.rst lines 133-142.

        Example:
            from sklearn.ensemble import GradientBoostingRegressor
            import emlearn

            reg = GradientBoostingRegressor(n_estimators=10, max_depth=3)
            reg.fit(X_train, y_train)

            cmodel = emlearn.convert(reg, method='inline')
            cmodel.save(file='gbr_model.h')
        """
        X_train, y_train = datasets.make_regression(
            n_samples=100, n_features=10, random_state=42
        )

        reg = GradientBoostingRegressor(n_estimators=10, max_depth=3)
        reg.fit(X_train, y_train)

        cmodel = emlearn.convert(reg, method='inline')
        file_path = tmp_path / 'gbr_model.h'
        cmodel.save(file=str(file_path))

        assert file_path.exists()
        content = file_path.read_text()
        assert 'gbr_model_predict' in content

    def test_inference_methods_example(self):
        """Validates tree_based_models.rst lines 158-164.

        Example:
            # Inline method (default) - works with float features
            cmodel = emlearn.convert(clf, method='inline')

            # Loadable method - requires int16 features
            cmodel = emlearn.convert(clf, method='loadable', dtype='int16_t')
        """
        X_train, y_train = datasets.make_classification(
            n_samples=100, n_features=10, random_state=42
        )

        # Train on float data for inline
        clf_inline = GradientBoostingClassifier(n_estimators=5, random_state=42)
        clf_inline.fit(X_train, y_train)
        cmodel_inline = emlearn.convert(clf_inline, method='inline')
        assert cmodel_inline is not None

        # Train on int16 data for loadable
        X_int16 = (X_train * 1000).astype(numpy.int16)
        clf_loadable = GradientBoostingClassifier(n_estimators=5, random_state=42)
        clf_loadable.fit(X_int16, y_train)
        cmodel_loadable = emlearn.convert(clf_loadable, method='loadable', dtype='int16_t')
        assert cmodel_loadable is not None

    def test_prediction_matches_sklearn(self):
        """Ensure converted model predictions match sklearn."""
        X, y = datasets.make_classification(
            n_samples=100, n_features=10, random_state=42
        )

        clf = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X, y)

        cmodel = emlearn.convert(clf, method='inline')

        # Predictions should match
        pred_sklearn = clf.predict(X[:20])
        pred_emlearn = cmodel.predict(X[:20])
        numpy.testing.assert_equal(pred_emlearn, pred_sklearn)

        # Probabilities should be close
        proba_sklearn = clf.predict_proba(X[:20])
        proba_emlearn = cmodel.predict_proba(X[:20])
        numpy.testing.assert_allclose(proba_emlearn, proba_sklearn, rtol=0.02, atol=0.02)

    @pytest.mark.parametrize("n_features", [100])
    def test_high_feature_count_prediction_count(self, n_features):
        """Regression test: GBT with many features."""
        X, y = datasets.make_classification(
            n_samples=50, n_features=n_features,
            n_informative=min(20, n_features // 2),
            n_classes=2, random_state=42
        )

        clf = GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42)
        clf.fit(X, y)

        cmodel = emlearn.convert(clf, method='inline')

        X_test = X[:10]
        pred_emlearn = cmodel.predict(X_test)
        assert len(pred_emlearn) == len(X_test), (
            f"Prediction count mismatch: got {len(pred_emlearn)}, expected {len(X_test)}."
        )

        pred_sklearn = clf.predict(X_test)
        numpy.testing.assert_equal(pred_emlearn, pred_sklearn)
