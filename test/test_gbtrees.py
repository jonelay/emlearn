
"""Tests for gradient boosting conversion."""

import re

import numpy
from numpy.testing import assert_allclose, assert_equal
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import datasets

import emlearn

RANDOM_SEED = 42

CLASSIFICATION_DATASETS = {
    'binary': datasets.make_classification(n_classes=2, n_samples=100, random_state=RANDOM_SEED),
    '5way': datasets.make_classification(n_classes=5, n_informative=5, n_samples=100, random_state=RANDOM_SEED),
}

REGRESSION_DATASETS = {
    '1out': datasets.make_regression(n_targets=1, n_samples=100, random_state=RANDOM_SEED),
}

MULTICLASS_DATASETS = {
    '3way': datasets.make_classification(n_classes=3, n_informative=3, n_redundant=0, n_samples=100, random_state=RANDOM_SEED),
}

METHODS = ['loadable', 'inline']


def build_classifier(n_classes=2, n_estimators=3, random_state=1, **kwargs):
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=4, n_redundant=0,
        n_classes=n_classes, random_state=random_state,
    )
    estimator = GradientBoostingClassifier(
        n_estimators=n_estimators, random_state=random_state, **kwargs)
    estimator.fit(X, y)
    return estimator, X, y


def prepare_data_for_method(X, method):
    """Prepare features for the given compilation method.

    loadable requires int16_t features: scale by 1000 (3 decimal places of
    precision, within int16 bounds for typical normalized feature ranges).
    """
    if method == 'loadable':
        X_scaled = (X * 1000).astype(numpy.int16)
        return X_scaled, 'int16_t'
    else:
        return X, 'float'


# --- Initial scores ---

def test_binary_default_init_log_odds():
    estimator, X, y = build_classifier(n_classes=2)
    cmodel = emlearn.convert(estimator)

    prior = estimator.init_.class_prior_
    expected = float(numpy.log(prior[1] / prior[0]))
    assert cmodel.initial_value == pytest.approx(expected)


def test_multiclass_default_init_log_priors():
    estimator, X, y = build_classifier(n_classes=3)
    cmodel = emlearn.convert(estimator)

    prior = estimator.init_.class_prior_
    expected = [float(numpy.log(p)) for p in prior]
    assert_allclose(cmodel.initial_value, expected)


def test_regression_default_init_constant():
    X, y = make_regression(n_samples=100, n_features=5, random_state=1)
    estimator = GradientBoostingRegressor(n_estimators=3, random_state=1)
    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator)

    expected = float(estimator.init_.constant_[0, 0])
    assert cmodel.initial_value == pytest.approx(expected)


def test_custom_init_rejected():
    from sklearn.tree import DecisionTreeClassifier
    estimator, X, y = build_classifier(n_classes=2,
        init=DecisionTreeClassifier(max_depth=1))

    with pytest.raises(ValueError, match='custom init'):
        emlearn.convert(estimator)


def test_zero_init_binary():
    estimator, X, y = build_classifier(n_classes=2, init='zero')
    cmodel = emlearn.convert(estimator)

    assert cmodel.initial_value == 0.0
    assert_equal(cmodel.predict(X[:20]), estimator.predict(X[:20]))


def test_zero_init_multiclass():
    estimator, X, y = build_classifier(n_classes=3, init='zero')
    cmodel = emlearn.convert(estimator)

    assert cmodel.initial_value == [0.0, 0.0, 0.0]
    assert_equal(cmodel.predict(X[:20]), estimator.predict(X[:20]))


def test_zero_init_regression():
    X, y = make_regression(n_samples=100, n_features=5, random_state=1)
    estimator = GradientBoostingRegressor(
        n_estimators=3, init='zero', random_state=1)
    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator)

    assert cmodel.initial_value == 0.0
    assert_allclose(cmodel.predict(X[:20]), estimator.predict(X[:20]),
        rtol=1e-3, atol=1e-3)


# --- Multiclass tree ordering ---

def test_inline_codegen_class_tree_indices():
    n_classes = 3
    n_estimators = 2
    estimator, X, y = build_classifier(
        n_classes=n_classes, n_estimators=n_estimators)
    code = emlearn.convert(estimator).save(name='mygbt')

    for k in range(n_classes):
        pattern = r'scores\[{k}\] = .*'.format(k=k)
        line = re.search(pattern, code)
        assert line is not None, code
        called = re.findall(r'mygbt_tree_(\d+)\(', line.group(0))
        expected = [str(i * n_classes + k) for i in range(n_estimators)]
        assert called == expected, line.group(0)


@pytest.mark.parametrize('method', ['inline', 'loadable'])
def test_multiclass_proba_parity(method):
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, random_state=1,
    )
    X_method, dtype = prepare_data_for_method(X, method)
    estimator = GradientBoostingClassifier(n_estimators=5, random_state=1)
    estimator.fit(X_method, y)
    cmodel = emlearn.convert(estimator, method=method, dtype=dtype)

    proba = cmodel.predict_proba(X_method[:20])
    expected = estimator.predict_proba(X_method[:20])
    assert_allclose(proba, expected, rtol=1e-5, atol=1e-5)


# --- Generated code symbols ---

def test_inline_classifier_symbols():
    estimator, X, y = build_classifier(n_classes=2)
    code = emlearn.convert(estimator, method='inline').save(name='mygbt')

    assert '#include <stdint.h>' in code
    assert '#include <math.h>' in code
    assert 'float mygbt_raw_score(' in code
    assert 'int32_t mygbt_predict(' in code
    assert 'int mygbt_predict_proba(' in code
    assert 'float mygbt_predict_proba_class1(' in code


def test_inline_regressor_symbols():
    X, y = make_regression(n_samples=100, n_features=5, random_state=1)
    estimator = GradientBoostingRegressor(n_estimators=3, random_state=1)
    estimator.fit(X, y)
    code = emlearn.convert(estimator, method='inline').save(name='mygbt')

    assert 'float mygbt_predict(' in code
    assert 'predict_proba' not in code


def test_loadable_symbols_and_include():
    estimator, X, y = build_classifier(n_classes=3)
    code = emlearn.convert(estimator, method='loadable').save(name='mygbt')

    assert '#include <eml_gbtrees.h>' in code
    assert 'EmlGradientBoosting mygbt =' in code
    assert 'static const int32_t mygbt_tree_roots[' in code
    assert 'mygbt_nodes' in code
    assert 'mygbt_leaves[' in code
    assert 'static const float mygbt_initial_values[' in code
    assert 'eml_gbtrees_predict(' in code
    assert 'eml_gbtrees_predict_proba(' in code


def test_loadable_regressor_symbols():
    X, y = make_regression(n_samples=100, n_features=5, random_state=1)
    estimator = GradientBoostingRegressor(n_estimators=3, random_state=1)
    estimator.fit(X, y)
    code = emlearn.convert(estimator, method='loadable').save(name='mygbt')

    assert 'eml_gbtrees_regress(' in code


# --- Float leaf serialization ---

def test_bytelist_roundtrip():
    from emlearn.trees import leaves_to_bytelist

    leaves = [0.5, -1.25, 3.75e-3, 1e10, -0.0]
    data = leaves_to_bytelist(leaves, leaf_bits=32)
    assert len(data) == 4 * len(leaves)

    decoded = numpy.frombuffer(bytes(data), dtype=numpy.float32)
    assert_allclose(decoded, numpy.array(leaves, dtype=numpy.float32),
        rtol=0, atol=0)


def test_loadable_end_to_end_regression_parity():
    X, y = make_regression(n_samples=100, n_features=5, random_state=1)
    Xi = (X * 100).astype(int)
    estimator = GradientBoostingRegressor(n_estimators=5, random_state=1)
    estimator.fit(Xi, y)
    cmodel = emlearn.convert(estimator, method='loadable')

    out = cmodel.predict(Xi[:20])
    expected = estimator.predict(Xi[:20])
    assert_allclose(out, expected, rtol=1e-4)


# --- Class label remapping ---

def test_noncontiguous_labels():
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=4, n_redundant=0,
        n_classes=2, random_state=1)
    y = numpy.where(y == 0, 3, 7)

    estimator = GradientBoostingClassifier(n_estimators=3, random_state=1)
    estimator.fit(X, y)
    assert_equal(estimator.classes_, [3, 7])

    cmodel = emlearn.convert(estimator)
    out = numpy.array(cmodel.predict(X[:20]))
    sklearn_labels = estimator.predict(X[:20])

    assert set(numpy.unique(out)) <= {3, 7}
    assert_equal(out, sklearn_labels)


# --- Feature validation ---

def test_too_many_features():
    estimator, X, y = build_classifier()
    cmodel = emlearn.convert(estimator)
    X_extra = numpy.hstack([X, X[:, :1]])
    with pytest.raises(ValueError, match="Expected .* features"):
        cmodel.predict(X_extra)


def test_too_few_features():
    estimator, X, y = build_classifier()
    cmodel = emlearn.convert(estimator)
    with pytest.raises(ValueError, match="Expected .* features"):
        cmodel.predict(X[:, :2])


# --- Convert defaults ---

def test_inline_default_dtype_float():
    estimator, X, y = build_classifier()
    cmodel = emlearn.convert(estimator)
    assert cmodel.method == 'inline'
    assert cmodel.dtype == 'float'


def test_loadable_default_dtype_int16():
    estimator, X, y = build_classifier()
    cmodel = emlearn.convert(estimator, method='loadable')
    assert cmodel.dtype == 'int16_t'


def test_loadable_rejects_float_dtype():
    estimator, X, y = build_classifier()
    with pytest.raises(ValueError, match='int16_t'):
        emlearn.convert(estimator, method='loadable', dtype='float')


def test_invalid_method_rejected():
    estimator, X, y = build_classifier()
    with pytest.raises(ValueError, match='method'):
        emlearn.convert(estimator, method='nosuchmethod')


def test_loadable_rejects_too_many_features():
    X, y = make_classification(n_samples=50, n_features=130, n_informative=10, random_state=42)
    estimator = GradientBoostingClassifier(n_estimators=3, random_state=42)
    estimator.fit(X, y)
    with pytest.raises(ValueError, match='Maximum features'):
        emlearn.convert(estimator, method='loadable')


def test_unsupported_kwargs_rejected():
    estimator, X, y = build_classifier()
    with pytest.raises(ValueError, match='leaf_bits'):
        emlearn.convert(estimator, leaf_bits=8)


# --- Parity tests ---

@pytest.mark.parametrize("data", CLASSIFICATION_DATASETS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_classifier_predict_parity(data, method):
    """Converted classifier predictions match sklearn."""
    X, y = CLASSIFICATION_DATASETS[data]
    X_method, dtype = prepare_data_for_method(X, method)

    estimator = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=RANDOM_SEED)
    estimator.fit(X_method, y)
    cmodel = emlearn.convert(estimator, method=method, dtype=dtype)

    pred_original = estimator.predict(X_method[:10])
    pred_c = cmodel.predict(X_method[:10])
    assert_equal(pred_c, pred_original)


@pytest.mark.parametrize("data", CLASSIFICATION_DATASETS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_classifier_proba_parity(data, method):
    """Converted classifier probabilities match sklearn."""
    X, y = CLASSIFICATION_DATASETS[data]
    X_method, dtype = prepare_data_for_method(X, method)

    estimator = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=RANDOM_SEED)
    estimator.fit(X_method, y)
    cmodel = emlearn.convert(estimator, method=method, dtype=dtype)

    proba_original = estimator.predict_proba(X_method[:10])
    proba_c = cmodel.predict_proba(X_method[:10])

    assert_allclose(proba_c, proba_original, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("data", MULTICLASS_DATASETS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_multiclass_parity(data, method):
    """Converted multi-class classifier matches sklearn predictions and probabilities."""
    X, y = MULTICLASS_DATASETS[data]
    X_method, dtype = prepare_data_for_method(X, method)

    estimator = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=RANDOM_SEED)
    estimator.fit(X_method, y)
    cmodel = emlearn.convert(estimator, method=method, dtype=dtype)

    pred_original = estimator.predict(X_method[:10])
    pred_c = cmodel.predict(X_method[:10])
    assert_equal(pred_c, pred_original)

    proba_original = estimator.predict_proba(X_method[:10])
    proba_c = cmodel.predict_proba(X_method[:10])
    assert_allclose(proba_c, proba_original, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("data", REGRESSION_DATASETS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_regressor_parity(data, method):
    """Converted regressor predictions match sklearn."""
    X, y = REGRESSION_DATASETS[data]
    X_method, dtype = prepare_data_for_method(X, method)

    estimator = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=RANDOM_SEED)
    estimator.fit(X_method, y)
    cmodel = emlearn.convert(estimator, method=method, dtype=dtype)

    pred_original = estimator.predict(X_method[:10])
    pred_c = cmodel.predict(X_method[:10])

    rtol = 0.05 if method == 'loadable' else 1e-3
    atol = 1.0 if method == 'loadable' else 1e-3
    assert_allclose(pred_c, pred_original, rtol=rtol, atol=atol)


# --- Edge cases ---

def test_single_estimator():
    """n_estimators=1 works for both classifier and regressor."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    clf = GradientBoostingClassifier(n_estimators=1, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')
    assert_equal(cmodel.predict(X[:10]), clf.predict(X[:10]))

    X_reg, y_reg = make_regression(n_samples=50, n_features=4, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=1, random_state=42)
    reg.fit(X_reg, y_reg)
    cmodel_reg = emlearn.convert(reg, method='inline')
    assert_allclose(cmodel_reg.predict(X_reg[:10]), reg.predict(X_reg[:10]),
        rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("learning_rate", [0.001, 0.1, 1.0])
def test_learning_rates(learning_rate):
    """Prediction parity holds across learning rates."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    clf = GradientBoostingClassifier(n_estimators=5, learning_rate=learning_rate, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')

    assert_equal(cmodel.predict(X[:10]), clf.predict(X[:10]))
    assert_allclose(cmodel.predict_proba(X[:10]), clf.predict_proba(X[:10]),
        rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_estimators,max_depth,learning_rate", [
    (1, 2, 0.01),
    (20, 5, 0.5),
    (100, 10, 1.0),
])
def test_numerical_precision(n_estimators, max_depth, learning_rate):
    """C and Python predictions agree closely across model configurations."""
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
    )
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')

    assert_equal(cmodel.predict(X), clf.predict(X))
    assert_allclose(cmodel.predict_proba(X), clf.predict_proba(X),
        rtol=1e-5, atol=1e-5)


def test_extreme_class_prior_rejected():
    """Class priors below 1e-10 are rejected (would give log(0) in init score)."""
    from unittest.mock import Mock

    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=3, random_state=42)
    clf.fit(X, y)

    mock_init = Mock()
    mock_init.class_prior_ = numpy.array([1e-15, 1 - 1e-15])
    clf.init_ = mock_init

    with pytest.raises(ValueError, match='class priors'):
        emlearn.convert(clf, method='inline')


# --- Save / serialization ---

def test_save_to_file_inline(tmp_path):
    """save(file=...) writes the generated inline code to disk."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=3, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')

    file_path = tmp_path / "gradient_boosting_model.h"
    code = cmodel.save(file=str(file_path))

    assert file_path.exists()
    content = file_path.read_text()
    assert '// !!! This file is generated using emlearn !!!' in content
    assert 'gradient_boosting_model_predict' in content
    assert code == content


def test_save_to_file_loadable(tmp_path):
    """save(file=...) writes the generated loadable code to disk."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    X_scaled = (X * 1000).astype(numpy.int16)
    clf = GradientBoostingClassifier(n_estimators=3, random_state=42)
    clf.fit(X_scaled, y)
    cmodel = emlearn.convert(clf, method='loadable')

    file_path = tmp_path / "gb_loadable_model.h"
    code = cmodel.save(file=str(file_path))

    assert file_path.exists()
    content = file_path.read_text()
    assert '// !!! This file is generated using emlearn !!!' in content
    assert 'EmlGradientBoosting' in content
    assert 'eml_gbtrees.h' in content
    assert code == content


@pytest.mark.parametrize('method', METHODS)
def test_save_include_proba_false(method):
    """save(include_proba=False) omits predict_proba from generated code."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    X_method, dtype = prepare_data_for_method(X, method)
    clf = GradientBoostingClassifier(n_estimators=3, random_state=42)
    clf.fit(X_method, y)
    cmodel = emlearn.convert(clf, method=method, dtype=dtype)

    code_with = cmodel.save(name='test', include_proba=True)
    code_without = cmodel.save(name='test', include_proba=False)

    assert 'test_predict_proba' in code_with
    assert 'test_predict_proba' not in code_without
    assert 'test_predict' in code_without


# --- Input validation ---

def test_nan_input_raises():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')

    X_nan = X[:5].copy()
    X_nan[0, 0] = numpy.nan
    with pytest.raises(ValueError, match="NaN"):
        cmodel.predict(X_nan)
    with pytest.raises(ValueError, match="NaN"):
        cmodel.predict_proba(X_nan)


def test_inf_input_raises():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')

    X_inf = X[:5].copy()
    X_inf[0, 0] = numpy.inf
    with pytest.raises(ValueError, match="infinity"):
        cmodel.predict(X_inf)


def test_valid_input_works():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')
    assert len(cmodel.predict(X[:5])) == 5


# --- Validation guards ---

def test_exponential_loss_rejected():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=5, loss='exponential', random_state=42)
    clf.fit(X, y)
    with pytest.raises(ValueError, match="exponential"):
        emlearn.convert(clf, method='inline')


def test_loadable_max_classes_rejected():
    n_classes = 31
    X, y = make_classification(
        n_samples=300, n_features=40, n_informative=35,
        n_classes=n_classes, n_clusters_per_class=1, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42)
    clf.fit(X, y)
    with pytest.raises(ValueError, match="30 classes"):
        emlearn.convert(clf, method='loadable', dtype='int16_t')


def test_inline_many_classes_accepted():
    n_classes = 31
    X, y = make_classification(
        n_samples=300, n_features=40, n_informative=35,
        n_classes=n_classes, n_clusters_per_class=1, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')
    pred = cmodel.predict(X[:5])
    assert len(pred) == 5


# --- Float precision ---

def test_small_learning_rate_preserved():
    from emlearn import cgen
    val = 1e-7
    c_literal = cgen.constant(val, dtype='float')
    recovered = float(c_literal.rstrip('f'))
    assert recovered != 0.0, f"Value {val} was truncated to {c_literal}"
    assert abs(recovered - val) / val < 1e-6


def test_regression_small_learning_rate():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    clf = GradientBoostingRegressor(n_estimators=5, learning_rate=1e-5, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')
    pred_sklearn = clf.predict(X[:10])
    pred_emlearn = cmodel.predict(X[:10])
    assert_allclose(pred_emlearn, pred_sklearn, rtol=0.01)
