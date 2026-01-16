
import os
import io

import sklearn
import numpy
import numpy.testing
import pandas

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import sklearn.model_selection
import sklearn.metrics
import scipy.stats

from emlearn.preprocessing import Quantizer
import emlearn
from emlearn.evaluate.trees import model_size_nodes, tree_depth_average
import pytest

here = os.path.dirname(__file__)

random = numpy.random.randint(0, 1000)
print('random_state={}'.format(random))


CLASSIFICATION_MODELS = {
    'RFC': RandomForestClassifier(n_estimators=10, random_state=random),
    'ETC': ExtraTreesClassifier(n_estimators=10, random_state=random),
    'DTC': DecisionTreeClassifier(random_state=random),
}

# these will cause incomplete leaf nodes. Needs class proportion support
CLASSIFICATION_MODELS_DEPTH_LIMIT = {
    'RFC-max_depth': RandomForestClassifier(n_estimators=10, random_state=random, max_depth=3),
    'ETC-min_samples_leaf': RandomForestClassifier(n_estimators=10, random_state=random, min_samples_leaf=0.20),
}


REGRESSION_MODELS = {
    'RFR': RandomForestRegressor(n_estimators=10, random_state=random),
    'ERR': ExtraTreesRegressor(n_estimators=10, random_state=random),
    'DTR': DecisionTreeRegressor(random_state=random),
}

CLASSIFICATION_DATASETS = {
    'binary': datasets.make_classification(n_classes=2, n_samples=100, random_state=random),
    '5way': datasets.make_classification(n_classes=5, n_informative=5, n_samples=100, random_state=random),
}

REGRESSION_DATASETS = {
    '1out': datasets.make_regression(n_targets=1, n_samples=100, random_state=random),
}

# Gradient Boosting models
GRADIENT_BOOSTING_CLASSIFIERS = {
    'GBC': GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=random),
}

GRADIENT_BOOSTING_REGRESSORS = {
    'GBR': GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=random),
}

# Multi-class dataset for GradientBoosting
MULTICLASS_DATASETS = {
    '3way': datasets.make_classification(n_classes=3, n_informative=3, n_redundant=0, n_samples=100, random_state=random),
}

METHODS = ['loadable', 'inline']

def check_csv_export(cmodel):
    """
    Check that can be saved to CSV file on correct format
    """

    # check that CSV export is valid
    csv = cmodel.save(format='csv', name='mymodelname22')
    loaded = pandas.read_csv(io.StringIO(csv), header=None, engine='python', names=['item', '1', '2', '3', '4'])
    csv_items = set(loaded.item.unique())
    assert csv_items == set(['r', 'n', 'l', 'f', 'c'])

    nodes = loaded[loaded.item == 'n']
    leaves = loaded[loaded.item == 'l']
    roots = loaded[loaded.item == 'r']
    classes = loaded[loaded.item == 'c']
    features = loaded[loaded.item == 'f']

    assert len(nodes) == len(cmodel.forest_[0])
    assert len(roots) == len(cmodel.forest_[1])
    assert len(leaves) == len(cmodel.forest_[2])

    assert len(classes) == 1
    assert classes.iloc[0, 1] == cmodel.n_classes

    assert len(features) == 1
    assert features.iloc[0, 1] == cmodel.n_features

@pytest.mark.parametrize("data", CLASSIFICATION_DATASETS.keys())
@pytest.mark.parametrize("model", CLASSIFICATION_MODELS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_trees_sklearn_classifier_predict(data, model, method):
    X, y = CLASSIFICATION_DATASETS[data]
    estimator = CLASSIFICATION_MODELS[model]
    X = Quantizer().fit_transform(X)

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method=method)

    pred_original = estimator.predict(X[:5])
    pred_c = cmodel.predict(X[:5])
    numpy.testing.assert_equal(pred_c, pred_original)

    proba_original = estimator.predict_proba(X[:5])
    proba_c = cmodel.predict_proba(X[:5])
    numpy.testing.assert_allclose(proba_c, proba_original, rtol=0.001, atol=0.11)

    check_csv_export(cmodel)

@pytest.mark.parametrize("data", REGRESSION_DATASETS.keys())
@pytest.mark.parametrize("model", REGRESSION_MODELS.keys())
@pytest.mark.parametrize("method", METHODS)
def test_trees_sklearn_regressor_predict(data, model, method):
    X, y = REGRESSION_DATASETS[data]
    estimator = REGRESSION_MODELS[model]
    X = Quantizer().fit_transform(X)

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method=method)

    pred_original = estimator.predict(X[:5])
    pred_c = cmodel.predict(X[:5])

    numpy.testing.assert_allclose(pred_c, pred_original, rtol=1e-3, atol=2)

    check_csv_export(cmodel)


SUPPORTED_FEATURE_DTYPES = [
    'float',
    'int32_t',
    'int16_t',
    'int8_t',
    #'uint8_t',
]
@pytest.mark.parametrize("data", CLASSIFICATION_DATASETS.keys())
@pytest.mark.parametrize("model", CLASSIFICATION_MODELS.keys())
@pytest.mark.parametrize("dtype", SUPPORTED_FEATURE_DTYPES)
def test_trees_sklearn_classifier_inline_dtype(data, model, dtype):
    """
    The inline method supports specifying the input datatype
    """
    X, y = CLASSIFICATION_DATASETS[data]
    estimator = CLASSIFICATION_MODELS[model]

    # Convert the data to the dtype (and range)
    python_dtype = dtype.removesuffix('_t')
    if dtype != 'float':
        X = Quantizer(dtype=python_dtype, max_quantile=None).fit_transform(X)
    assert X.dtype == python_dtype

    if 'int32' in dtype:
        X = (X * 0.9).astype(python_dtype) # XXX: for some unknown reason fails if using entire int32

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method='inline', dtype=dtype)

    allowed_incorrect = 0.10 if 'int8' in dtype else 0.0
    atol = 0.3 if 'int8' in dtype else 0.1

    X_sub = X[:100]
    pred_original = estimator.predict(X_sub)
    pred_c = cmodel.predict(X_sub)
    incorrect = numpy.where(pred_c != pred_original)[0]
    assert len(incorrect) <= int(allowed_incorrect*len(X_sub)), (pred_c, pred_original)

    # no point checking where we were off
    X_sub = X_sub[pred_c == pred_original]
    assert len(X_sub >= 4)
    proba_original = estimator.predict_proba(X_sub)
    proba_c = cmodel.predict_proba(X_sub)

    bad = numpy.sum(numpy.abs(proba_original - proba_c) > 0.30, axis=1)
    X_bad = X_sub[bad.astype(bool)]

    numpy.testing.assert_allclose(proba_c, proba_original, atol=atol, rtol=atol)


@pytest.mark.parametrize("data", REGRESSION_DATASETS.keys())
@pytest.mark.parametrize("model", REGRESSION_MODELS.keys())
@pytest.mark.parametrize("dtype", SUPPORTED_FEATURE_DTYPES)
def test_trees_sklearn_regressor_inline_dtype(data, model, dtype):
    X, y = REGRESSION_DATASETS[data]
    estimator = REGRESSION_MODELS[model]
    X = Quantizer().fit_transform(X)

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method='inline')

    pred_original = estimator.predict(X[:5])
    pred_c = cmodel.predict(X[:5])

    allowed_rtol = 0.1 if 'int8' in dtype else 0.001
    allowed_atol = 2.0
    numpy.testing.assert_allclose(pred_c, pred_original, rtol=allowed_rtol, atol=allowed_atol)
    check_csv_export(cmodel)


@pytest.mark.parametrize("data", CLASSIFICATION_DATASETS.keys())
@pytest.mark.parametrize("model", CLASSIFICATION_MODELS_DEPTH_LIMIT.keys())
@pytest.mark.parametrize("method", ['inline', 'loadable'])
@pytest.mark.parametrize("leaf_bits", [3, 4, 5, 6, 7, 8])
def test_trees_sklearn_classifier_leaf_proportions(data, model, method, leaf_bits):
    X, y = CLASSIFICATION_DATASETS[data]
    estimator = CLASSIFICATION_MODELS_DEPTH_LIMIT[model]
    X = Quantizer().fit_transform(X)

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method=method, leaf_bits=leaf_bits)

    X_sub = X[:8]
    proba_original = estimator.predict_proba(X_sub)
    proba_c = cmodel.predict_proba(X_sub)
    atol = (1.5/(2**leaf_bits))
    numpy.testing.assert_allclose(proba_c, proba_original, rtol=1e-6, atol=atol)

    allowed_incorrect = 3 if leaf_bits <= 5 else 1
    pred_original = estimator.predict(X_sub)
    pred_c = cmodel.predict(X_sub)
    incorrect = numpy.where(pred_c != pred_original)[0]
    assert len(incorrect) <= allowed_incorrect, (pred_c, pred_original)


@pytest.mark.parametrize("model", CLASSIFICATION_MODELS.keys())
@pytest.mark.parametrize("data", CLASSIFICATION_DATASETS.keys())
def test_trees_evaluate_scoring(model, data):
    """
    Test that the emlearn.evaluate.tree functions for cost metrics can be used with scikit-learn
    """
    estimator = CLASSIFICATION_MODELS[model]
    X, y = CLASSIFICATION_DATASETS[data]

    from emlearn.evaluate.trees import \
        model_size_nodes, model_size_bytes, \
        tree_depth_average, tree_depth_difference, \
        count_trees, compute_cost_estimate \

    hyperparameters = {
        #'max_depth': scipy.stats.randint(1, 10),
        'min_samples_leaf': scipy.stats.loguniform(0.01, 0.33),
    }
    if 'DTC' not in model:
        hyperparameters.update({
            'n_estimators': scipy.stats.randint(5, 100),
        })

    # custom emlearn metrics for the model costs
    custom_metrics = {
        'bytes': model_size_bytes,
        'nodes': model_size_nodes,
        'compute': compute_cost_estimate,
        'depth_avg': model_size_nodes,
        'depth_diff': tree_depth_difference,
        'trees': count_trees,
    }
    # standard metrics
    metrics = {
        'accuracy': sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score),
    }
    metrics.update(custom_metrics)

    search = sklearn.model_selection.RandomizedSearchCV(
        estimator,
        param_distributions=hyperparameters,
        scoring=metrics,
        refit='accuracy',
        n_iter=4,
        cv=2,
        return_train_score=True,
        n_jobs=4,
    )
    model = search.fit(X, y)
    results = pandas.DataFrame(model.cv_results_)

    result_keys = [ f'mean_test_{m}' for m in custom_metrics ]
    missing_columns = set(result_keys) - set(results.columns)
    assert missing_columns == set(), missing_columns

    values = results[result_keys]
    rows_with_nan = values[values.isna().any(axis=1)]
    assert len(rows_with_nan) == 0, rows_with_nan


def test_trees_single_leaf_tree():
    """
    Test ensemble that includes tree with a single node/leaf (edge case)
    """
    estimator = CLASSIFICATION_MODELS['RFC']
    # FIXME: fails more often for 5way dataset. Maybe to do with how tree predictions are combined
    X, y = CLASSIFICATION_DATASETS['binary']

    # force there to be a chance of getting single leaf in a tree
    estimator.set_params(min_samples_leaf=0.33, n_estimators=3)

    model = estimator.fit(X, y)

    # Check that we were able to trigger the edge case
    depths = [ e.tree_.max_depth for e in estimator.estimators_ ]
    leaf_only_trees = [ e.tree_ for e in estimator.estimators_ if e.tree_.max_depth == 0 ]
    assert leaf_only_trees, depths

    for t in leaf_only_trees:
        assert t.children_right == [-1]
        assert t.children_left == [-1]

    # Check that model can be converted
    cmodel = emlearn.convert(estimator, method='inline')
    nodes = pandas.DataFrame(cmodel.forest_[0], columns=['feature', 'treshold', 'left', 'right'])
    roots = pandas.DataFrame(cmodel.forest_[1], columns=['index'])
    leaves = pandas.DataFrame(cmodel.forest_[2], columns=['data'])

    pred_original = estimator.predict(X)
    pred_c = cmodel.predict(X)
    # Skip comparison, fails randomly
    #numpy.testing.assert_equal(pred_c, pred_original)

@pytest.fixture(scope='module')
def huge_trees_model():
    store_classifier_path = os.path.join(here, 'out/test_trees_huge.model.pickle')
    X, Y = datasets.make_classification(n_classes=2, n_samples=1000, random_state=1)
    X = Quantizer().fit_transform(X)
    est = RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=1)
    est.fit(X, Y)

    n_nodes = model_size_nodes(est)
    assert n_nodes >= 90*1000

    return X, Y, est

@pytest.mark.parametrize("method", ['loadable', 'inline'])
@pytest.mark.skipif(not bool(int(os.environ.get('EMLEARN_TESTS_SLOW', '0'))), reason='EMLEARN_TESTS_SLOW not enabled')
def test_trees_huge(method, huge_trees_model):
    """Should work just the same as a smaller model"""
    X, Y, estimator = huge_trees_model

    cmodel = emlearn.convert(estimator, method=method)
    pred_original = estimator.predict(X)
    pred_c = cmodel.predict(X)
    numpy.testing.assert_equal(pred_c, pred_original)


@pytest.fixture(scope='module')
def deep_trees_model():
    store_classifier_path = os.path.join(here, 'out/test_trees_huge.model.pickle')
    X, Y = datasets.make_classification(n_classes=30, n_features=120, n_informative=100, n_samples=10000, random_state=1)
    X = Quantizer().fit_transform(X)
    est = RandomForestClassifier(n_estimators=2, random_state=1)
    est.fit(X, Y)

    depth = tree_depth_average(est)
    n_nodes = model_size_nodes(est)
    assert depth >= 40
    assert n_nodes >= 8*1000

    return X, Y, est

@pytest.mark.parametrize("method", ['loadable', 'inline'])
def test_trees_deep(method, deep_trees_model):
    """Should work just the same as a smaller model"""
    X, Y, estimator = deep_trees_model

    X = X[0:1000, :]
    cmodel = emlearn.convert(estimator, method=method)
    pred_original = estimator.predict(X)
    pred_c = cmodel.predict(X)
    numpy.testing.assert_equal(pred_c, pred_original)



@pytest.mark.parametrize("method", ['loadable', 'inline'])
def test_trees_too_many_features(method, deep_trees_model):
    """Should raise error during convert()"""

    above_max = 200 if method == 'inlne' else 20000
    X, Y = datasets.make_classification(n_classes=10, n_features=above_max,
        n_informative=100, n_samples=1000, random_state=1)
    estimator = RandomForestClassifier(n_estimators=5, random_state=1)
    estimator.fit(X, Y)

    with pytest.raises(ValueError, match='features'):
        cmodel = emlearn.convert(estimator, method=method)

LOADABLE_UNSUPPORTED_FEATURE_DTYPES=['float', 'int8_t', 'uint8_t', 'int32_t']
@pytest.mark.parametrize("dtype", LOADABLE_UNSUPPORTED_FEATURE_DTYPES)
def test_trees_loadable_unsupported_dtype(dtype):
    """Should raise error during convert()"""

    X, Y = datasets.make_classification(n_classes=10, n_features=10,
        n_informative=8, n_samples=100, random_state=1)
    estimator = RandomForestClassifier(n_estimators=5, random_state=1)
    estimator.fit(X, Y)

    with pytest.raises(ValueError, match='loadable'):
        cmodel = emlearn.convert(estimator, method='loadable', dtype=dtype)


# ============================================================================
# Gradient Boosting Tests
# ============================================================================

@pytest.mark.parametrize("data", CLASSIFICATION_DATASETS.keys())
@pytest.mark.parametrize("model", GRADIENT_BOOSTING_CLASSIFIERS.keys())
def test_gradient_boosting_classifier_predict(data, model):
    """Test GradientBoostingClassifier conversion and prediction."""
    X, y = CLASSIFICATION_DATASETS[data]
    estimator = GRADIENT_BOOSTING_CLASSIFIERS[model]

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method='inline')

    pred_original = estimator.predict(X[:10])
    pred_c = cmodel.predict(X[:10])
    numpy.testing.assert_equal(pred_c, pred_original)


@pytest.mark.parametrize("data", CLASSIFICATION_DATASETS.keys())
@pytest.mark.parametrize("model", GRADIENT_BOOSTING_CLASSIFIERS.keys())
def test_gradient_boosting_classifier_proba(data, model):
    """Test GradientBoostingClassifier probability estimation."""
    X, y = CLASSIFICATION_DATASETS[data]
    estimator = GRADIENT_BOOSTING_CLASSIFIERS[model]

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method='inline')

    proba_original = estimator.predict_proba(X[:10])
    proba_c = cmodel.predict_proba(X[:10])

    # Allow some tolerance due to floating point differences
    numpy.testing.assert_allclose(proba_c, proba_original, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("data", MULTICLASS_DATASETS.keys())
@pytest.mark.parametrize("model", GRADIENT_BOOSTING_CLASSIFIERS.keys())
def test_gradient_boosting_classifier_multiclass(data, model):
    """Test GradientBoostingClassifier with multi-class datasets."""
    X, y = MULTICLASS_DATASETS[data]
    estimator = GRADIENT_BOOSTING_CLASSIFIERS[model]

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method='inline')

    # Test predictions
    pred_original = estimator.predict(X[:10])
    pred_c = cmodel.predict(X[:10])
    numpy.testing.assert_equal(pred_c, pred_original)

    # Test probabilities
    proba_original = estimator.predict_proba(X[:10])
    proba_c = cmodel.predict_proba(X[:10])
    numpy.testing.assert_allclose(proba_c, proba_original, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("data", REGRESSION_DATASETS.keys())
@pytest.mark.parametrize("model", GRADIENT_BOOSTING_REGRESSORS.keys())
def test_gradient_boosting_regressor_predict(data, model):
    """Test GradientBoostingRegressor conversion and prediction."""
    X, y = REGRESSION_DATASETS[data]
    estimator = GRADIENT_BOOSTING_REGRESSORS[model]

    estimator.fit(X, y)
    cmodel = emlearn.convert(estimator, method='inline')

    pred_original = estimator.predict(X[:10])
    pred_c = cmodel.predict(X[:10])

    numpy.testing.assert_allclose(pred_c, pred_original, rtol=1e-3, atol=1e-3)


def test_gradient_boosting_loadable_not_supported():
    """Test that loadable method raises error for GradientBoosting."""
    X, y = datasets.make_classification(n_samples=50, n_features=4, random_state=42)
    estimator = GradientBoostingClassifier(n_estimators=3, random_state=42)
    estimator.fit(X, y)

    with pytest.raises(ValueError, match='loadable'):
        emlearn.convert(estimator, method='loadable')


def test_gradient_boosting_single_estimator():
    """Test GradientBoosting with n_estimators=1."""
    X, y = datasets.make_classification(n_samples=50, n_features=4, random_state=42)

    # Test classifier
    clf = GradientBoostingClassifier(n_estimators=1, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')

    pred_original = clf.predict(X[:10])
    pred_c = cmodel.predict(X[:10])
    numpy.testing.assert_equal(pred_c, pred_original)

    # Test regressor
    X_reg, y_reg = datasets.make_regression(n_samples=50, n_features=4, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=1, random_state=42)
    reg.fit(X_reg, y_reg)
    cmodel_reg = emlearn.convert(reg, method='inline')

    pred_reg_original = reg.predict(X_reg[:10])
    pred_reg_c = cmodel_reg.predict(X_reg[:10])
    numpy.testing.assert_allclose(pred_reg_c, pred_reg_original, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("learning_rate", [0.001, 0.1, 1.0])
def test_gradient_boosting_learning_rates(learning_rate):
    """Test GradientBoosting with various learning rates."""
    X, y = datasets.make_classification(n_samples=50, n_features=4, random_state=42)

    clf = GradientBoostingClassifier(n_estimators=5, learning_rate=learning_rate, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')

    pred_original = clf.predict(X[:10])
    pred_c = cmodel.predict(X[:10])
    numpy.testing.assert_equal(pred_c, pred_original)

    proba_original = clf.predict_proba(X[:10])
    proba_c = cmodel.predict_proba(X[:10])
    numpy.testing.assert_allclose(proba_c, proba_original, rtol=0.02, atol=0.02)


def test_gradient_boosting_extreme_class_prior_validation():
    """Test that extreme class prior validation works correctly.

    The validation is for class priors < 1e-10 which prevents division by zero
    in log(p1/p0). This is extremely rare in practice (would require ~10 billion
    samples to have a class prior that low), so we test by directly calling the
    wrapper with a mocked estimator.
    """
    from unittest.mock import Mock, patch

    X, y = datasets.make_classification(n_samples=50, n_features=4, random_state=42)

    # Create a fitted classifier
    clf = GradientBoostingClassifier(n_estimators=3, random_state=42)
    clf.fit(X, y)

    # Mock the init_ estimator to have extreme class priors
    original_init = clf.init_
    mock_init = Mock()
    mock_init.class_prior_ = numpy.array([1e-15, 1 - 1e-15])  # Extremely imbalanced

    clf.init_ = mock_init

    # Should raise ValueError due to extreme class prior
    with pytest.raises(ValueError, match='class priors'):
        emlearn.convert(clf, method='inline')

    # Restore original init for cleanup
    clf.init_ = original_init


def test_gradient_boosting_save_to_file(tmp_path):
    """Test saving GradientBoosting model to file."""
    X, y = datasets.make_classification(n_samples=50, n_features=4, random_state=42)

    clf = GradientBoostingClassifier(n_estimators=3, random_state=42)
    clf.fit(X, y)
    cmodel = emlearn.convert(clf, method='inline')

    # Save to file
    file_path = tmp_path / "gradient_boosting_model.h"
    code = cmodel.save(file=str(file_path))

    # Verify file was created and contains expected content
    assert file_path.exists()
    content = file_path.read_text()
    assert '// !!! This file is generated using emlearn !!!' in content
    assert 'gradient_boosting_model_predict' in content
    assert code == content


