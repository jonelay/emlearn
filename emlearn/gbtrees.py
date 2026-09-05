
"""
Gradient boosted trees
=========================
"""

import math
import os.path

import numpy
import sklearn.base

from . import common, cgen
from .trees import (
    flatten_forest,
    remove_duplicate_leaves,
    generate_c_inline_tree_functions,
    generate_c_nodes,
    leaves_to_bytelist,
)

SUPPORTED_ESTIMATORS = [
    'GradientBoostingClassifier',
    'GradientBoostingRegressor',
]


def generate_c_inlined(forest, name, n_features, learning_rate,
                       initial_value, n_classes=0, dtype='float',
                       classifier=True, include_proba=True):
    """Generate inline C code for gradient boosting inference."""
    nodes, roots, leaves = forest

    cgen.assert_valid_identifier(name)

    tree_names = [name + '_tree_{}'.format(i) for i, _ in enumerate(roots)]
    ctype = dtype

    # GBT leaves are raw float contribution values
    tree_funcs = generate_c_inline_tree_functions(forest, tree_names, dtype,
        leaf_dtype='float', return_type='float')

    # Generate tree prediction accumulation
    def tree_accumulate(name):
        return 'score += {}(features, features_length);'.format(name)

    if classifier and n_classes == 2:
        # Binary classification: raw score accumulation (shared by predict and predict_proba)
        # initial_value is the log-odds from class prior
        raw_score_func = """float {function_name}(const {ctype} *features, int32_t features_length) {{
        if (features_length != {n_features}) return NAN;
        float score = 0.0f;

        {tree_predictions}

        // Apply learning rate and add initial log-odds
        return {initial_value} + {learning_rate} * score;
    }}
    """.format(**{
            'function_name': name + "_raw_score",
            'initial_value': cgen.constant(initial_value, dtype='float'),
            'learning_rate': cgen.constant(learning_rate, dtype='float'),
            'tree_predictions': '\n        '.join([tree_accumulate(n) for n in tree_names]),
            'ctype': ctype,
            'n_features': n_features,
        })

        # predict_proba_class1 applies sigmoid to the raw score
        forest_func = """float {function_name}(const {ctype} *features, int32_t features_length) {{
        const float score = {name}_raw_score(features, features_length);
        return 1.0f / (1.0f + expf(-score));
    }}
    """.format(**{
            'function_name': name + "_predict_proba_class1",
            'name': name,
            'ctype': ctype,
        })

        # predict skips sigmoid: sigmoid(x) >= 0.5 iff x >= 0
        predict_func = """int32_t {function_name}(const {ctype} *features, int32_t features_length) {{
        if (features_length != {n_features}) return -1;
        const float score = {name}_raw_score(features, features_length);
        return (score >= 0.0f) ? 1 : 0;
    }}
    """.format(**{
            'function_name': name + "_predict",
            'name': name,
            'n_features': n_features,
            'ctype': ctype,
        })

        proba_func = """int {function_name}(const {ctype} *features, int32_t features_length, float *out, int out_length) {{
        if (features_length != {n_features}) return -1;
        if (out_length != 2) return -1;
        const float prob1 = {name}_predict_proba_class1(features, features_length);
        out[0] = 1.0f - prob1;
        out[1] = prob1;
        return 0;
    }}
    """.format(**{
            'function_name': name + "_predict_proba",
            'name': name,
            'n_features': n_features,
            'ctype': ctype,
        })

        forest_funcs = [raw_score_func, predict_func]
        if include_proba:
            forest_funcs = [raw_score_func, forest_func, predict_func, proba_func]

    elif classifier and n_classes > 2:
        # Multi-class: trees are organized as [n_estimators, n_classes]
        # Need to accumulate scores per class and apply softmax
        n_estimators = len(roots) // n_classes

        class_score_accum = []
        for k in range(n_classes):
            tree_indices = [i * n_classes + k for i in range(n_estimators)]
            tree_calls = ' + '.join([f'{tree_names[idx]}(features, features_length)' for idx in tree_indices])
            init_val = cgen.constant(initial_value[k], dtype='float')
            lr_val = cgen.constant(learning_rate, dtype='float')
            class_score_accum.append(f'scores[{k}] = {init_val} + {lr_val} * ({tree_calls});')

        forest_func = """void {function_name}_raw_scores(const {ctype} *features, int32_t features_length, float *scores) {{
        {class_scores}
    }}
    """.format(**{
            'function_name': name,
            'class_scores': '\n        '.join(class_score_accum),
            'ctype': ctype,
        })

        predict_func = """int32_t {function_name}(const {ctype} *features, int32_t features_length) {{
        if (features_length != {n_features}) return -1;
        float scores[{n_classes}];
        {name}_raw_scores(features, features_length, scores);

        // Find argmax
        int32_t best_class = 0;
        float best_score = scores[0];
        for (int32_t i = 1; i < {n_classes}; i++) {{
            if (scores[i] > best_score) {{
                best_score = scores[i];
                best_class = i;
            }}
        }}
        return best_class;
    }}
    """.format(**{
            'function_name': name + "_predict",
            'name': name,
            'n_classes': n_classes,
            'n_features': n_features,
            'ctype': ctype,
        })

        # Multi-class predict_proba with softmax
        proba_func = """int {function_name}(const {ctype} *features, int32_t features_length, float *out, int out_length) {{
        if (features_length != {n_features}) return -1;
        if (out_length != {n_classes}) return -1;

        float scores[{n_classes}];
        {name}_raw_scores(features, features_length, scores);

        // Apply softmax
        float max_score = scores[0];
        for (int i = 1; i < {n_classes}; i++) {{
            if (scores[i] > max_score) max_score = scores[i];
        }}

        float sum_exp = 0.0f;
        for (int i = 0; i < {n_classes}; i++) {{
            out[i] = expf(scores[i] - max_score);
            sum_exp += out[i];
        }}

        // Numerical stability: prevent division by zero
        if (sum_exp < 1e-10f) sum_exp = 1e-10f;

        for (int i = 0; i < {n_classes}; i++) {{
            out[i] /= sum_exp;
        }}

        return 0;
    }}
    """.format(**{
            'function_name': name + "_predict_proba",
            'name': name,
            'n_classes': n_classes,
            'n_features': n_features,
            'ctype': ctype,
        })

        forest_funcs = [forest_func, predict_func]
        if include_proba:
            forest_funcs.append(proba_func)

    else:
        # Regression: simple sum with learning rate and initial value
        forest_func = """float {function_name}(const {ctype} *features, int32_t features_length) {{
        if (features_length != {n_features}) return NAN;
        float score = 0.0f;

        {tree_predictions}

        return {initial_value} + {learning_rate} * score;
    }}
    """.format(**{
            'function_name': name + "_predict",
            'initial_value': cgen.constant(initial_value, dtype='float'),
            'learning_rate': cgen.constant(learning_rate, dtype='float'),
            'tree_predictions': '\n        '.join([tree_accumulate(n) for n in tree_names]),
            'ctype': ctype,
            'n_features': n_features,
        })

        forest_funcs = [forest_func]

    head = """
    // !!! This file is generated using emlearn !!!

    #include <stdint.h>
    #include <math.h>
    """

    parts = [head] + tree_funcs + forest_funcs
    out = '\n\n'.join(parts)

    return out


def generate_c_loadable(forest, name, n_features, learning_rate,
        initial_value, n_classes, dtype='int16_t', classifier=True,
        include_proba=True):
    """Generate loadable C code for gradient boosting model.

    Creates struct-based representation using EmlGradientBoosting from eml_gbtrees.h.

    Args:
        forest: Flattened forest tuple (nodes, roots, leaves).
        name: Name for C model variable.
        n_features: Number of input features.
        learning_rate: Learning rate from estimator.
        initial_value: Initial prediction value(s).
        n_classes: Number of classes (0 for regression).
        dtype: Feature data type.
        classifier: True for classifier, False for regressor.

    Returns:
        str: Generated C code.
    """
    nodes, roots, leaves = forest

    cgen.assert_valid_identifier(name)

    # Generate nodes array
    nodes_name = name + '_nodes'
    nodes_c = generate_c_nodes(nodes, nodes_name, dtype=dtype, modifiers='static const')

    # Generate tree roots array
    tree_roots_name = name + '_tree_roots'
    tree_roots_values = ', '.join(str(t) for t in roots)
    tree_roots_c = 'static const int32_t {name}[{length}] = {{ {values} }};'.format(
        name=tree_roots_name, length=len(roots), values=tree_roots_values
    )

    # Generate leaves array (32-bit floats for gradient boosting)
    leaves_name = name + '_leaves'
    leaves_array = leaves_to_bytelist(leaves, leaf_bits=32)
    leaves_c = cgen.array_declare(
        leaves_name, len(leaves_array),
        modifiers='static const', dtype='uint8_t', values=leaves_array
    )

    # Generate initial values array for multi-class
    initial_values_c = ''
    initial_values_ptr = 'NULL'
    if classifier and n_classes > 2 and isinstance(initial_value, (list, tuple)):
        initial_values_name = name + '_initial_values'
        initial_values_str = ', '.join(cgen.constant(v, dtype='float') for v in initial_value)
        initial_values_c = 'static const float {name}[{length}] = {{ {values} }};'.format(
            name=initial_values_name, length=len(initial_value), values=initial_values_str
        )
        initial_values_ptr = initial_values_name

    # Generate EmlGradientBoosting struct
    initial_val = initial_value if not isinstance(initial_value, (list, tuple)) else 0.0
    model_struct = """EmlGradientBoosting {name} = {{
    // EmlTrees trees
    {{
        {n_nodes},
        (EmlTreesNode *)({nodes_name}),
        {n_trees},
        (int32_t *)({tree_roots_name}),
        {n_leaves},
        (uint8_t *)({leaves_name}),
        32,  // leaf_bits
        {n_features},
        {n_classes},
    }},
    {learning_rate},  // learning_rate
    {initial_value},  // initial_value
    {initial_values_ptr},  // initial_values
    {n_classes},  // n_classes
}};""".format(
        name=name,
        n_nodes=len(nodes),
        nodes_name=nodes_name,
        n_trees=len(roots),
        tree_roots_name=tree_roots_name,
        n_leaves=len(leaves_array),
        leaves_name=leaves_name,
        n_features=n_features,
        n_classes=n_classes,
        learning_rate=cgen.constant(learning_rate, dtype='float'),
        initial_value=cgen.constant(initial_val, dtype='float'),
        initial_values_ptr=initial_values_ptr,
    )

    # Generate wrapper functions
    ctype = dtype

    if classifier:
        predict_func = """int32_t {name}_predict(const {ctype} *features, int32_t features_length) {{
    return eml_gbtrees_predict(&{model}, features, features_length);
}}""".format(name=name, ctype=ctype, model=name)

        proba_func = """int {name}_predict_proba(const {ctype} *features, int32_t features_length, float *out, int out_length) {{
    return eml_gbtrees_predict_proba(&{model}, features, features_length, out, out_length);
}}""".format(name=name, ctype=ctype, model=name)

        funcs = [predict_func]
        if include_proba:
            funcs.append(proba_func)
    else:
        regress_func = """float {name}_predict(const {ctype} *features, int32_t features_length) {{
    return eml_gbtrees_regress(&{model}, features, features_length);
}}""".format(name=name, ctype=ctype, model=name)

        funcs = [regress_func]

    head = """
// !!! This file is generated using emlearn !!!

#include <eml_gbtrees.h>
"""

    parts = [head, nodes_c, tree_roots_c, leaves_c]
    if initial_values_c:
        parts.append(initial_values_c)
    parts.append(model_struct)
    parts.extend(funcs)

    return '\n\n'.join(parts)


def _extract_initial_scores(estimator, is_classifier, n_classes):
    """Extract and validate initial raw scores from a fitted sklearn estimator.

    Supports the default init (DummyClassifier/DummyRegressor prior/mean)
    and init='zero'. Custom init estimators are rejected: their raw
    predictions depend on input features, which the generated C cannot
    reproduce.

    Args:
        estimator: Fitted sklearn GradientBoosting estimator.
        is_classifier: True for classifier, False for regressor.
        n_classes: Number of classes (0 for regression).

    Returns:
        float for binary classification (log-odds) and regression (constant),
        list of floats (per-class log-priors) for multi-class.

    Raises:
        ValueError: On custom init estimators or non-finite/invalid priors.
    """
    init = estimator.init_

    if isinstance(init, str) and init == 'zero':
        if is_classifier and n_classes > 2:
            return [0.0] * n_classes
        return 0.0

    if is_classifier:
        if not hasattr(init, 'class_prior_'):
            raise ValueError(
                "GradientBoosting with custom init estimator not supported. "
                "Use default init or init='zero'."
            )
        class_prior = numpy.asarray(init.class_prior_)
        if class_prior.shape != (n_classes,):
            raise ValueError(
                f"Unexpected class_prior_ shape {class_prior.shape}, "
                f"expected ({n_classes},)"
            )
        if not numpy.all(numpy.isfinite(class_prior)):
            raise ValueError(
                f"GradientBoosting init has non-finite class priors: {class_prior}"
            )

        if n_classes == 2:
            # Binary: log-odds log(p1/p0)
            # Validate class priors are not extreme (would cause log(0) or division by zero)
            if class_prior[0] < 1e-10 or class_prior[1] < 1e-10:
                raise ValueError(
                    "GradientBoosting conversion requires non-extreme class priors. "
                    f"Got class_prior={class_prior}. Consider rebalancing your dataset."
                )
            return float(numpy.log(class_prior[1] / class_prior[0]))

        # Multi-class: one initial log-prior per class
        return [float(numpy.log(max(p, 1e-10))) for p in class_prior]

    # Regression: init_ is DummyRegressor with constant_
    if not hasattr(init, 'constant_'):
        raise ValueError(
            "GradientBoosting with custom init estimator not supported. "
            "Use default init or init='zero'."
        )
    value = float(init.constant_[0, 0])
    if not math.isfinite(value):
        raise ValueError(
            f"GradientBoosting init has non-finite constant: {value}"
        )
    return value


def _extract_trees(estimator, is_classifier, n_classes):
    """Flatten the sklearn 2-D estimator array into a tree list.

    Multi-class trees are ordered iteration-major/class-minor
    ([iter0_class0, iter0_class1, ..., iter1_class0, ...]) to match
    the layout assumed by both inline codegen and the C runtime.
    """
    # GradientBoosting uses 2D array: [n_estimators, n_outputs]
    # For regression/binary: n_outputs=1, for multiclass: n_outputs=n_classes
    estimators_2d = estimator.estimators_
    n_estimators, n_outputs = estimators_2d.shape

    if is_classifier and n_classes > 2:
        if n_outputs != n_classes:
            raise ValueError(
                f"Multi-class GBT tree layout mismatch: "
                f"n_outputs={n_outputs} != n_classes={n_classes}"
            )
        trees = []
        for i in range(n_estimators):
            for k in range(n_outputs):
                trees.append(estimators_2d[i, k].tree_)
        return trees

    # Binary/regression: single tree per iteration
    return [estimators_2d[i, 0].tree_ for i in range(n_estimators)]


class Wrapper:
    def __init__(self, estimator, method, dtype=None, **kwargs):
        if kwargs:
            raise ValueError(
                "Unsupported arguments for GradientBoosting conversion: "
                f"{sorted(kwargs)}"
            )

        # Check sklearn version (class_prior_ attribute requires >= 1.0.0)
        import sklearn
        version_parts = sklearn.__version__.split('.')
        try:
            major, minor = int(version_parts[0]), int(version_parts[1])
            if (major, minor) < (1, 0):
                raise ImportError(
                    f"gbtrees.Wrapper requires scikit-learn >= 1.0.0, "
                    f"but {sklearn.__version__} is installed."
                )
        except (IndexError, ValueError):
            pass  # Unusual version format, proceed and let it fail naturally

        if method is None:
            method = 'inline'

        self.dtype = dtype
        if self.dtype is None:
            # loadable inference only supports int16_t features
            self.dtype = 'int16_t' if method == 'loadable' else 'float'

        self.is_classifier = sklearn.base.is_classifier(estimator)
        self.out_dtype = "int" if self.is_classifier else "float"
        self.classes_ = estimator.classes_ if self.is_classifier else None

        if self.is_classifier and getattr(estimator, 'loss', None) == 'exponential':
            raise ValueError(
                "GradientBoosting with loss='exponential' is not supported "
                "(uses a different link function). Use loss='log_loss' (default)."
            )

        # Extract learning rate
        self.learning_rate = estimator.learning_rate

        self.n_classes = estimator.n_classes_ if self.is_classifier else 0
        self.initial_value = _extract_initial_scores(
            estimator, self.is_classifier, self.n_classes)
        trees = _extract_trees(estimator, self.is_classifier, self.n_classes)

        # Flatten forest using 'value' leaf type (regression values)
        self.forest_ = flatten_forest(trees, leaf='value', leaf_bits=32)
        self.forest_ = remove_duplicate_leaves(self.forest_)

        self.n_features = estimator.n_features_in_
        self.method = method

        if self.method not in ('loadable', 'inline'):
            raise ValueError("Unsupported inference method '{}'".format(self.method))

        if self.method == 'loadable' and self.dtype != 'int16_t':
            raise ValueError("Inference method='loadable' only supports dtype='int16_t'. Use method='inline' for others")

        max_features = 127 if self.method == 'loadable' else 10000
        if self.n_features > max_features:
            raise ValueError(f"Maximum features exceeded. features={self.n_features} max={max_features}")

        if self.method == 'loadable' and self.is_classifier and self.n_classes > 30:
            raise ValueError(
                f"Loadable method supports at most 30 classes, got {self.n_classes}. "
                "Use method='inline' for more classes."
            )

        self.classifier_ = None

    def _build_classifier(self):
        if self.classifier_ is not None:
            return

        name = 'mygbt'
        n_features = self.n_features
        n_classes = self.n_classes
        feature_dtype = self.dtype

        model_init = self.save(name=name)

        return_type = 'int32_t' if self.is_classifier else 'float'

        if self.is_classifier:
            wrapper_functions = [
                f"""
                {return_type}
                predict_wrapper(const float *values, int length) {{
                    if (length != {n_features}) return -1;
                    {feature_dtype} features[{n_features}];
                    for (int i=0; i<length; i++) {{
                        features[i] = ({feature_dtype})values[i];
                    }}
                    return {name}_predict(features, length);
                }}""",
                f"""
                int
                predict_proba_wrapper(const float *values, int length, float *outputs, int n_outputs) {{
                    if (length != {n_features}) return -1;
                    {feature_dtype} features[{n_features}];
                    for (int i=0; i<length; i++) {{
                        features[i] = ({feature_dtype})values[i];
                    }}
                    return {name}_predict_proba(features, length, outputs, n_outputs);
                }}
                """,
            ]
            proba_func = 'predict_proba_wrapper(values, length, outputs, N_CLASSES)'
        else:
            wrapper_functions = [
                f"""
                float
                regress_wrapper(const float *values, int length) {{
                    if (length != {n_features}) return NAN;
                    {feature_dtype} features[{n_features}];
                    for (int i=0; i<length; i++) {{
                        features[i] = ({feature_dtype})values[i];
                    }}
                    return {name}_predict(features, length);
                }}
                """,
            ]
            proba_func = None

        code = '\n'.join([model_init] + wrapper_functions)

        predict_func = 'predict_wrapper(values, length)'
        regress_func = 'regress_wrapper(values, length)'

        if self.is_classifier:
            call_func = predict_func
        else:
            call_func = regress_func
            proba_func = None

        self.classifier_ = common.CompiledClassifier(code, name=name,
            call=call_func, proba_call=proba_func,
            out_dtype=self.out_dtype, n_classes=self.n_classes,
        )

    def _check_features(self, X):
        import numpy
        X = numpy.atleast_2d(X)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}")
        return X

    def predict(self, X):
        X = self._check_features(X)
        self._build_classifier()

        if self.is_classifier:
            predictions = self.classifier_.predict(X)
            if self.classes_ is not None:
                predictions = self.classes_[predictions]
        else:
            predictions = self.classifier_.regress(X)

        return predictions

    def predict_proba(self, X):
        X = self._check_features(X)
        self._build_classifier()

        if not self.is_classifier:
            raise ValueError("Cannot call predict_proba on a Regressor")

        probabilities = self.classifier_.predict_proba(X)
        return probabilities

    def save(self, name=None, file=None, format='c', include_proba=True):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        if format != 'c':
            raise ValueError(f"Only format='c' is supported for GradientBoosting, got '{format}'")

        if self.method == 'inline':
            code = generate_c_inlined(
                forest=self.forest_,
                name=name,
                n_features=self.n_features,
                learning_rate=self.learning_rate,
                initial_value=self.initial_value,
                n_classes=self.n_classes,
                dtype=self.dtype,
                classifier=self.is_classifier,
                include_proba=include_proba,
            )
        else:  # loadable
            code = generate_c_loadable(
                forest=self.forest_,
                name=name,
                n_features=self.n_features,
                learning_rate=self.learning_rate,
                initial_value=self.initial_value,
                n_classes=self.n_classes,
                dtype=self.dtype,
                classifier=self.is_classifier,
                include_proba=include_proba,
            )

        if file:
            with open(file, 'w') as f:
                f.write(code)

        return code
