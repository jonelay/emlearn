#!/usr/bin/env python
# coding: utf-8

"""
Inference time for tree ensembles
=================================

How long does a tree ensemble take to classify one sample, and what does
that buy in accuracy? This example compares `GradientBoosting` against
`RandomForest` on two reference datasets, timing the generated C code
rather than the scikit-learn estimator.

Timings are measured on the host, by compiling the model into a benchmark
program. Relative differences between models carry over to a
microcontroller; absolute numbers do not.
"""

import os.path

import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

import emlearn
from emlearn.evaluate.performance import benchmark_inference_time
from emlearn.examples.datasets.sonar import load_sonar_dataset

try:
    # When executed as regular .py script
    here = os.path.dirname(__file__)
except NameError:
    # When executed as Jupyter notebook / Sphinx Gallery
    here = os.getcwd()

results_file = os.path.join(here, 'tree_ensemble_perf_benchmark.csv')

# Timing repetitions. Higher is more stable, but slower to run
N_REPETITIONS = 100

# %%
# Datasets
# --------
#
# Sonar is a wide binary problem (60 features), Wine is a narrow
# multi-class one (13 features, 3 classes). Feature count and class count
# are the two things that move tree ensemble inference cost the most.

def load_sonar():
    data = load_sonar_dataset()
    feature_columns = [c for c in data.columns if c.startswith('b.')]

    X = data[feature_columns].values.astype('float32')
    y = LabelEncoder().fit_transform(data['label'].astype(str))
    return X, y


def load_wine_data():
    wine = load_wine()
    return wine.data.astype('float32'), wine.target


DATASETS = {
    'sonar': load_sonar,
    'wine': load_wine_data,
}

CONFIGS = [
    dict(n_estimators=10, max_depth=3),
    dict(n_estimators=20, max_depth=5),
    dict(n_estimators=50, max_depth=7),
]

MODELS = {
    'gbt': GradientBoostingClassifier,
    'rf': RandomForestClassifier,
}

METHODS = ['inline', 'loadable']


# %%
# Measuring one configuration
# ---------------------------
#
# The ``loadable`` strategy only supports int16 features, so the data is
# quantized before training. Training on the quantized values keeps the
# split thresholds consistent with what the C model will see.

def prepare_data(X, method):
    """Quantize features when the inference method requires it."""

    if method == 'loadable':
        return (X * 100).astype(numpy.int16), 'int16_t'
    return X, 'float'


def evaluate(model_cls, config, method, X_train, X_test, y_train, y_test):
    """Train, convert, and measure accuracy and inference time."""

    X_train_m, dtype = prepare_data(X_train, method)
    X_test_m, _ = prepare_data(X_test, method)

    model = model_cls(random_state=42, **config)
    model.fit(X_train_m.astype(float), y_train)

    cmodel = emlearn.convert(model, method=method, dtype=dtype)

    # Accuracy of the generated C model, and of the estimator it came from
    c_accuracy = accuracy_score(y_test, cmodel.predict(X_test_m.astype(float)))
    sklearn_accuracy = accuracy_score(
        y_test, model.predict(X_test_m.astype(float)),
    )

    timing = benchmark_inference_time(
        cmodel=cmodel,
        X=X_test_m,
        n_reps=N_REPETITIONS,
        is_classifier=True,
        feature_dtype=dtype,
    )

    return dict(
        accuracy=c_accuracy,
        sklearn_accuracy=sklearn_accuracy,
        time_mean_us=timing['mean_us'],
        time_std_us=timing['std_us'],
    )


# %%
# Running the sweep
# -----------------
#
# Both the C accuracy and the estimator accuracy are recorded, so that the
# cost of conversion stays visible instead of being folded into the
# comparison. The assertion below is a coarse sanity check for a broken
# conversion, not a claim that the two agree exactly.

ACCURACY_TOLERANCE = 0.15


def run_experiments():
    rows = []

    for dataset_name, loader in DATASETS.items():
        X, y = loader()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42,
        )

        for model_type, model_cls in MODELS.items():
            for config in CONFIGS:
                for method in METHODS:
                    measured = evaluate(
                        model_cls, config, method,
                        X_train, X_test, y_train, y_test,
                    )

                    deviation = abs(
                        measured['accuracy'] - measured['sklearn_accuracy']
                    )
                    assert deviation <= ACCURACY_TOLERANCE, \
                        (f"{dataset_name}/{model_type}/{method}: C model"
                         f" deviates from estimator by {deviation:.3f}")

                    rows.append(dict(
                        dataset=dataset_name,
                        model_type=model_type,
                        method=method,
                        n_features=X.shape[1],
                        **config,
                        **measured,
                    ))

    return pandas.DataFrame.from_records(rows)


results = run_experiments()
results.to_csv(results_file, index=False)
print("Ran experiments. Results written to", results_file)

print(results[['dataset', 'model_type', 'method',
               'n_estimators', 'accuracy', 'time_mean_us']].to_string())


# %%
# Inference time
# --------------
#
# ``inline`` compiles the tree into nested if/else, which the C compiler can
# lay out well. ``loadable`` walks a node array instead, trading speed for
# the ability to swap the model out without recompiling.

g = seaborn.catplot(
    data=results,
    kind='bar',
    x='n_estimators',
    y='time_mean_us',
    hue='model_type',
    col='method',
    row='dataset',
    height=3.5,
    aspect=1.3,
    errorbar=None,
    sharey=False,
)
g.set_axis_labels("Number of estimators", "Inference time (µs)")
g.figure.suptitle("Tree ensemble inference time", y=1.02)

for ax in g.axes.flat:
    ax.grid(True, which='major', axis='y')
    ax.set_axisbelow(True)

g.figure.savefig(os.path.join(here, 'example-tree-ensemble-perf-time.png'),
                 bbox_inches='tight')


# %%
# Accuracy against inference time
# -------------------------------
#
# The useful question is not which model is fastest, but which reaches an
# accuracy target for the least time. Points towards the upper left are the
# better trade-offs.

fig, axes = plt.subplots(
    1, len(DATASETS), figsize=(6 * len(DATASETS), 4.5), squeeze=False,
)

for ax, dataset_name in zip(axes.flat, DATASETS):
    subset = results[results.dataset == dataset_name]

    seaborn.scatterplot(
        data=subset,
        x='time_mean_us',
        y='accuracy',
        hue='model_type',
        style='method',
        s=110,
        ax=ax,
    )
    ax.set_xscale('log')
    ax.set_title(dataset_name)
    ax.set_xlabel("Inference time (µs, log scale)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, which='major')
    ax.set_axisbelow(True)

fig.suptitle("Accuracy vs inference time")
fig.tight_layout()
fig.savefig(os.path.join(here, 'example-tree-ensemble-perf-pareto.png'),
            bbox_inches='tight')


# %%
# Summary
# -------
#
# Averaged over the configurations tested. GradientBoosting reaches
# comparable accuracy with shallower trees, but pays for the probability
# conversion on every prediction.

summary = (
    results
    .groupby(['dataset', 'model_type', 'method'])[['accuracy', 'time_mean_us']]
    .mean()
    .round(3)
)
print(summary.to_string())
