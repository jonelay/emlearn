#!/usr/bin/env python
# coding: utf-8

"""
Model size for tree ensembles
=============================

`GradientBoosting` and `RandomForest` are both tree ensembles, but they
spend program memory very differently. This example measures the compiled
size of each on three microcontroller targets, for both inference
strategies that emlearn supports (``inline`` and ``loadable``).

The numbers are model size, not program size: an empty reference program is
compiled for every target and subtracted, so the toolchain and C runtime
overhead does not show up in the comparison.
"""

import os.path

import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.datasets import make_classification, make_regression

import emlearn
from emlearn.evaluate.size import get_program_size, check_build_tools

try:
    # When executed as regular .py script
    here = os.path.dirname(__file__)
except NameError:
    # When executed as Jupyter notebook / Sphinx Gallery
    here = os.getcwd()

results_file = os.path.join(here, 'tree_ensemble_size_benchmark.csv')

# %%
# What is measured
# ----------------
#
# Four estimators, so that classification and regression can be compared
# separately. Both tasks use the same synthetic dataset shape, which keeps
# the feature count out of the comparison.

N_FEATURES = 10

PLATFORMS = [
    ('avr', 'atmega2560'),
    ('arm', 'Cortex-M0'),
    ('arm', 'Cortex-M4F'),
]

MODELS = {
    'gbt_classifier': GradientBoostingClassifier,
    'gbt_regressor': GradientBoostingRegressor,
    'rf_classifier': RandomForestClassifier,
    'rf_regressor': RandomForestRegressor,
}

CONFIGS = [
    dict(n_estimators=5, max_depth=3),
    dict(n_estimators=10, max_depth=3),
    dict(n_estimators=10, max_depth=5),
]

METHODS = ['inline', 'loadable']


# %%
# Measuring one model
# -------------------
#
# The generated model is compiled into a minimal program that calls
# ``predict`` once. Without that call the linker would drop the model as
# unused, and the measurement would be zero for every configuration.

def build_program(model_code, model_name, features_length, dtype, is_classifier):
    """Wrap generated model code in a minimal program that uses it."""

    return_type = 'int32_t' if is_classifier else 'float'

    return f"""
    #include <stdint.h>
    #include <math.h>

    {model_code}

    static {dtype} features[{features_length}] = {{0, }};

    int main()
    {{
        {return_type} out = {model_name}_predict(features, {features_length});
        return (int)out;
    }}
    """


def empty_program():
    """Reference program with no model, used as the size baseline."""

    return """
    #include <stdint.h>
    #include <math.h>

    int main()
    {
        return 0;
    }
    """


def measure_model(model, method, platform, mcu, is_classifier):
    """Compiled flash/RAM size of one model, baseline already subtracted."""

    # 'loadable' stores the tree data as arrays, and only supports int16
    dtype = 'int16_t' if method == 'loadable' else 'float'

    model_name = f'sizecheck_{method}_{dtype}'
    cmodel = emlearn.convert(model, method=method, dtype=dtype)
    model_code = cmodel.save(name=model_name)

    program = build_program(
        model_code, model_name, N_FEATURES, dtype, is_classifier,
    )
    return get_program_size(program, platform=platform, mcu=mcu)


def train(model_cls, config, is_classifier, method):
    """Fit an estimator on synthetic data shaped for the target dtype."""

    if is_classifier:
        X, y = make_classification(
            n_samples=100, n_features=N_FEATURES, random_state=42,
        )
    else:
        X, y = make_regression(
            n_samples=100, n_features=N_FEATURES, random_state=42,
        )

    if method == 'loadable':
        # Scale into the int16 range that the loadable inference expects
        X = (X * 1000).astype(numpy.int16)

    model = model_cls(random_state=42, **config)
    model.fit(X, y)
    return model


# %%
# Running the sweep
# -----------------
#
# Cross-compilers are not available everywhere, so results are cached in a
# CSV next to this script. When a toolchain is missing the cached numbers
# are loaded instead, which keeps the documentation build reproducible.

def run_experiments():
    rows = []

    for platform, mcu in PLATFORMS:
        baseline = get_program_size(empty_program(), platform=platform, mcu=mcu)

        for model_name, model_cls in MODELS.items():
            model_type, task = model_name.split('_')
            is_classifier = task == 'classifier'

            for config in CONFIGS:
                for method in METHODS:
                    model = train(model_cls, config, is_classifier, method)

                    sizes = measure_model(
                        model, method, platform, mcu, is_classifier,
                    )

                    rows.append(dict(
                        model_type=model_type,
                        task=task,
                        method=method,
                        platform=platform,
                        mcu=mcu,
                        **config,
                        flash=sizes['flash'] - baseline['flash'],
                        ram=sizes['ram'] - baseline['ram'],
                    ))

    return pandas.DataFrame.from_records(rows)


# check_build_tools returns a human-readable string, or None when the
# toolchain is complete
missing_tools = sorted({
    check_build_tools(platform) for platform, _ in PLATFORMS
} - {None})

if missing_tools:
    for message in missing_tools:
        print(f"WARNING: {message}")
    print("Loading cached results from", results_file)
    results = pandas.read_csv(results_file)
else:
    results = run_experiments()
    results.to_csv(results_file, index=False)
    print("Ran experiments. Results written to", results_file)

results['target'] = results.platform + '/' + results.mcu
print(results.head())


# %%
# Model size by target
# --------------------
#
# ``inline`` generates the tree as C code, so it lands in flash and uses
# almost no RAM. ``loadable`` stores the same tree as data, which moves part
# of the cost into RAM but keeps a single shared inference function.

def plot_sizes(results, value='flash'):

    g = seaborn.catplot(
        data=results,
        kind='bar',
        x='model_type',
        y=value,
        hue='method',
        col='target',
        row='task',
        height=3.5,
        aspect=1.2,
        errorbar=None,
    )
    g.figure.suptitle(f"Tree ensemble {value} usage", y=1.02)

    for ax in g.axes.flat:
        ax.grid(True, which='major', axis='y')
        ax.set_axisbelow(True)

    return g.figure


fig = plot_sizes(results, value='flash')
fig.savefig(os.path.join(here, 'example-tree-ensemble-size-flash.png'),
            bbox_inches='tight')

# %%
# The same breakdown for RAM.

fig = plot_sizes(results, value='ram')
fig.savefig(os.path.join(here, 'example-tree-ensemble-size-ram.png'),
            bbox_inches='tight')


# %%
# How the two ensembles compare
# -----------------------------
#
# A GradientBoosting ensemble stores one float leaf value per node, and its
# classifier calls ``expf`` to turn the summed score into a probability.
# A RandomForest classifier can vote on class indices instead, which avoids
# both the float leaves and libm.

summary = (
    results
    .groupby(['task', 'model_type', 'method'])[['flash', 'ram']]
    .mean()
    .round(0)
    .astype(int)
)
print(summary.to_string())
