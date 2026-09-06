#!/usr/bin/env python3
"""
Generate benchmark models for emlearn Zephyr timing tests.

Creates RandomForest and GradientBoosting classifiers with configurable size
for benchmarking inference latency on MCU targets.
"""

import emlearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification


def generate_model(
    model_type='rf',
    n_estimators=10,
    max_depth=5,
    n_features=10,
    name='benchmark_model',
    method='inline',
):
    """Generate and save a model for benchmarking.

    Args:
        model_type: Model type ('rf' for RandomForest, 'gbt' for GradientBoosting).
        n_estimators: Number of trees/estimators.
        max_depth: Maximum tree depth.
        n_features: Number of input features.
        name: Model name for C header.
        method: Compilation method ('inline' or 'loadable').

    Returns:
        Tuple of (classifier, X, y, accuracy).
    """
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=2,
        random_state=42
    )

    # Scale to int16 range for embedded use
    X = (X * 100).astype(np.int16)

    # Train model based on type
    if model_type == 'gbt':
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    else:  # rf
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    clf.fit(X, y)

    # Verify accuracy
    accuracy = clf.score(X, y)
    print(f"Training accuracy: {accuracy:.3f}")

    # Convert to emlearn
    cmodel = emlearn.convert(clf, method=method)

    # Save model header
    header_path = f'{name}.h'
    cmodel.save(file=header_path, name=name)
    print(f"Saved model to {header_path}")

    # Save test data for benchmark
    test_data_path = f'{name}_testdata.h'
    save_test_data(X[:100], test_data_path, name, n_classes=len(np.unique(y)))
    print(f"Saved test data to {test_data_path}")

    return clf, X, y, accuracy


def save_test_data(X, path, name, n_classes=None):
    """Save test data as C header for embedded benchmark."""
    n_samples, n_features = X.shape

    with open(path, 'w') as f:
        f.write(f"// Auto-generated test data for {name}\n")
        f.write(f"#ifndef {name.upper()}_TESTDATA_H\n")
        f.write(f"#define {name.upper()}_TESTDATA_H\n\n")
        f.write(f"#define {name.upper()}_N_SAMPLES {n_samples}\n")
        f.write(f"#define {name.upper()}_N_FEATURES {n_features}\n")
        if n_classes:
            f.write(f"#define {name.upper()}_N_CLASSES {n_classes}\n")
        f.write("\n")
        f.write(f"static const int16_t {name}_test_data[{n_samples}][{n_features}] = {{\n")

        for i, row in enumerate(X):
            values = ', '.join(str(int(v)) for v in row)
            comma = ',' if i < n_samples - 1 else ''
            f.write(f"    {{ {values} }}{comma}\n")

        f.write("};\n\n")
        f.write("#endif\n")


def generate_all_configs(configs, output_dir='.'):
    """Generate models for all configurations.

    Args:
        configs: List of model configuration dictionaries.
        output_dir: Output directory for generated files.

    Returns:
        List of (config, accuracy) tuples.
    """
    import os
    results = []

    for config in configs:
        name = config['name']
        print(f"\nGenerating {name}...")

        # Change to output directory
        original_dir = os.getcwd()
        os.chdir(output_dir)

        try:
            _, _, _, accuracy = generate_model(
                model_type=config['type'],
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                n_features=config.get('n_features', 10),
                name=name,
                method=config.get('method', 'inline'),
            )
            results.append((config, accuracy))
        finally:
            os.chdir(original_dir)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate benchmark model')
    parser.add_argument('--type', choices=['rf', 'gbt'], default='rf',
                        help='Model type (rf=RandomForest, gbt=GradientBoosting)')
    parser.add_argument('--trees', type=int, default=10, help='Number of trees')
    parser.add_argument('--depth', type=int, default=5, help='Max tree depth')
    parser.add_argument('--features', type=int, default=10, help='Number of features')
    parser.add_argument('--name', default='benchmark_model', help='Model name')
    parser.add_argument('--method', choices=['inline', 'loadable'], default='inline',
                        help='Compilation method')
    parser.add_argument('--all', action='store_true',
                        help='Generate all standard benchmark configurations')

    args = parser.parse_args()

    if args.all:
        # Generate all standard configurations
        configs = [
            {'name': 'gbt_small', 'type': 'gbt', 'n_estimators': 5, 'max_depth': 3},
            {'name': 'rf_small', 'type': 'rf', 'n_estimators': 5, 'max_depth': 3},
            {'name': 'gbt_medium', 'type': 'gbt', 'n_estimators': 10, 'max_depth': 5},
            {'name': 'rf_medium', 'type': 'rf', 'n_estimators': 10, 'max_depth': 5},
            {'name': 'gbt_large', 'type': 'gbt', 'n_estimators': 20, 'max_depth': 5},
            {'name': 'rf_large', 'type': 'rf', 'n_estimators': 20, 'max_depth': 5},
        ]
        results = generate_all_configs(configs)
        print("\n" + "=" * 50)
        print("Summary:")
        for config, accuracy in results:
            print(f"  {config['name']}: accuracy={accuracy:.3f}")
    else:
        generate_model(
            model_type=args.type,
            n_estimators=args.trees,
            max_depth=args.depth,
            n_features=args.features,
            name=args.name,
            method=args.method,
        )
