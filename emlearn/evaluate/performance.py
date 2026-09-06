
"""
Performance benchmarking utilities for tree ensemble models
============================================================

Functions for measuring actual inference time of converted models.
"""

import os
import subprocess
import tempfile
import numpy as np
from typing import Dict, Optional, Tuple

from emlearn import common


# C benchmark template for timing tree model inference
BENCHMARK_TEMPLATE = '''
#include "{model_header}"
#include <eml_benchmark.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_SAMPLES {n_samples}
#define N_FEATURES {n_features}
#define N_REPS {n_reps}

// Test data
static {feature_dtype} test_features[N_SAMPLES][N_FEATURES] = {{
{features_data}
}};

int main() {{
    float times[N_REPS];

    // Warmup run
    for (int i = 0; i < N_SAMPLES; i++) {{
        volatile {return_type} pred = {predict_call}(test_features[i], N_FEATURES);
        (void)pred;
    }}

    // Timed runs
    for (int rep = 0; rep < N_REPS; rep++) {{
        int64_t start = eml_benchmark_micros();

        for (int i = 0; i < N_SAMPLES; i++) {{
            volatile {return_type} pred = {predict_call}(test_features[i], N_FEATURES);
            (void)pred;
        }}

        int64_t end = eml_benchmark_micros();
        times[rep] = (float)(end - start) / N_SAMPLES;
    }}

    // Calculate statistics
    float sum = 0.0f;
    float min_time = times[0];
    float max_time = times[0];

    for (int i = 0; i < N_REPS; i++) {{
        sum += times[i];
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }}

    float mean = sum / N_REPS;

    float variance = 0.0f;
    for (int i = 0; i < N_REPS; i++) {{
        float diff = times[i] - mean;
        variance += diff * diff;
    }}
    float std = sqrtf(variance / N_REPS);

    // Output: mean,std,min,max
    printf("%.6f,%.6f,%.6f,%.6f\\n", (double)mean, (double)std, (double)min_time, (double)max_time);

    return 0;
}}
'''


def format_features_array(X: np.ndarray, dtype: str = 'float') -> str:
    """Format numpy array as C array initializer.

    Args:
        X: 2D numpy array of shape (n_samples, n_features).
        dtype: C data type ('float' or 'int16_t').

    Returns:
        C array initializer string.
    """
    lines = []
    for row in X:
        if dtype == 'int16_t':
            values = ', '.join(str(int(v)) for v in row)
        else:
            values = ', '.join(f'{v:.8f}f' for v in row)
        lines.append(f'    {{ {values} }}')
    return ',\n'.join(lines)


def benchmark_inference_time(
    cmodel,
    X: np.ndarray,
    n_reps: int = 100,
    model_name: str = 'model',
    is_classifier: bool = True,
    feature_dtype: str = 'float',
) -> Dict[str, float]:
    """Measure actual C inference time for a converted model.

    Compiles and runs a benchmark program that times model inference
    using eml_benchmark_micros() for microsecond-precision timing.

    Args:
        cmodel: Converted emlearn model (e.g., from emlearn.convert()).
        X: Test features as 2D array of shape (n_samples, n_features).
            A subset of samples is used to keep benchmark fast.
        n_reps: Number of timing repetitions for statistical accuracy.
        model_name: Name used in generated C code.
        is_classifier: True for classifier, False for regressor.
        feature_dtype: C data type for features ('float' or 'int16_t').

    Returns:
        Dictionary with timing statistics in microseconds per sample:
            - mean_us: Mean inference time
            - std_us: Standard deviation
            - min_us: Minimum time
            - max_us: Maximum time

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import emlearn
        >>> clf = RandomForestClassifier(n_estimators=10)
        >>> clf.fit(X_train, y_train)
        >>> cmodel = emlearn.convert(clf)
        >>> timing = benchmark_inference_time(cmodel, X_test[:50])
        >>> print(f"Mean inference time: {timing['mean_us']:.2f} µs")
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_samples, n_features = X.shape

    # Limit samples to keep benchmark fast
    max_samples = 100
    if n_samples > max_samples:
        X = X[:max_samples]
        n_samples = max_samples

    # Generate model C code
    model_code = cmodel.save(name=model_name)

    # Determine return type and predict function
    return_type = 'int32_t' if is_classifier else 'float'
    predict_call = f'{model_name}_predict'

    # Generate benchmark C code
    features_data = format_features_array(X, dtype=feature_dtype)
    benchmark_code = BENCHMARK_TEMPLATE.format(
        model_header=f'{model_name}.h',
        n_samples=n_samples,
        n_features=n_features,
        n_reps=n_reps,
        features_data=features_data,
        feature_dtype=feature_dtype,
        return_type=return_type,
        predict_call=predict_call,
    )

    # Compile and run benchmark
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write model header
        model_path = os.path.join(temp_dir, f'{model_name}.h')
        with open(model_path, 'w') as f:
            f.write(model_code)

        # Write benchmark code
        bench_path = os.path.join(temp_dir, 'bench.c')
        with open(bench_path, 'w') as f:
            f.write(benchmark_code)

        # Compile
        include_dir = common.get_include_dir()
        bin_path = common.compile_executable(
            code_file=bench_path,
            out_dir=temp_dir,
            name='bench',
            include_dirs=[temp_dir, include_dir],
        )

        # Run benchmark
        result = subprocess.check_output([bin_path], text=True)

    # Parse output: mean,std,min,max
    values = result.strip().split(',')
    mean_us, std_us, min_us, max_us = [float(v) for v in values]

    return {
        'mean_us': mean_us,
        'std_us': std_us,
        'min_us': min_us,
        'max_us': max_us,
    }


def benchmark_model_performance(
    sklearn_model,
    X: np.ndarray,
    method: str = 'inline',
    n_reps: int = 100,
) -> Dict[str, float]:
    """Convenience function to benchmark a sklearn model end-to-end.

    Converts the model and measures inference time in a single call.

    Args:
        sklearn_model: Fitted sklearn model (RandomForest, GradientBoosting, etc.).
        X: Test features for benchmarking.
        method: Conversion method ('inline' or 'loadable').
        n_reps: Number of timing repetitions.

    Returns:
        Dictionary with timing statistics (see benchmark_inference_time).
    """
    import emlearn

    cmodel = emlearn.convert(sklearn_model, method=method)

    # Determine if classifier
    model_type = type(sklearn_model).__name__
    is_classifier = 'Classifier' in model_type

    return benchmark_inference_time(
        cmodel=cmodel,
        X=X,
        n_reps=n_reps,
        is_classifier=is_classifier,
    )
