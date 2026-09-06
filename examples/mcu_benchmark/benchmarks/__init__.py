"""
MCU Benchmark Modules
=====================

Individual benchmark implementations for GBT vs RF comparison across platforms.

Available benchmarks:
- sample_efficiency: Measure trees needed to reach accuracy thresholds
- pareto_efficiency: Find accuracy vs flash size Pareto frontiers
- latency: Measure inference timing across platforms
- regression_accuracy: Compare MSE/R² on regression tasks
- probability_calibration: Measure Brier score and ECE
- size_constrained: Find best accuracy at fixed flash budgets
"""

from .sample_efficiency import run_sample_efficiency_benchmark
from .pareto_efficiency import run_pareto_benchmark
from .latency import run_latency_benchmark
from .regression_accuracy import run_regression_benchmark
from .probability_calibration import run_calibration_benchmark
from .size_constrained import run_size_constrained_benchmark
from .multiclass_latency import run_multiclass_latency_benchmark

# List of all available benchmarks
BENCHMARK_NAMES = [
    'sample_efficiency',
    'pareto_efficiency',
    'latency',
    'regression_accuracy',
    'probability_calibration',
    'size_constrained',
    'multiclass_latency',
]

# Mapping of benchmark names to functions
BENCHMARK_FUNCTIONS = {
    'sample_efficiency': run_sample_efficiency_benchmark,
    'pareto_efficiency': run_pareto_benchmark,
    'latency': run_latency_benchmark,
    'regression_accuracy': run_regression_benchmark,
    'probability_calibration': run_calibration_benchmark,
    'size_constrained': run_size_constrained_benchmark,
    'multiclass_latency': run_multiclass_latency_benchmark,
}

__all__ = [
    'BENCHMARK_NAMES',
    'BENCHMARK_FUNCTIONS',
    'run_sample_efficiency_benchmark',
    'run_pareto_benchmark',
    'run_latency_benchmark',
    'run_regression_benchmark',
    'run_calibration_benchmark',
    'run_size_constrained_benchmark',
    'run_multiclass_latency_benchmark',
]
