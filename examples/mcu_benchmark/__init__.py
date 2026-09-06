"""
MCU Comparative Benchmark Suite
================================

Comparative benchmarks for emlearn inference across Host/Renode/Hardware.
"""

from .common import (
    BenchmarkConfig,
    TimingResult,
    save_results,
    load_results,
    RESULTS_DIR,
    FIGURES_DIR,
)
from .model_configs import (
    MODEL_CONFIGS,
    QUICK_MODEL_CONFIGS,
    get_model_configs,
    get_platforms,
)

__all__ = [
    'BenchmarkConfig',
    'TimingResult',
    'save_results',
    'load_results',
    'RESULTS_DIR',
    'FIGURES_DIR',
    'MODEL_CONFIGS',
    'QUICK_MODEL_CONFIGS',
    'get_model_configs',
    'get_platforms',
]
