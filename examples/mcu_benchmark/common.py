"""
Common utilities for MCU Comparative Benchmarks
================================================

Shared datasets, utilities, and result saving functions used across all benchmarks.
"""

from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


# =============================================================================
# Duration Parsing and Timeout Support
# =============================================================================


def parse_duration(duration_str: str | None) -> float | None:
    """Parse duration string to seconds.

    Supports:
        - "5m" (minutes)
        - "300s" (seconds)
        - "300" (seconds, bare number)
        - None (returns None)

    Args:
        duration_str: Duration string like "5m", "300s", or "300".

    Returns:
        Duration in seconds, or None if input is None.

    Raises:
        ValueError: If duration string is invalid.

    Examples:
        >>> parse_duration("5m")
        300.0
        >>> parse_duration("30s")
        30.0
        >>> parse_duration("120")
        120.0
        >>> parse_duration(None)
        None
    """
    if duration_str is None:
        return None

    duration_str = duration_str.strip()

    if duration_str.endswith('m'):
        return float(duration_str[:-1]) * 60
    elif duration_str.endswith('s'):
        return float(duration_str[:-1])
    else:
        return float(duration_str)


class TestTimeout(Exception):
    """Exception raised when a test exceeds its time limit."""

    pass


T = TypeVar('T')


def run_with_timeout(
    func: Callable[[], T],
    timeout_seconds: float | None,
    default: T | None = None,
) -> tuple[T | None, bool]:
    """Run a function with a timeout.

    Uses SIGALRM on Unix systems. On timeout, returns (default, True).
    On success, returns (result, False).

    Note: This only works on Unix-like systems. On Windows, no timeout
    is applied and the function runs normally.

    Args:
        func: Zero-argument callable to run.
        timeout_seconds: Maximum time in seconds, or None for no limit.
        default: Value to return on timeout.

    Returns:
        Tuple of (result, timed_out) where:
        - result: Function return value, or default on timeout
        - timed_out: True if timeout occurred

    Examples:
        >>> def slow_fn():
        ...     import time
        ...     time.sleep(10)
        ...     return "done"
        >>> result, timed_out = run_with_timeout(slow_fn, 0.1)
        >>> timed_out
        True
        >>> result is None
        True
    """
    if timeout_seconds is None:
        return func(), False

    # Check if SIGALRM is available (Unix only)
    if not hasattr(signal, 'SIGALRM'):
        # Windows: run without timeout
        return func(), False

    def handler(signum, frame):
        raise TestTimeout()

    old_handler = signal.signal(signal.SIGALRM, handler)
    # Use minimum of 1 second for alarm (sub-second timeouts get rounded up)
    signal.alarm(max(1, int(timeout_seconds)))
    try:
        result = func()
        signal.alarm(0)
        return result, False
    except TestTimeout:
        return default, True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# Directory containing this script (for relative path resolution)
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
RUNS_DIR = SCRIPT_DIR / "runs"

# Default random state for reproducibility
RANDOM_STATE = 42

# Zephyr benchmark app path
BENCHMARK_APP_PATH = Path(__file__).parent.parent.parent / "platform_examples" / "zephyr" / "benchmark"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        platforms: List of platforms to benchmark (host, renode_nrf52840, nrf52dk_nrf52832)
        model_configs: List of model configurations to test
        quick: Whether running in quick mode (reduced configs)
    """
    platforms: list[str] | None = None
    model_configs: list[dict] | None = None
    quick: bool = False

    @classmethod
    def for_quick_mode(cls) -> 'BenchmarkConfig':
        """Create configuration for quick benchmark runs (Renode only)."""
        from .model_configs import QUICK_MODEL_CONFIGS
        return cls(
            platforms=['host', 'renode_nrf52840'],
            model_configs=QUICK_MODEL_CONFIGS,
            quick=True,
        )

    @classmethod
    def for_full_mode(cls) -> 'BenchmarkConfig':
        """Create configuration for full benchmark runs."""
        from .model_configs import MODEL_CONFIGS
        return cls(
            platforms=['host', 'renode_nrf52840', 'nrf52dk_nrf52832'],
            model_configs=MODEL_CONFIGS,
            quick=False,
        )


@dataclass
class TimingResult:
    """Result from a timing measurement.

    Attributes:
        success: Whether measurement succeeded
        min_ns: Minimum inference time (nanoseconds)
        avg_ns: Average inference time (nanoseconds)
        min_cycles: Minimum CPU cycles
        avg_cycles: Average CPU cycles
        iterations: Number of iterations measured
        timing_mode: Timing source used (cffi, dwt, systick, posix)
        error: Error message if failed
    """
    success: bool
    min_ns: float
    avg_ns: float
    min_cycles: int
    avg_cycles: int
    iterations: int
    timing_mode: str
    error: str | None = None


def measure_host_latency(
    cmodel,
    X_sample: np.ndarray,
    max_iterations: int = 100,
    target_time_ms: float = 500.0,
) -> TimingResult:
    """Measure inference latency on host with adaptive iteration count.

    Uses fewer iterations for slow models to keep total time reasonable.
    Targets ~500ms total measurement time by default.

    Args:
        cmodel: Converted emlearn model with predict() method.
        X_sample: Single sample to predict (shape: [1, n_features]).
        max_iterations: Maximum iterations (capped by target time).
        target_time_ms: Target total measurement time in milliseconds.

    Returns:
        TimingResult with timing measurements (in nanoseconds).
    """
    import time

    # Ensure we have a single sample
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)
    elif X_sample.shape[0] > 1:
        X_sample = X_sample[:1]

    # Warm-up run and measure baseline
    _ = cmodel.predict(X_sample)

    start = time.perf_counter_ns()
    _ = cmodel.predict(X_sample)
    single_ns = time.perf_counter_ns() - start

    # Calculate adaptive iteration count
    # Target total time / single inference time, capped at max_iterations
    if single_ns > 0:
        target_iterations = int((target_time_ms * 1_000_000) / single_ns)
        iterations = max(3, min(target_iterations, max_iterations))
    else:
        iterations = max_iterations

    # Run timed iterations
    times_ns = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        _ = cmodel.predict(X_sample)
        end = time.perf_counter_ns()
        times_ns.append(end - start)

    min_ns = float(min(times_ns))
    avg_ns = float(sum(times_ns) / len(times_ns))

    return TimingResult(
        success=True,
        min_ns=min_ns,
        avg_ns=avg_ns,
        min_cycles=int(min(times_ns)),  # Rough estimate assuming 1GHz
        avg_cycles=int(sum(times_ns) / len(times_ns)),
        iterations=iterations,
        timing_mode='cffi',
    )


def create_benchmark_result(
    platform: str,
    dataset: str,
    task: str,
    model_type: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float | None,
    score: float,
    score_type: str,
    flash_bytes: int,
    latency_ns: float = 0.0,
    method: str = 'inline',
    config_hash: str = '',
    **extra_fields,
) -> dict:
    """Create a standardized benchmark result dictionary.

    All benchmarks should use this to ensure consistent result schema.

    Args:
        platform: Platform name (host, renode_nrf52840, native_sim, etc.).
        dataset: Dataset name.
        task: Task type ('classification' or 'regression').
        model_type: Model type ('gbt' or 'rf').
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        learning_rate: Learning rate (None for RF).
        score: Model score (accuracy for classification, metric for regression).
        score_type: Score metric name ('accuracy', 'neg_mse', 'r2', etc.).
        flash_bytes: Model flash size in bytes.
        latency_ns: Inference latency in nanoseconds (0 if not measured).
        method: Compilation method ('inline' or 'loadable').
        config_hash: Configuration hash for deduplication.
        **extra_fields: Additional benchmark-specific fields.

    Returns:
        Standardized result dictionary.
    """
    result = {
        # Identity
        'platform': platform,
        'dataset': dataset,
        'task': task,
        'config_hash': config_hash,
        # Model configuration
        'model_type': model_type,
        'method': method,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate if learning_rate is not None else 0.0,
        # Performance metrics
        'score': score,
        'score_type': score_type,
        'flash_bytes': flash_bytes,
        'latency_ns': latency_ns,
    }
    # Add any extra benchmark-specific fields
    result.update(extra_fields)
    return result


def make_benchmark_dataset(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 2,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset for benchmarking.

    Creates a dataset suitable for embedded ML benchmarking with features
    scaled to int16 range.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        n_classes: Number of classes.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) where X is int16 features and y is labels.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=random_state,
    )

    # Scale to int16 range for embedded use
    X = (X * 100).astype(np.int16)

    return X, y


def get_benchmark_data(
    n_features: int = 10,
    n_samples: int = 500,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get train/test split for benchmark dataset.

    Args:
        n_features: Number of features.
        n_samples: Total number of samples.
        test_size: Fraction of data for testing.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X, y = make_benchmark_dataset(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_platform_type(platform: str) -> str:
    """Get platform type (host, renode, native, hardware) from platform name.

    Args:
        platform: Platform identifier (e.g., 'host', 'renode_nrf52840', 'native_sim', 'nrf52dk_nrf52832')

    Returns:
        Platform type: 'host', 'renode', 'native', or 'hardware'
    """
    if platform == 'host':
        return 'host'
    elif platform.startswith('renode'):
        return 'renode'
    elif platform.startswith('native'):
        return 'native'
    else:
        return 'hardware'


def get_timing_mode(platform: str) -> str:
    """Get timing mode for a platform.

    Args:
        platform: Platform identifier

    Returns:
        Timing mode: 'cffi', 'systick', or 'dwt'
    """
    if platform == 'host':
        return 'cffi'
    elif platform == 'native_sim':
        return 'systick'
    else:
        return 'dwt'


def save_results(
    df: pd.DataFrame,
    name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save benchmark results to CSV.

    Args:
        df: Results DataFrame.
        name: Base name for output file (without extension).
        output_dir: Directory for output files (defaults to results/).

    Returns:
        Path to saved CSV file.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{name}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return csv_path


def load_results(name: str, results_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load benchmark results from CSV.

    Args:
        name: Base name of results file (without extension).
        results_dir: Directory containing results (defaults to results/).

    Returns:
        Results DataFrame, or empty DataFrame if not found.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    csv_path = results_dir / f'{name}.csv'
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return pd.DataFrame()

    return pd.read_csv(csv_path)


def get_csv_schema() -> list[str]:
    """Return the column names for mcu_inference_timing.csv.

    Returns:
        List of column names in order.
    """
    return [
        'model_name',       # Model ID (e.g., "gbt_n10_d5")
        'model_type',       # "gbt" or "rf"
        'method',           # "inline" or "loadable"
        'n_estimators',     # Number of trees
        'max_depth',        # Max tree depth
        'n_features',       # Input features
        'platform',         # "host", "renode_nrf52840", "nrf52dk_nrf52832"
        'platform_type',    # "host", "renode", "native", "hardware"
        'timing_mode',      # "cffi", "systick", "dwt", "posix"
        'min_ns',           # Min inference time (ns)
        'avg_ns',           # Avg inference time (ns)
        'min_cycles',       # Min CPU cycles
        'avg_cycles',       # Avg CPU cycles
        'accuracy',         # Model accuracy
        'flash_bytes',      # Flash size
        'success',          # Whether measurement succeeded
        'error',            # Error message if failed
    ]


def create_result_row(
    model_config: dict,
    platform: str,
    timing: TimingResult,
    accuracy: float,
    flash_bytes: int,
) -> dict:
    """Create a result row dictionary.

    Args:
        model_config: Model configuration dict.
        platform: Platform name.
        timing: Timing result.
        accuracy: Model accuracy.
        flash_bytes: Flash size in bytes.

    Returns:
        Dictionary with all CSV columns.
    """
    return {
        'model_name': model_config['name'],
        'model_type': model_config['type'],
        'method': model_config.get('method', 'inline'),
        'n_estimators': model_config['n_estimators'],
        'max_depth': model_config['max_depth'],
        'n_features': model_config.get('n_features', 10),
        'platform': platform,
        'platform_type': get_platform_type(platform),
        'timing_mode': timing.timing_mode,
        'min_ns': timing.min_ns,
        'avg_ns': timing.avg_ns,
        'min_cycles': timing.min_cycles,
        'avg_cycles': timing.avg_cycles,
        'accuracy': accuracy,
        'flash_bytes': flash_bytes,
        'success': timing.success,
        'error': timing.error,
    }


def print_summary(df: pd.DataFrame) -> None:
    """Print summary of benchmark results.

    Args:
        df: Results DataFrame.
    """
    print("\n" + "=" * 70)
    print("MCU Benchmark Summary")
    print("=" * 70)

    for platform in df['platform'].unique():
        platform_df = df[df['platform'] == platform]
        print(f"\n{platform}:")
        print(f"  Models tested: {len(platform_df)}")

        for model_type in platform_df['model_type'].unique():
            type_df = platform_df[platform_df['model_type'] == model_type]
            print(f"  {model_type.upper()}:")
            print(f"    Avg inference: {type_df['avg_ns'].mean():.1f} ns")
            print(f"    Min inference: {type_df['min_ns'].min():.1f} ns")
            print(f"    Avg accuracy:  {type_df['accuracy'].mean():.3f}")


# =============================================================================
# Run Context and Artifact Management
# =============================================================================


@dataclass
class RunContext:
    """Context for a benchmark run with artifact management.

    Manages timestamped run directories, configuration persistence,
    and result saving.

    Attributes:
        run_dir: Path to the run directory.
        config: Run configuration dictionary.
        checkpoint_path: Path to checkpoint JSON.
    """

    run_dir: Path
    config: dict
    checkpoint_path: Path

    @classmethod
    def create(
        cls,
        base_dir: Path | None = None,
        config: dict | None = None,
        name: str = "sweep",
    ) -> RunContext:
        """Create new run context with timestamped directory.

        Args:
            base_dir: Base directory for runs (defaults to examples/mcu_benchmark).
            config: Configuration dictionary to save.
            name: Name suffix for run directory.

        Returns:
            New RunContext with initialized directories.
        """
        if base_dir is None:
            base_dir = SCRIPT_DIR

        if config is None:
            config = {}

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = base_dir / "runs" / f"{timestamp}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "results").mkdir(exist_ok=True)
        (run_dir / "figures").mkdir(exist_ok=True)

        # Save config
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2, default=str))

        return cls(
            run_dir=run_dir,
            config=config,
            checkpoint_path=run_dir / "checkpoint.json",
        )

    @classmethod
    def resume(cls, run_dir: Path) -> RunContext:
        """Resume from existing run directory.

        Args:
            run_dir: Path to existing run directory.

        Returns:
            RunContext loaded from existing run.

        Raises:
            FileNotFoundError: If run directory doesn't exist.
        """
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        config_path = run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
        else:
            config = {}

        return cls(
            run_dir=run_dir,
            config=config,
            checkpoint_path=run_dir / "checkpoint.json",
        )

    def save_results(self, benchmark: str, df: pd.DataFrame) -> Path:
        """Save benchmark results to CSV.

        Args:
            benchmark: Benchmark name (used as filename).
            df: Results DataFrame.

        Returns:
            Path to saved CSV file.
        """
        path = self.run_dir / "results" / f"{benchmark}.csv"
        df.to_csv(path, index=False)
        print(f"Results saved to {path}")
        return path

    def load_results(self, benchmark: str) -> pd.DataFrame:
        """Load benchmark results from CSV.

        Args:
            benchmark: Benchmark name.

        Returns:
            Results DataFrame, or empty DataFrame if not found.
        """
        path = self.run_dir / "results" / f"{benchmark}.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def save_figure(self, name: str, fig: Any) -> Path:
        """Save figure to run directory.

        Args:
            name: Figure name (without extension).
            fig: Plotly figure or matplotlib figure.

        Returns:
            Path to saved figure.
        """
        path = self.run_dir / "figures" / f"{name}.png"

        # Try plotly first, then matplotlib
        if hasattr(fig, 'write_image'):
            fig.write_image(str(path))
        elif hasattr(fig, 'savefig'):
            fig.savefig(str(path), dpi=150, bbox_inches='tight')
        else:
            raise TypeError(f"Unknown figure type: {type(fig)}")

        print(f"Figure saved to {path}")
        return path

    def get_log_path(self, name: str) -> Path:
        """Get path for a log file.

        Args:
            name: Log file name (without extension).

        Returns:
            Path to log file.
        """
        return self.run_dir / "logs" / f"{name}.log"

    def save_combined_results(self, results: list[pd.DataFrame]) -> Path:
        """Combine and save all benchmark results.

        Args:
            results: List of result DataFrames.

        Returns:
            Path to combined CSV file.
        """
        if not results:
            return self.run_dir / "results" / "combined.csv"

        combined = pd.concat(results, ignore_index=True)
        path = self.run_dir / "results" / "combined.csv"
        combined.to_csv(path, index=False)
        print(f"Combined results saved to {path}")
        return path


# =============================================================================
# Prepared Model and Extended Result Types
# =============================================================================


@dataclass
class PreparedModel:
    """Model prepared for MCU execution.

    Attributes:
        config: Model configuration dictionary.
        headers: Dict mapping filename to C header content.
        sklearn_model: Trained sklearn model.
        config_hash: Hash of the configuration.
        accuracy: Model accuracy on test set.
        flash_bytes: Estimated flash size in bytes.
    """

    config: dict
    headers: dict[str, str]
    sklearn_model: Any
    config_hash: str
    accuracy: float = 0.0
    flash_bytes: int = 0


@dataclass
class SweepResult:
    """Result from a single benchmark execution.

    Attributes:
        config_hash: Hash of the configuration.
        platform: Platform name (host, renode_nrf52840, etc.).
        dataset: Dataset name.
        model_type: Model type (gbt or rf).
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        learning_rate: Learning rate (GBT only).
        method: Compilation method (inline or loadable).
        accuracy: Model accuracy.
        timing_ns: Inference time in nanoseconds.
        flash_bytes: Flash size in bytes.
        ram_bytes: RAM usage in bytes.
        success: Whether execution succeeded.
        error: Error message if failed.
    """

    config_hash: str
    platform: str
    dataset: str
    model_type: str
    n_estimators: int
    max_depth: int
    learning_rate: float | None
    method: str
    accuracy: float
    timing_ns: float
    flash_bytes: int
    ram_bytes: int = 0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return asdict(self)


@dataclass
class SampleEfficiencyResult(SweepResult):
    """Result from sample efficiency benchmark.

    Additional attributes:
        threshold_reached: Whether accuracy threshold was reached.
        threshold: Target accuracy threshold.
    """

    threshold_reached: bool = False
    threshold: float = 0.0


@dataclass
class ParetoResult(SweepResult):
    """Result from Pareto efficiency benchmark.

    Additional attributes:
        is_pareto: Whether this point is on the Pareto frontier.
        dominated_by: List of config hashes that dominate this point.
    """

    is_pareto: bool = False
    dominated_by: list[str] = field(default_factory=list)


@dataclass
class CalibrationResult(SweepResult):
    """Result from probability calibration benchmark.

    Additional attributes:
        brier_score: Brier score (lower is better).
        expected_calibration_error: ECE (lower is better).
    """

    brier_score: float = 0.0
    expected_calibration_error: float = 0.0
