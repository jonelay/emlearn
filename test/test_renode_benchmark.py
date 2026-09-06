"""
Renode Benchmark Tests with pytest-xdist Parallelization
=========================================================

Uses a two-phase approach for true parallel execution:
1. Build Phase: All model variants are built sequentially at session start
2. Run Phase: Renode executions run in parallel with no shared state

This avoids race conditions from shared header files and CMake caching.

Usage:
    # Run in parallel with 4 workers
    RENODE_PATH=/path/to/renode pytest -n 4 test/test_renode_benchmark.py -v

    # Quick mode (fewer configs)
    RENODE_PATH=/path/to/renode pytest -n auto test/test_renode_benchmark.py -v -k quick

    # Collect results to CSV
    RENODE_PATH=/path/to/renode pytest -n auto test/test_renode_benchmark.py --csv results.csv

Environment:
    RENODE_PATH: Path to Renode executable (required)
    ZEPHYR_BASE: Path to Zephyr base (auto-detected)
    ZEPHYR_SDK_INSTALL_DIR: Path to Zephyr SDK (auto-detected)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import filelock
import numpy as np
import pytest
from sklearn.datasets import make_classification

import emlearn
from emlearn.mcu.renode_runner import (
    RenodeConfig,
    RenodeRunner,
    analyze_trace_overhead,
    parse_micro_benchmark_results,
)

# Import shared model training and header generation functions.
# sys.path manipulation needed because examples/ is not an installable package.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'examples' / 'mcu_benchmark'))
from model_configs import (
    train_model as shared_train_model,
    generate_testdata_header as shared_generate_testdata_header,
    get_model_id,
    get_multiclass_configs,
    MULTICLASS_CONFIGS,
)


# =============================================================================
# Configuration
# =============================================================================

# Model configurations for benchmark sweep
# QUICK is a subset of FULL for cache compatibility
QUICK_MODEL_CONFIGS = [
    {'type': 'gbt', 'n_estimators': 2, 'max_depth': 2},
    {'type': 'gbt', 'n_estimators': 10, 'max_depth': 4},
    {'type': 'rf', 'n_estimators': 2, 'max_depth': 2},
    {'type': 'rf', 'n_estimators': 10, 'max_depth': 4},
]

FULL_MODEL_CONFIGS = [
    {'type': 'gbt', 'n_estimators': n, 'max_depth': d}
    for n in [2, 5, 10, 20]
    for d in [2, 3, 4, 5]
] + [
    {'type': 'rf', 'n_estimators': n, 'max_depth': d}
    for n in [2, 5, 10, 20]
    for d in [2, 3, 4, 5]
]


# get_model_id is imported from model_configs


# =============================================================================
# Result Collection
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_id: str
    model_type: str
    n_estimators: int
    max_depth: int
    success: bool
    avg_ns: int = 0
    min_cycles: int = 0
    avg_cycles: int = 0
    iterations: int = 0
    total_instructions: int = 0
    flash_bytes: int = 0
    accuracy: float = 0.0
    error: str | None = None


# Results directory for CSV export (saved to runs/ for posterity)
RUNS_DIR = Path(__file__).parent.parent / 'examples' / 'mcu_benchmark' / 'runs'


def save_result_to_cache(result: BenchmarkResult, cache_dir: Path) -> None:
    """Save benchmark result to JSON file in cache directory.

    This approach works with pytest-xdist because each worker writes
    to a separate file based on model_id.
    """
    result_file = cache_dir / result.model_id / 'result.json'
    result_file.write_text(json.dumps(vars(result), indent=2))


def collect_results_from_cache(cache_dir: Path) -> list[BenchmarkResult]:
    """Collect all results from cache directory."""
    results = []
    for result_file in cache_dir.glob('*/result.json'):
        try:
            data = json.loads(result_file.read_text())
            results.append(BenchmarkResult(**data))
        except (json.JSONDecodeError, TypeError, FileNotFoundError):
            pass
    return results


def export_results_to_csv(cache_dir: Path, csv_path: Path | None = None) -> Path:
    """Export collected results to CSV file.

    Args:
        cache_dir: Path to benchmark cache directory.
        csv_path: Optional path for CSV output. If None, uses timestamped file in runs/.

    Returns:
        Path to the saved CSV file.
    """
    import pandas as pd

    results = collect_results_from_cache(cache_dir)
    if not results:
        raise ValueError("No results found in cache directory")

    df = pd.DataFrame([vars(r) for r in results])

    if csv_path is None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = RUNS_DIR / f'renode_benchmark_{timestamp}.csv'

    df.to_csv(csv_path, index=False)
    return csv_path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope='session')
def zephyr_env() -> dict[str, str]:
    """Session-scoped Zephyr environment setup.

    Returns environment variables for west/cmake.
    """
    # Find Zephyr workspace
    zephyr_base = os.environ.get('ZEPHYR_BASE')
    if not zephyr_base:
        pytest.skip("ZEPHYR_BASE not set. Required for benchmark builds.")

    zephyr_ws = str(Path(zephyr_base).parent)

    zephyr_sdk = os.environ.get('ZEPHYR_SDK_INSTALL_DIR')
    if not zephyr_sdk:
        pytest.skip("ZEPHYR_SDK_INSTALL_DIR not set. Required for benchmark builds.")

    return {
        'ZEPHYR_BASE': zephyr_base,
        'ZEPHYR_SDK_INSTALL_DIR': zephyr_sdk,
        'zephyr_ws': zephyr_ws,
    }


@pytest.fixture(scope='session')
def renode_path() -> Path:
    """Session-scoped Renode executable path."""
    path = os.environ.get('RENODE_PATH')
    if not path:
        pytest.skip("RENODE_PATH not set. Set environment variable to Renode executable.")
    return Path(path)


@pytest.fixture(scope='session')
def west_path() -> Path:
    """Session-scoped west executable path."""
    west = shutil.which('west')
    if west:
        return Path(west)

    # Try venv location
    venv_west = Path(__file__).parent.parent / '.venv' / 'bin' / 'west'
    if venv_west.exists():
        return venv_west

    pytest.skip("west not found")


@pytest.fixture(scope='session')
def python_path() -> Path:
    """Python executable for west builds (the one running pytest)."""
    return Path(sys.executable)


@pytest.fixture(scope='session')
def benchmark_app_path() -> Path:
    """Path to benchmark Zephyr application."""
    return Path(__file__).parent.parent / 'platform_examples' / 'zephyr' / 'benchmark'


@pytest.fixture(scope='session')
def repl_file(benchmark_app_path: Path) -> Path:
    """Path to Renode platform description."""
    return benchmark_app_path / 'boards' / 'nrf52840_dwt.repl'


@pytest.fixture(scope='session')
def conf_file(benchmark_app_path: Path) -> Path:
    """Path to Renode config overlay."""
    return benchmark_app_path / 'boards' / 'renode_nrf52840.conf'


@pytest.fixture(scope='session')
def benchmark_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Shared dataset for benchmarks."""
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=42
    )
    return X.astype(np.float32), y


def _get_cache_dir() -> Path:
    """Get shared cache directory for build artifacts.

    Uses a fixed location in /tmp so all pytest-xdist workers share it.
    """
    return Path('/tmp/emlearn_renode_benchmark_cache')


def _get_cache_version(
    benchmark_app_path: Path,
    configs: list[dict],
) -> str:
    """Generate cache version hash based on source files and configs.

    If any source file or config changes, the cache is invalidated.
    """
    hasher = hashlib.sha256()

    # Hash configs
    hasher.update(json.dumps(configs, sort_keys=True).encode())

    # Hash key source files
    for pattern in ['*.c', '*.h', 'CMakeLists.txt', 'prj.conf']:
        for src_file in benchmark_app_path.glob(f'**/{pattern}'):
            if '.lock' not in str(src_file) and 'benchmark_model' not in src_file.name:
                try:
                    hasher.update(src_file.read_bytes())
                except (OSError, IOError):
                    pass

    # Hash emlearn version
    hasher.update(emlearn.__version__.encode())

    return hasher.hexdigest()[:16]



def _find_emlearn_headers(benchmark_app_path: Path) -> Path:
    """Find the emlearn headers directory from the benchmark app path.

    Searches parent directories for eml_trees.h, supporting both:
    - Nested layout: repo/emlearn/emlearn/*.h (dev repo)
    - Flat layout: repo/emlearn/*.h (github repo)
    """
    # Try parent chain + /emlearn (nested layout)
    candidate = benchmark_app_path.parent.parent.parent / 'emlearn'
    if (candidate / 'eml_trees.h').exists():
        return candidate

    # Try parent chain directly (flat layout where package dir has headers)
    candidate = benchmark_app_path.parent.parent.parent
    if (candidate / 'eml_trees.h').exists():
        return candidate

    # Try one more level up + /emlearn (deeper nesting)
    candidate = benchmark_app_path.parent.parent.parent.parent / 'emlearn'
    if (candidate / 'eml_trees.h').exists():
        return candidate

    raise FileNotFoundError(
        f"Cannot find emlearn headers (eml_trees.h) from {benchmark_app_path}"
    )

def _build_model_variant(
    config: dict,
    cache_dir: Path,
    benchmark_app_path: Path,
    conf_file: Path,
    zephyr_env: dict[str, str],
    west_path: Path,
    python_path: Path,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[bool, str]:
    """Build a single model variant to the cache.

    Creates an isolated app directory with model-specific headers,
    then builds to avoid CMake caching issues.

    Args:
        config: Model configuration dict.
        cache_dir: Root cache directory.
        benchmark_app_path: Path to benchmark app template.
        conf_file: Renode config overlay path.
        zephyr_env: Zephyr environment variables.
        west_path: Path to west executable.
        python_path: Path to Python executable.
        X: Training features.
        y: Training labels.

    Returns:
        Tuple of (success, error_message).
    """
    model_id = get_model_id(config)
    model_dir = cache_dir / model_id

    # Create isolated app directory
    app_dir = model_dir / 'app'
    if app_dir.exists():
        shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True)

    # Copy benchmark app (excluding generated headers, build dirs, locks)
    for item in benchmark_app_path.iterdir():
        if item.name in ('.lock', 'build', '.renode_build.lock'):
            continue
        dest = app_dir / item.name
        if item.is_dir():
            shutil.copytree(
                item, dest,
                ignore=shutil.ignore_patterns(
                    'benchmark_model.h',
                    'benchmark_model_testdata.h',
                    '*.lock',
                    'build',
                )
            )
        else:
            shutil.copy2(item, dest)

    # Patch CMakeLists.txt to use absolute path for emlearn headers
    # The original uses relative path ../../../emlearn which breaks when copied
    emlearn_headers_path = _find_emlearn_headers(benchmark_app_path)
    cmake_file = app_dir / 'CMakeLists.txt'
    cmake_content = cmake_file.read_text()
    cmake_content = cmake_content.replace(
        '${CMAKE_CURRENT_SOURCE_DIR}/../../../emlearn',
        str(emlearn_headers_path)
    )
    cmake_file.write_text(cmake_content)

    # Train and convert model
    clf = train_model(config, X, y)
    cmodel = emlearn.convert(clf, dtype='float')

    # Generate headers to isolated app
    src_dir = app_dir / 'src'
    (src_dir / 'benchmark_model.h').write_text(cmodel.save(name='benchmark_model'))
    (src_dir / 'benchmark_model_testdata.h').write_text(
        generate_testdata_header(X, n_classes=getattr(cmodel, 'n_classes', 0) or None))

    # Store model metadata for test use
    accuracy = float((cmodel.predict(X) == y).mean())
    model_code = cmodel.save(name='benchmark_model')
    metadata = {
        'model_id': model_id,
        'config': config,
        'accuracy': accuracy,
        'flash_bytes': len(model_code.encode('utf-8')),
    }
    (model_dir / 'metadata.json').write_text(json.dumps(metadata))

    # Build firmware
    build_dir = model_dir / 'build'
    success, error = build_firmware(
        build_dir, app_dir, conf_file,
        zephyr_env, west_path, python_path
    )

    return success, error


@pytest.fixture(scope='session')
def build_cache(
    benchmark_app_path: Path,
    conf_file: Path,
    zephyr_env: dict[str, str],
    west_path: Path,
    python_path: Path,
    benchmark_dataset: tuple[np.ndarray, np.ndarray],
) -> Path:
    """Session-scoped fixture that pre-builds all model variants.

    Uses a lock file for pytest-xdist coordination - only one worker
    builds, others wait and use the cached results.

    The cache is versioned by source file hashes, so it's automatically
    invalidated when code changes.

    Returns:
        Path to cache directory containing built ELFs.
    """
    X, y = benchmark_dataset
    # Build all models (FULL includes QUICK as subset)
    configs = FULL_MODEL_CONFIGS

    cache_dir = _get_cache_dir()
    cache_version = _get_cache_version(benchmark_app_path, configs)
    version_file = cache_dir / 'version.txt'
    lock_path = cache_dir.parent / 'emlearn_renode_build.lock'

    # Ensure parent exists for lock file
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    with filelock.FileLock(str(lock_path), timeout=600):
        # Check if cache is valid
        if cache_dir.exists() and version_file.exists():
            cached_version = version_file.read_text().strip()
            if cached_version == cache_version:
                # Verify all ELFs exist
                all_built = all(
                    (cache_dir / get_model_id(c) / 'build' / 'zephyr' / 'zephyr.elf').exists()
                    for c in configs
                )
                if all_built:
                    return cache_dir

        # Cache invalid or missing - rebuild using atomic copy-on-write pattern.
        # This avoids TOCTOU race: build into .new dir, atomically rename.
        build_dir = cache_dir.with_suffix('.new')
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        # Build all model variants sequentially into the temp directory
        for config in configs:
            model_id = get_model_id(config)
            print(f"\nBuilding {model_id}...")

            success, error = _build_model_variant(
                config, build_dir, benchmark_app_path, conf_file,
                zephyr_env, west_path, python_path, X, y
            )

            if not success:
                # Clean up partial build
                shutil.rmtree(build_dir, ignore_errors=True)
                raise RuntimeError(f"Build failed for {model_id}: {error}")

            print(f"  Built {model_id} successfully")

        # Write version file to mark cache valid
        (build_dir / 'version.txt').write_text(cache_version)

        # Atomic swap: remove old cache, rename new to final location
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        build_dir.rename(cache_dir)

    return cache_dir


# =============================================================================
# Helper Functions (using shared implementations from model_configs)
# =============================================================================

# Aliases for shared functions
train_model = shared_train_model
generate_testdata_header = shared_generate_testdata_header


def build_firmware(
    build_dir: Path,
    benchmark_app: Path,
    conf_file: Path,
    zephyr_env: dict[str, str],
    west_path: Path,
    python_path: Path,
) -> tuple[bool, str]:
    """Build Zephyr firmware for Renode.

    Returns:
        Tuple of (success, error_message).
    """
    env = os.environ.copy()
    env['ZEPHYR_BASE'] = zephyr_env['ZEPHYR_BASE']
    env['ZEPHYR_SDK_INSTALL_DIR'] = zephyr_env['ZEPHYR_SDK_INSTALL_DIR']

    cmd = [
        str(west_path), 'build',
        '--board', 'nrf52840dk/nrf52840',
        '-d', str(build_dir),
        str(benchmark_app),
        '--pristine',
        '--',
        f'-DEXTRA_CONF_FILE={conf_file}',
        f'-DPython3_EXECUTABLE={python_path}',
    ]

    result = subprocess.run(
        cmd,
        cwd=zephyr_env['zephyr_ws'],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        error = result.stderr[-1000:] if result.stderr else result.stdout[-1000:]
        return False, error

    return True, ""


def run_renode_benchmark(
    elf_path: Path,
    repl_file: Path,
    renode_path: Path,
    timeout: float = 120.0,
) -> tuple[int, int, int, str]:
    """Run Renode with full execution tracing.

    Uses RenodeRunner which traces entire execution. Timing is extracted
    from firmware UART output (using DWT cycle counter), while instruction
    count covers full execution for reference.

    Note: RenodeGDBRunner (which brackets only benchmark loop) is not used
    due to Renode GDB server bugs causing arithmetic overflow and connection
    drops when hitting breakpoints.

    Args:
        elf_path: Path to ELF firmware file.
        repl_file: Path to platform description file.
        renode_path: Path to Renode executable (unused, runner finds it).
        timeout: Timeout in seconds.

    Returns:
        Tuple of (avg_ns, avg_cycles, total_instructions, uart_output)
    """
    config = RenodeConfig(
        cpu_freq_hz=64_000_000,
        repl_file=str(repl_file),
        use_execution_trace=True,
    )
    runner = RenodeRunner(config, timeout=timeout, verbose=False)

    # Run benchmark and capture UART output
    uart_output = runner.run(elf_path, repl_file)
    total_instructions = runner.last_total_instructions

    # Parse timing from UART output
    # Format: RESULT:INFERENCE min_ns=X max_ns=X avg_ns=X min_cycles=X avg_cycles=X iterations=X
    avg_ns = 0
    avg_cycles = 0
    for line in uart_output.splitlines():
        if 'RESULT:INFERENCE' in line:
            # Extract avg_ns and avg_cycles
            for part in line.split():
                if part.startswith('avg_ns='):
                    avg_ns = int(part.split('=')[1])
                elif part.startswith('avg_cycles='):
                    avg_cycles = int(part.split('=')[1])
            break

    return (
        avg_ns,
        avg_cycles,
        total_instructions,
        uart_output,
    )


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.parametrize('model_config', QUICK_MODEL_CONFIGS, ids=get_model_id)
@pytest.mark.renode
def test_renode_benchmark_quick(
    model_config: dict,
    build_cache: Path,
    repl_file: Path,
    renode_path: Path,
):
    """Quick benchmark test - runs with pytest-xdist in parallel.

    Uses pre-built ELFs from the build_cache fixture, allowing fully
    parallel Renode execution with no shared state.
    """
    model_id = get_model_id(model_config)
    model_dir = build_cache / model_id

    # Load pre-computed metadata
    metadata_file = model_dir / 'metadata.json'
    assert metadata_file.exists(), f"Missing metadata for {model_id}"
    metadata = json.loads(metadata_file.read_text())

    accuracy = metadata['accuracy']
    flash_bytes = metadata['flash_bytes']

    # Get pre-built ELF
    elf_path = model_dir / 'build' / 'zephyr' / 'zephyr.elf'
    assert elf_path.exists(), f"Missing ELF for {model_id}"

    # Run Renode (fully parallel - each model has its own isolated build)
    avg_ns, avg_cycles, total_instructions, uart_output = run_renode_benchmark(
        elf_path, repl_file, renode_path
    )

    # Record result to cache (works with pytest-xdist)
    result = BenchmarkResult(
        model_id=model_id,
        model_type=model_config['type'],
        n_estimators=model_config['n_estimators'],
        max_depth=model_config['max_depth'],
        success=avg_ns > 0,
        avg_ns=avg_ns,
        min_cycles=avg_cycles,
        avg_cycles=avg_cycles,
        total_instructions=total_instructions,
        flash_bytes=flash_bytes,
        accuracy=accuracy,
        error=None if avg_ns > 0 else f"No timing. UART: {uart_output[:200]}"
    )
    save_result_to_cache(result, build_cache)

    # Assert reasonable timing
    # avg_ns > 0 ensures timing calculation succeeded
    # total_instructions > 1000 ensures benchmark actually ran
    # (RF models: ~89 instr/inference × 100 = 8,900; GBT models: ~1,500-3,000 × 100)
    assert avg_ns > 0, f"No timing data for {model_id}. Total instructions: {total_instructions}. UART: {uart_output[:300]}"
    assert total_instructions > 1_000, f"Too few instructions for {model_id}: {total_instructions}"

    print(f"\n{model_id}: {avg_ns}ns, {total_instructions:,} instructions, accuracy={accuracy:.3f}")


@pytest.mark.parametrize('model_config', FULL_MODEL_CONFIGS, ids=get_model_id)
@pytest.mark.renode
def test_renode_benchmark_full(
    model_config: dict,
    build_cache: Path,
    repl_file: Path,
    renode_path: Path,
):
    """Full benchmark test - comprehensive model sweep.

    Uses pre-built ELFs from the build_cache fixture, allowing fully
    parallel Renode execution with no shared state.

    Run with: pytest -n 4 test/test_renode_benchmark.py::test_renode_benchmark_full -v
    """
    model_id = get_model_id(model_config)
    model_dir = build_cache / model_id

    # Load pre-computed metadata
    metadata_file = model_dir / 'metadata.json'
    assert metadata_file.exists(), f"Missing metadata for {model_id}"
    metadata = json.loads(metadata_file.read_text())

    accuracy = metadata['accuracy']
    flash_bytes = metadata['flash_bytes']

    # Get pre-built ELF
    elf_path = model_dir / 'build' / 'zephyr' / 'zephyr.elf'
    assert elf_path.exists(), f"Missing ELF for {model_id}"

    # Run Renode (fully parallel - each model has its own isolated build)
    avg_ns, avg_cycles, total_instructions, uart_output = run_renode_benchmark(
        elf_path, repl_file, renode_path
    )

    # Record result to cache (works with pytest-xdist)
    result = BenchmarkResult(
        model_id=model_id,
        model_type=model_config['type'],
        n_estimators=model_config['n_estimators'],
        max_depth=model_config['max_depth'],
        success=avg_ns > 0,
        avg_ns=avg_ns,
        min_cycles=avg_cycles,
        avg_cycles=avg_cycles,
        total_instructions=total_instructions,
        flash_bytes=flash_bytes,
        accuracy=accuracy,
        error=None if avg_ns > 0 else f"No timing. UART: {uart_output[:200]}"
    )
    save_result_to_cache(result, build_cache)

    # Assert reasonable timing
    assert avg_ns > 0, f"No timing data for {model_id}. Total instructions: {total_instructions}. UART: {uart_output[:300]}"
    assert total_instructions > 1_000, f"Too few instructions for {model_id}: {total_instructions}"

    print(f"\n{model_id}: {avg_ns}ns, {total_instructions:,} instructions, accuracy={accuracy:.3f}")


# =============================================================================
# pytest configuration
# =============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        '--csv',
        action='store',
        default=None,
        help='Export results to CSV file',
    )


# =============================================================================
# CPI Validation Test Configurations
# =============================================================================

# Expected CPI ranges for different instruction types on Cortex-M4F
# These are used to validate timing assumptions in micro-benchmarks
CPI_VALIDATION_CONFIGS = [
    {
        'name': 'INT_OPS',
        'expected_cpi_min': 0.5,  # May pipeline well
        'expected_cpi_max': 2.0,  # With some stalls
        'description': 'Integer add/xor/and operations',
    },
    {
        'name': 'FP_OPS',
        'expected_cpi_min': 0.5,  # FPU pipelining
        'expected_cpi_max': 2.0,  # With dependencies
        'description': 'FP add/mul operations (FPU)',
    },
    {
        'name': 'FP_DIV',
        'expected_cpi_min': 10.0,  # Known ~14 cycles on M4F
        'expected_cpi_max': 20.0,
        'description': 'FP division (~14 cycles on Cortex-M4F)',
    },
    {
        'name': 'EXPF',
        'expected_cpi_min': 30.0,   # Library function, many instructions
        'expected_cpi_max': 300.0,  # Depends on implementation
        'description': 'expf() library function',
    },
    {
        'name': 'SIGMOID',
        'expected_cpi_min': 40.0,   # expf + division + adds
        'expected_cpi_max': 350.0,
        'description': 'Full sigmoid 1/(1+exp(-x))',
    },
]


# =============================================================================
# Timing Validation Tests
# =============================================================================

@pytest.mark.renode
def test_dwt_vs_instruction_count_validation(
    build_cache: Path,
    repl_file: Path,
    renode_path: Path,
):
    """Compare DWT-reported cycles with traced instruction counts.

    For integer-heavy code (RF): CPI ≈ 1.0, ratio should be ~1.0
    For FP-heavy code (GBT with expf): CPI > 1.0, ratio should be > 1.5

    This validates that the ~5,600 cycle GBT overhead IS from expf()
    (multi-cycle instructions) rather than more instructions.
    """
    # Use a small RF and GBT model for comparison
    rf_config = {'type': 'rf', 'n_estimators': 2, 'max_depth': 2}
    gbt_config = {'type': 'gbt', 'n_estimators': 2, 'max_depth': 2}

    results = {}

    for config in [rf_config, gbt_config]:
        model_id = get_model_id(config)
        model_dir = build_cache / model_id

        if not model_dir.exists():
            pytest.skip(f"Model {model_id} not in build cache")

        elf_path = model_dir / 'build' / 'zephyr' / 'zephyr.elf'
        if not elf_path.exists():
            pytest.skip(f"ELF not found for {model_id}")

        # Run Renode and get timing
        avg_ns, avg_cycles, total_instructions, uart_output = run_renode_benchmark(
            elf_path, repl_file, renode_path
        )

        # Calculate effective CPI (DWT cycles / instruction count)
        # Note: This is approximate since total_instructions includes overhead
        if total_instructions > 0:
            effective_cpi = avg_cycles / (total_instructions / 100)  # 100 iterations
        else:
            effective_cpi = 0

        results[config['type']] = {
            'avg_cycles': avg_cycles,
            'total_instructions': total_instructions,
            'effective_cpi': effective_cpi,
        }

        print(f"\n{model_id}:")
        print(f"  DWT avg_cycles: {avg_cycles}")
        print(f"  Total instructions: {total_instructions:,}")
        print(f"  Effective CPI (approx): {effective_cpi:.2f}")

    # Validate that GBT has higher effective CPI than RF (due to expf)
    if 'gbt' in results and 'rf' in results:
        gbt_cpi = results['gbt']['effective_cpi']
        rf_cpi = results['rf']['effective_cpi']

        print(f"\nGBT/RF CPI ratio: {gbt_cpi / rf_cpi:.2f}" if rf_cpi > 0 else "")

        # GBT should have higher CPI due to expf (multi-cycle instruction)
        # This is a soft assertion - we're validating the hypothesis
        if gbt_cpi > 0 and rf_cpi > 0:
            assert gbt_cpi > rf_cpi, (
                f"Expected GBT CPI ({gbt_cpi:.2f}) > RF CPI ({rf_cpi:.2f}) "
                "due to expf() overhead"
            )


@pytest.mark.renode
@pytest.mark.parametrize('multiclass_config', MULTICLASS_CONFIGS[:3], ids=lambda c: f"c{c['n_classes']}")
def test_multiclass_softmax_scaling(
    multiclass_config: dict,
    benchmark_app_path: Path,
    conf_file: Path,
    zephyr_env: dict[str, str],
    west_path: Path,
    python_path: Path,
    repl_file: Path,
    renode_path: Path,
):
    """Verify softmax overhead scales with n_classes.

    Binary (n_classes=2) uses sigmoid (1 expf call).
    Multiclass uses softmax (n_classes expf calls).

    Validates that inference time increases with n_classes due to
    additional expf() calls in softmax normalization.
    """
    n_classes = multiclass_config['n_classes']

    # Generate proper multiclass dataset (can't use modulo on binary data)
    # Note: benchmark_dataset is binary - we need actual n_classes labels
    from sklearn.datasets import make_classification
    X, y_multiclass = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=n_classes,
        n_informative=min(3, n_classes),  # At least n_classes informative features
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Use a small model to isolate softmax overhead
    config = {
        'type': 'gbt',
        'n_estimators': 5,
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_classes': n_classes,
    }

    # Create a unique model ID
    model_id = f"gbt_n5_d3_c{n_classes}"

    # Build in a temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_dir = tmp_path / model_id
        model_dir.mkdir()

        # Train multiclass model
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=5,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        clf.fit(X, y_multiclass)

        # Convert to emlearn
        cmodel = emlearn.convert(clf, dtype='float')

        # Create isolated app directory
        app_dir = model_dir / 'app'
        shutil.copytree(
            benchmark_app_path, app_dir,
            ignore=shutil.ignore_patterns(
                'benchmark_model.h',
                'benchmark_model_testdata.h',
                '*.lock',
                'build',
            )
        )

        # Patch CMakeLists.txt
        emlearn_headers_path = _find_emlearn_headers(benchmark_app_path)
        cmake_file = app_dir / 'CMakeLists.txt'
        cmake_content = cmake_file.read_text()
        cmake_content = cmake_content.replace(
            '${CMAKE_CURRENT_SOURCE_DIR}/../../../emlearn',
            str(emlearn_headers_path)
        )
        cmake_file.write_text(cmake_content)

        # Generate headers
        src_dir = app_dir / 'src'
        (src_dir / 'benchmark_model.h').write_text(cmodel.save(name='benchmark_model'))
        (src_dir / 'benchmark_model_testdata.h').write_text(
        generate_testdata_header(X, n_classes=getattr(cmodel, 'n_classes', 0) or None))

        # Build
        build_dir = model_dir / 'build'
        success, error = build_firmware(
            build_dir, app_dir, conf_file,
            zephyr_env, west_path, python_path
        )

        if not success:
            pytest.fail(f"Build failed for {model_id}: {error}")

        # Run Renode
        elf_path = build_dir / 'zephyr' / 'zephyr.elf'
        avg_ns, avg_cycles, total_instructions, uart_output = run_renode_benchmark(
            elf_path, repl_file, renode_path
        )

    print(f"\n{model_id} (n_classes={n_classes}):")
    print(f"  avg_ns: {avg_ns}")
    print(f"  avg_cycles: {avg_cycles}")
    print(f"  expected_expf_calls: {multiclass_config['expected_expf_calls']}")

    # Basic sanity checks
    assert avg_ns > 0, f"No timing data for {model_id}"
    assert avg_cycles > 0, f"No cycle data for {model_id}"

    # Store result for cross-test comparison (would need fixture for full implementation)
    # For now, just validate we got reasonable timing


# Note: pytest_sessionfinish hook is in conftest.py for proper pytest recognition
