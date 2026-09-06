"""
MCU Comparative Benchmark Tests
===============================

Unit tests for the MCU comparative benchmark suite.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

# Add examples to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Unit Tests (always run)
# =============================================================================

class TestMCUBenchmarkModule:
    """Test that the MCU benchmark module can be imported and works."""

    def test_import_benchmark_module(self):
        """Test that mcu_benchmark package can be imported."""
        from examples.mcu_benchmark import (
            BenchmarkConfig,
            TimingResult,
            MODEL_CONFIGS,
            get_model_configs,
            get_platforms,
        )
        assert BenchmarkConfig is not None
        assert TimingResult is not None
        assert len(MODEL_CONFIGS) > 0

    def test_model_configs(self):
        """Test model configuration structure."""
        from examples.mcu_benchmark.model_configs import (
            MODEL_CONFIGS,
            QUICK_MODEL_CONFIGS,
            get_model_configs,
            get_model_id,
        )

        # Check we have both GBT and RF configs
        types = set(c['type'] for c in MODEL_CONFIGS)
        assert 'gbt' in types
        assert 'rf' in types

        # Check quick mode has fewer configs
        assert len(QUICK_MODEL_CONFIGS) <= len(MODEL_CONFIGS)

        # Check model ID generation
        config = {'type': 'gbt', 'n_estimators': 10, 'max_depth': 5}
        assert get_model_id(config) == 'gbt_n10_d5'

    def test_platforms(self):
        """Test platform configuration."""
        from examples.mcu_benchmark.model_configs import (
            get_platforms,
            PLATFORMS,
            QUICK_PLATFORMS,
            FULL_PLATFORMS,
        )

        # Check platforms are defined
        assert 'host' in PLATFORMS
        assert 'renode_nrf52840' in PLATFORMS
        assert 'nrf52dk_nrf52832' in PLATFORMS

        # Check quick mode excludes hardware
        quick = get_platforms(quick=True)
        full = get_platforms(quick=False)

        assert 'host' in quick
        assert 'renode_nrf52840' in quick
        assert len(quick) <= len(full)

    def test_common_utilities(self):
        """Test common utility functions."""
        from examples.mcu_benchmark.common import (
            get_platform_type,
            get_timing_mode,
            get_csv_schema,
        )

        # Test platform type detection
        assert get_platform_type('host') == 'host'
        assert get_platform_type('renode_nrf52840') == 'renode'
        assert get_platform_type('nrf52dk_nrf52832') == 'hardware'

        # Test timing mode detection
        assert get_timing_mode('host') == 'cffi'
        assert get_timing_mode('renode_nrf52840') == 'dwt'
        assert get_timing_mode('nrf52dk_nrf52832') == 'dwt'

        # Test CSV schema
        schema = get_csv_schema()
        assert 'model_name' in schema
        assert 'platform' in schema
        assert 'avg_ns' in schema

    def test_make_benchmark_dataset(self):
        """Test benchmark dataset generation."""
        from examples.mcu_benchmark.common import make_benchmark_dataset

        X, y = make_benchmark_dataset(
            n_samples=100,
            n_features=10,
            n_classes=2,
        )

        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert X.dtype == np.int16  # Scaled for embedded use
        assert set(y).issubset({0, 1})


# =============================================================================
# Figure Generation Tests (no MCU required)
# =============================================================================

class TestFigureGeneration:
    """Test figure generation functionality."""

    def test_check_dependencies(self):
        """Test dependency checking."""
        from examples.mcu_benchmark.generate_figures import check_dependencies

        # Should not raise, may return True or False depending on install
        result = check_dependencies()
        assert isinstance(result, bool)

    def test_generate_figures_no_data(self, tmp_path):
        """Test that figure generation handles missing data gracefully."""
        # This test just verifies the code doesn't crash with no data
        # In practice, figures are generated from CSV results
        from examples.mcu_benchmark.generate_figures import (
            load_csv,
            RESULTS_DIR,
        )

        # With no results, should return empty DataFrame
        df = load_csv("nonexistent_file.csv")
        assert df.empty
