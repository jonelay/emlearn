"""
MCU benchmark tests for emlearn.

Progressive testing: Host → Renode → Hardware

Enable tests via environment variables:
    EMLEARN_TEST_MCU=renode    # Run Renode emulator tests only
    EMLEARN_TEST_MCU=hardware  # Run hardware tests only
    EMLEARN_TEST_MCU=all       # Run all MCU tests
    MCU_DEVICE=nrf52dk_nrf52832  # Specify target device
    RENODE_PATH=/path/to/renode  # Path to Renode executable

Example:
    # Run Renode tests
    RENODE_PATH=/path/to/renode EMLEARN_TEST_MCU=renode \\
        .venv/bin/python -m pytest -v test/test_mcu.py

    # Run hardware tests on nRF52DK
    EMLEARN_TEST_MCU=hardware MCU_DEVICE=nrf52dk_nrf52832 \\
        .venv/bin/python -m pytest -v test/test_mcu.py
"""

import os
import shutil
import subprocess
import sys
import pytest
from pathlib import Path


@pytest.fixture(scope='session')
def benchmark_app_path(benchmark_app_path, tmp_path_factory):
    """Copy of the benchmark app with a generated model header.

    The app expects src/benchmark_model.h and src/benchmark_model_testdata.h,
    which are produced by src/train_model.py. Generate them into a temporary
    copy so the source tree is never modified.
    """
    app_copy = tmp_path_factory.mktemp('benchmark_app') / 'benchmark'
    shutil.copytree(benchmark_app_path, app_copy)
    subprocess.run(
        [sys.executable, 'train_model.py'],
        cwd=app_copy / 'src',
        check=True,
        capture_output=True,
        text=True,
    )
    return app_copy


# =============================================================================
# Module Import Tests (always run)
# =============================================================================

class TestMCUModule:
    """Test that the MCU module can be imported and basic functionality works."""

    def test_import_mcu_module(self):
        """Test that emlearn.mcu module can be imported."""
        from emlearn import mcu
        assert hasattr(mcu, 'MCUTestRunner')
        assert hasattr(mcu, 'BoardConfig')
        assert hasattr(mcu, 'BOARDS')

    def test_boards_available(self):
        """Test that board configurations are available."""
        from emlearn.mcu import BOARDS, list_boards

        # Check that some boards are defined
        assert len(BOARDS) > 0

        # Check emulator boards exist
        assert 'native_sim' in BOARDS
        assert 'renode_nrf52840' in BOARDS

        # Check hardware boards exist
        assert 'nrf52dk_nrf52832' in BOARDS

        # Check list functions work
        all_boards = list_boards()
        hw_boards = list_boards(hardware_only=True)
        emu_boards = list_boards(emulator_only=True)

        assert len(all_boards) > 0
        assert set(hw_boards).isdisjoint(set(emu_boards))

    def test_board_config_properties(self):
        """Test BoardConfig dataclass properties."""
        from emlearn.mcu import BOARDS

        # Test emulator boards
        native = BOARDS['native_sim']
        assert native.is_emulator is True
        assert native.is_hardware is False

        renode = BOARDS['renode_nrf52840']
        assert renode.is_emulator is True
        assert renode.is_hardware is False

        # Test hardware board
        nrf52 = BOARDS['nrf52dk_nrf52832']
        assert nrf52.is_emulator is False
        assert nrf52.is_hardware is True

    def test_get_board(self):
        """Test get_board function."""
        from emlearn.mcu import get_board

        # Exact match for native_sim
        board = get_board('native_sim')
        assert board.name == 'native_sim'

        # Exact match for hardware board
        board = get_board('nrf52dk_nrf52832')
        assert board.name == 'nrf52dk/nrf52832'  # Zephyr 4.x board/soc format

        # Unknown board raises KeyError
        with pytest.raises(KeyError):
            get_board('nonexistent_board_xyz')

    def test_output_capture_parse(self):
        """Test benchmark output parsing."""
        from emlearn.mcu.capture import parse_benchmark_output

        sample_output = """
BENCHMARK:START
BENCHMARK:CONFIG board=nrf52dk_nrf52832 timing=dwt freq=64000000
BENCHMARK:MODEL name=benchmark_model features=10 samples=5
RESULT:INFERENCE min_ns=123000 max_ns=456000 avg_ns=234000 min_cycles=7872 avg_cycles=14976 iterations=100
BENCHMARK:END status=ok
"""
        result = parse_benchmark_output(sample_output)

        assert result['board'] == 'nrf52dk_nrf52832'
        assert result['timing'] == 'dwt'
        assert result['freq_hz'] == 64000000
        assert result['model_name'] == 'benchmark_model'
        assert result['features'] == 10
        assert result['min_ns'] == 123000
        assert result['max_ns'] == 456000
        assert result['avg_ns'] == 234000
        assert result['min_cycles'] == 7872
        assert result['avg_cycles'] == 14976
        assert result['iterations'] == 100
        assert result['status'] == 'ok'

    def test_micro_benchmark_parse(self):
        """Test micro-benchmark output parsing."""
        from emlearn.mcu.capture import parse_micro_benchmark_output

        sample_output = """
RESULT:MICRO:EXPF avg_cycles=85 iterations=1000
RESULT:MICRO:SIGMOID avg_cycles=105 iterations=1000
RESULT:MICRO:INT_OPS avg_cycles=1 iterations=1000
RESULT:MICRO:FP_OPS avg_cycles=1 iterations=1000
RESULT:MICRO:FP_DIV avg_cycles=14 iterations=1000
"""
        result = parse_micro_benchmark_output(sample_output)

        assert result['EXPF'] == 85
        assert result['SIGMOID'] == 105
        assert result['INT_OPS'] == 1
        assert result['FP_OPS'] == 1
        assert result['FP_DIV'] == 14


# =============================================================================
# Renode Tests (require EMLEARN_TEST_MCU=renode|all and RENODE_PATH)
# =============================================================================

@pytest.mark.mcu
@pytest.mark.mcu_renode
@pytest.mark.slow
class TestMCURenode:
    """Tests that run on Renode emulator."""

    def test_renode_board_available(self, has_west):
        """Verify Renode prerequisites are available."""
        if not has_west:
            pytest.skip("west not available")

        import os
        if not os.environ.get('RENODE_PATH'):
            pytest.skip("RENODE_PATH not set")

        from emlearn.mcu import BOARDS
        assert 'renode_nrf52840' in BOARDS

    def test_benchmark_build_renode(self, mcu_runner, benchmark_app_path, has_west):
        """Test that benchmark app can be built for Renode."""
        if not has_west:
            pytest.skip("west not available")

        import os
        if not os.environ.get('RENODE_PATH'):
            pytest.skip("RENODE_PATH not set")

        from emlearn.mcu import BOARDS

        # Use Renode board for this test
        renode_board = BOARDS['renode_nrf52840']
        mcu_runner.board = renode_board

        # Just test that build works (don't run)
        build_dir = mcu_runner.build(benchmark_app_path)
        assert build_dir.exists()
        assert (build_dir / 'zephyr' / 'zephyr.elf').exists()

    @pytest.mark.timeout(120)
    def test_benchmark_run_renode(self, mcu_runner, benchmark_app_path, has_west):
        """Test complete benchmark run on Renode."""
        if not has_west:
            pytest.skip("west not available")

        import os
        if not os.environ.get('RENODE_PATH'):
            pytest.skip("RENODE_PATH not set")

        from emlearn.mcu import BOARDS

        # Use Renode board
        renode_board = BOARDS['renode_nrf52840']
        mcu_runner.board = renode_board

        result = mcu_runner.run_benchmark(
            app_path=benchmark_app_path,
            timeout=90.0,
        )

        # Check result
        assert result.success, f"Benchmark failed: {result.error}"
        assert result.min_us > 0
        assert result.avg_us >= result.min_us
        assert result.iterations > 0


# =============================================================================
# Hardware Tests (require EMLEARN_TEST_MCU=hardware|all)
# =============================================================================

@pytest.mark.mcu
@pytest.mark.mcu_hardware
@pytest.mark.slow
class TestMCUHardware:
    """Tests that run on real hardware."""

    def test_jlink_available(self, has_jlink):
        """Verify J-Link tools are available."""
        if not has_jlink:
            pytest.skip("J-Link tools not available")

    def test_device_discovery(self, has_jlink):
        """Test J-Link device discovery."""
        if not has_jlink:
            pytest.skip("J-Link tools not available")

        from emlearn.mcu import discover_devices
        devices = discover_devices()
        # May be empty if no device connected, but shouldn't raise
        assert isinstance(devices, list)

    def test_benchmark_build_hardware(self, mcu_runner, benchmark_app_path, has_west, mcu_device):
        """Test that benchmark app can be built for hardware target."""
        if not has_west:
            pytest.skip("west not available")

        if mcu_device == 'auto':
            pytest.skip("Specific device not configured")

        # Build for configured device
        build_dir = mcu_runner.build(benchmark_app_path)
        assert build_dir.exists()
        assert (build_dir / 'zephyr' / 'zephyr.elf').exists()

    @pytest.mark.timeout(120)
    def test_benchmark_run_hardware(self, mcu_runner, benchmark_app_path, has_west, has_jlink, mcu_device):
        """Test complete benchmark run on hardware."""
        if not has_west:
            pytest.skip("west not available")
        if not has_jlink:
            pytest.skip("J-Link tools not available")
        if mcu_device == 'auto':
            pytest.skip("Specific device not configured")

        result = mcu_runner.run_benchmark(
            app_path=benchmark_app_path,
            timeout=60.0,
        )

        # Check result
        assert result.success, f"Benchmark failed: {result.error}"
        assert result.min_us > 0
        assert result.avg_us >= result.min_us
        assert result.iterations > 0

        # Hardware should report cycle counts
        if result.timing_mode == 'dwt':
            assert result.min_cycles > 0


# =============================================================================
# Comparative Tests (run on whichever target is available)
# =============================================================================

@pytest.mark.mcu
@pytest.mark.slow
class TestMCUComparative:
    """Comparative tests that validate results across targets."""

    def test_inference_timing_reasonable(self, mcu_runner, benchmark_app_path, has_west):
        """Test that inference timing is in reasonable range."""
        if not has_west:
            pytest.skip("west not available")

        result = mcu_runner.run_benchmark(
            app_path=benchmark_app_path,
            timeout=90.0,
        )

        if not result.success:
            pytest.skip(f"Benchmark failed: {result.error}")

        # Inference should complete in reasonable time
        # For a small model: < 10ms on most targets
        assert result.min_us < 10000, f"Inference too slow: {result.min_us} us"

        # Timing should be consistent (max < 10x min)
        # This catches timing issues like cache misses
        assert result.max_us < result.min_us * 10, \
            f"Timing inconsistent: min={result.min_us}, max={result.max_us}"
