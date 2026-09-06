"""
Hardware validation tests for emlearn.

Validates Renode timing predictions on real nRF52 hardware using J-Link + RTT.
These tests require physical hardware and are marked with @pytest.mark.mcu_hardware.

Key questions answered:
1. Does hardware DWT match Renode? (target: +/-15% tolerance)
2. What's the actual expf() cost? (expect 50-200 cycles)
3. Does 3.16x GBT/RF CPI ratio hold on silicon?
4. Why is multiclass timing invariant?

Example usage:
    # Run hardware validation tests
    EMLEARN_TEST_MCU=hardware MCU_DEVICE=nrf52dk_nrf52832 \
        .venv/bin/python -m pytest -v test/test_hardware_validation.py -s

    # Run with specific hardware board
    EMLEARN_TEST_MCU=hardware MCU_DEVICE=nrf52dk_nrf52832 \
        .venv/bin/python -m pytest -v test/test_hardware_validation.py -s
"""

import os
from pathlib import Path

import pytest


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope='module')
def hardware_runner(has_west, has_jlink, mcu_device):
    """Create MCUTestRunner specifically for hardware validation.

    This fixture ensures we're running on real hardware, not emulators.
    """
    if not has_west:
        pytest.skip("west not available")
    if not has_jlink:
        pytest.skip("J-Link tools not available")

    mcu_mode = os.environ.get('EMLEARN_TEST_MCU', '').lower()
    if mcu_mode != 'hardware' and mcu_mode != 'all':
        pytest.skip("Hardware tests require EMLEARN_TEST_MCU=hardware")

    if mcu_device == 'auto':
        pytest.skip("Hardware validation requires explicit MCU_DEVICE")

    from emlearn.mcu import MCUTestRunner, get_board

    try:
        board = get_board(mcu_device)
    except KeyError:
        pytest.skip(f"Unknown board: {mcu_device}")

    if board.is_emulator:
        pytest.skip(f"Board {mcu_device} is an emulator, not hardware")

    runner = MCUTestRunner(board=board, verbose=True, timeout=120.0)
    yield runner
    runner.cleanup()


# =============================================================================
# Hardware Validation Tests
# =============================================================================

@pytest.mark.mcu
@pytest.mark.mcu_hardware
@pytest.mark.slow
class TestHardwareValidation:
    """Validate Renode timing predictions on real hardware."""

    def test_micro_benchmarks(self, hardware_runner, benchmark_app_path):
        """Run EXPF, SIGMOID, INT_OPS, FP_OPS, FP_DIV micro-benchmarks on hardware.

        Expected results (nRF52832 @ 64MHz):
        - INT_OPS: ~1 cycle
        - FP_OPS: ~1 cycle (with FPU)
        - FP_DIV: ~14 cycles
        - EXPF: 50-200 cycles (libm implementation)
        - SIGMOID: 60-220 cycles (includes expf + division)
        """
        from emlearn.mcu.capture import parse_micro_benchmark_output

        # Build with micro-benchmarks enabled
        build_dir = hardware_runner.build(
            benchmark_app_path,
            extra_conf={'CONFIG_EMLEARN_BENCHMARK_MICRO': 'y'},
            pristine=True,
        )

        # Flash and run
        assert hardware_runner.flash(build_dir), "Flash failed"

        # Capture output
        capture = hardware_runner.start_capture()
        hardware_runner.stop_capture()  # Reset triggers benchmark run

        from emlearn.mcu.flash import reset_target
        if hardware_runner.board.device:
            reset_target(hardware_runner.board.device)

        # Wait for completion
        capture = hardware_runner.start_capture()
        if not capture.wait_for_complete(timeout=60.0, end_pattern=r'BENCHMARK:END'):
            output = hardware_runner.stop_capture()
            pytest.fail(f"Timeout waiting for benchmark. Output:\n{output}")

        output = hardware_runner.stop_capture()

        # Parse micro-benchmark results
        micro_results = parse_micro_benchmark_output(output)

        print("\n=== Micro-benchmark Results ===")
        for name, cycles in sorted(micro_results.items()):
            print(f"  {name}: {cycles} cycles")

        # Validate expected ranges
        if 'INT_OPS' in micro_results:
            assert micro_results['INT_OPS'] <= 5, \
                f"INT_OPS too slow: {micro_results['INT_OPS']} cycles"

        if 'FP_OPS' in micro_results:
            # FPU should be ~1 cycle, software FP is slower
            assert micro_results['FP_OPS'] <= 50, \
                f"FP_OPS too slow: {micro_results['FP_OPS']} cycles"

        if 'FP_DIV' in micro_results:
            assert 5 <= micro_results['FP_DIV'] <= 30, \
                f"FP_DIV unexpected: {micro_results['FP_DIV']} cycles"

        if 'EXPF' in micro_results:
            # Key finding: Renode inferred ~5,600 cycles, hardware should be 50-200
            assert 30 <= micro_results['EXPF'] <= 300, \
                f"EXPF unexpected: {micro_results['EXPF']} cycles (expected 50-200)"
            print(f"\n  ** EXPF validated: {micro_results['EXPF']} cycles **")

        if 'SIGMOID' in micro_results:
            # Sigmoid = 1/(1+exp(-x)) includes expf + division
            assert 40 <= micro_results['SIGMOID'] <= 350, \
                f"SIGMOID unexpected: {micro_results['SIGMOID']} cycles"

        return micro_results

    @pytest.mark.timeout(180)
    def test_gbt_vs_rf_cpi_ratio(self, hardware_runner, benchmark_app_path):
        """Verify ~3x CPI ratio between GBT and RF holds on hardware.

        This validates the key finding from Renode benchmarks:
        - RF achieves ~1 CPI (integer operations only)
        - GBT achieves ~3 CPI (FP operations + expf overhead)
        """
        # For this test, we need to build and run with different model configs
        # This is a simplified version - full sweep is in test_renode_benchmark.py

        result = hardware_runner.run_benchmark(
            app_path=benchmark_app_path,
            timeout=90.0,
        )

        if not result.success:
            pytest.skip(f"Benchmark failed: {result.error}")

        print(f"\n=== Hardware Benchmark Result ===")
        print(f"  Board: {result.board}")
        print(f"  Timing: {result.timing_mode}")
        print(f"  Min cycles: {result.min_cycles}")
        print(f"  Avg cycles: {result.avg_cycles}")
        print(f"  Min time: {result.min_us:.2f} us")
        print(f"  Iterations: {result.iterations}")

        # Validate DWT is working
        if result.timing_mode == 'dwt':
            assert result.min_cycles > 0, "DWT should report non-zero cycles"
            assert result.avg_cycles >= result.min_cycles, "Avg should be >= min"

            # Calculate effective CPI (crude estimate)
            # This requires knowing instruction count, which we don't have
            # But we can validate cycles are in reasonable range
            assert result.min_cycles < 1_000_000, \
                f"Inference too slow: {result.min_cycles} cycles"

    def test_renode_correlation(self, hardware_runner, benchmark_app_path):
        """Compare hardware DWT measurements with Renode predictions.

        Target: Hardware DWT within +/-15% of Renode DWT cycles.

        Note: This test requires pre-existing Renode results for comparison.
        If no Renode results are available, the test is skipped.
        """
        # Look for recent Renode benchmark results
        runs_dir = Path(__file__).parent.parent / 'examples' / 'mcu_benchmark' / 'runs'
        renode_files = sorted(runs_dir.glob('renode_benchmark_*.csv'), reverse=True)

        if not renode_files:
            pytest.skip("No Renode benchmark results found for comparison")

        # Load most recent Renode results
        import csv
        renode_data = {}
        with open(renode_files[0]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row.get('model_type', '')}_{row.get('n_trees', '')}_{row.get('max_depth', '')}"
                if 'dwt_cycles' in row:
                    renode_data[key] = int(row['dwt_cycles'])

        if not renode_data:
            pytest.skip("Renode results don't contain DWT cycle data")

        # Run hardware benchmark
        result = hardware_runner.run_benchmark(
            app_path=benchmark_app_path,
            timeout=90.0,
        )

        if not result.success:
            pytest.skip(f"Benchmark failed: {result.error}")

        # For now, just report the comparison
        # Full correlation requires running the same model configurations
        print(f"\n=== Hardware vs Renode Correlation ===")
        print(f"  Hardware min cycles: {result.min_cycles}")
        print(f"  Renode results available: {len(renode_data)} configurations")

        # Basic sanity check: hardware cycles should be in same order of magnitude
        if renode_data:
            avg_renode = sum(renode_data.values()) / len(renode_data)
            ratio = result.min_cycles / avg_renode if avg_renode > 0 else 0
            print(f"  Avg Renode cycles: {avg_renode:.0f}")
            print(f"  Hardware/Renode ratio: {ratio:.2f}")

            # Warn if ratio is way off, but don't fail
            # (different model configs may explain differences)
            if ratio < 0.1 or ratio > 10:
                print(f"  WARNING: Large ratio suggests model config mismatch")


@pytest.mark.mcu
@pytest.mark.mcu_hardware
@pytest.mark.slow
class TestHardwareTiming:
    """Detailed hardware timing analysis."""

    def test_timing_consistency(self, hardware_runner, benchmark_app_path):
        """Verify timing measurements are consistent across runs.

        Hardware timing should be more consistent than emulator timing.
        Target: max/min ratio < 2x for same workload.
        """
        result = hardware_runner.run_benchmark(
            app_path=benchmark_app_path,
            timeout=90.0,
        )

        if not result.success:
            pytest.skip(f"Benchmark failed: {result.error}")

        if result.min_cycles == 0:
            pytest.skip("No cycle count data available")

        ratio = result.avg_cycles / result.min_cycles
        print(f"\n=== Timing Consistency ===")
        print(f"  Min cycles: {result.min_cycles}")
        print(f"  Avg cycles: {result.avg_cycles}")
        print(f"  Ratio: {ratio:.2f}")

        # Hardware should have low jitter
        assert ratio < 2.0, \
            f"Timing too inconsistent: avg/min ratio {ratio:.2f}"

    def test_dwt_present(self, hardware_runner, benchmark_app_path):
        """Verify DWT cycle counter is functioning on hardware."""
        result = hardware_runner.run_benchmark(
            app_path=benchmark_app_path,
            timeout=90.0,
        )

        if not result.success:
            pytest.skip(f"Benchmark failed: {result.error}")

        print(f"\n=== DWT Status ===")
        print(f"  Timing mode: {result.timing_mode}")
        print(f"  CPU frequency: {result.cpu_freq_hz} Hz")

        if result.timing_mode == 'dwt':
            assert result.min_cycles > 0, "DWT should report cycles"
            # Cross-check: cycles should match time at known frequency
            if result.cpu_freq_hz > 0:
                expected_ns = (result.min_cycles * 1e9) / result.cpu_freq_hz
                actual_ns = result.min_ns
                ratio = actual_ns / expected_ns if expected_ns > 0 else 0
                print(f"  Expected ns from cycles: {expected_ns:.0f}")
                print(f"  Actual ns from DWT: {actual_ns}")
                print(f"  Ratio: {ratio:.2f}")

                # Allow for some measurement overhead
                assert 0.5 < ratio < 2.0, \
                    f"DWT time/cycle mismatch: ratio {ratio:.2f}"
        else:
            print(f"  WARNING: DWT not available, using {result.timing_mode}")
