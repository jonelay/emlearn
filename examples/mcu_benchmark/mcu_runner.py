"""
MCU Benchmark Runner - Renode/native_sim timing measurement.
============================================================

Provides functions to measure inference timing on emulated MCU platforms.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from emlearn.mcu.boards import BOARDS, FlashMethod
from emlearn.mcu.renode_runner import RenodeRunner, RenodeConfig, get_renode_executable

from .common import TimingResult, BENCHMARK_APP_PATH
from .model_configs import generate_testdata_header


def get_west_workspace() -> Path | None:
    """Get the west workspace directory from ZEPHYR_BASE.

    Returns:
        Path to west workspace, or None if not found.
    """
    zephyr_base = os.environ.get('ZEPHYR_BASE')
    if not zephyr_base:
        return None

    # ZEPHYR_BASE is typically <workspace>/zephyr
    # West workspace is the parent directory
    zephyr_path = Path(zephyr_base)
    workspace = zephyr_path.parent

    # Verify it's a valid west workspace
    if (workspace / '.west').is_dir():
        return workspace

    return None


def generate_model_header(cmodel, name: str = 'benchmark_model') -> str:
    """Generate C header from converted emlearn model.

    Args:
        cmodel: Converted emlearn model.
        name: Model name for C functions.

    Returns:
        C header code as string.
    """
    return cmodel.save(name=name)


def estimate_timeout(cmodel, base_timeout: float = 120.0) -> float:
    """Estimate appropriate timeout based on model complexity.

    Larger models (more trees, deeper) need more time in Renode due to
    emulation overhead. Scales timeout based on estimated node count.

    Args:
        cmodel: Converted emlearn model.
        base_timeout: Base timeout for small models in seconds.

    Returns:
        Scaled timeout in seconds (minimum base_timeout).
    """
    try:
        # Get model size metrics if available
        n_nodes = getattr(cmodel, 'n_nodes', None)
        if n_nodes is None:
            # Try to estimate from model code size
            code = cmodel.save(name='_estimate')
            code_size = len(code.encode('utf-8'))
            # Rough heuristic: >50KB code = large model, scale up
            if code_size > 50_000:
                scale = code_size / 50_000
                return base_timeout * min(scale, 5.0)
            elif code_size > 20_000:
                scale = code_size / 20_000
                return base_timeout * min(scale, 3.0)
            return base_timeout
        # Scale based on node count: >500 nodes gets extra time
        if n_nodes > 500:
            scale = n_nodes / 500
            return base_timeout * min(scale, 5.0)
    except Exception:
        pass
    return base_timeout


def measure_mcu_timing(
    cmodel,
    X_test: np.ndarray,
    platform: str,
    timeout: float = 120.0,
    verbose: bool = False,
    extra_defines: list[str] | None = None,
    benchmark_mode: str = 'predict',
) -> TimingResult:
    """Measure inference timing on Renode or native_sim.

    Builds the Zephyr benchmark app with the provided model and test data,
    runs it on the specified emulator platform, and parses timing results.

    Args:
        cmodel: Converted emlearn model.
        X_test: Test features for benchmark (int16 or will be converted).
        platform: Platform name ('renode_nrf52840', 'native_sim', etc.).
        timeout: Timeout for benchmark run in seconds.
        verbose: Print verbose output.
        extra_defines: Additional C defines for Kconfig options.
        benchmark_mode: 'predict' for class labels, 'predict_proba' for probabilities.

    Returns:
        TimingResult with timing measurements or error information.
    """
    # Handle benchmark_mode by adding to extra_defines
    if benchmark_mode == 'predict_proba':
        extra_defines = list(extra_defines or [])
        extra_defines.append('EML_BENCHMARK_PREDICT_PROBA')
    # Validate platform
    if platform not in BOARDS:
        return TimingResult(
            success=False,
            min_ns=0,
            avg_ns=0,
            min_cycles=0,
            avg_cycles=0,
            iterations=0,
            timing_mode='unknown',
            error=f'Unknown platform: {platform}. Available: {list(BOARDS.keys())}',
        )

    board = BOARDS[platform]

    # Only support emulator platforms
    if not board.is_emulator:
        return TimingResult(
            success=False,
            min_ns=0,
            avg_ns=0,
            min_cycles=0,
            avg_cycles=0,
            iterations=0,
            timing_mode='unknown',
            error=f'Platform {platform} is not an emulator. Use hardware runner.',
        )

    # Verify west workspace is available
    workspace = get_west_workspace()
    if workspace is None:
        return TimingResult(
            success=False,
            min_ns=0,
            avg_ns=0,
            min_cycles=0,
            avg_cycles=0,
            iterations=0,
            timing_mode='unknown',
            error='ZEPHYR_BASE not set or west workspace not found. '
                  'Source .env.local before running MCU benchmarks.',
        )

    # Copy app to temp dir to avoid writing generated headers into source tree
    import shutil as _shutil
    import tempfile as _tempfile
    app_temp = Path(_tempfile.mkdtemp(prefix='emlearn_bench_'))
    _shutil.copytree(BENCHMARK_APP_PATH, app_temp / 'benchmark', dirs_exist_ok=True)
    app_dir = app_temp / 'benchmark'
    app_src = app_dir / 'src'

    # Ensure src directory exists
    app_src.mkdir(parents=True, exist_ok=True)

    # Generate and write headers
    model_code = generate_model_header(cmodel)
    # Classifiers expose n_classes; regressors report 0 (no predict_proba)
    testdata_code = generate_testdata_header(
        X_test, n_samples=100, n_classes=getattr(cmodel, 'n_classes', 0) or None,
    )

    model_header_path = app_src / 'benchmark_model.h'
    testdata_header_path = app_src / 'benchmark_model_testdata.h'

    model_header_path.write_text(model_code)
    testdata_header_path.write_text(testdata_code)

    if verbose:
        print(f"Wrote model header: {model_header_path}")
        print(f"Wrote test data header: {testdata_header_path}")

    # Build and run directly since the benchmark app is outside the west workspace
    import shutil
    import subprocess
    from emlearn import temp_manager

    west = shutil.which('west')
    if not west:
        return TimingResult(
            success=False,
            min_ns=0,
            avg_ns=0,
            min_cycles=0,
            avg_cycles=0,
            iterations=0,
            timing_mode='unknown',
            error='west not found. Install Zephyr SDK.',
        )

    build_dir = temp_manager.create_build_dir(board.name)

    try:
        # Build from temp copy (generated headers were written there)
        app_path = app_dir.resolve()
        build_cmd = [
            west, 'build',
            '-b', board.name,
            '-d', str(build_dir),
            str(app_path),
            '--pristine',
        ]

        # Add cmake arguments: Python executable and optional Renode config
        # Use the current Python to ensure pyelftools and other deps are available
        import sys
        cmake_args = [f'-DPython3_EXECUTABLE={sys.executable}']

        # Add Renode config for Renode platforms (UART console, DWT timing)
        conf_files = []
        if board.flash_method == FlashMethod.RENODE:
            renode_conf = app_path / 'boards' / 'renode_nrf52840.conf'
            if renode_conf.exists():
                conf_files.append(str(renode_conf))

        # Add extra Kconfig options via temporary overlay file
        # Maps C defines to Kconfig options (e.g., EML_TREES_USE_POINTER_TRAVERSAL -> CONFIG_EMLEARN_POINTER_TRAVERSAL)
        kconfig_map = {
            'EML_TREES_USE_POINTER_TRAVERSAL': 'CONFIG_EMLEARN_POINTER_TRAVERSAL',
            'EML_TREES_CMSIS_DSP': 'CONFIG_EMLEARN_USE_CMSIS_DSP',
            'EML_BENCHMARK_PREDICT_PROBA': 'CONFIG_EMLEARN_BENCHMARK_PREDICT_PROBA',
        }
        if extra_defines:
            overlay_lines = []
            for define in extra_defines:
                if define in kconfig_map:
                    overlay_lines.append(f'{kconfig_map[define]}=y')
                else:
                    # Unknown define — skip with warning
                    if verbose:
                        print(f"Warning: Unknown define {define}, not mapped to Kconfig")
            if overlay_lines:
                # Write overlay to parent directory to survive --pristine clean
                # (--pristine deletes the build_dir before CMake processes conf files)
                overlay_path = build_dir.parent / f'{build_dir.name}_extra_defines.conf'
                overlay_path.write_text('\n'.join(overlay_lines) + '\n')
                conf_files.append(str(overlay_path))

        # Add conf files using west's --extra-conf option (before --)
        for conf_file in conf_files:
            build_cmd.extend(['--extra-conf', conf_file])

        build_cmd.extend(['--'] + cmake_args)

        if verbose:
            print(f"Building: {' '.join(build_cmd)}")

        result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=workspace,
        )

        if result.returncode != 0:
            return TimingResult(
                success=False,
                min_ns=0,
                avg_ns=0,
                min_cycles=0,
                avg_cycles=0,
                iterations=0,
                timing_mode='unknown',
                error=f'Build failed:\n{result.stderr}',
            )

        # Run the executable: native_sim or Renode
        if platform == 'native_sim':
            exe_path = build_dir / 'zephyr' / 'zephyr.exe'
            if not exe_path.exists():
                return TimingResult(
                    success=False,
                    min_ns=0,
                    avg_ns=0,
                    min_cycles=0,
                    avg_cycles=0,
                    iterations=0,
                    timing_mode='unknown',
                    error=f'Executable not found: {exe_path}',
                )
            run_cmd = [str(exe_path)]
        elif board.flash_method == FlashMethod.RENODE:
            # Use RenodeRunner for Renode platforms
            elf_path = build_dir / 'zephyr' / 'zephyr.elf'
            if not elf_path.exists():
                return TimingResult(
                    success=False,
                    min_ns=0,
                    avg_ns=0,
                    min_cycles=0,
                    avg_cycles=0,
                    iterations=0,
                    timing_mode='unknown',
                    error=f'ELF not found: {elf_path}',
                )

            # Find platform description file
            if board.renode_repl:
                repl_path = BENCHMARK_APP_PATH.parent.parent.parent / board.renode_repl
                if not repl_path.exists():
                    repl_path = Path(board.renode_repl)
            else:
                return TimingResult(
                    success=False,
                    min_ns=0,
                    avg_ns=0,
                    min_cycles=0,
                    avg_cycles=0,
                    iterations=0,
                    timing_mode='unknown',
                    error=f'No renode_repl configured for platform {platform}',
                )

            if not repl_path.exists():
                return TimingResult(
                    success=False,
                    min_ns=0,
                    avg_ns=0,
                    min_cycles=0,
                    avg_cycles=0,
                    iterations=0,
                    timing_mode='unknown',
                    error=f'Platform file not found: {repl_path}',
                )

            # Check Renode is available
            renode_exe = get_renode_executable()
            if renode_exe is None:
                return TimingResult(
                    success=False,
                    min_ns=0,
                    avg_ns=0,
                    min_cycles=0,
                    avg_cycles=0,
                    iterations=0,
                    timing_mode='unknown',
                    error='Renode not found. Set RENODE_PATH or install Renode.',
                )

            if verbose:
                print(f"Running Renode: {elf_path}")
                print(f"Platform: {repl_path}")

            # Run with RenodeRunner
            config = RenodeConfig(
                platform=board.renode_platform or 'nrf52840',
                cpu_freq_hz=board.cpu_freq_hz,
                repl_file=str(repl_path),
            )
            runner = RenodeRunner(config, timeout=timeout, verbose=verbose)

            try:
                stdout = runner.run(elf_path, repl_path)
            except Exception as e:
                return TimingResult(
                    success=False,
                    min_ns=0,
                    avg_ns=0,
                    min_cycles=0,
                    avg_cycles=0,
                    iterations=0,
                    timing_mode='unknown',
                    error=f'Renode execution failed: {e}',
                )

            if verbose:
                print(f"Renode output:\n{stdout}")

            # Parse results directly and return (skip the native_sim parsing below)
            from emlearn.mcu.capture import parse_benchmark_output
            parsed = parse_benchmark_output(stdout)

            if parsed['status'] != 'ok':
                return TimingResult(
                    success=False,
                    min_ns=parsed.get('min_ns', 0),
                    avg_ns=parsed.get('avg_ns', 0),
                    min_cycles=parsed.get('min_cycles', 0),
                    avg_cycles=parsed.get('avg_cycles', 0),
                    iterations=parsed.get('iterations', 0),
                    timing_mode=parsed.get('timing', 'dwt'),
                    error=f"Benchmark status: {parsed['status']}",
                )

            return TimingResult(
                success=True,
                min_ns=parsed.get('min_ns', 0),
                avg_ns=parsed.get('avg_ns', 0),
                min_cycles=parsed.get('min_cycles', 0),
                avg_cycles=parsed.get('avg_cycles', 0),
                iterations=parsed.get('iterations', 0),
                timing_mode=parsed.get('timing', 'dwt'),
            )
        else:
            # Fallback: use west build -t run for other emulators
            run_cmd = [west, 'build', '-d', str(build_dir), '-t', 'run']

        if verbose:
            print(f"Running: {' '.join(run_cmd)}")

        # Use pty to force line buffering (native_sim buffers output otherwise)
        import pty
        import os as os_module
        import time
        import select

        master_fd, slave_fd = pty.openpty()

        proc = subprocess.Popen(
            run_cmd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=workspace,
            close_fds=True,
        )

        os_module.close(slave_fd)  # Close slave end in parent

        start_time = time.time()
        output_buffer = []  # Collect all output fragments
        benchmark_complete = False

        try:
            while (time.time() - start_time) < timeout:
                # Check if process exited
                if proc.poll() is not None:
                    # Read remaining output
                    while True:
                        if select.select([master_fd], [], [], 0.1)[0]:
                            try:
                                data = os_module.read(master_fd, 4096)
                                if not data:
                                    break
                                text = data.decode('utf-8', errors='replace')
                                output_buffer.append(text)
                            except OSError:
                                break
                        else:
                            break

                    # Check combined output for completion
                    full_output = ''.join(output_buffer)
                    if 'BENCHMARK:END' in full_output:
                        benchmark_complete = True
                    break

                # Read available data
                if select.select([master_fd], [], [], 0.5)[0]:
                    try:
                        data = os_module.read(master_fd, 4096)
                        if data:
                            text = data.decode('utf-8', errors='replace')
                            output_buffer.append(text)

                            # Check combined output for completion marker
                            # Wait for full line (status=ok or status=error) after BENCHMARK:END
                            full_output = ''.join(output_buffer)
                            if 'BENCHMARK:END' in full_output and 'status=' in full_output:
                                benchmark_complete = True
                                if verbose:
                                    # Print clean output
                                    for line in full_output.splitlines():
                                        if line.strip():
                                            print(f"  {line}")
                                # Read any remaining data
                                time.sleep(0.1)
                                while select.select([master_fd], [], [], 0.1)[0]:
                                    try:
                                        extra = os_module.read(master_fd, 4096)
                                        if extra:
                                            output_buffer.append(extra.decode('utf-8', errors='replace'))
                                        else:
                                            break
                                    except OSError:
                                        break
                                break
                    except OSError:
                        break

            if not benchmark_complete and (time.time() - start_time) >= timeout:
                proc.kill()
                proc.wait()
                try:
                    os_module.close(master_fd)
                except OSError:
                    pass
                return TimingResult(
                    success=False,
                    min_ns=0,
                    avg_ns=0,
                    min_cycles=0,
                    avg_cycles=0,
                    iterations=0,
                    timing_mode='unknown',
                    error='Timeout waiting for benchmark completion',
                )

        finally:
            # Always clean up the process
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            try:
                os_module.close(master_fd)
            except OSError:
                pass  # Already closed

        stdout = ''.join(output_buffer)

        # Clean up fragmented output (emulators may send character by character)
        import re

        # First, remove all whitespace to join fragments
        clean = ''.join(stdout.split())

        # Extract benchmark lines using patterns (without whitespace)
        lines = []

        # BENCHMARK:START
        if 'BENCHMARK:START' in clean:
            lines.append('BENCHMARK:START')

        # BENCHMARK:CONFIG board=X timing=X freq=X
        config_match = re.search(r'BENCHMARK:CONFIGboard=(\S+?)timing=(\S+?)freq=(\d+)', clean)
        if config_match:
            lines.append(f'BENCHMARK:CONFIG board={config_match.group(1)} timing={config_match.group(2)} freq={config_match.group(3)}')

        # BENCHMARK:MODEL name=X features=X samples=X
        model_match = re.search(r'BENCHMARK:MODELname=(\S+?)features=(\d+)samples=(\d+)', clean)
        if model_match:
            lines.append(f'BENCHMARK:MODEL name={model_match.group(1)} features={model_match.group(2)} samples={model_match.group(3)}')

        # RESULT:INFERENCE min_ns=X max_ns=X avg_ns=X min_cycles=X avg_cycles=X iterations=X
        result_match = re.search(
            r'RESULT:INFERENCEmin_ns=(\d+)max_ns=(\d+)avg_ns=(\d+)min_cycles=(\d+)avg_cycles=(\d+)iterations=(\d+)',
            clean
        )
        if result_match:
            lines.append(f'RESULT:INFERENCE min_ns={result_match.group(1)} max_ns={result_match.group(2)} avg_ns={result_match.group(3)} min_cycles={result_match.group(4)} avg_cycles={result_match.group(5)} iterations={result_match.group(6)}')

        # BENCHMARK:END status=X
        end_match = re.search(r'BENCHMARK:ENDstatus=(\S+)', clean)
        if end_match:
            lines.append(f'BENCHMARK:END status={end_match.group(1)}')

        # Reconstruct clean output
        stdout = '\n'.join(lines)

        if verbose:
            print(f"Parsed output:\n{stdout}")

        # Parse results
        from emlearn.mcu.capture import parse_benchmark_output
        parsed = parse_benchmark_output(stdout)

        if parsed['status'] != 'ok':
            return TimingResult(
                success=False,
                min_ns=parsed.get('min_ns', 0),
                avg_ns=parsed.get('avg_ns', 0),
                min_cycles=parsed.get('min_cycles', 0),
                avg_cycles=parsed.get('avg_cycles', 0),
                iterations=parsed.get('iterations', 0),
                timing_mode=parsed.get('timing', 'unknown'),
                error=f"Benchmark status: {parsed['status']}",
            )

        return TimingResult(
            success=True,
            min_ns=parsed.get('min_ns', 0),
            avg_ns=parsed.get('avg_ns', 0),
            min_cycles=parsed.get('min_cycles', 0),
            avg_cycles=parsed.get('avg_cycles', 0),
            iterations=parsed.get('iterations', 0),
            timing_mode=parsed.get('timing', 'unknown'),
        )

    except Exception as e:
        return TimingResult(
            success=False,
            min_ns=0,
            avg_ns=0,
            min_cycles=0,
            avg_cycles=0,
            iterations=0,
            timing_mode='unknown',
            error=str(e),
        )

    finally:
        # Cleanup build directory
        temp_manager.cleanup_subdir(build_dir)


def is_emulator_platform(platform: str) -> bool:
    """Check if a platform is an emulator (Renode/native_sim).

    Args:
        platform: Platform name.

    Returns:
        True if platform is an emulator.
    """
    if platform == 'host':
        return False

    if platform not in BOARDS:
        return False

    return BOARDS[platform].is_emulator


def get_emulator_platforms() -> list[str]:
    """Get list of available emulator platforms.

    Returns:
        List of emulator platform names.
    """
    return [name for name, board in BOARDS.items() if board.is_emulator]
