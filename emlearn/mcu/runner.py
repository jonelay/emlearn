"""
MCU Test Runner for emlearn benchmarks.

Orchestrates the build → flash → capture → parse workflow.
"""

import subprocess
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

from .. import temp_manager
from .boards import BoardConfig, OutputMethod, FlashMethod, BOARDS
from .capture import (
    OutputCapture, RTTCapture, SerialCapture, NativeCapture,
    parse_benchmark_output
)
from .flash import flash_target, reset_target


def find_west_workspace(app_path: Path) -> Optional[Path]:
    """Find west workspace directory.

    Search order:
    1. ZEPHYR_BASE environment variable (parent of zephyr/)
    2. Upward search from app path for .west/ folder

    Args:
        app_path: Path to Zephyr application.

    Returns:
        Path to west workspace or None if not found.
    """
    # Prefer ZEPHYR_BASE environment variable
    zephyr_base = os.environ.get('ZEPHYR_BASE')
    if zephyr_base:
        # ZEPHYR_BASE points to zephyr/, workspace is parent
        workspace = Path(zephyr_base).parent
        if (workspace / '.west').is_dir():
            return workspace

    # Fall back to upward search
    current = Path(app_path).resolve()
    for parent in [current] + list(current.parents):
        if (parent / '.west').is_dir():
            return parent
    return None


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.

    All timing values are in nanoseconds for maximum precision.
    Use the _us and _ms properties for convenience conversions.
    """
    success: bool
    board: str
    timing_mode: str
    cpu_freq_hz: int
    min_ns: int
    max_ns: int
    avg_ns: int
    min_cycles: int
    avg_cycles: int
    iterations: int
    raw_output: str
    error: Optional[str] = None

    @property
    def min_us(self) -> float:
        """Minimum time in microseconds."""
        return self.min_ns / 1000.0

    @property
    def max_us(self) -> float:
        """Maximum time in microseconds."""
        return self.max_ns / 1000.0

    @property
    def avg_us(self) -> float:
        """Average time in microseconds."""
        return self.avg_ns / 1000.0

    @property
    def min_ms(self) -> float:
        """Minimum time in milliseconds."""
        return self.min_ns / 1_000_000.0

    @property
    def avg_ms(self) -> float:
        """Average time in milliseconds."""
        return self.avg_ns / 1_000_000.0


class MCUTestRunner:
    """Orchestrates MCU benchmark runs.

    Handles the complete workflow:
    1. Build firmware (west build)
    2. Flash to target (J-Link, Renode, or native_sim)
    3. Capture output (RTT, serial, native stdout, or Renode UART)
    4. Parse results

    Supports:
    - Hardware targets: nrf52dk (J-Link/RTT)
    - Renode emulator: Cycle-accurate nRF52840 emulation with DWT timing
    - native_sim: Host-native validation (no real timing)

    Example:
        runner = MCUTestRunner(board=BOARDS['renode_nrf52840'])
        result = runner.run_benchmark(
            app_path='platform_examples/zephyr/benchmark',
            model_header='model.h'
        )
        print(f"Inference: {result.min_us} us")
    """

    def __init__(self, board: BoardConfig, zephyr_base: Optional[Path] = None,
                 timeout: float = 60.0, verbose: bool = False):
        """Initialize MCU test runner.

        Args:
            board: Target board configuration
            zephyr_base: Path to Zephyr base (auto-detect if None)
            timeout: Default timeout for operations
            verbose: Print verbose output
        """
        self.board = board
        self.timeout = timeout
        self.verbose = verbose

        # Find Zephyr base
        if zephyr_base:
            self.zephyr_base = zephyr_base
        else:
            self.zephyr_base = Path(os.environ.get('ZEPHYR_BASE', ''))

        self._build_dir: Optional[Path] = None
        self._capture: Optional[OutputCapture] = None
        self._native_process: Optional[subprocess.Popen] = None
        self._workspace: Optional[Path] = None

    def build(self, app_path: Path, extra_conf: Optional[Dict[str, Any]] = None,
              extra_conf_files: Optional[list[str]] = None,
              pristine: bool = False) -> Path:
        """Build firmware for target board.

        Args:
            app_path: Path to Zephyr application
            extra_conf: Extra Kconfig options (e.g., {'CONFIG_FOO': 'y'})
            extra_conf_files: Extra config files (e.g., ['boards/renode.conf'])
            pristine: Clean build directory first

        Returns:
            Path to build directory
        """
        west = shutil.which('west')
        if not west:
            raise RuntimeError("west not found. Install Zephyr SDK.")

        # Find west workspace (required for west build to work)
        app_path = Path(app_path).resolve()
        self._workspace = find_west_workspace(app_path)
        if not self._workspace:
            raise RuntimeError(
                f"No west workspace found for {app_path}. "
                "Expected .west/ directory in app path or parent."
            )

        # Create build directory using centralized temp manager
        self._build_dir = temp_manager.create_build_dir(self.board.name)

        cmd = [
            west, 'build',
            '-b', self.board.name,
            '-d', str(self._build_dir),
            str(app_path),
        ]

        if pristine:
            cmd.append('--pristine')

        # Add extra config files and options
        # Always pass Python executable to ensure pyelftools and other deps are available
        import sys
        cmake_args = [f'-DPython3_EXECUTABLE={sys.executable}']
        if extra_conf_files:
            # EXTRA_CONF_FILE accepts semicolon-separated list
            cmake_args.append(f'-DEXTRA_CONF_FILE={";".join(extra_conf_files)}')
        if extra_conf:
            cmake_args.extend(f'-D{k}={v}' for k, v in extra_conf.items())
        cmd.extend(['--'] + cmake_args)

        if self.verbose:
            print(f"Building: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=self._workspace,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Build failed:\n{result.stderr}")

        return self._build_dir

    def flash(self, build_dir: Optional[Path] = None) -> bool:
        """Flash firmware to target.

        Args:
            build_dir: Build directory (uses last build if None)

        Returns:
            True on success
        """
        if build_dir is None:
            build_dir = self._build_dir

        if build_dir is None:
            raise ValueError("No build directory. Call build() first.")

        elf_path = build_dir / 'zephyr' / 'zephyr.elf'
        if not elf_path.exists():
            raise FileNotFoundError(f"ELF file not found: {elf_path}")

        # Renode and native_sim don't need flashing
        if self.board.flash_method in (FlashMethod.NONE, FlashMethod.RENODE):
            return True

        method = self.board.flash_method.value
        return flash_target(
            elf_path=elf_path,
            device=self.board.device or '',
            method=method,
            build_dir=build_dir,
        )

    def start_capture(self) -> OutputCapture:
        """Start output capture for the target.

        Returns:
            OutputCapture instance
        """
        if self.board.output_method == OutputMethod.RTT:
            if not self.board.device:
                raise ValueError("RTT capture requires device name")
            self._capture = RTTCapture(
                device=self.board.device,
                channel=self.board.rtt_channel,
                timeout=self.timeout,
            )
        elif self.board.output_method == OutputMethod.SERIAL:
            if not self.board.serial_port:
                raise ValueError("Serial capture requires serial_port")
            self._capture = SerialCapture(
                port=self.board.serial_port,
                timeout=self.timeout,
            )
        elif self.board.output_method == OutputMethod.NATIVE:
            if not self._native_process:
                raise ValueError("Native process not started")
            self._capture = NativeCapture(
                process=self._native_process,
                timeout=self.timeout,
            )
        elif self.board.output_method == OutputMethod.RENODE:
            # Renode capture is handled in _run_renode via file backend
            raise ValueError("Renode capture uses file backend, not OutputCapture")
        else:
            raise ValueError(f"Unsupported output method: {self.board.output_method}")

        self._capture.start()
        return self._capture

    def stop_capture(self) -> str:
        """Stop output capture.

        Returns:
            Captured text
        """
        if self._capture:
            self._capture.stop()
            return self._capture.text
        return ''

    def run_native_sim(self, build_dir: Optional[Path] = None) -> subprocess.Popen:
        """Run firmware in native_sim.

        Args:
            build_dir: Build directory

        Returns:
            Native process
        """
        if build_dir is None:
            build_dir = self._build_dir

        if build_dir is None:
            raise ValueError("No build directory. Call build() first.")

        # Run the native executable directly
        exe_path = build_dir / 'zephyr' / 'zephyr.exe'
        if not exe_path.exists():
            raise FileNotFoundError(f"Native executable not found: {exe_path}")

        if self.verbose:
            print(f"Running native_sim: {exe_path}")

        self._native_process = subprocess.Popen(
            [str(exe_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return self._native_process

    def stop_native_sim(self) -> None:
        """Stop native_sim process."""
        if self._native_process:
            self._native_process.terminate()
            try:
                self._native_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._native_process.kill()
            self._native_process = None

    def _run_renode(self, build_dir: Path, timeout: float) -> str:
        """Run firmware in Renode emulator.

        Args:
            build_dir: Build directory containing zephyr.elf
            timeout: Timeout for benchmark execution

        Returns:
            Captured UART output text
        """
        from .renode_runner import RenodeRunner, RenodeConfig

        elf_path = build_dir / 'zephyr' / 'zephyr.elf'
        if not elf_path.exists():
            raise FileNotFoundError(f"ELF file not found: {elf_path}")

        # Get project root for relative paths
        project_root = Path(__file__).parent.parent.parent

        # Resolve repl path
        if self.board.renode_repl:
            repl_path = project_root / self.board.renode_repl
        else:
            raise ValueError("Board config must specify renode_repl path")

        if not repl_path.exists():
            raise FileNotFoundError(f"Platform file not found: {repl_path}")

        config = RenodeConfig(
            platform=self.board.renode_platform or 'nrf52840',
            cpu_freq_hz=self.board.cpu_freq_hz,
            repl_file=str(repl_path),
        )

        if self.verbose:
            print(f"Running Renode: {elf_path}")

        runner = RenodeRunner(config, timeout=timeout, verbose=self.verbose)
        return runner.run(elf_path, repl_path)

    def run_benchmark(self, app_path: Path, model_header: Optional[Path] = None,
                      extra_conf: Optional[Dict[str, Any]] = None,
                      timeout: Optional[float] = None) -> BenchmarkResult:
        """Run complete benchmark workflow.

        Args:
            app_path: Path to benchmark application
            model_header: Path to model header file (optional)
            extra_conf: Extra Kconfig options
            timeout: Timeout for benchmark run

        Returns:
            BenchmarkResult with timing data
        """
        if timeout is None:
            timeout = self.timeout

        try:
            # Build with appropriate config
            extra_conf_files = None
            if self.board.conf_file:
                # Use board-specific config file if specified
                conf_path = app_path / self.board.conf_file
                if conf_path.exists():
                    extra_conf_files = [self.board.conf_file]
                    if self.verbose:
                        print(f"Using board config: {conf_path}")
            elif self.board.output_method == OutputMethod.RENODE:
                # Fallback: Renode needs UART console (not RTT) for output capture
                # Check for boards/renode_nrf52840.conf in app directory
                renode_conf = app_path / 'boards' / 'renode_nrf52840.conf'
                if renode_conf.exists():
                    extra_conf_files = ['boards/renode_nrf52840.conf']
                    if self.verbose:
                        print(f"Using Renode config: {renode_conf}")

            if self.verbose:
                print(f"Building for {self.board.name}...")
            build_dir = self.build(app_path, extra_conf, extra_conf_files)

            # Dispatch based on output method
            if self.board.output_method == OutputMethod.RENODE:
                # Renode workflow: uses LoggingUartAnalyzer for output capture
                if self.verbose:
                    print("Starting Renode emulator...")
                output = self._run_renode(build_dir, timeout)

            elif self.board.output_method == OutputMethod.NATIVE:
                # native_sim workflow: run executable and capture stdout
                if self.verbose:
                    print("Starting native_sim...")
                self.run_native_sim(build_dir)
                capture = self.start_capture()

                # Wait for completion
                if not capture.wait_for_complete(timeout=timeout):
                    self.stop_native_sim()
                    return BenchmarkResult(
                        success=False,
                        board=self.board.name,
                        timing_mode='unknown',
                        cpu_freq_hz=0,
                        min_ns=0, max_ns=0, avg_ns=0,
                        min_cycles=0, avg_cycles=0,
                        iterations=0,
                        raw_output=capture.text,
                        error='Timeout waiting for benchmark completion',
                    )

                output = self.stop_capture()
                self.stop_native_sim()

            else:
                # Hardware workflow (RTT, serial)
                if self.verbose:
                    print("Flashing target...")
                if not self.flash(build_dir):
                    return BenchmarkResult(
                        success=False,
                        board=self.board.name,
                        timing_mode='unknown',
                        cpu_freq_hz=0,
                        min_ns=0, max_ns=0, avg_ns=0,
                        min_cycles=0, avg_cycles=0,
                        iterations=0,
                        raw_output='',
                        error='Flash failed',
                    )

                if self.verbose:
                    print("Capturing output...")
                capture = self.start_capture()

                # Reset to start benchmark
                if self.board.device:
                    reset_target(self.board.device)

                # Wait for completion
                if not capture.wait_for_complete(timeout=timeout):
                    self.stop_capture()
                    return BenchmarkResult(
                        success=False,
                        board=self.board.name,
                        timing_mode='unknown',
                        cpu_freq_hz=0,
                        min_ns=0, max_ns=0, avg_ns=0,
                        min_cycles=0, avg_cycles=0,
                        iterations=0,
                        raw_output=capture.text,
                        error='Timeout waiting for benchmark completion',
                    )

                output = self.stop_capture()

            # Parse results
            parsed = parse_benchmark_output(output)

            if parsed['status'] != 'ok':
                return BenchmarkResult(
                    success=False,
                    board=self.board.name,
                    timing_mode=parsed.get('timing', 'unknown'),
                    cpu_freq_hz=parsed.get('freq_hz', 0),
                    min_ns=parsed.get('min_ns', 0),
                    max_ns=parsed.get('max_ns', 0),
                    avg_ns=parsed.get('avg_ns', 0),
                    min_cycles=parsed.get('min_cycles', 0),
                    avg_cycles=parsed.get('avg_cycles', 0),
                    iterations=parsed.get('iterations', 0),
                    raw_output=output,
                    error=f"Benchmark status: {parsed['status']}",
                )

            required = ['avg_ns', 'iterations', 'freq_hz']
            missing = [k for k in required if not parsed.get(k)]
            if missing:
                return BenchmarkResult(
                    success=False,
                    board=self.board.name,
                    timing_mode=parsed.get('timing', 'unknown'),
                    cpu_freq_hz=parsed.get('freq_hz', 0),
                    min_ns=parsed.get('min_ns', 0),
                    max_ns=parsed.get('max_ns', 0),
                    avg_ns=parsed.get('avg_ns', 0),
                    min_cycles=parsed.get('min_cycles', 0),
                    avg_cycles=parsed.get('avg_cycles', 0),
                    iterations=parsed.get('iterations', 0),
                    raw_output=output,
                    error=f"Missing benchmark fields: {missing}",
                )

            return BenchmarkResult(
                success=True,
                board=self.board.name,
                timing_mode=parsed.get('timing', 'unknown'),
                cpu_freq_hz=parsed.get('freq_hz', 0),
                min_ns=parsed.get('min_ns', 0),
                max_ns=parsed.get('max_ns', 0),
                avg_ns=parsed.get('avg_ns', 0),
                min_cycles=parsed.get('min_cycles', 0),
                avg_cycles=parsed.get('avg_cycles', 0),
                iterations=parsed.get('iterations', 0),
                raw_output=output,
            )

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
                FileNotFoundError, RuntimeError, ValueError) as e:
            import traceback
            traceback.print_exc()
            return BenchmarkResult(
                success=False,
                board=self.board.name,
                timing_mode='unknown',
                cpu_freq_hz=0,
                min_ns=0, max_ns=0, avg_ns=0,
                min_cycles=0, avg_cycles=0,
                iterations=0,
                raw_output='',
                error=str(e),
            )

        finally:
            # Cleanup
            if self._native_process:
                self.stop_native_sim()
            if self._capture:
                self.stop_capture()

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._build_dir and self._build_dir.exists():
            temp_manager.cleanup_subdir(self._build_dir)
            self._build_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        """Auto-cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass  # Best effort cleanup

    def run_with_gdb(
        self,
        build_dir: Path,
        gdb_port: int = 3333,
        wait_for_attach: bool = True,
    ) -> dict:
        """Start emulator/debugger with GDB server.

        Starts the target platform with a GDB server running, allowing
        a GDB client to attach for debugging.

        For Renode: Adds 'machine StartGdbServer <port>' to script
        For Hardware: Requires external JLinkGDBServer
        For native_sim: Use gdb directly on the executable

        Args:
            build_dir: Build directory containing zephyr.elf
            gdb_port: Port for GDB server (default 3333)
            wait_for_attach: If True, halt at start for GDB attach

        Returns:
            Dict with 'process', 'port', and 'elf_path' keys

        Raises:
            NotImplementedError: For unsupported platforms
        """
        elf_path = build_dir / 'zephyr' / 'zephyr.elf'
        if not elf_path.exists():
            raise FileNotFoundError(f"ELF file not found: {elf_path}")

        if self.board.output_method == OutputMethod.RENODE:
            # Renode with GDB server
            from .renode_runner import RenodeRunner, RenodeConfig

            project_root = Path(__file__).parent.parent.parent
            if self.board.renode_repl:
                repl_path = project_root / self.board.renode_repl
            else:
                raise ValueError("Board config must specify renode_repl path")

            config = RenodeConfig(
                platform=self.board.renode_platform or 'nrf52840',
                cpu_freq_hz=self.board.cpu_freq_hz,
                repl_file=str(repl_path),
            )

            # Build Renode command with GDB server
            from .renode_runner import get_renode_executable
            renode_exe = get_renode_executable()
            if not renode_exe:
                raise RuntimeError("Renode not found")

            quantum_str = f"00:00:00.{config.quantum_ns:09d}"
            gdb_commands = [
                f'emulation SetGlobalQuantum "{quantum_str}"',
                'mach add "benchmark"',
                'mach set "benchmark"',
                f'machine LoadPlatformDescription @{repl_path}',
                f'sysbus LoadELF @{elf_path}',
                f'machine StartGdbServer {gdb_port}',
            ]
            if wait_for_attach:
                gdb_commands.append('machine Pause')

            cmd = [
                str(renode_exe),
                '--disable-xwt',
                '--console',
                '-e', '; '.join(gdb_commands),
            ]

            if self.verbose:
                print(f"Starting Renode with GDB on port {gdb_port}")
                print(f"Connect with: arm-none-eabi-gdb {elf_path} -ex 'target remote :{gdb_port}'")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            return {
                'process': process,
                'port': gdb_port,
                'elf_path': elf_path,
                'platform': 'renode',
            }

        elif self.board.output_method == OutputMethod.NATIVE:
            # native_sim: just return info for direct gdb
            exe_path = build_dir / 'zephyr' / 'zephyr.exe'
            if not exe_path.exists():
                raise FileNotFoundError(f"Native executable not found: {exe_path}")

            if self.verbose:
                print(f"Debug with: gdb {exe_path}")

            return {
                'process': None,
                'port': None,
                'elf_path': exe_path,
                'platform': 'native',
            }

        else:
            # Hardware requires external GDB server (JLinkGDBServer)
            if self.verbose:
                print(f"Start JLinkGDBServer separately, then connect GDB to port {gdb_port}")
                print(f"Debug with: arm-none-eabi-gdb {elf_path} -ex 'target remote :{gdb_port}'")

            return {
                'process': None,
                'port': gdb_port,
                'elf_path': elf_path,
                'platform': 'hardware',
            }
