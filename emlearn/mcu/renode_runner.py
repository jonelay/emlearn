"""
Renode runners for emlearn MCU benchmarks.

Provides two runners for nRF52 target emulation:

1. RenodeRunner: Simple runner that traces entire execution. Good for quick
   tests but includes boot/warmup/print overhead in instruction counts.

2. RenodeGDBRunner: Uses GDB breakpoints on firmware marker functions to
   bracket only the benchmark loop. Provides precise instruction counts
   without overhead estimation. Recommended for accurate timing measurements.

DWT cycle counter in Renode is approximate (only advances at quantum boundaries),
so execution tracing provides more accurate relative timing between models.

Note: FileBackend for UART capture doesn't work reliably with nRF52840_UART.
We parse UART output from Renode's console log instead.

Trace Analysis
--------------
The module provides functions to analyze execution traces and break down
instruction counts by execution phase (boot, warmup, benchmark, output).
This helps identify overhead and validate timing assumptions.
"""

import atexit
import os
import re
import shutil
import socket
import subprocess
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

from .. import temp_manager


# =============================================================================
# Process Registry for Cleanup on Exit
# =============================================================================

_active_processes: Set[weakref.ref] = set()


def _register_process(proc: subprocess.Popen) -> None:
    """Register a subprocess for cleanup on exit."""
    ref = weakref.ref(proc, lambda r: _active_processes.discard(r))
    _active_processes.add(ref)


def _cleanup_all_processes() -> None:
    """Terminate all registered processes on interpreter exit.

    This catches zombie processes from failed tests or interrupted sessions.
    """
    for ref in list(_active_processes):
        proc = ref()
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass


atexit.register(_cleanup_all_processes)


def _find_free_port() -> int:
    """Find an available port for GDB server.

    Uses OS ephemeral port allocation to avoid conflicts when running
    multiple Renode instances in parallel (e.g., pytest -n 4).

    Returns:
        Available TCP port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# =============================================================================
# Trace Analysis for Overhead Isolation
# =============================================================================

@dataclass
class TraceOverheadAnalysis:
    """Result from trace overhead analysis.

    Attributes:
        total_instructions: Total instructions in trace.
        boot_instructions: Instructions from start to marker_boot_end.
        warmup_instructions: Instructions from marker_warmup_start to marker_warmup_end.
        benchmark_instructions: Instructions from marker_benchmark_start to marker_benchmark_end.
        output_instructions: Remaining instructions after benchmark.
        markers_found: List of marker names that were found in trace.
    """
    total_instructions: int = 0
    boot_instructions: int = 0
    warmup_instructions: int = 0
    benchmark_instructions: int = 0
    output_instructions: int = 0
    markers_found: list[str] = field(default_factory=list)


def get_marker_addresses(elf_path: Path) -> dict[str, int]:
    """Extract marker function addresses from ELF file.

    Uses nm to find addresses of marker functions used for overhead isolation.

    Args:
        elf_path: Path to ELF firmware file.

    Returns:
        Dictionary mapping marker names to addresses (as ints).
        Missing markers are not included.
    """
    markers = [
        'marker_boot_end',
        'marker_warmup_start',
        'marker_warmup_end',
        'marker_benchmark_start',
        'marker_benchmark_end',
        'benchmark_loop_start',
        'benchmark_loop_end',
    ]

    # Try to find nm executable
    nm_exe = shutil.which('arm-zephyr-eabi-nm')
    if not nm_exe:
        # Try Zephyr SDK location
        sdk_dir = os.environ.get('ZEPHYR_SDK_INSTALL_DIR')
        if sdk_dir:
            nm_path = Path(sdk_dir) / 'arm-zephyr-eabi' / 'bin' / 'arm-zephyr-eabi-nm'
            if nm_path.exists():
                nm_exe = str(nm_path)

    if not nm_exe:
        # Fall back to generic nm
        nm_exe = shutil.which('nm')

    if not nm_exe:
        return {}

    try:
        result = subprocess.run(
            [nm_exe, str(elf_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return {}

        addresses = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                addr_str, symbol_type, name = parts[0], parts[1], parts[2]
                if name in markers:
                    try:
                        addresses[name] = int(addr_str, 16)
                    except ValueError:
                        pass

        return addresses

    except (subprocess.TimeoutExpired, OSError):
        return {}


def analyze_trace_overhead(
    trace_file: Path,
    elf_path: Path,
    verbose: bool = False,
) -> TraceOverheadAnalysis:
    """Analyze trace to break down instruction counts by phase.

    Reads a PC-mode execution trace and counts instructions between marker
    functions to isolate overhead from boot, warmup, benchmark, and output.

    Args:
        trace_file: Path to PC-mode trace file (one PC per line, hex format).
        elf_path: Path to ELF file (for extracting marker addresses).
        verbose: Print detailed analysis info.

    Returns:
        TraceOverheadAnalysis with instruction counts per phase.
    """
    result = TraceOverheadAnalysis()

    # Get marker addresses
    markers = get_marker_addresses(elf_path)
    if not markers:
        # Can't analyze without markers, just count total
        try:
            with open(trace_file) as f:
                result.total_instructions = sum(1 for _ in f)
        except (IOError, OSError):
            pass
        return result

    if verbose:
        print(f"Marker addresses: {markers}")

    # Build address-to-marker lookup
    addr_to_marker = {addr: name for name, addr in markers.items()}

    # Read trace and find marker positions
    marker_positions = {}  # marker_name -> (first_line, last_line)
    line_count = 0

    try:
        with open(trace_file) as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()
                if not line:
                    continue

                # Parse hex address (format: 0x00000000 or just 00000000)
                try:
                    if line.startswith('0x'):
                        addr = int(line[2:], 16)
                    else:
                        addr = int(line, 16)
                except ValueError:
                    continue

                # Check if this is a marker
                if addr in addr_to_marker:
                    marker_name = addr_to_marker[addr]
                    if marker_name not in marker_positions:
                        marker_positions[marker_name] = (line_num, line_num)
                    else:
                        # Update last occurrence
                        first, _ = marker_positions[marker_name]
                        marker_positions[marker_name] = (first, line_num)

    except (IOError, OSError):
        return result

    result.total_instructions = line_count
    result.markers_found = list(marker_positions.keys())

    if verbose:
        print(f"Total instructions: {line_count}")
        print(f"Marker positions: {marker_positions}")

    # Calculate phase instruction counts
    if 'marker_boot_end' in marker_positions:
        first_boot_end, _ = marker_positions['marker_boot_end']
        result.boot_instructions = first_boot_end

    if 'marker_warmup_start' in marker_positions and 'marker_warmup_end' in marker_positions:
        _, warmup_start = marker_positions['marker_warmup_start']
        warmup_end, _ = marker_positions['marker_warmup_end']
        result.warmup_instructions = max(0, warmup_end - warmup_start)

    if 'marker_benchmark_start' in marker_positions and 'marker_benchmark_end' in marker_positions:
        _, bench_start = marker_positions['marker_benchmark_start']
        bench_end, _ = marker_positions['marker_benchmark_end']
        result.benchmark_instructions = max(0, bench_end - bench_start)

    # Output = everything after benchmark_end
    if 'marker_benchmark_end' in marker_positions:
        bench_end, _ = marker_positions['marker_benchmark_end']
        result.output_instructions = max(0, line_count - bench_end)

    return result


def parse_micro_benchmark_results(uart_output: str) -> dict[str, dict]:
    """Parse micro-benchmark results from UART output.

    Args:
        uart_output: Raw UART output string.

    Returns:
        Dictionary mapping benchmark name to result dict with 'avg_cycles' and 'iterations'.
        Example: {'EXPF': {'avg_cycles': 150, 'iterations': 1000}}
    """
    results = {}

    # Pattern: RESULT:MICRO:<name> avg_cycles=<cyc> iterations=<n>
    pattern = re.compile(
        r'RESULT:MICRO:(\w+)\s+avg_cycles=(\d+)\s+iterations=(\d+)'
    )

    for match in pattern.finditer(uart_output):
        name = match.group(1)
        avg_cycles = int(match.group(2))
        iterations = int(match.group(3))
        results[name] = {
            'avg_cycles': avg_cycles,
            'iterations': iterations,
        }

    return results


@dataclass
class RenodeResult:
    """Result from Renode benchmark run.

    Attributes:
        uart_output: Raw UART output text from firmware.
        total_instructions: Total instructions executed (from profiler).
        iterations: Number of benchmark iterations (parsed from output).
        instructions_per_inference: Calculated instructions per inference.
        cycles_per_inference: Estimated cycles (assuming ~1 CPI).
        ns_per_inference: Estimated nanoseconds per inference.
    """
    uart_output: str
    total_instructions: int = 0
    iterations: int = 0
    instructions_per_inference: int = 0
    cycles_per_inference: int = 0
    ns_per_inference: int = 0


@dataclass
class RenodeConfig:
    """Renode platform configuration.

    Attributes:
        platform: Platform name (e.g., 'nrf52840').
        cpu_freq_hz: CPU frequency in Hz.
        uart_path: Renode UART peripheral path.
        repl_file: Path to platform description file.
        quantum_ns: Global quantum in nanoseconds (16ns = 1 cycle at 64MHz).
        use_execution_trace: Enable execution tracing for accurate instruction counts.
        warmup_iterations: Number of warmup iterations (to subtract from overhead).
        benchmark_iterations: Number of benchmark iterations (default from Kconfig).
    """
    platform: str = 'nrf52840'
    cpu_freq_hz: int = 64_000_000
    uart_path: str = 'sysbus.uart0'
    repl_file: Optional[str] = None
    quantum_ns: int = 16  # 16ns = 1 cycle at 64MHz
    use_execution_trace: bool = True
    warmup_iterations: int = 10
    benchmark_iterations: int = 100


def get_renode_executable() -> Optional[Path]:
    """Find Renode executable.

    Search order:
    1. RENODE_PATH environment variable
    2. 'renode' on PATH
    3. Common install locations

    Returns:
        Path to renode executable, or None if not found.
    """
    # Check environment variable first
    env_path = os.environ.get('RENODE_PATH')
    if env_path:
        path = Path(env_path)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        elif path.is_dir():
            # RENODE_PATH is a directory - look for executable inside
            exe_path = path / 'renode'
            if exe_path.is_file() and os.access(exe_path, os.X_OK):
                return exe_path

    # Check PATH
    which_result = shutil.which('renode')
    if which_result:
        return Path(which_result)

    # Check common install locations
    home = Path.home()
    common_paths = [
        # Portable installs in home directory
        *sorted(home.glob('renode_*/renode'), reverse=True),
        *sorted(home.glob('renode-*/renode'), reverse=True),
        # System installs
        Path('/opt/renode/renode'),
        Path('/usr/local/bin/renode'),
        Path('/usr/bin/renode'),
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


class RenodeRunner:
    """Run Zephyr firmware in Renode emulator (simple mode).

    Runs firmware and captures UART output. If execution tracing is enabled,
    counts total instructions executed (including boot, warmup, and output).

    For precise benchmark timing, use RenodeGDBRunner instead - it uses GDB
    breakpoints to trace only the benchmark loop, eliminating overhead.

    Output is captured via LoggingUartAnalyzer which writes to Renode's console
    log. We parse this log for UART messages prefixed with the peripheral name.

    Example:
        config = RenodeConfig(cpu_freq_hz=64_000_000)
        runner = RenodeRunner(config, timeout=30.0)
        output = runner.run(Path('/tmp/build/zephyr/zephyr.elf'))
        print(output)
    """

    def __init__(
        self,
        config: RenodeConfig,
        timeout: float = 60.0,
        verbose: bool = False,
    ):
        """Initialize Renode runner.

        Args:
            config: Renode platform configuration.
            timeout: Timeout for benchmark execution in seconds.
            verbose: Print verbose output.
        """
        self.config = config
        self.timeout = timeout
        self.verbose = verbose
        self._process: Optional[subprocess.Popen] = None
        self.last_total_instructions: int = 0  # Instruction count from last run

    def _build_renode_commands(
        self,
        elf_path: Path,
        repl_path: Path,
        trace_file: Optional[Path] = None,
    ) -> str:
        """Build Renode inline commands.

        Args:
            elf_path: Path to ELF firmware file.
            repl_path: Path to platform description file.
            trace_file: Optional path to write execution trace (PC mode).

        Returns:
            Renode command string for -e flag.
        """
        # Convert quantum to Renode time format (HH:MM:SS.fffffffff)
        quantum_str = f"00:00:00.{self.config.quantum_ns:09d}"

        # Calculate MIPS from CPU frequency (assuming ~1 CPI)
        mips = self.config.cpu_freq_hz // 1_000_000

        commands = [
            # Set global quantum for cycle-accurate timing
            f'emulation SetGlobalQuantum "{quantum_str}"',
            # Create machine
            'mach add "benchmark"',
            'mach set "benchmark"',
            # Load platform description
            f'machine LoadPlatformDescription @{repl_path}',
            # Set CPU performance to match clock frequency
            f'cpu PerformanceInMips {mips}',
        ]

        # Add execution tracing if enabled
        if trace_file is not None:
            commands.append(
                f'cpu CreateExecutionTracing "trace" @{trace_file} PC'
            )

        commands.extend([
            # Use LoggingUartAnalyzer to capture output to console log
            # (FileBackend doesn't work reliably with nRF52840_UART)
            f'showAnalyzer {self.config.uart_path} Antmicro.Renode.Analyzers.LoggingUartAnalyzer',
            # Load firmware
            f'sysbus LoadELF @{elf_path}',
            # Start emulation
            'start',
        ])

        return '; '.join(commands)

    def run(
        self,
        elf_path: Path,
        repl_path: Optional[Path] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Run firmware and capture output.

        Uses execution tracing to count instructions, then calculates timing.
        Output is captured from LoggingUartAnalyzer which writes to Renode's
        console log.

        Args:
            elf_path: Path to ELF firmware file.
            repl_path: Path to platform description file (uses config default if None).
            timeout: Timeout in seconds (uses self.timeout if None).

        Returns:
            Captured UART output text with timing corrected from instruction counts.

        Raises:
            RuntimeError: If Renode executable not found.
            FileNotFoundError: If ELF or platform file not found.
        """
        if timeout is None:
            timeout = self.timeout

        # Find Renode executable
        renode_exe = get_renode_executable()
        if renode_exe is None:
            raise RuntimeError(
                "Renode not found. Install from https://renode.io or set RENODE_PATH."
            )

        # Validate ELF file
        elf_path = Path(elf_path).resolve()
        if not elf_path.exists():
            raise FileNotFoundError(f"ELF file not found: {elf_path}")

        # Determine platform file
        if repl_path is None and self.config.repl_file:
            repl_path = Path(self.config.repl_file)
        if repl_path is None:
            raise ValueError("No platform description file specified")

        repl_path = Path(repl_path).resolve()
        if not repl_path.exists():
            raise FileNotFoundError(f"Platform file not found: {repl_path}")

        # Create temp file for execution trace if enabled
        trace_file = None
        if self.config.use_execution_trace:
            trace_file = temp_manager.create_trace_file(suffix='.log')

        try:
            # Build command
            renode_commands = self._build_renode_commands(elf_path, repl_path, trace_file)

            cmd = [
                str(renode_exe),
                '--disable-xwt',  # No GUI
                '--console',
                '-e', renode_commands,
            ]

            if self.verbose:
                print(f"Running Renode: {' '.join(cmd)}")

            # Start Renode process with merged stdout/stderr
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
            )
            _register_process(self._process)

            # Wait for completion and capture output
            output = self._wait_for_completion(timeout)

            # Count instructions from trace (if enabled)
            # Note: For precise timing, use RenodeGDBRunner which brackets only
            # the benchmark loop. This runner traces the entire execution.
            if trace_file is not None and trace_file.exists():
                total_instructions = self._count_trace_instructions(trace_file)
                self.last_total_instructions = total_instructions
                if self.verbose:
                    print(f"Total instructions (full execution): {total_instructions:,}")

            return output

        finally:
            # Clean up trace file
            if trace_file is not None and trace_file.exists():
                try:
                    trace_file.unlink()
                except OSError:
                    pass

    def _count_trace_instructions(self, trace_file: Path) -> int:
        """Count instructions in execution trace file.

        Args:
            trace_file: Path to PC-mode trace file (one PC per line).

        Returns:
            Total number of instructions executed.
        """
        try:
            with open(trace_file) as f:
                return sum(1 for _ in f)
        except (IOError, OSError):
            return 0

    def _wait_for_completion(self, timeout: float) -> str:
        """Wait for benchmark completion or timeout.

        Reads Renode console output and parses UART messages from the
        LoggingUartAnalyzer. UART lines have format:
            uart0: [host: ...|virt: ...] <message>

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            Captured UART output text (messages only, without timing prefix).
        """
        start_time = time.monotonic()
        end_marker = 'BENCHMARK:END'
        output_lines = []
        uart_messages = []

        # Pattern to extract UART messages from LoggingUartAnalyzer output
        # Format: uart0: [host: 2.06s (+2.06s)|virt: 1.72ms (+1.72ms)] MESSAGE
        uart_pattern = re.compile(
            r'uart0:\s+\[host:.*?\|virt:.*?\]\s*(.*)',
            re.IGNORECASE
        )

        while time.monotonic() - start_time < timeout:
            # Check if process exited unexpectedly
            if self._process and self._process.poll() is not None:
                break

            # Read available output (non-blocking would be better but this works)
            try:
                # Use select or just readline with timeout
                line = self._process.stdout.readline()
                if line:
                    output_lines.append(line)

                    # Check for UART messages
                    match = uart_pattern.search(line)
                    if match:
                        message = match.group(1).strip()
                        uart_messages.append(message)

                        if self.verbose:
                            print(f"UART: {message}")

                        # Check for completion
                        if end_marker in message:
                            self._quit_and_wait()
                            return '\n'.join(uart_messages)
                else:
                    time.sleep(0.05)
            except (IOError, OSError):
                time.sleep(0.05)

        # Timeout - return whatever we have
        self._quit_and_wait()
        return '\n'.join(uart_messages)

    def _quit_and_wait(self) -> None:
        """Send quit command and wait for Renode to exit cleanly.

        If execution tracing was enabled, we first call DisableExecutionTracing
        to flush the trace buffer to disk before quitting. This avoids the race
        condition where the trace file is incomplete when we try to read it.
        """
        if self._process and self._process.poll() is None:
            try:
                # Flush trace buffer before quit (if tracing was enabled)
                # DisableExecutionTracing ensures all buffered trace data is written
                self._process.stdin.write('cpu DisableExecutionTracing\n')
                self._process.stdin.flush()
                time.sleep(0.1)  # Brief wait for buffer flush

                # Send quit command
                self._process.stdin.write('quit\n')
                self._process.stdin.flush()
                # Wait for clean exit
                self._process.wait(timeout=5.0)
            except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
                # Force terminate if quit doesn't work
                self._terminate()
        self._process = None

    def _terminate(self) -> None:
        """Terminate Renode process forcefully."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None


def get_gdb_executable() -> Optional[Path]:
    """Find ARM GDB executable.

    Search order:
    1. ZEPHYR_SDK_INSTALL_DIR environment variable
    2. arm-zephyr-eabi-gdb on PATH
    3. Common Zephyr SDK locations

    Returns:
        Path to GDB executable, or None if not found.
    """
    # Check Zephyr SDK first
    sdk_dir = os.environ.get('ZEPHYR_SDK_INSTALL_DIR')
    if sdk_dir:
        gdb_path = Path(sdk_dir) / 'arm-zephyr-eabi' / 'bin' / 'arm-zephyr-eabi-gdb'
        if gdb_path.exists():
            return gdb_path

    # Check PATH
    which_result = shutil.which('arm-zephyr-eabi-gdb')
    if which_result:
        return Path(which_result)

    # Check common locations
    home = Path.home()
    common_paths = [
        *sorted(home.glob('zephyr-sdk-*/arm-zephyr-eabi/bin/arm-zephyr-eabi-gdb'), reverse=True),
        Path('/opt/zephyr-sdk/arm-zephyr-eabi/bin/arm-zephyr-eabi-gdb'),
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


@dataclass
class GDBBenchmarkResult:
    """Result from GDB-controlled benchmark run.

    Attributes:
        uart_output: Raw UART output text from firmware.
        total_instructions: Instructions executed during benchmark loop only.
        iterations: Number of benchmark iterations.
        instructions_per_inference: Calculated instructions per inference.
        ns_per_inference: Calculated nanoseconds per inference.
    """
    uart_output: str
    total_instructions: int = 0
    iterations: int = 100
    instructions_per_inference: int = 0
    ns_per_inference: int = 0


class RenodeGDBRunner:
    """Run Zephyr firmware in Renode with GDB-controlled execution tracing.

    Uses GDB breakpoints on firmware marker functions to precisely bracket
    the benchmark loop, eliminating overhead estimation. This provides
    accurate instruction counts for just the timed code.

    Workflow:
    1. Start Renode with GDB server enabled
    2. Connect GDB and set breakpoints on benchmark_loop_start/end
    3. Continue to benchmark_loop_start, enable execution tracing
    4. Continue to benchmark_loop_end, disable tracing
    5. Continue to completion, capture UART output
    6. Count instructions in trace file (= benchmark loop only)

    Note: Renode's stdout is redirected to a temp file to avoid pipe buffer
    deadlocks. The subprocess pipe buffer is limited (~64KB on Linux) and
    Renode produces lots of output. If not drained, Renode blocks and
    GDB commands hang indefinitely.

    Example:
        config = RenodeConfig(cpu_freq_hz=64_000_000)
        runner = RenodeGDBRunner(config, timeout=60.0)
        result = runner.run(Path('/tmp/build/zephyr/zephyr.elf'))
        print(f"Instructions per inference: {result.instructions_per_inference}")
    """

    def __init__(
        self,
        config: RenodeConfig,
        timeout: float = 60.0,
        verbose: bool = False,
    ):
        """Initialize GDB-controlled Renode runner.

        Args:
            config: Renode platform configuration.
            timeout: Timeout for benchmark execution in seconds.
            verbose: Print verbose output.
        """
        self.config = config
        self.timeout = timeout
        self.verbose = verbose
        self._gdb_port = _find_free_port()  # Dynamic port for parallel execution
        self._renode_process: Optional[subprocess.Popen] = None
        self._gdb_process: Optional[subprocess.Popen] = None
        self._trace_file: Optional[Path] = None
        self._renode_log_file: Optional[Path] = None

    def run(
        self,
        elf_path: Path,
        repl_path: Optional[Path] = None,
    ) -> GDBBenchmarkResult:
        """Run firmware with GDB-controlled tracing.

        Args:
            elf_path: Path to ELF firmware file.
            repl_path: Path to platform description file.

        Returns:
            GDBBenchmarkResult with precise instruction counts.

        Raises:
            RuntimeError: If Renode or GDB not found.
            FileNotFoundError: If ELF or platform file not found.
        """
        # Find executables
        renode_exe = get_renode_executable()
        if renode_exe is None:
            raise RuntimeError("Renode not found")

        gdb_exe = get_gdb_executable()
        if gdb_exe is None:
            raise RuntimeError("ARM GDB not found")

        # Validate paths
        elf_path = Path(elf_path).resolve()
        if not elf_path.exists():
            raise FileNotFoundError(f"ELF file not found: {elf_path}")

        if repl_path is None and self.config.repl_file:
            repl_path = Path(self.config.repl_file)
        if repl_path is None:
            raise ValueError("No platform description file specified")

        repl_path = Path(repl_path).resolve()
        if not repl_path.exists():
            raise FileNotFoundError(f"Platform file not found: {repl_path}")

        # Create trace file
        self._trace_file = temp_manager.create_trace_file(suffix='.log')

        try:
            # Start Renode with GDB server (no execution tracing yet)
            self._start_renode(renode_exe, elf_path, repl_path)
            # Wait for Renode GDB server to become ready
            import socket
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(('localhost', self._gdb_port), timeout=0.5):
                        break
                except OSError:
                    time.sleep(0.2)
            else:
                raise TimeoutError(
                    f"Renode GDB server not ready on port {self._gdb_port} after 10s"
                )

            # Run GDB-controlled benchmark
            uart_output = self._run_gdb_benchmark(gdb_exe, elf_path)

            # Count instructions (benchmark loop only)
            total_instructions = self._count_trace_instructions()

            # Calculate timing (no overhead estimation needed)
            iterations = self.config.benchmark_iterations
            instr_per_inference = total_instructions // iterations if iterations > 0 else 0
            ns_per_inference = (instr_per_inference * 1_000_000_000) // self.config.cpu_freq_hz

            if self.verbose:
                print(f"GDB benchmark: {total_instructions:,} instructions for {iterations} iterations")
                print(f"  Instructions per inference: {instr_per_inference:,}")
                print(f"  ns per inference: {ns_per_inference:,}")

            return GDBBenchmarkResult(
                uart_output=uart_output,
                total_instructions=total_instructions,
                iterations=iterations,
                instructions_per_inference=instr_per_inference,
                ns_per_inference=ns_per_inference,
            )

        finally:
            self._cleanup()

    def _start_renode(self, renode_exe: Path, elf_path: Path, repl_path: Path) -> None:
        """Start Renode with GDB server enabled.

        Redirects Renode's stdout to a temp file to avoid pipe buffer deadlocks.
        The subprocess pipe buffer is limited (~64KB on Linux) and if Renode's
        output is not drained, it blocks indefinitely.
        """
        quantum_str = f"00:00:00.{self.config.quantum_ns:09d}"
        mips = self.config.cpu_freq_hz // 1_000_000

        commands = [
            # Don't set custom quantum when using GDB - causes arithmetic overflow
            # f'emulation SetGlobalQuantum "{quantum_str}"',
            'mach add "benchmark"',
            'mach set "benchmark"',
            f'machine LoadPlatformDescription @{repl_path}',
            f'cpu PerformanceInMips {mips}',
            f'showAnalyzer {self.config.uart_path} Antmicro.Renode.Analyzers.LoggingUartAnalyzer',
            f'sysbus LoadELF @{elf_path}',
            f'machine StartGdbServer {self._gdb_port}',
            # Start emulation - GDB will halt it on connect
            'start',
        ]

        cmd = [
            str(renode_exe),
            '--disable-xwt',
            '--console',
            '-e', '; '.join(commands),
        ]

        if self.verbose:
            print(f"Starting Renode with GDB server on port {self._gdb_port}")

        # Create temp file for Renode output to avoid pipe buffer deadlock
        self._renode_log_file = temp_manager.create_trace_file(suffix='_renode.log')
        renode_log_handle = open(self._renode_log_file, 'w')

        try:
            self._renode_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=renode_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _register_process(self._renode_process)
        finally:
            # Close our handle - subprocess has its own copy via fd duplication
            renode_log_handle.close()

    def _run_gdb_benchmark(self, gdb_exe: Path, elf_path: Path) -> str:
        """Run benchmark with GDB-controlled tracing."""
        # GDB commands to execute
        gdb_commands = [
            # Connect to Renode GDB server
            f'-target-select remote localhost:{self._gdb_port}',
            # Set breakpoints on marker functions
            '-break-insert benchmark_loop_start',
            '-break-insert benchmark_loop_end',
            # Continue to benchmark_loop_start
            '-exec-continue',
        ]

        # Start GDB in MI mode
        cmd = [str(gdb_exe), '--interpreter=mi3', str(elf_path)]

        if self.verbose:
            print(f"Starting GDB: {' '.join(cmd)}")

        self._gdb_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _register_process(self._gdb_process)

        try:
            # Execute initial commands
            for gdb_cmd in gdb_commands:
                self._gdb_command(gdb_cmd)

            # Wait for benchmark_loop_start breakpoint
            self._wait_for_stop("benchmark_loop_start")

            # Enable execution tracing via GDB monitor command
            # Use escaped quotes for GDB MI interface
            trace_cmd = f'cpu CreateExecutionTracing \\"trace\\" @{self._trace_file} PC'
            self._gdb_command(f'-interpreter-exec console "monitor {trace_cmd}"')

            if self.verbose:
                print("Tracing enabled at benchmark_loop_start")

            # Continue to benchmark_loop_end
            self._gdb_command('-exec-continue')
            self._wait_for_stop("benchmark_loop_end")

            # Disable execution tracing (flushes file)
            self._gdb_command('-interpreter-exec console "monitor cpu DisableExecutionTracing"')
            time.sleep(0.1)  # Brief wait for flush

            if self.verbose:
                print("Tracing disabled at benchmark_loop_end")

            # Continue to completion to capture UART output
            self._gdb_command('-exec-continue')

            # Wait for benchmark completion via Renode console
            uart_output = self._wait_for_benchmark_complete()

            return uart_output

        finally:
            # Clean up GDB
            try:
                self._gdb_command('-gdb-exit')
            except (BrokenPipeError, OSError):
                pass

    def _gdb_command(self, cmd: str) -> str:
        """Send command to GDB MI interface and read response."""
        if self._gdb_process is None or self._gdb_process.stdin is None:
            raise RuntimeError("GDB not running")

        if self.verbose:
            print(f"GDB> {cmd}")

        self._gdb_process.stdin.write(f'{cmd}\n')
        self._gdb_process.stdin.flush()

        return self._read_gdb_response()

    def _read_gdb_response(self, timeout: float = 30.0) -> str:
        """Read GDB MI response until result or async notification.

        Args:
            timeout: Maximum seconds to wait for a response line.

        Raises:
            TimeoutError: If no response received within timeout.
        """
        import select

        if self._gdb_process is None or self._gdb_process.stdout is None:
            return ""

        lines = []
        while True:
            fd = self._gdb_process.stdout.fileno()
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                raise TimeoutError(
                    f"GDB response timeout after {timeout}s. "
                    f"Partial response: {''.join(lines)}"
                )

            line = self._gdb_process.stdout.readline()
            if not line:
                break
            lines.append(line.strip())

            if self.verbose and line.strip():
                print(f"GDB< {line.strip()}")

            # GDB MI result records start with ^
            # Async exec records start with *
            if line.startswith('^') or line.startswith('(gdb)'):
                break

        return '\n'.join(lines)

    def _wait_for_stop(self, expected_function: str) -> None:
        """Wait for GDB to report stopped state at expected breakpoint."""
        if self._gdb_process is None or self._gdb_process.stdout is None:
            return

        start_time = time.monotonic()
        while time.monotonic() - start_time < self.timeout:
            line = self._gdb_process.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue

            if self.verbose:
                print(f"GDB< {line.strip()}")

            # Check for GDB internal errors or thread exit (connection lost)
            if '=thread-exited' in line or '=thread-group-exited' in line:
                raise RuntimeError(
                    f"GDB lost connection to Renode while waiting for {expected_function}"
                )

            if 'internal-error' in line:
                raise RuntimeError(f"GDB internal error: {line.strip()}")

            # Look for stopped notification
            if '*stopped' in line:
                if expected_function in line or 'breakpoint-hit' in line:
                    return
                # Unexpected stop
                if self.verbose:
                    print(f"Stopped but not at {expected_function}: {line}")
                return

        raise TimeoutError(f"Timeout waiting for breakpoint at {expected_function}")

    def _wait_for_benchmark_complete(self) -> str:
        """Wait for benchmark completion via Renode UART output.

        Reads from the Renode log file (stdout is redirected there to avoid
        pipe buffer deadlocks).
        """
        if self._renode_log_file is None or not self._renode_log_file.exists():
            return ""

        uart_messages = []
        uart_pattern = re.compile(r'uart0:\s+\[host:.*?\|virt:.*?\]\s*(.*)', re.IGNORECASE)
        end_marker = 'BENCHMARK:END'
        start_time = time.monotonic()
        last_position = 0

        while time.monotonic() - start_time < self.timeout:
            # Check if process has exited
            if self._renode_process and self._renode_process.poll() is not None:
                # Process exited - read remaining output
                pass

            # Read new content from log file
            try:
                with open(self._renode_log_file, 'r') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    last_position = f.tell()
            except (IOError, OSError):
                time.sleep(0.1)
                continue

            if not new_content:
                time.sleep(0.1)
                continue

            # Process new lines
            for line in new_content.splitlines():
                match = uart_pattern.search(line)
                if match:
                    message = match.group(1).strip()
                    uart_messages.append(message)

                    if self.verbose:
                        print(f"UART: {message}")

                    if end_marker in message:
                        return '\n'.join(uart_messages)

        return '\n'.join(uart_messages)

    def _count_trace_instructions(self) -> int:
        """Count instructions in the trace file."""
        if self._trace_file is None or not self._trace_file.exists():
            return 0

        try:
            with open(self._trace_file) as f:
                return sum(1 for _ in f)
        except (IOError, OSError):
            return 0

    def _cleanup(self) -> None:
        """Clean up Renode and GDB processes and temp files."""
        # Terminate GDB
        if self._gdb_process:
            try:
                self._gdb_process.terminate()
                self._gdb_process.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, OSError):
                self._gdb_process.kill()
            self._gdb_process = None

        # Terminate Renode
        if self._renode_process:
            try:
                self._renode_process.stdin.write('quit\n')
                self._renode_process.stdin.flush()
                self._renode_process.wait(timeout=5.0)
            except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
                self._renode_process.terminate()
                try:
                    self._renode_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._renode_process.kill()
            self._renode_process = None

        # Clean up trace file
        if self._trace_file and self._trace_file.exists():
            try:
                self._trace_file.unlink()
            except OSError:
                pass
            self._trace_file = None

        # Clean up Renode log file
        if self._renode_log_file and self._renode_log_file.exists():
            try:
                self._renode_log_file.unlink()
            except OSError:
                pass
            self._renode_log_file = None


def run_renode_benchmark(
    elf_path: Path,
    repl_path: Path,
    cpu_freq_hz: int = 64_000_000,
    timeout: float = 60.0,
    verbose: bool = False,
) -> str:
    """Convenience function to run benchmark in Renode.

    Args:
        elf_path: Path to ELF firmware file.
        repl_path: Path to platform description file.
        cpu_freq_hz: CPU frequency in Hz.
        timeout: Timeout in seconds.
        verbose: Print verbose output.

    Returns:
        Captured UART output text.
    """
    config = RenodeConfig(cpu_freq_hz=cpu_freq_hz, repl_file=str(repl_path))
    runner = RenodeRunner(config, timeout=timeout, verbose=verbose)
    return runner.run(elf_path, repl_path)
