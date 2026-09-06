"""
Output capture utilities for MCU testing.

Provides RTT and serial capture for benchmark output.
"""

import subprocess
import threading
import time
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable
from pathlib import Path


@dataclass
class CapturedLine:
    """A captured line of output with metadata."""
    text: str
    timestamp: float
    source: str  # 'rtt', 'serial', 'native'


class OutputCapture(ABC):
    """Abstract base class for output capture methods."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._lines: List[CapturedLine] = []
        self._lines_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0

    @abstractmethod
    def start(self) -> None:
        """Start capturing output."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing output."""
        pass

    @property
    def lines(self) -> List[CapturedLine]:
        """Get all captured lines."""
        with self._lines_lock:
            return self._lines.copy()

    @property
    def text(self) -> str:
        """Get all captured text as a single string."""
        return '\n'.join(line.text for line in self._lines)

    def wait_for_pattern(self, pattern: str, timeout: Optional[float] = None) -> Optional[str]:
        """Wait for a line matching the regex pattern.

        Args:
            pattern: Regex pattern to match
            timeout: Timeout in seconds (default: self.timeout)

        Returns:
            Matching line text, or None if timeout
        """
        if timeout is None:
            timeout = self.timeout

        regex = re.compile(pattern)
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            for line in self._lines:
                if regex.search(line.text):
                    return line.text
            time.sleep(0.05)

        return None

    def wait_for_complete(self, end_pattern: str = r'BENCHMARK:END',
                          timeout: Optional[float] = None) -> bool:
        """Wait for benchmark completion.

        Args:
            end_pattern: Pattern indicating completion
            timeout: Timeout in seconds

        Returns:
            True if completion detected, False on timeout
        """
        return self.wait_for_pattern(end_pattern, timeout) is not None

    def _add_line(self, text: str, source: str) -> None:
        """Add a captured line (thread-safe)."""
        line = CapturedLine(
            text=text.rstrip('\r\n'),
            timestamp=time.monotonic() - self._start_time,
            source=source,
        )
        with self._lines_lock:
            self._lines.append(line)


class RTTCapture(OutputCapture):
    """Capture output via SEGGER RTT using JLinkRTTLogger.

    Requires J-Link tools installed and a J-Link debugger connected.
    """

    def __init__(self, device: str, channel: int = 0, speed: int = 4000,
                 timeout: float = 30.0, jlink_path: Optional[str] = None):
        """Initialize RTT capture.

        Args:
            device: J-Link device name (e.g., 'nRF52832_xxAA')
            channel: RTT channel number
            speed: J-Link interface speed in kHz
            timeout: Default timeout for wait operations
            jlink_path: Path to JLinkRTTLogger (auto-detect if None)
        """
        super().__init__(timeout)
        self.device = device
        self.channel = channel
        self.speed = speed
        self._process: Optional[subprocess.Popen] = None

        # Find JLinkRTTLogger
        if jlink_path:
            self.jlink_rtt_logger = jlink_path
        else:
            self.jlink_rtt_logger = shutil.which('JLinkRTTLogger')
            if not self.jlink_rtt_logger:
                # Try common installation paths
                for path in ['/opt/SEGGER/JLink/JLinkRTTLogger',
                             '/usr/bin/JLinkRTTLogger']:
                    if Path(path).exists():
                        self.jlink_rtt_logger = path
                        break

    def start(self) -> None:
        """Start RTT capture via JLinkRTTLogger."""
        if not self.jlink_rtt_logger:
            raise RuntimeError("JLinkRTTLogger not found. Install J-Link tools.")

        self._lines = []
        self._start_time = time.monotonic()
        self._running = True

        # JLinkRTTLogger -Device <device> -If SWD -Speed <speed> -RTTChannel <ch>
        cmd = [
            self.jlink_rtt_logger,
            '-Device', self.device,
            '-If', 'SWD',
            '-Speed', str(self.speed),
            '-RTTChannel', str(self.channel),
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Start reader thread
        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop RTT capture."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _reader_thread(self) -> None:
        """Thread to read RTT output."""
        if not self._process or not self._process.stdout:
            return

        try:
            for line in self._process.stdout:
                if not self._running:
                    break
                self._add_line(line, 'rtt')
        except Exception:
            pass  # Process terminated


class SerialCapture(OutputCapture):
    """Capture output via serial port."""

    def __init__(self, port: str, baudrate: int = 115200,
                 timeout: float = 30.0):
        """Initialize serial capture.

        Args:
            port: Serial port path (e.g., '/dev/ttyACM0')
            baudrate: Baud rate
            timeout: Default timeout for wait operations
        """
        super().__init__(timeout)
        self.port = port
        self.baudrate = baudrate
        self._serial = None

    def start(self) -> None:
        """Start serial capture."""
        try:
            import serial
        except ImportError:
            raise RuntimeError("pyserial not installed. Run: pip install pyserial")

        self._lines = []
        self._start_time = time.monotonic()
        self._running = True

        self._serial = serial.Serial(
            self.port,
            baudrate=self.baudrate,
            timeout=0.1,
        )

        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop serial capture."""
        self._running = False
        if self._serial:
            self._serial.close()
            self._serial = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _reader_thread(self) -> None:
        """Thread to read serial output."""
        if not self._serial:
            return

        buffer = ''
        try:
            while self._running:
                data = self._serial.read(self._serial.in_waiting or 1)
                if data:
                    buffer += data.decode('utf-8', errors='replace')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        self._add_line(line, 'serial')
        except Exception:
            pass


class NativeCapture(OutputCapture):
    """Capture output from native_sim process stdout."""

    def __init__(self, process: subprocess.Popen, timeout: float = 30.0):
        """Initialize native capture.

        Args:
            process: Native subprocess (zephyr.exe)
            timeout: Default timeout for wait operations
        """
        super().__init__(timeout)
        self._process = process

    def start(self) -> None:
        """Start capturing native output."""
        self._lines = []
        self._start_time = time.monotonic()
        self._running = True

        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _reader_thread(self) -> None:
        """Thread to read native process output."""
        if not self._process or not self._process.stdout:
            return

        try:
            for line in self._process.stdout:
                if not self._running:
                    break
                if isinstance(line, bytes):
                    line = line.decode('utf-8', errors='replace')
                self._add_line(line, 'native')
        except Exception:
            pass


def parse_benchmark_output(text: str) -> dict:
    """Parse structured benchmark output.

    Args:
        text: Raw benchmark output text

    Returns:
        Dictionary with parsed results (timing in nanoseconds)
    """
    result = {
        'board': None,
        'timing': None,
        'freq_hz': None,
        'model_name': None,
        'features': None,
        'min_ns': None,
        'max_ns': None,
        'avg_ns': None,
        'min_cycles': None,
        'avg_cycles': None,
        'iterations': None,
        'status': None,
        'raw': text,
    }

    # Parse CONFIG line
    config_match = re.search(
        r'BENCHMARK:CONFIG\s+board=(\S+)\s+timing=(\S+)\s+freq=(\d+)',
        text
    )
    if config_match:
        result['board'] = config_match.group(1)
        result['timing'] = config_match.group(2)
        result['freq_hz'] = int(config_match.group(3))

    # Parse MODEL line
    model_match = re.search(
        r'BENCHMARK:MODEL\s+name=(\S+)\s+features=(\d+)',
        text
    )
    if model_match:
        result['model_name'] = model_match.group(1)
        result['features'] = int(model_match.group(2))

    # Parse RESULT line (nanoseconds format)
    result_match = re.search(
        r'RESULT:INFERENCE\s+min_ns=(\d+)\s+max_ns=(\d+)\s+avg_ns=(\d+)\s+'
        r'min_cycles=(\d+)\s+avg_cycles=(\d+)\s+iterations=(\d+)',
        text
    )
    if result_match:
        result['min_ns'] = int(result_match.group(1))
        result['max_ns'] = int(result_match.group(2))
        result['avg_ns'] = int(result_match.group(3))
        result['min_cycles'] = int(result_match.group(4))
        result['avg_cycles'] = int(result_match.group(5))
        result['iterations'] = int(result_match.group(6))

    # Parse END line
    end_match = re.search(r'BENCHMARK:END\s+status=(\S+)', text)
    if end_match:
        result['status'] = end_match.group(1)

    return result


def parse_micro_benchmark_output(text: str) -> dict[str, int]:
    """Parse micro-benchmark output lines.

    Micro-benchmarks are used to measure the cost of primitive operations
    like expf(), sigmoid, integer ops, floating-point ops, etc.

    Args:
        text: Raw output text containing RESULT:MICRO lines

    Returns:
        Dictionary mapping benchmark names to average cycle counts.
        Example: {'EXPF': 85, 'SIGMOID': 105, 'INT_OPS': 1, 'FP_OPS': 1, 'FP_DIV': 14}
    """
    results = {}

    # Pattern: RESULT:MICRO:EXPF avg_cycles=85 iterations=1000
    # Also support: RESULT:MICRO:EXPF cycles=85 (without avg_ prefix)
    pattern = re.compile(
        r'RESULT:MICRO:(\w+)\s+(?:avg_)?cycles=(\d+)',
        re.IGNORECASE
    )

    for match in pattern.finditer(text):
        name = match.group(1).upper()
        cycles = int(match.group(2))
        results[name] = cycles

    return results
