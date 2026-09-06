"""
emlearn MCU testing module.

Provides infrastructure for running emlearn benchmarks on MCU targets:
- Renode emulation for cycle-accurate testing
- Hardware targets with J-Link/RTT for accurate timing

Example usage:
    from emlearn.mcu import MCUTestRunner, BOARDS

    runner = MCUTestRunner(board=BOARDS['nrf52dk'])
    result = runner.run_benchmark(model_path='model.h')
    print(f"Inference time: {result.min_us} us")
"""

from .boards import BoardConfig, BOARDS, get_board, list_boards
from .runner import MCUTestRunner, BenchmarkResult
from .capture import (
    OutputCapture, RTTCapture, SerialCapture,
    parse_benchmark_output, parse_micro_benchmark_output,
)
from .flash import flash_target, discover_devices

__all__ = [
    'BoardConfig',
    'BOARDS',
    'get_board',
    'list_boards',
    'MCUTestRunner',
    'BenchmarkResult',
    'OutputCapture',
    'RTTCapture',
    'SerialCapture',
    'parse_benchmark_output',
    'parse_micro_benchmark_output',
    'flash_target',
    'discover_devices',
]
