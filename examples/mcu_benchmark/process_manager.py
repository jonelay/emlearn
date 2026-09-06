"""
Process management for MCU benchmarks.

Provides parallel execution with clean termination on SIGINT/SIGTERM.
"""

from __future__ import annotations

import signal
import sys
from contextlib import contextmanager
from types import FrameType
from typing import Callable, Generator, TypeVar

# Check if joblib is available
try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    Parallel = None  # type: ignore[assignment,misc]
    delayed = None  # type: ignore[assignment]

T = TypeVar("T")
R = TypeVar("R")


@contextmanager
def process_group_manager() -> Generator[None, None, None]:
    """Context manager for signal handling during parallel execution.

    Intercepts SIGINT/SIGTERM to restore original handlers before
    re-raising, allowing clean propagation of interrupts.
    """
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def signal_handler(signum: int, frame: FrameType | None) -> None:
        print(f"\nReceived signal {signum}, terminating all processes...")
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


def run_parallel(
    items: list[T],
    process_fn: Callable[[T], R],
    n_jobs: int = -1,
    verbose: int = 10,
) -> list[R]:
    """Run function on items in parallel with managed cleanup.

    Args:
        items: List of items to process.
        process_fn: Function to apply to each item.
        n_jobs: Number of parallel jobs (-1 for all cores).
        verbose: Verbosity level.

    Returns:
        List of results.
    """
    if not JOBLIB_AVAILABLE or n_jobs == 1:
        if not JOBLIB_AVAILABLE:
            print("Warning: joblib not available, running serially")
        return [process_fn(item) for item in items]

    with process_group_manager():
        parallel = Parallel(
            n_jobs=n_jobs,
            verbose=verbose,
            batch_size=1,
            backend="loky",
        )
        return parallel(delayed(process_fn)(item) for item in items)


# Re-export delayed for convenience
if JOBLIB_AVAILABLE:
    __all__ = ["run_parallel", "process_group_manager", "delayed"]
else:
    __all__ = ["run_parallel", "process_group_manager"]
