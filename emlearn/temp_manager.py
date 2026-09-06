"""
Temporary directory management for emlearn.

Provides centralized management of temporary files and directories,
with automatic cleanup on program exit.

All temp files are organized under a single session directory:
    /tmp/emlearn_<session_id>/
        compile/     - CompiledClassifier temp files
        mcu_build/   - MCU build directories
        traces/      - Renode execution traces
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import uuid
from pathlib import Path

# Module state
_session_dir: Path | None = None
_session_id: str = uuid.uuid4().hex[:12]
_keep: bool = False

__all__ = [
    "get_session_dir",
    "create_compile_dir",
    "create_build_dir",
    "create_trace_file",
    "set_keep_temp_files",
    "cleanup",
    "cleanup_subdir",
]


def get_session_dir() -> Path:
    """Get or create the session temp directory."""
    global _session_dir
    if _session_dir is None:
        _session_dir = Path(tempfile.gettempdir()) / f"emlearn_{_session_id}"
        _session_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(cleanup)
    return _session_dir


def create_compile_dir(name: str | None = None) -> Path:
    """Create a temp directory for C compilation."""
    return _create_subdir("compile", name)


def create_build_dir(name: str | None = None) -> Path:
    """Create a temp directory for MCU builds."""
    return _create_subdir("mcu_build", name)


def create_trace_file(name: str | None = None, suffix: str = ".log") -> Path:
    """Create a temp file for execution traces."""
    traces_dir = get_session_dir() / "traces"
    traces_dir.mkdir(exist_ok=True)
    prefix = f"{name}_" if name else "trace_"
    trace_path = traces_dir / f"{prefix}{uuid.uuid4().hex[:8]}{suffix}"
    trace_path.touch()
    return trace_path


def _create_subdir(category: str, name: str | None = None) -> Path:
    """Create a subdirectory within the session directory."""
    category_dir = get_session_dir() / category
    category_dir.mkdir(exist_ok=True)
    subdir_name = uuid.uuid4().hex[:8]
    if name:
        safe_name = name.replace("/", "_").replace("\\", "_")
        subdir_name = f"{safe_name}_{subdir_name}"
    subdir = category_dir / subdir_name
    subdir.mkdir(exist_ok=True)
    return subdir


def set_keep_temp_files(keep: bool) -> None:
    """Set whether to keep temp files on cleanup."""
    global _keep
    _keep = keep


def cleanup() -> None:
    """Clean up the session temp directory."""
    global _session_dir
    if _session_dir is None:
        return
    if _keep:
        print(f"Temp files preserved at: {_session_dir}")
        return
    shutil.rmtree(_session_dir, ignore_errors=True)
    _session_dir = None


def cleanup_subdir(subdir: Path) -> None:
    """Clean up a specific subdirectory."""
    if _keep:
        return
    shutil.rmtree(subdir, ignore_errors=True)
