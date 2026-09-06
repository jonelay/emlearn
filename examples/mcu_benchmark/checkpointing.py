"""
Checkpoint and Resume Infrastructure for MCU Benchmarks
=======================================================

Provides checkpoint state management for long-running benchmark sweeps,
enabling resume after interruption.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar('T')


@dataclass
class CheckpointState:
    """Checkpoint state for resumable benchmark runs.

    Attributes:
        completed_configs: Config hashes that completed successfully.
        failed_configs: Config hashes that failed.
        current_benchmark: Currently running benchmark name.
        current_platform: Currently running platform.
        results_so_far: Accumulated results from completed configs.
    """

    completed_configs: list[str] = field(default_factory=list)
    failed_configs: list[str] = field(default_factory=list)
    current_benchmark: str | None = None
    current_platform: str | None = None
    results_so_far: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Save checkpoint to disk atomically.

        Args:
            path: Path to save checkpoint JSON.
        """
        data = {
            'completed_configs': self.completed_configs,
            'failed_configs': self.failed_configs,
            'current_benchmark': self.current_benchmark,
            'current_platform': self.current_platform,
            'results_count': len(self.results_so_far),
        }
        # Atomic write via temp file
        tmp_path = path.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.rename(path)

    @classmethod
    def load(cls, path: Path) -> CheckpointState:
        """Load checkpoint from disk.

        Args:
            path: Path to checkpoint JSON.

        Returns:
            CheckpointState loaded from file, or empty state if not found.
        """
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(
                completed_configs=data.get('completed_configs', []),
                failed_configs=data.get('failed_configs', []),
                current_benchmark=data.get('current_benchmark'),
                current_platform=data.get('current_platform'),
                results_so_far=[],  # Results loaded separately from CSV
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load checkpoint from {path}: {e}")
            return cls()

    def is_completed(self, config_hash: str) -> bool:
        """Check if config was already completed.

        Args:
            config_hash: Hash of the config to check.

        Returns:
            True if config completed successfully.
        """
        return config_hash in self.completed_configs

    def is_failed(self, config_hash: str) -> bool:
        """Check if config previously failed.

        Args:
            config_hash: Hash of the config to check.

        Returns:
            True if config failed previously.
        """
        return config_hash in self.failed_configs

    def mark_completed(self, config_hash: str) -> None:
        """Mark config as completed.

        Args:
            config_hash: Hash of completed config.
        """
        if config_hash not in self.completed_configs:
            self.completed_configs.append(config_hash)

    def mark_failed(self, config_hash: str) -> None:
        """Mark config as failed.

        Args:
            config_hash: Hash of failed config.
        """
        if config_hash not in self.failed_configs:
            self.failed_configs.append(config_hash)


def config_hash(config: dict) -> str:
    """Generate stable hash for config deduplication.

    Args:
        config: Configuration dictionary.

    Returns:
        16-character hex hash string.
    """
    # Sort keys for deterministic serialization
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def run_with_checkpointing(
    configs: list[dict],
    run_fn: Callable[[dict], T | None],
    checkpoint: CheckpointState,
    checkpoint_path: Path,
    skip_failed: bool = True,
    verbose: bool = False,
) -> list[T]:
    """Run configs with checkpoint/resume support.

    Args:
        configs: List of configurations to process.
        run_fn: Function to run on each config. Returns result or None on failure.
        checkpoint: CheckpointState to track progress.
        checkpoint_path: Path to save checkpoint after each config.
        skip_failed: Whether to skip configs that previously failed.
        verbose: Print progress information.

    Returns:
        List of results from successful configs.
    """
    results: list[T] = []
    total = len(configs)

    for i, config in enumerate(configs):
        cfg_hash = config_hash(config)
        config_name = config.get('name', config.get('model_name', f'config_{i}'))

        # Skip already completed
        if checkpoint.is_completed(cfg_hash):
            if verbose:
                print(f"[{i + 1}/{total}] {config_name}: skipping (completed)")
            continue

        # Skip known failures
        if skip_failed and checkpoint.is_failed(cfg_hash):
            if verbose:
                print(f"[{i + 1}/{total}] {config_name}: skipping (previously failed)")
            continue

        if verbose:
            print(f"[{i + 1}/{total}] {config_name}: running...")

        try:
            result = run_fn(config)
            if result is not None:
                results.append(result)
                checkpoint.mark_completed(cfg_hash)
                if verbose:
                    print(f"[{i + 1}/{total}] {config_name}: completed")
            else:
                checkpoint.mark_failed(cfg_hash)
                if verbose:
                    print(f"[{i + 1}/{total}] {config_name}: failed (returned None)")
        except Exception as e:
            print(f"[{i + 1}/{total}] {config_name}: failed with error: {e}")
            checkpoint.mark_failed(cfg_hash)

        # Save checkpoint after each config
        checkpoint.results_so_far = [r if isinstance(r, dict) else {} for r in results]
        checkpoint.save(checkpoint_path)

    return results


def merge_checkpoints(
    primary: CheckpointState,
    secondary: CheckpointState,
) -> CheckpointState:
    """Merge two checkpoint states.

    Useful when combining results from parallel runs.

    Args:
        primary: Primary checkpoint (takes precedence).
        secondary: Secondary checkpoint to merge in.

    Returns:
        Merged CheckpointState.
    """
    merged = CheckpointState(
        completed_configs=list(set(primary.completed_configs + secondary.completed_configs)),
        failed_configs=list(set(primary.failed_configs + secondary.failed_configs)),
        current_benchmark=primary.current_benchmark or secondary.current_benchmark,
        current_platform=primary.current_platform or secondary.current_platform,
        results_so_far=primary.results_so_far + secondary.results_so_far,
    )
    return merged
