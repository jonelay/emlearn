"""
Board configurations for MCU testing.

Provides dataclass-based board definitions with type safety and IDE support.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List


class OutputMethod(Enum):
    """Output capture method for benchmark results."""
    RTT = "rtt"        # SEGGER RTT (requires J-Link)
    SERIAL = "serial"  # UART serial port
    NATIVE = "native"  # Native stdout capture (native_sim)
    RENODE = "renode"  # Renode UART file backend


class TimingSource(Enum):
    """Timing source for benchmark measurements."""
    DWT = "dwt"        # DWT cycle counter (Cortex-M3+)
    SYSTICK = "systick"  # Zephyr kernel systick


class FlashMethod(Enum):
    """Flash method for programming targets."""
    JLINK = "jlink"    # SEGGER J-Link
    WEST = "west"      # Zephyr west flash command
    NONE = "none"      # No flash needed (native_sim)
    RENODE = "renode"  # Renode emulator (no flash, just load ELF)


@dataclass
class BoardConfig:
    """Configuration for a target board.

    Attributes:
        name: Zephyr board name (e.g., 'nrf52dk_nrf52832')
        device: J-Link device name (e.g., 'nRF52832_xxAA')
        cpu_freq_hz: CPU frequency in Hz
        has_dwt: Whether DWT cycle counter is available
        timing_source: Preferred timing source
        output_method: Preferred output capture method
        flash_method: How to program the device
        serial_port: Serial port path (if using serial output)
        rtt_channel: RTT channel number for output
        is_emulator: Whether this is an emulated target
        renode_platform: Renode platform name (e.g., 'nrf52840')
        renode_repl: Path to custom .repl platform description file
        conf_file: Optional board-specific config file (e.g., 'boards/renode_nrf52840.conf')
    """
    name: str
    device: Optional[str] = None
    cpu_freq_hz: int = 64000000
    has_dwt: bool = True
    timing_source: TimingSource = TimingSource.DWT
    output_method: OutputMethod = OutputMethod.RTT
    flash_method: FlashMethod = FlashMethod.JLINK
    serial_port: Optional[str] = None
    rtt_channel: int = 0
    is_emulator: bool = False
    renode_platform: Optional[str] = None
    renode_repl: Optional[str] = None
    conf_file: Optional[str] = None

    @property
    def is_hardware(self) -> bool:
        """Returns True if this is a real hardware target."""
        return not self.is_emulator


# Pre-defined board configurations
BOARDS: Dict[str, BoardConfig] = {
    # Emulators
    'native_sim': BoardConfig(
        name='native_sim',
        device=None,
        cpu_freq_hz=1000000000,  # ~1GHz host (not meaningful for timing)
        has_dwt=False,           # No ARM DWT on native
        timing_source=TimingSource.SYSTICK,  # Use Zephyr systick
        output_method=OutputMethod.NATIVE,   # Native stdout
        flash_method=FlashMethod.NONE,       # No flash needed
        is_emulator=True,
    ),
    'renode_nrf52840': BoardConfig(
        name='nrf52840dk/nrf52840',  # Zephyr 4.x format: board/soc; matches the emulated SoC
        device=None,
        cpu_freq_hz=64_000_000,
        has_dwt=True,
        timing_source=TimingSource.DWT,
        output_method=OutputMethod.RENODE,
        flash_method=FlashMethod.RENODE,
        is_emulator=True,
        renode_platform='nrf52840',
        renode_repl='platform_examples/zephyr/benchmark/boards/nrf52840_dwt.repl',
        conf_file='boards/renode_nrf52840.conf',
    ),

    # Nordic nRF52 Series
    'nrf52dk_nrf52832': BoardConfig(
        name='nrf52dk/nrf52832',  # Zephyr 4.x format: board/soc
        device='nRF52832_xxAA',
        cpu_freq_hz=64000000,
        has_dwt=True,
        timing_source=TimingSource.DWT,
        output_method=OutputMethod.RTT,
        flash_method=FlashMethod.JLINK,
    ),
    'nrf52840dk_nrf52840': BoardConfig(
        name='nrf52840dk/nrf52840',  # Zephyr 4.x format: board/soc
        device='nRF52840_xxAA',
        cpu_freq_hz=64000000,
        has_dwt=True,
        timing_source=TimingSource.DWT,
        output_method=OutputMethod.RTT,
        flash_method=FlashMethod.JLINK,
    ),
}


def get_board(name: str) -> BoardConfig:
    """Get board configuration by name.

    Args:
        name: Board name (Zephyr board name or alias)

    Returns:
        BoardConfig for the specified board

    Raises:
        KeyError: If board not found
    """
    if name in BOARDS:
        return BOARDS[name]

    # Try to find by partial match
    for board_name, config in BOARDS.items():
        if name in board_name or board_name in name:
            return config

    raise KeyError(f"Unknown board: {name}. Available: {list(BOARDS.keys())}")


def list_boards(hardware_only: bool = False, emulator_only: bool = False) -> List[str]:
    """List available board configurations.

    Args:
        hardware_only: Only return hardware targets
        emulator_only: Only return emulator targets

    Returns:
        List of board names
    """
    boards = []
    for name, config in BOARDS.items():
        if hardware_only and config.is_emulator:
            continue
        if emulator_only and not config.is_emulator:
            continue
        boards.append(name)
    return sorted(boards)


def register_board(name: str, config: BoardConfig) -> None:
    """Register a custom board configuration.

    Args:
        name: Board name to register
        config: Board configuration
    """
    BOARDS[name] = config
