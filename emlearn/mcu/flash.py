"""
Flash utilities for MCU testing.

Provides device discovery and programming via J-Link and west.
"""

import subprocess
import re
import shutil
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class JLinkDevice:
    """Information about a connected J-Link device."""
    serial: str
    product: str
    connection: str  # 'USB' or 'IP'


def discover_devices() -> List[JLinkDevice]:
    """Discover connected J-Link devices.

    Returns:
        List of connected J-Link devices
    """
    jlink_exe = shutil.which('JLinkExe') or shutil.which('JLink')
    if not jlink_exe:
        # Try common paths
        for path in ['/opt/SEGGER/JLink/JLinkExe', '/usr/bin/JLinkExe']:
            if Path(path).exists():
                jlink_exe = path
                break

    if not jlink_exe:
        return []

    # Run JLinkExe with ShowEmuList command
    try:
        result = subprocess.run(
            [jlink_exe, '-CommandFile', '/dev/stdin'],
            input='ShowEmuList\nqc\n',
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return []

    devices = []
    # Parse output for connected probes
    # Format: J-Link[0]: Connection: USB, Serial number: 12345678, ProductName: J-Link EDU Mini
    pattern = r'J-Link\[\d+\]:\s+Connection:\s+(\S+),\s+Serial number:\s+(\d+),\s+ProductName:\s+(.+)'

    for match in re.finditer(pattern, result.stdout):
        devices.append(JLinkDevice(
            connection=match.group(1),
            serial=match.group(2),
            product=match.group(3).strip(),
        ))

    return devices


def flash_jlink(elf_path: Path, device: str, speed: int = 4000,
                serial: Optional[str] = None) -> bool:
    """Flash firmware using J-Link.

    Args:
        elf_path: Path to ELF file to flash
        device: J-Link device name (e.g., 'nRF52832_xxAA')
        speed: Interface speed in kHz
        serial: J-Link serial number (for selecting specific probe)

    Returns:
        True on success, False on failure

    Raises:
        ValueError: If elf_path contains command injection characters
    """
    # Validate path doesn't contain command injection characters
    path_str = str(elf_path)
    if any(c in path_str for c in '\n\r;'):
        raise ValueError(f"Invalid characters in ELF path: {elf_path}")

    jlink_exe = shutil.which('JLinkExe') or shutil.which('JLink')
    if not jlink_exe:
        for path in ['/opt/SEGGER/JLink/JLinkExe', '/usr/bin/JLinkExe']:
            if Path(path).exists():
                jlink_exe = path
                break

    if not jlink_exe:
        raise RuntimeError("JLinkExe not found. Install J-Link tools.")

    # Create command script
    commands = f"""
si SWD
speed {speed}
loadfile {elf_path}
r
g
qc
"""

    cmd = [jlink_exe, '-Device', device, '-If', 'SWD', '-Speed', str(speed),
           '-AutoConnect', '1', '-CommandFile', '/dev/stdin']

    if serial:
        cmd.extend(['-SelectEmuBySN', serial])

    try:
        result = subprocess.run(
            cmd,
            input=commands,
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Check for success indicators
        return 'O.K.' in result.stdout or 'Downloading file' in result.stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def flash_west(build_dir: Path, board: str, runner: str = 'jlink') -> bool:
    """Flash firmware using west flash command.

    Args:
        build_dir: Path to Zephyr build directory
        board: Board name
        runner: Flash runner (jlink, openocd, etc.)

    Returns:
        True on success, False on failure
    """
    west = shutil.which('west')
    if not west:
        raise RuntimeError("west not found. Install Zephyr SDK.")

    cmd = ['west', 'flash', '-d', str(build_dir), '--runner', runner]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=build_dir.parent,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def flash_target(elf_path: Path, device: str, method: str = 'jlink',
                 build_dir: Optional[Path] = None, **kwargs) -> bool:
    """Flash firmware to target device.

    Args:
        elf_path: Path to ELF file
        device: Device name
        method: Flash method ('jlink', 'west', 'renode', 'none')
        build_dir: Build directory (for west method)
        **kwargs: Additional arguments passed to flash function

    Returns:
        True on success, False on failure
    """
    if method == 'jlink':
        return flash_jlink(elf_path, device, **kwargs)
    elif method == 'west':
        if build_dir is None:
            raise ValueError("build_dir required for west flash method")
        return flash_west(build_dir, device, **kwargs)
    elif method in ('renode', 'none'):
        return True  # No flash needed for emulators
    else:
        raise ValueError(f"Unknown flash method: {method}")


def reset_target(device: str, serial: Optional[str] = None) -> bool:
    """Reset target device via J-Link.

    Args:
        device: J-Link device name (e.g., 'nRF52832_xxAA').
            Should be from BOARDS config or other trusted source.
        serial: J-Link serial number (for selecting specific probe).

    Returns:
        True on success.
    """
    jlink_exe = shutil.which('JLinkExe') or shutil.which('JLink')
    if not jlink_exe:
        for path in ['/opt/SEGGER/JLink/JLinkExe', '/usr/bin/JLinkExe']:
            if Path(path).exists():
                jlink_exe = path
                break

    if not jlink_exe:
        return False

    commands = """
si SWD
r
g
qc
"""

    cmd = [jlink_exe, '-Device', device, '-If', 'SWD', '-AutoConnect', '1',
           '-CommandFile', '/dev/stdin']

    if serial:
        cmd.extend(['-SelectEmuBySN', serial])

    try:
        result = subprocess.run(
            cmd,
            input=commands,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False
