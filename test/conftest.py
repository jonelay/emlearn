"""Shared fixtures and datasets for tree model tests."""

import os
import io
import shutil
from pathlib import Path

import pytest
import numpy
import pandas
from sklearn import datasets

from emlearn.preprocessing import Quantizer

# Fixed random seed for reproducible tests
RANDOM_SEED = 42

# Tolerance for int16 quantization (20% allowed incorrect)
ALLOWED_INCORRECT_INT16 = 0.20

here = os.path.dirname(__file__)
emlearn_root = Path(here).parent


# =============================================================================
# MCU Testing Infrastructure
# =============================================================================

def pytest_configure(config):
    """Register MCU markers."""
    config.addinivalue_line(
        "markers", "mcu: mark test as requiring MCU infrastructure"
    )
    config.addinivalue_line(
        "markers", "mcu_renode: mark test as requiring Renode emulation"
    )
    config.addinivalue_line(
        "markers", "mcu_hardware: mark test as requiring real hardware"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselected by default)"
    )
    config.addinivalue_line(
        "markers", "renode: mark test as requiring Renode emulator"
    )


def pytest_collection_modifyitems(config, items):
    """Skip MCU tests unless explicitly enabled."""
    mcu_mode = os.environ.get('EMLEARN_TEST_MCU', '').lower()
    renode_path = os.environ.get('RENODE_PATH')

    # Determine which MCU tests to run
    run_renode = mcu_mode in ('renode', 'all')
    run_hardware = mcu_mode in ('hardware', 'all')

    skip_mcu = pytest.mark.skip(reason="MCU tests disabled (set EMLEARN_TEST_MCU=renode|hardware|all)")
    skip_renode = pytest.mark.skip(reason="Renode tests disabled (set EMLEARN_TEST_MCU=renode|all)")
    skip_hardware = pytest.mark.skip(reason="Hardware tests disabled (set EMLEARN_TEST_MCU=hardware|all)")
    skip_no_renode = pytest.mark.skip(reason="RENODE_PATH not set")

    for item in items:
        # Skip general MCU tests if not enabled
        if "mcu" in item.keywords and not (run_renode or run_hardware):
            item.add_marker(skip_mcu)
        # Skip Renode-specific tests
        if "mcu_renode" in item.keywords and not run_renode:
            item.add_marker(skip_renode)
        # Skip hardware-specific tests
        if "mcu_hardware" in item.keywords and not run_hardware:
            item.add_marker(skip_hardware)
        # Skip @pytest.mark.renode tests if RENODE_PATH not set
        if "renode" in item.keywords and not renode_path:
            item.add_marker(skip_no_renode)


@pytest.fixture(scope='session')
def mcu_test_mode():
    """Get MCU test mode from environment.

    Returns:
        'renode', 'hardware', 'all', or None
    """
    return os.environ.get('EMLEARN_TEST_MCU', '').lower() or None


@pytest.fixture(scope='session')
def mcu_device():
    """Get MCU device from environment.

    Returns:
        Device name string or 'auto' for auto-detection
    """
    return os.environ.get('MCU_DEVICE', 'auto')


@pytest.fixture(scope='session')
def benchmark_app_path():
    """Path to the benchmark Zephyr application."""
    return emlearn_root / 'platform_examples' / 'zephyr' / 'benchmark'


@pytest.fixture(scope='session')
def has_west():
    """Check if west (Zephyr build tool) is available."""
    return shutil.which('west') is not None


@pytest.fixture(scope='session')
def has_jlink():
    """Check if J-Link tools are available."""
    jlink = shutil.which('JLinkExe') or shutil.which('JLink')
    if jlink:
        return True
    # Check common paths
    for path in ['/opt/SEGGER/JLink/JLinkExe', '/usr/bin/JLinkExe']:
        if Path(path).exists():
            return True
    return False


@pytest.fixture(scope='session')
def has_renode():
    """Check if Renode is available."""
    import os
    renode_path = os.environ.get('RENODE_PATH')
    if renode_path:
        return Path(renode_path).exists()
    return shutil.which('renode') is not None


@pytest.fixture
def mcu_runner(mcu_device, has_west, has_jlink):
    """Create MCUTestRunner for the configured device.

    Automatically selects Renode or hardware based on environment.
    """
    if not has_west:
        pytest.skip("west not available")

    try:
        from emlearn.mcu import MCUTestRunner, BOARDS, get_board
    except ImportError:
        pytest.skip("emlearn.mcu module not available")

    mcu_mode = os.environ.get('EMLEARN_TEST_MCU', '').lower()

    if mcu_mode == 'renode' or (mcu_mode == 'all' and mcu_device == 'auto'):
        # Use Renode
        if not os.environ.get('RENODE_PATH'):
            pytest.skip("RENODE_PATH not set")
        board = BOARDS.get('renode_nrf52840')
        if not board:
            pytest.skip("Renode board config not found")
    elif mcu_device == 'auto':
        # Try to detect connected hardware
        from emlearn.mcu import discover_devices
        devices = discover_devices()
        if not devices:
            pytest.skip("No J-Link devices found")
        # Default to nrf52dk if available
        board = BOARDS.get('nrf52dk_nrf52832')
    else:
        try:
            board = get_board(mcu_device)
        except KeyError:
            pytest.skip(f"Unknown board: {mcu_device}")

    runner = MCUTestRunner(board=board, verbose=True)
    yield runner
    runner.cleanup()

# Shared datasets
CLASSIFICATION_DATASETS = {
    'binary': datasets.make_classification(n_classes=2, n_samples=100, random_state=RANDOM_SEED),
    '5way': datasets.make_classification(n_classes=5, n_informative=5, n_samples=100, random_state=RANDOM_SEED),
}

REGRESSION_DATASETS = {
    '1out': datasets.make_regression(n_targets=1, n_samples=100, random_state=RANDOM_SEED),
}

MULTICLASS_DATASETS = {
    '3way': datasets.make_classification(n_classes=3, n_informative=3, n_redundant=0, n_samples=100, random_state=RANDOM_SEED),
}

METHODS = ['loadable', 'inline']


def check_csv_export(cmodel):
    """Check that can be saved to CSV file on correct format."""

    # check that CSV export is valid
    csv = cmodel.save(format='csv', name='mymodelname22')
    loaded = pandas.read_csv(io.StringIO(csv), header=None, engine='python', names=['item', '1', '2', '3', '4'])
    csv_items = set(loaded.item.unique())
    assert csv_items == set(['r', 'n', 'l', 'f', 'c'])

    nodes = loaded[loaded.item == 'n']
    leaves = loaded[loaded.item == 'l']
    roots = loaded[loaded.item == 'r']
    classes = loaded[loaded.item == 'c']
    features = loaded[loaded.item == 'f']

    assert len(nodes) == len(cmodel.forest_[0])
    assert len(roots) == len(cmodel.forest_[1])
    assert len(leaves) == len(cmodel.forest_[2])

    assert len(classes) == 1
    assert classes.iloc[0, 1] == cmodel.n_classes

    assert len(features) == 1
    assert features.iloc[0, 1] == cmodel.n_features


