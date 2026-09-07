import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs the GPU (Warp/JAX); slow")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


def pytest_collection_modifyitems(config, items):
    if os.getenv("TRAINER_GPU_TESTS", "") in ("1", "true"):
        return
    skip = pytest.mark.skip(reason="set TRAINER_GPU_TESTS=1 to run GPU tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
