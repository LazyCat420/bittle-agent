"""Keep simultaneous Warp workers in separate private kernel caches."""
import os
from pathlib import Path


def pytest_configure(config):
    worker = os.getenv('PYTEST_XDIST_WORKER', 'gw0')
    cache = Path(os.environ['BITTLE_GPU_CACHE']) / 'warp' / worker
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache.chmod(0o700)
    os.environ['WARP_CACHE_PATH'] = str(cache)
