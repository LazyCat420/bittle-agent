"""Verify the real training entry point waits before executing even argparse."""
import fcntl
from pathlib import Path
import subprocess
import sys


def test_training_entry_waits_for_shared_gpu_lock(tmp_path):
    lock_path = tmp_path / 'gpu.lock'
    probe = tmp_path / 'entry.py'
    probe.write_text('''
import builtins
import runpy
import sys
original_open = builtins.open
lock_path = sys.argv[1]
def isolated_open(path, *args, **kwargs):
    if path == "/tmp/sun-rtx3090ti.gpu.lock":
        path = lock_path
        print("attempting GPU lock", flush=True)
    return original_open(path, *args, **kwargs)
builtins.open = isolated_open
sys.argv = ["trainer.train.run", "--help"]
try:
    runpy.run_module("trainer.train.run", run_name="__main__")
finally:
    assert "torch" not in sys.modules and "jax" not in sys.modules
''')
    import os
    repo = Path(__file__).resolve().parents[2]
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        child = subprocess.Popen([sys.executable, str(probe), str(lock_path)], cwd=repo,
                                 env=dict(os.environ, PYTHONPATH=str(repo)),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            import select
            readable, _, _ = select.select([child.stdout], [], [], 10)
            assert readable, 'entry did not attempt GPU lock'
            assert child.stdout.readline().strip() == 'attempting GPU lock'
            try:
                child.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            else:
                raise AssertionError('training bypassed the held GPU lock')
            fcntl.flock(lock, fcntl.LOCK_UN)
            stdout, stderr = child.communicate(timeout=10)
            assert child.returncode == 0, stderr
            assert '--run-id' in stdout
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
