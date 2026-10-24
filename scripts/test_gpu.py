#!/usr/bin/env python3
"""Run all terrain GPU gates with measured background/fast resource profiles."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def passed_gates(report):
    cases = list(ET.parse(report).iter('testcase'))
    return len(cases) == 10 and all(
        all(case.find(tag) is None for tag in ('failure', 'error', 'skipped'))
        for case in cases
    ) and len({case.get('name') for case in cases}) == 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('background', 'fast'), default='background')
    parser.add_argument('--gpu-duty', type=int, choices=range(10, 101), metavar='10..100',
                        help='Cooperative submission duty; background default 50, fast 100')
    args = parser.parse_args()
    duty = args.gpu_duty if args.gpu_duty is not None else (50 if args.profile == 'background' else 100)
    if importlib.util.find_spec('filelock') is None:
        parser.error('Install trainer/requirements-test.txt in this environment; bounded JAX caching requires filelock')
    mem = 6 if args.profile == 'background' else 9
    runner = Path.home() / '.local/bin/testrun'
    unit = os.getenv('TESTRUN_UNIT')
    if not unit:
        os.execvpe(str(runner), [str(runner), '--mem', f'{mem}G', '--gpu', '4G',
                    '--max-time', '15m', '--wait', '5m', '--', sys.executable,
                    str(Path(__file__).resolve()), '--profile', args.profile, '--gpu-duty', str(duty)], os.environ)
    # Nested testrun ignores new budget flags. Verify the outer scope instead.
    state = Path(os.getenv('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'testrun'
    try:
        registration = json.loads((state / 'runs' / f'{unit}.json').read_text())
        cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
        memory_max = int((cgroup / 'memory.max').read_text())
        if registration['gpu'] < 4 * GIB or memory_max < mem * GIB:
            raise ValueError('insufficient outer RAM/GPU admission')
    except (OSError, KeyError, ValueError) as error:
        parser.error(f'Use testrun --mem {mem}G --gpu 4G for this profile: {error}')
    cache = Path.home() / '.cache/bittle-gpu-tests'
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache.chmod(0o700)
    env = dict(os.environ, TRAINER_GPU_TESTS='1', JAX_PLATFORMS='cuda',
               GPU_TEST_DUTY=str(duty),
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               BITTLE_GPU_CACHE=str(cache), JAX_ENABLE_COMPILATION_CACHE='true',
               JAX_RAISE_PERSISTENT_CACHE_ERRORS='true',
               # Bounded caches need atime metadata absent in unbounded caches.
               JAX_COMPILATION_CACHE_DIR=str(cache / 'jax-512m'),
               JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',
               JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES='-1',
               JAX_COMPILATION_CACHE_MAX_SIZE=str(512 * 1024 ** 2))
    with tempfile.TemporaryDirectory(prefix='bittle-gpu-gates-') as directory:
        report = Path(directory) / 'cases.xml'
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'trainer.tests.gpu_profile_plugin',
                   'trainer/tests/test_gpu_terrain.py', f'--junitxml={report}']
        if args.profile == 'fast':
            command += ['-n', '2']  # Requires trainer/requirements-test.txt.
        code = subprocess.call(command, cwd=REPO, env=env)
        if code:
            return code if code > 0 else 128 - code
        if not report.exists() or not passed_gates(report):
            print('GPU gates incomplete: require all 10 unique tests passed with no skips.', file=sys.stderr)
            return 1
    print(f'GPU gates: 10/10 passed ({args.profile}, submission duty {duty}%).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
