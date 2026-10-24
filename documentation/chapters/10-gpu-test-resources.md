---
part: Test efficiency
status: in-progress
updated: 2026-10-05
---

# GPU tests with foreground headroom

## Draft status

Throughput/cache baselines are measured, but the user reports foreground video
lag during high GPU utilization. The new cooperative pacing draft is not
validated or landed. Its background50% duty is a starting hypothesis, not a
GPU driver cap or proven optimum. Full-speed mode remains asynchronous.
All declared physics inputs, steps and assertions are retained. Compare video
playback stalls/dropped frames in muted headless Chrome before choosing a
background duty. No owned runs/browsers remain at this regrouping checkpoint.

## Measured workload

The terrain GPU lane retains all ten tests, including the 512-environment,
1000-step random-action simulation. Inputs, seeds, precision and assertions
are unchanged. CUDA is required; skipped gates cannot produce a successful run.

The shared workstation has 24 logical CPUs, 32 GiB WSL RAM and one 24 GiB RTX
3090 Ti. Three valid fresh-process trials per worker setting passed all ten
gates with matching case identities and no scope swap. Competing builds and
native games were excluded from timings. Background desktop applications
remained, so device-total VRAM includes them.

| Workers, JAX cache disabled | Median seconds | Median peak RAM GiB | Median CPU seconds | Median device-total peak VRAM MiB |
| --- | --- | --- | --- | --- |
| 1 | 169.655 | 3.239 | 256.186 | 3598 |
| 2 | 116.254 | 4.454 | 318.487 | 3989 |

One process reduces duplicated memory; two processes shorten completion by
31.5%. GPU VRAM is well below the card's capacity in this lane. These results
do not measure desktop frame rate or prove a universal GPU worker count.

A separate prime took 179.325 seconds. Three warm single-worker passes took a
median 131.103 seconds, 3.292 GiB RAM and 175.868 CPU seconds, with 406 cache
hits and nine misses each. Caching reduced elapsed time by 22.7% and CPU work
by 31.4%; RAM increased 1.6%. It improves compilation efficiency. Cached serial
execution uses 26.1% less RAM than the uncached two-worker baseline, taking
12.8% longer. Warm two-worker performance was not measured in three repeats.

## Commands

```bash
# scripts/test_gpu.py starts testrun itself when invoked outside a scope.
testrun --mem 1G -- .venv-trainer/bin/python -m pip install -r trainer/requirements-test.txt
.venv-trainer/bin/python scripts/test_gpu.py

# The test dependencies also enable two-worker fast mode.
.venv-trainer/bin/python scripts/test_gpu.py --profile fast

# Explicit outer-scope equivalents:
testrun --mem 6G --gpu 4G --max-time 15m -- .venv-trainer/bin/python scripts/test_gpu.py
testrun --mem 9G --gpu 4G --max-time 15m -- .venv-trainer/bin/python scripts/test_gpu.py --profile fast
```

Background uses one process and a 6 GiB RAM cap. Fast uses two processes and
9 GiB. These ceilings include growth margin above observed peaks; they do not
allocate that RAM in advance. The runner verifies the outer scope and refuses
insufficient RAM or missing GPU admission. The 4 GiB GPU request is a free-VRAM
admission threshold and holds testrun's GPU test lock; it is not a VRAM cap.

The lane forces CUDA, disables JAX preallocation and sets OpenMP/OpenBLAS/MKL
threads to one. Warp workers use separate private caches. Persistent JAX code
is stored beneath `~/.cache/bittle-gpu-tests`, with a private parent directory
and a 512 MiB cache limit. This reuses compiled kernels; test assertions still
execute on every run. The first run builds entries, and dependency/backend
changes can require rebuilding them. See the
[JAX cache documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

Test dependencies are separate from training/runtime dependencies. The runner
requires filelock for the cache limit and raises cache errors instead of
silently recompiling after a configuration failure.
The bounded cache uses its own `jax-512m` directory; unbounded-cache entries
lack the access-time metadata needed by JAX's bounded cache. A new namespace
starts cold and then reuses valid entries across processes.

Music Player's benchmark documentation holds raw baseline, cache-prime and
warm-trial evidence, exact versions, source hashes and the comparison protocol:
[benchmark sheet](http://127.0.0.1:8910/#/music-player/documentation/chapters/test-resource-benchmark.md).

The shared RAM/swap/CPU limits, training services and simulation remain unchanged.
The GPU test lock does not coordinate the separate music-generation lock or
native Windows applications. Continue checking `testrun status` and coordinate
other GPU owners before heavy work. Tests stop with their scope when finished.
