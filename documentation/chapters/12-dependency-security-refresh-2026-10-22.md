---
part: Agent tools
status: shipped
updated: 2026-10-22
---

# Dependency security refresh 2026-10-22

The Python pins in `requirements.txt` (the sim-only control plane, `python:3.11-slim` image) and `trainer/requirements.txt` (the trainer, `python:3.12-slim`) were re-audited with `pip-audit` on the **fully resolved** dependency set. An earlier `--no-deps` pass reported "clean" only because it skipped transitive packages such as `starlette`.

## What changed

Control plane (`requirements.txt`):

| Package | From | To | Why |
|---|---|---|---|
| fastapi | 0.115.6 | 0.161.0 | 0.115.6 required `starlette<0.42`, which held starlette 0.41.3 (14 advisories) |
| starlette (new explicit pin) | 0.41.3 (transitive) | 1.9.0 | fix is >= 1.3.1 |
| pydantic | 2.10.4 | 2.14.1 | matches the new fastapi |
| uvicorn[standard] | 0.34.0 | 0.54.0 | newest stable |
| certifi (new explicit pin) | 2026.7.22 | 2026.10.19 | PYSEC-2026-4320 |
| setuptools (new explicit pin) | 79.0.1 (bundled by the 3.11 `venv`) | 84.2.1 | PYSEC-2026-3447 fixed in 83.0.0; it ships in the image's `/opt/venv` |
| h11, idna (new explicit pins) | 0.16.0, 3.20 | unchanged | pinned for reproducibility |

Trainer (`trainer/requirements.txt`), new explicit pins only: `pillow==13.0.0` (was 12.3.0 via brax/matplotlib/mediapy: 35 advisories), `werkzeug==3.1.9` (was 3.1.8 via Flask: PYSEC-2026-4312), `certifi==2026.10.19`. fastapi 0.141.1 / starlette 1.6.0 / pydantic 2.13.5 were already clean and stay.

## Audit before and after

- Control plane, before (venv): 27 advisories in 3 packages (starlette 14, pip 12, certifi 1). After, a fresh Python 3.11 environment installed from the new file: **No known vulnerabilities found**.
- Trainer, before (`.venv-trainer`): 37 advisories in 3 packages (pillow 35, certifi 1, werkzeug 1). After, the resolved 109-package set from `pip install --dry-run --report`: **No known vulnerabilities found**. `.venv-trainer` itself was not rebuilt.

pip 24.0 in the local `.venv` is not shipped in the image and is outside the pins.
