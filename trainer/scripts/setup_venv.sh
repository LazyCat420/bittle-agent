#!/bin/bash
# Create the trainer venv on the GPU box (WSL2 / Linux, Python 3.12, NVIDIA driver present).
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m venv .venv-trainer
.venv-trainer/bin/pip install --upgrade pip
.venv-trainer/bin/pip install -r trainer/requirements.txt
.venv-trainer/bin/python -c "import jax; print('jax devices:', jax.devices())"
echo "trainer venv ready: .venv-trainer"
