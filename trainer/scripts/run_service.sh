#!/bin/bash
# Start the trainer service on the GPU box. Shared box: 8 cores, nice 10, half the VRAM.
set -euo pipefail
cd "$(dirname "$0")/../.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export TRAINER_CPU_CORES="${TRAINER_CPU_CORES:-0-7}"
export TRAINER_RUNS_DIR="${TRAINER_RUNS_DIR:-$PWD/runs}"
export TRAINER_PORT="${TRAINER_PORT:-8009}"
exec nice -n 10 .venv-trainer/bin/python -m trainer.service
