# bittle-trainer — secondary path (the venv runner is primary on WSL2).
# Needs the NVIDIA Container Toolkit; the JAX cuda wheels bundle CUDA, only the driver is required.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY trainer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY trainer/ ./trainer/
COPY static/assets/ ./static/assets/
COPY hardware_profiles/ ./hardware_profiles/
COPY app/motion/ ./app/motion/
COPY app/__init__.py ./app/__init__.py
ENV PYTHONPATH=/app XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 TRAINER_RUNS_DIR=/runs
EXPOSE 8009
CMD ["python", "-m", "trainer.service"]
