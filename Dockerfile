# ============================================================
# bittle-agent — Docker Build
# ============================================================
# Safety-gated control plane for a Petoi Bittle. Exposes HTTP API on :8008.
#
# Ships sim-only by default: no serial device is mapped in, and
# BITTLE_ALLOW_REAL_HARDWARE defaults to false. See SAFETY.md before arming.
# ============================================================

FROM python:3.11-slim AS deps

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Production runner ─────────────────────────────────────────
FROM python:3.11-slim AS runner
WORKDIR /app

# Non-root user. Note: talking to a real serial device additionally requires the
# container user to be in the host's dialout group (see docker-compose.yml).
RUN groupadd --system --gid 1001 appgrp \
    && useradd --system --uid 1001 --gid appgrp -m -d /home/appusr appusr

COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

COPY app/ ./app/
COPY static/ ./static/
COPY mcp/ ./mcp/
COPY hardware_profiles/ ./hardware_profiles/
COPY upstream.lock ./upstream.lock
# The generated docs page + media, served at /documentation (index.html is committed).
COPY documentation/ ./documentation/

RUN chown -R appusr:appgrp /app

ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    HOST="0.0.0.0" \
    PORT="8008"

USER appusr

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD wget --no-verbose --tries=1 -O /dev/null http://localhost:8008/health || exit 1

# Single worker, deliberately. The serial port is a single exclusive resource
# guarded by an in-process lock; a second worker would open its own connection
# and the two would interleave commands on one robot.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008", "--workers", "1"]
