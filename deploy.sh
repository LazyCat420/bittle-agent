#!/bin/bash
# ============================================================
# Bittle Agent — Build & Deploy to Synology NAS
#
# Thin wrapper — all logic lives in ../deploy-kit/lib.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="bittle-agent"
DISPLAY_NAME="🐕 Bittle Agent"

EXTRA_VALIDATE() {
  # The safety layer is the whole point of this service, so a deploy that
  # can't prove the clamps still work should not reach the robot.
  if command -v python3 >/dev/null 2>&1 && [ -d "${SCRIPT_DIR}/.venv" ]; then
    step "Running safety tests"
    if "${SCRIPT_DIR}/.venv/bin/python" -m pytest "${SCRIPT_DIR}/tests" -q > /tmp/bittle-tests.log 2>&1; then
      ok "safety tests passed"
    else
      tail -20 /tmp/bittle-tests.log
      fail "safety tests FAILED — refusing to deploy robot control code"
    fi
  else
    warn "no .venv found; skipping safety tests (run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt)"
  fi
}

EXTRA_SSH_SYNC() {
  # Ship .env.example so the remote has the documented knobs, but never
  # overwrite an existing .env — that's where the hardware arming lives.
  info "Syncing .env.example..."
  cat "${SCRIPT_DIR}/.env.example" | ssh "$DEPLOY_SSH_HOST" "cat > '${DEPLOY_COMPOSE_DIR}/.env.example'"
  ssh "$DEPLOY_SSH_HOST" "[ -f '${DEPLOY_COMPOSE_DIR}/.env' ] || cp '${DEPLOY_COMPOSE_DIR}/.env.example' '${DEPLOY_COMPOSE_DIR}/.env'"
  ok ".env.example synced"
}

source "${SCRIPT_DIR}/../deploy-kit/lib.sh"
