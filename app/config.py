"""Configuration.

Defaults are the safe ones. Reaching real hardware requires deliberately
flipping two independent switches (env flag + per-request token), so no single
misconfiguration or prompt injection is enough to move a real servo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8008"))

    #: Master switch. False => the serial backend is never even constructed.
    allow_real_hardware: bool = _flag("BITTLE_ALLOW_REAL_HARDWARE", False)

    #: Second switch: requests targeting real hardware must present this token.
    #: Empty + allow_real_hardware=true means "no token required", which we warn
    #: about at startup rather than silently accepting.
    confirm_token: str = os.getenv("BITTLE_CONFIRM_TOKEN", "")

    serial_port: str = os.getenv("BITTLE_SERIAL_PORT", "/dev/ttyUSB0")
    serial_baud: int = int(os.getenv("BITTLE_SERIAL_BAUD", "115200"))

    #: Gaits carry the robot across a surface; can be disabled entirely.
    allow_locomotion: bool = _flag("BITTLE_ALLOW_LOCOMOTION", True)

    rate_per_sec: float = float(os.getenv("BITTLE_RATE_PER_SEC", "8"))
    rate_burst: int = int(os.getenv("BITTLE_RATE_BURST", "16"))

    #: Local GLM model configuration (strictly local / internal GoldSpark cluster endpoint)
    llm_api_base: str = os.getenv("BITTLE_LLM_API_BASE", "http://10.0.0.141:8000/v1")
    llm_model: str = os.getenv("BITTLE_LLM_MODEL", "GLM-5.3-Flash-EXL3")
    llm_api_key: str = os.getenv("BITTLE_LLM_API_KEY", "EMPTY")
    llm_timeout: float = float(os.getenv("BITTLE_LLM_TIMEOUT", "120.0"))

    @property
    def requires_confirm_token(self) -> bool:
        return bool(self.confirm_token)


settings = Settings()
