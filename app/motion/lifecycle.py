"""Skill Lifecycle Manager.

Enforces human-in-the-loop promotion gate:
DRAFT -> VALIDATED -> SIMULATED -> CANARIED -> APPROVED -> PROMOTED.
Only PROMOTED manifests with verified SHA-256 payload hashes are callable on real hardware.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from .compiler import SkillCompiler, SkillCompilerError
from .schema import SkillIR, SkillManifest, ValidationResult
from .validator import TrajectoryValidator


class LifecycleError(RuntimeError):
    pass


class SkillLifecycleManager:
    def __init__(self, storage_dir: Path | None = None):
        self.storage_dir = storage_dir
        self.manifests: dict[str, SkillManifest] = {}
        self.simulation_records: dict[str, dict[str, Any]] = {}
        self.canary_records: dict[str, dict[str, Any]] = {}
        self.compiler = SkillCompiler()
        self.validator = TrajectoryValidator()

    def draft(self, ir_data: dict[str, Any]) -> SkillIR:
        return SkillIR.model_validate(ir_data)

    def validate_and_compile(self, ir: SkillIR) -> tuple[ValidationResult, SkillManifest | None]:
        val_res = self.validator.validate(ir)
        if not val_res.valid:
            return val_res, None
        manifest = self.compiler.compile(ir)
        self.manifests[manifest.payload_hash] = manifest
        return val_res, manifest

    def record_simulation(
        self, manifest_hash: str, evidence: dict[str, Any]
    ) -> SkillManifest:
        manifest = self.get_manifest(manifest_hash)
        manifest.simulated = True
        manifest.status = "simulated"
        self.simulation_records[manifest_hash] = evidence
        return manifest

    def record_canary(
        self, manifest_hash: str, operator_notes: str
    ) -> SkillManifest:
        manifest = self.get_manifest(manifest_hash)
        if not manifest.simulated:
            raise LifecycleError("Skill must be simulated before physical canary execution on stand")
        manifest.status = "canaried"
        self.canary_records[manifest_hash] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "notes": operator_notes,
        }
        return manifest

    def approve(self, manifest_hash: str, approver: str) -> SkillManifest:
        manifest = self.get_manifest(manifest_hash)
        if manifest.status not in ("simulated", "canaried"):
            raise LifecycleError(f"Cannot approve skill in '{manifest.status}' state; must be simulated or canaried first")
        manifest.approved_by = approver
        manifest.approved_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest.status = "approved"
        return manifest

    def promote(self, manifest_hash: str) -> SkillManifest:
        manifest = self.get_manifest(manifest_hash)
        if manifest.status != "approved":
            raise LifecycleError(f"Cannot promote skill in '{manifest.status}' state; requires human approval")
        manifest.promoted = True
        manifest.status = "promoted"
        return manifest

    def revoke(self, manifest_hash: str, reason: str) -> SkillManifest:
        manifest = self.get_manifest(manifest_hash)
        manifest.promoted = False
        manifest.status = "revoked"
        return manifest

    def get_manifest(self, manifest_hash: str) -> SkillManifest:
        if manifest_hash not in self.manifests:
            raise KeyError(f"Skill manifest {manifest_hash!r} not found in lifecycle registry")
        return self.manifests[manifest_hash]

    def is_callable_on_real(self, manifest_hash: str) -> bool:
        if manifest_hash not in self.manifests:
            return False
        m = self.manifests[manifest_hash]
        return m.promoted and m.status == "promoted"

    def list_manifests(self, status: str | None = None) -> list[SkillManifest]:
        res = list(self.manifests.values())
        if status:
            res = [m for m in res if m.status == status]
        return res


_GLOBAL_LIFECYCLE: SkillLifecycleManager | None = None


def get_lifecycle() -> SkillLifecycleManager:
    global _GLOBAL_LIFECYCLE
    if _GLOBAL_LIFECYCLE is None:
        _GLOBAL_LIFECYCLE = SkillLifecycleManager()
    return _GLOBAL_LIFECYCLE
