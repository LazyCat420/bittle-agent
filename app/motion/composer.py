"""Composition & Iteration Engine for GLM motion authoring.

Provides high-level functions for GLM tool calls:
- compose: snap primitives together into a moveset
- iterate: tweak an existing moveset's parameters
- evaluate: dry-run validate and score a moveset
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
from pathlib import Path
from typing import Any

from .builtin_library import (
    BUILTIN_PRIMITIVES,
    get_builtin_primitive,
    list_builtin_primitives,
    list_builtin_primitives_by_group,
)
from .primitives import (
    JOINT_GROUPS,
    CompositionError,
    compose,
    evaluate_frames,
)

logger = logging.getLogger("bittle-agent.composer")


class MovesetComposer:
    """GLM-facing composition and iteration layer."""

    def __init__(self, storage_dir: Path | None = None):
        if storage_dir is None:
            storage_dir = Path(__file__).resolve().parent.parent.parent / "storage"
        self.movesets_dir = storage_dir / "movesets"
        self.primitives_dir = storage_dir / "primitives"
        self.movesets_dir.mkdir(parents=True, exist_ok=True)
        self.primitives_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self._primitives: dict[str, dict[str, Any]] = dict(BUILTIN_PRIMITIVES)
        self._movesets: dict[str, dict[str, Any]] = {}
        self._load_user_primitives()
        self._load_user_movesets()

    def _load_user_primitives(self) -> None:
        """Load user-created primitives from disk."""
        for p in self.primitives_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                self._primitives[data["name"]] = data
            except Exception:
                logger.warning("Failed to load primitive %s", p)

    def _load_user_movesets(self) -> None:
        """Load user-created movesets from disk."""
        for p in self.movesets_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                self._movesets[data.get("name", p.stem)] = data
            except Exception:
                logger.warning("Failed to load moveset %s", p)

    # ── Primitive CRUD ──────────────────────────────────────────────────

    def get_primitive(self, name: str) -> dict[str, Any] | None:
        return self._primitives.get(name)

    def list_primitives(self, group: str | None = None) -> list[dict[str, Any]]:
        prims = list(self._primitives.values())
        if group:
            prims = [p for p in prims if p.get("group") == group]
        return prims

    def save_primitive(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        data["name"] = name
        data.setdefault("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
        self._primitives[name] = data
        try:
            path = self.primitives_dir / f"{name}.json"
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            logger.warning("Failed to save primitive %s to disk", name)
        return data

    def delete_primitive(self, name: str) -> bool:
        if name in BUILTIN_PRIMITIVES:
            raise CompositionError(f"Cannot delete built-in primitive {name!r}")
        existed = name in self._primitives
        self._primitives.pop(name, None)
        path = self.primitives_dir / f"{name}.json"
        if path.is_file():
            path.unlink()
            existed = True
        return existed

    # ── Moveset CRUD ────────────────────────────────────────────────────

    def get_moveset(self, name: str) -> dict[str, Any] | None:
        return self._movesets.get(name)

    def list_movesets(self) -> list[dict[str, Any]]:
        return list(self._movesets.values())

    def save_moveset(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        data["name"] = name
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        data.setdefault("created_at", now)
        data["updated_at"] = now

        # Version history: append current version
        history = data.get("version_history", [])
        history.append({
            "timestamp": now,
            "frame_count": len(data.get("frames", [])),
        })
        data["version_history"] = history[-20:]  # Keep last 20 versions

        self._movesets[name] = data
        try:
            path = self.movesets_dir / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            logger.warning("Failed to save moveset %s to disk", name)
        return data

    def delete_moveset(self, name: str) -> bool:
        existed = name in self._movesets
        self._movesets.pop(name, None)
        path = self.movesets_dir / f"{name}.json"
        if path.is_file():
            path.unlink()
            existed = True
        return existed

    # ── Composition ─────────────────────────────────────────────────────

    def compose_move(
        self,
        name: str,
        primitive_names: list[str],
        mode: str = "sequential",
        *,
        transition_blend_ms: int = 100,
        blend_frames: int = 2,
        description: str = "",
    ) -> dict[str, Any]:
        """Compose a new moveset from named primitives.

        Args:
            name: Identifier for the resulting moveset
            primitive_names: Ordered list of primitive names to compose
            mode: 'sequential', 'parallel', or 'blend'
            transition_blend_ms: ms for sequential transition smoothing
            blend_frames: number of interpolation frames for blend mode
            description: human-readable description

        Returns:
            Complete moveset dict with frames, metadata, and evaluation metrics

        Raises:
            CompositionError: if primitives not found or composition fails
        """
        # Resolve all primitives
        primitives = []
        for pname in primitive_names:
            prim = self.get_primitive(pname)
            if prim is None:
                raise CompositionError(
                    f"Primitive {pname!r} not found. Available: {sorted(self._primitives.keys())}"
                )
            primitives.append(prim)

        # Compose frames
        frames = compose(
            primitives,
            mode=mode,
            transition_blend_ms=transition_blend_ms,
            blend_frames=blend_frames,
        )

        # Evaluate quality
        metrics = evaluate_frames(frames)

        moveset = {
            "name": name,
            "description": description or f"Composed from: {', '.join(primitive_names)}",
            "source_primitives": primitive_names,
            "composition_mode": mode,
            "frames": frames,
            "metrics": metrics,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        return moveset

    # ── Iteration ───────────────────────────────────────────────────────

    def iterate_move(
        self,
        name: str,
        adjustments: dict[str, Any],
    ) -> dict[str, Any]:
        """Tweak an existing moveset's parameters.

        Args:
            name: Name of the moveset to iterate on
            adjustments: Dict with optional keys:
                - delay_scale: float — multiply all delays by this factor
                - speed_scale: float — multiply all speeds by this factor
                - angle_offsets: {joint_idx: offset_deg} — add offsets to specific joints
                - insert_frame: {index: int, frame: dict} — insert a frame at position
                - remove_frame: int — remove frame at position
                - swap_primitive: {old: str, new: str} — swap a source primitive and recompose

        Returns:
            Updated moveset dict with new frames and metrics

        Raises:
            CompositionError: if moveset not found or adjustment invalid
        """
        moveset = self.get_moveset(name)
        if moveset is None:
            raise CompositionError(f"Moveset {name!r} not found in library")

        moveset = copy.deepcopy(moveset)
        frames = moveset.get("frames", [])

        # Handle swap_primitive: recompose from source primitives
        if "swap_primitive" in adjustments:
            swap = adjustments["swap_primitive"]
            old_name = swap.get("old", "")
            new_name = swap.get("new", "")
            source = moveset.get("source_primitives", [])
            if old_name not in source:
                raise CompositionError(
                    f"Primitive {old_name!r} not in source list: {source}"
                )
            new_source = [new_name if s == old_name else s for s in source]
            mode = moveset.get("composition_mode", "sequential")
            return self.compose_move(name, new_source, mode=mode, description=moveset.get("description", ""))

        # Scale delays
        if "delay_scale" in adjustments:
            scale = float(adjustments["delay_scale"])
            for frame in frames:
                frame["delay_ms"] = max(50, int(frame.get("delay_ms", 150) * scale))

        # Scale speeds
        if "speed_scale" in adjustments:
            scale = float(adjustments["speed_scale"])
            for frame in frames:
                raw = int(frame.get("speed_deg_per_step", 8) * scale)
                frame["speed_deg_per_step"] = max(1, min(30, raw))

        # Apply angle offsets
        if "angle_offsets" in adjustments:
            offsets = adjustments["angle_offsets"]
            for frame in frames:
                angles = frame.get("angles", {})
                for j_str, offset in offsets.items():
                    j = int(j_str)
                    if j in angles:
                        angles[j] = int(angles[j] + offset)

        # Insert frame
        if "insert_frame" in adjustments:
            ins = adjustments["insert_frame"]
            idx = int(ins.get("index", len(frames)))
            new_frame = ins.get("frame", {})
            new_frame.setdefault("speed_deg_per_step", 8)
            new_frame.setdefault("delay_ms", 150)
            if "angles" in new_frame:
                new_frame["angles"] = {int(k): v for k, v in new_frame["angles"].items()}
            frames.insert(min(idx, len(frames)), new_frame)

        # Remove frame
        if "remove_frame" in adjustments:
            rm_idx = int(adjustments["remove_frame"])
            if 0 <= rm_idx < len(frames):
                frames.pop(rm_idx)

        moveset["frames"] = frames
        moveset["metrics"] = evaluate_frames(frames)
        moveset["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return moveset

    # ── Evaluation ──────────────────────────────────────────────────────

    def evaluate_move(
        self,
        name: str | None = None,
        frames: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Dry-run evaluate a moveset or raw frames without executing.

        Provide either a moveset name (looks up from library) or raw frames.
        """
        if frames is None and name:
            moveset = self.get_moveset(name)
            if moveset is None:
                raise CompositionError(f"Moveset {name!r} not found")
            frames = moveset.get("frames", [])
        if frames is None:
            raise CompositionError("Either name or frames must be provided")

        return evaluate_frames(frames)


# ── Singleton ───────────────────────────────────────────────────────────

_GLOBAL_COMPOSER: MovesetComposer | None = None


def get_composer() -> MovesetComposer:
    global _GLOBAL_COMPOSER
    if _GLOBAL_COMPOSER is None:
        _GLOBAL_COMPOSER = MovesetComposer()
    return _GLOBAL_COMPOSER
