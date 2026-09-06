"""FastAPI surface for bittle-agent."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import joints as joints_mod
from . import skills as skills_mod
from .agent import GLMAgentHarness
from .config import settings
from .controller import Controller, TargetUnavailable
from .motion import CompositionError, get_composer
from .safety import EStopEngaged, SafetyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("bittle-agent")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

controller = Controller(settings)
agent_harness = GLMAgentHarness(controller, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await controller.startup()
    if settings.allow_real_hardware and not settings.requires_confirm_token:
        logger.warning(
            "REAL HARDWARE IS ENABLED WITH NO CONFIRM TOKEN. "
            "Set BITTLE_CONFIRM_TOKEN to require per-request confirmation."
        )
    yield
    await controller.shutdown()


app = FastAPI(
    title="bittle-agent",
    description="Safety-gated control plane for a Petoi Bittle. The LLM never touches serial.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Error handling ────────────────────────────────────────────────────────
# Safety refusals are surfaced as structured, machine-readable errors so an LLM
# caller can tell "you asked for something unsafe" apart from "the service broke"
# and correct itself rather than blindly retrying.


@app.exception_handler(SafetyError)
async def _safety_handler(_request, exc: SafetyError):
    status = 409 if isinstance(exc, EStopEngaged) else 400
    if exc.reason == "rate_limited":
        status = 429
    return JSONResponse(
        status_code=status,
        content={"error": "safety_refused", "reason": exc.reason, "detail": exc.detail},
    )


@app.exception_handler(TargetUnavailable)
async def _target_handler(_request, exc: TargetUnavailable):
    return JSONResponse(
        status_code=403,
        content={"error": "target_unavailable", "detail": str(exc)},
    )


# ── Models ────────────────────────────────────────────────────────────────


class MoveRequest(BaseModel):
    angles: dict[str, float] = Field(
        ...,
        description="Joint index or name -> angle in degrees, e.g. {\"8\": -40, \"head\": 20}",
    )
    target: str = Field("sim", description="'sim' (default) or 'real'")
    simultaneous: bool = Field(True, description="Move joints together (safer) vs sequentially")
    confirm: str | None = Field(None, description="Confirm token, required for target='real'")


class SkillRequest(BaseModel):
    skill: str = Field(..., description="Skill name, e.g. 'sit' or 'ksit'")
    target: str = "sim"
    ack_locomotion: bool = Field(
        False, description="Required for gaits that drive the robot across the floor"
    )
    confirm: str | None = None


class EStopRequest(BaseModel):
    reason: str = "manual"


class ControlledStopRequest(BaseModel):
    reason: str = "controlled_stop"


class SelectProfileRequest(BaseModel):
    profile_id: str


class SimulateSkillRequest(BaseModel):
    manifest_hash: str


class CanarySkillRequest(BaseModel):
    manifest_hash: str
    operator_notes: str = "operator_tested_on_stand"


class ApproveSkillRequest(BaseModel):
    manifest_hash: str
    approver: str = "operator"


class PromoteSkillRequest(BaseModel):
    manifest_hash: str


class RunApprovedSkillRequest(BaseModel):
    manifest_hash: str
    target: str = "sim"
    confirm: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "bittle-agent"}


@app.get("/api/status")
async def status(target: str = Query("sim")):
    return await controller.status(target)


@app.get("/api/joints")
async def list_joints():
    return {
        "dof": joints_mod.DOF,
        "walking_dof": joints_mod.WALKING_DOF,
        "wire_range": [joints_mod.WIRE_MIN, joints_mod.WIRE_MAX],
        "joints": controller.describe_joints(),
    }


@app.get("/api/joints/state")
async def joint_state(target: str = Query("sim"), confirm: str | None = Query(None)):
    return {"target": target, "angles": await controller.joint_state(target, confirm)}


@app.get("/api/skills")
async def list_skills():
    return {
        "skills": [
            {
                "name": s.name,
                "token": s.token,
                "label": s.label,
                "kind": s.kind,
                "locomotes": s.locomotes,
            }
            for s in skills_mod.ALL
        ]
    }


@app.post("/api/move")
async def move(req: MoveRequest):
    result, validated = await controller.move(
        dict(req.angles),
        target=req.target,
        simultaneous=req.simultaneous,
        confirm=req.confirm,
    )
    return {
        "ok": result.ok,
        "target": req.target,
        "sent": result.sent,
        "response": result.response,
        "detail": result.detail,
        "clamped": validated.clamped,
        "adjustments": [a.__dict__ for a in validated.adjustments],
        "applied": {str(i): a for i, a in validated.pairs},
        "meta": result.meta,
    }


@app.post("/api/skill")
async def skill(req: SkillRequest):
    result, resolved = await controller.skill(
        req.skill,
        target=req.target,
        ack_locomotion=req.ack_locomotion,
        confirm=req.confirm,
    )
    return {
        "ok": result.ok,
        "target": req.target,
        "skill": resolved.name,
        "token": resolved.token,
        "sent": result.sent,
        "response": result.response,
        "detail": result.detail,
        "meta": result.meta,
    }


@app.post("/api/estop")
async def estop(req: EStopRequest = Body(default=EStopRequest())):
    """Trip the latching E-stop and relax servos on every backend.

    Never rate-limited and never gated on a confirm token: an emergency stop
    that can be refused is not one. Note `d` releases torque, so a standing
    robot will sit down abruptly -- intended, and documented in SAFETY.md.
    """
    return await controller.estop(req.reason)


@app.post("/api/estop/clear")
async def clear_estop():
    return controller.clear_estop()


@app.post("/api/stop/controlled")
async def controlled_stop(req: ControlledStopRequest = Body(default_factory=ControlledStopRequest)):
    return await controller.controlled_stop(req.reason)


@app.get("/api/profiles")
async def list_profiles():
    from .profiles import get_registry
    reg = get_registry()
    return {
        "active_profile": controller.profile.profile_id,
        "profiles": reg.list_profiles(),
    }


@app.post("/api/profiles/select")
async def select_profile(req: SelectProfileRequest):
    try:
        prof = controller.set_profile(req.profile_id)
        return {"ok": True, "active_profile": prof.profile_id, "installed_joints": list(prof.installed_joints)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/skills/draft")
async def draft_skill(skill: dict[str, Any] = Body(...)):
    from .motion import get_lifecycle
    lifecycle = get_lifecycle()
    payload = dict(skill.get("skill", skill))
    if "profile_id" not in payload:
        payload["profile_id"] = controller.profile.profile_id
    ir = lifecycle.draft(payload)
    return {"ok": True, "skill_ir": ir.model_dump(mode="json")}


@app.post("/api/skills/validate")
async def validate_skill(skill: dict[str, Any] = Body(...)):
    from .motion import get_lifecycle
    from .motion.schema import SkillIR
    lifecycle = get_lifecycle()
    payload = dict(skill.get("skill", skill))
    if "profile_id" not in payload:
        payload["profile_id"] = controller.profile.profile_id
    ir = SkillIR.model_validate(payload)
    val_res, manifest = lifecycle.validate_and_compile(ir)
    return {
        "ok": val_res.valid,
        "errors": val_res.errors,
        "warnings": val_res.warnings,
        "budget": val_res.budget.model_dump() if val_res.budget else None,
        "manifest_hash": manifest.payload_hash if manifest else None,
        "manifest": manifest.model_dump(mode="json") if manifest else None,
    }


@app.post("/api/skills/simulate")
async def simulate_skill(req: SimulateSkillRequest):
    from .motion import get_lifecycle
    lifecycle = get_lifecycle()
    try:
        manifest = lifecycle.get_manifest(req.manifest_hash)
        evidence = {"simulated": True, "frames": len(manifest.ir.frames)}
        m = lifecycle.record_simulation(req.manifest_hash, evidence)
        return {"ok": True, "manifest": m.model_dump(mode="json")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/skills/canary")
async def canary_skill(req: CanarySkillRequest):
    from .motion import get_lifecycle
    lifecycle = get_lifecycle()
    try:
        m = lifecycle.record_canary(req.manifest_hash, req.operator_notes)
        return {"ok": True, "manifest": m.model_dump(mode="json")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/skills/manifests")
async def list_manifests(status: str | None = Query(None)):
    from .motion import get_lifecycle
    lifecycle = get_lifecycle()
    return {"manifests": [m.model_dump(mode="json") for m in lifecycle.list_manifests(status)]}


@app.post("/api/skills/approve")
async def approve_skill(req: ApproveSkillRequest):
    from .motion import get_lifecycle
    lifecycle = get_lifecycle()
    try:
        m = lifecycle.approve(req.manifest_hash, req.approver)
        return {"ok": True, "manifest": m.model_dump(mode="json")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/skills/promote")
async def promote_skill(req: PromoteSkillRequest):
    from .motion import get_lifecycle
    lifecycle = get_lifecycle()
    try:
        m = lifecycle.promote(req.manifest_hash)
        return {"ok": True, "manifest": m.model_dump(mode="json")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/skills/run-approved")
async def run_approved_skill(req: RunApprovedSkillRequest):
    result, manifest = await controller.run_approved_skill(req.manifest_hash, target=req.target, confirm=req.confirm)
    return {
        "ok": result.ok,
        "sent": result.sent,
        "response": result.response,
        "manifest_hash": req.manifest_hash,
        "skill_name": manifest.ir.name,
    }


@app.post("/api/preview")
async def preview(
    angles: dict[str, int] = Body(...), simultaneous: bool = Body(True)
):
    """Show the exact wire bytes a move would produce, without sending anything.

    Useful for a model or operator to inspect a command before committing to it.
    """
    try:
        resolved = {joints_mod.resolve(k).index: v for k, v in angles.items()}
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"wire": controller.preview(resolved, simultaneous)}


# ── Moveset & Animation Library Endpoints ─────────────────────────────────


@app.get("/api/movesets")
async def list_movesets():
    from .motion.builtin_library import list_builtin_movesets
    from .motion import get_lifecycle, get_composer
    composer = get_composer()
    lifecycle = get_lifecycle()
    builtins = list_builtin_movesets()
    customs = composer.list_movesets()
    seen = {c.get("name") for c in customs if isinstance(c, dict) and "name" in c}
    for lm in lifecycle.list_movesets():
        if isinstance(lm, dict) and "name" in lm and lm["name"] not in seen:
            customs.append(lm)
            seen.add(lm["name"])
    for c in customs:
        c.setdefault("kind", "custom")
    return {
        "builtins": builtins,
        "customs": customs,
        "movesets": builtins + customs,
    }


class SaveMovesetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str = ""
    frames: list[dict[str, Any]] = Field(..., min_length=1)


@app.post("/api/movesets/save")
async def save_moveset_endpoint(req: SaveMovesetRequest):
    from .motion import get_composer
    composer = get_composer()
    saved = composer.save_moveset(req.name, {
        "description": req.description,
        "frames": req.frames,
        "kind": "custom",
    })
    return {"ok": True, "name": req.name, "moveset": saved}


@app.delete("/api/movesets/{name}")
async def delete_moveset_endpoint(name: str):
    from .motion import get_composer, get_lifecycle
    composer = get_composer()
    lifecycle = get_lifecycle()
    d1 = composer.delete_moveset(name)
    d2 = lifecycle.delete_moveset(name)
    return {"ok": True, "deleted": d1 or d2}


class PlayMovesetRequest(BaseModel):
    name: str | None = None
    frames: list[dict[str, Any]] | None = None
    target: str = "sim"
    confirm: str | None = None


@app.post("/api/movesets/play")
async def play_moveset_endpoint(req: PlayMovesetRequest):
    from .motion.builtin_library import get_builtin_moveset
    from .motion import get_composer, get_lifecycle
    frames = req.frames
    if not frames and req.name:
        composer = get_composer()
        lifecycle = get_lifecycle()
        m = composer.get_moveset(req.name) or lifecycle.get_moveset(req.name) or get_builtin_moveset(req.name)
        if m:
            frames = m.get("frames", [])

    if not frames:
        raise HTTPException(status_code=404, detail="Moveset frames not found")

    steps = []
    for f in frames:
        steps.append({
            "type": "move",
            "angles": f.get("angles", {}),
            "delay_ms": f.get("delay_ms", 150),
            "speed_deg_per_step": f.get("speed_deg_per_step", 8),
        })

    res = await agent_harness.execute_tool("bittle_execute_sequence", {
        "name": req.name or "custom_playback",
        "steps": steps,
        "target": req.target,
    }, target_override=req.target, confirm_token=req.confirm)
    return res


# ── Obstacle Course & Terrain Endpoints ────────────────────────────────────


@app.get("/api/obstacles/presets")
async def get_obstacle_presets():
    from .motion.obstacles import COURSE_PRESETS
    return {"presets": COURSE_PRESETS}


@app.get("/api/obstacles/layout/{preset}")
async def get_obstacle_layout_endpoint(preset: str):
    from .motion.obstacles import get_course_layout
    layout = get_course_layout(preset)
    return {"preset": preset, "layout": layout}


class EvaluateObstacleRequest(BaseModel):
    course_preset: str
    angles: dict[str, float] | None = None
    sequence: list[dict[str, Any]] | None = None
    moveset_name: str | None = None


@app.post("/api/obstacles/evaluate")
async def evaluate_obstacle_endpoint(req: EvaluateObstacleRequest):
    from .motion.builtin_library import get_builtin_moveset
    from .motion.obstacles import evaluate_terrain_clearance
    from .motion import get_composer, get_lifecycle
    data = None
    if req.sequence:
        data = req.sequence
    elif req.angles:
        data = req.angles
    elif req.moveset_name:
        composer = get_composer()
        lifecycle = get_lifecycle()
        m = composer.get_moveset(req.moveset_name) or lifecycle.get_moveset(req.moveset_name) or get_builtin_moveset(req.moveset_name)
        if m:
            data = m.get("frames", [])
        else:
            raise HTTPException(status_code=404, detail=f"Moveset '{req.moveset_name}' not found in library")
    else:
        raise HTTPException(status_code=400, detail="Must specify 'sequence', 'angles', or 'moveset_name'")

    return evaluate_terrain_clearance(req.course_preset, data)


class ExecuteSequenceRequest(BaseModel):
    name: str = "custom_sequence"
    steps: list[dict[str, Any]]
    target: str = "sim"
    confirm: str | None = None


@app.post("/api/sequence")
async def execute_sequence_endpoint(req: ExecuteSequenceRequest):
    res = await agent_harness.execute_tool(
        "bittle_execute_sequence",
        {"name": req.name, "steps": req.steps, "target": req.target},
        target_override=req.target,
        confirm_token=req.confirm,
    )
    return res


# ── Motion Primitives & Composition Endpoints ────────────────────────────


@app.get("/api/primitives")
async def list_primitives(group: str | None = Query(None)):
    composer = get_composer()
    prims = composer.list_primitives(group=group)
    return {
        "ok": True,
        "primitives": [
            {
                "name": p["name"],
                "group": p["group"],
                "joints": p["joints"],
                "description": p.get("description", ""),
                "tags": p.get("tags", []),
                "frame_count": len(p.get("frames", [])),
                "frames": p.get("frames", []),
            }
            for p in prims
        ],
    }


@app.get("/api/primitives/{name}")
async def get_primitive(name: str):
    composer = get_composer()
    prim = composer.get_primitive(name)
    if prim is None:
        raise HTTPException(status_code=404, detail=f"Primitive {name!r} not found")
    return {"ok": True, "primitive": prim}


class ComposeRequest(BaseModel):
    name: str
    primitives: list[str]
    mode: str = "sequential"
    description: str = ""
    transition_blend_ms: int = 100
    blend_frames: int = 2


@app.post("/api/compose")
async def compose_endpoint(req: ComposeRequest):
    composer = get_composer()
    try:
        moveset = composer.compose_move(
            req.name,
            req.primitives,
            mode=req.mode,
            description=req.description,
            transition_blend_ms=req.transition_blend_ms,
            blend_frames=req.blend_frames,
        )
        return {"ok": True, "moveset": moveset}
    except CompositionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class IterateRequest(BaseModel):
    name: str
    adjustments: dict[str, Any]


@app.post("/api/iterate")
async def iterate_endpoint(req: IterateRequest):
    composer = get_composer()
    try:
        moveset = composer.iterate_move(req.name, req.adjustments)
        return {"ok": True, "moveset": moveset}
    except CompositionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class EvaluateRequest(BaseModel):
    name: str | None = None
    frames: list[dict[str, Any]] | None = None


@app.post("/api/evaluate")
async def evaluate_endpoint(req: EvaluateRequest):
    composer = get_composer()
    try:
        metrics = composer.evaluate_move(name=req.name, frames=req.frames)
        return {"ok": True, "metrics": metrics}
    except CompositionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/library")
async def list_library():
    composer = get_composer()
    movesets = composer.list_movesets()
    return {
        "ok": True,
        "movesets": [
            {
                "name": m.get("name", ""),
                "description": m.get("description", ""),
                "frame_count": len(m.get("frames", [])),
                "source_primitives": m.get("source_primitives", []),
                "created_at": m.get("created_at", ""),
                "updated_at": m.get("updated_at", ""),
            }
            for m in movesets
        ],
    }


@app.get("/api/library/{name}")
async def get_library_moveset(name: str):
    composer = get_composer()
    moveset = composer.get_moveset(name)
    if moveset is None:
        raise HTTPException(status_code=404, detail=f"Moveset {name!r} not found")
    return {"ok": True, "moveset": moveset}


@app.delete("/api/library/{name}")
async def delete_library_moveset(name: str):
    composer = get_composer()
    deleted = composer.delete_moveset(name)
    return {"ok": True, "deleted": deleted}


class SaveToLibraryRequest(BaseModel):
    name: str
    description: str = ""
    frames: list[dict[str, Any]]
    source_primitives: list[str] = Field(default_factory=list)
    composition_mode: str = ""


@app.post("/api/library")
async def save_to_library(req: SaveToLibraryRequest):
    composer = get_composer()
    data = {
        "description": req.description,
        "frames": req.frames,
        "source_primitives": req.source_primitives,
        "composition_mode": req.composition_mode,
    }
    saved = composer.save_moveset(req.name, data)
    return {"ok": True, "name": req.name, "moveset": saved}


# ── Agent Harness Endpoints ───────────────────────────────────────────────


class AgentChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt / goal instruction")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Previous messages")
    target: str = Field("sim", description="'sim' or 'real'")
    confirm: str | None = None
    stream: bool = Field(True, description="Whether to stream response via SSE")


class AgentStepRequest(BaseModel):
    tool: str = Field(..., description="Tool name, e.g. 'bittle_do_skill'")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    target: str = Field("sim", description="'sim' or 'real'")
    confirm: str | None = None


@app.get("/api/agent/config")
async def agent_config():
    resolved_base, resolved_model = await agent_harness.resolve_endpoint_and_model()
    return {
        "api_base": resolved_base,
        "model": resolved_model,
        "configured_base": settings.llm_api_base,
        "configured_model": settings.llm_model,
        "timeout": settings.llm_timeout,
    }


@app.post("/api/agent/step")
async def agent_step(req: AgentStepRequest):
    return await agent_harness.execute_tool(
        req.tool,
        req.args,
        target_override=req.target,
        confirm_token=req.confirm,
    )


@app.post("/api/agent/chat")
async def agent_chat(req: AgentChatRequest):
    messages = list(req.history)
    messages.append({"role": "user", "content": req.prompt})

    if not req.stream:
        events = []
        async for ev in agent_harness.chat_stream(
            messages, target=req.target, confirm_token=req.confirm
        ):
            events.append(ev)
        return {"events": events}

    async def event_generator():
        async for ev in agent_harness.chat_stream(
            messages, target=req.target, confirm_token=req.confirm
        ):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if STATIC_DIR.is_dir():
    assets_dir = STATIC_DIR / "assets"
    vendor_dir = STATIC_DIR / "vendor"
    js_dir = STATIC_DIR / "js"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    if vendor_dir.is_dir():
        app.mount("/vendor", StaticFiles(directory=str(vendor_dir)), name="vendor")
    if js_dir.is_dir():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")

    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

    @app.get("/")
    async def index():
        return FileResponse(str(STATIC_DIR / "index.html"))
