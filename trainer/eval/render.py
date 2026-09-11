"""Headless clip rendering (OSMesa; EGL is not available under WSL2).

``render_episode`` re-simulates one benchmark episode on the CPU env and writes an annotated GIF.
The benchmark writes ``best.gif`` for its best episode next to the report; ``render_docs.py`` uses
the same function for the documentation clips.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

W, H, FPS, SECONDS = 300, 225, 12, 5.0


def _camera(d: mujoco.MjData, m: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.42 if m.nbody < 40 else 0.55  # the terrain variants carry 32 box bodies: frame wider
    cam.azimuth = 150
    cam.elevation = -18
    cam.lookat[:] = d.xpos[m.body("torso").id] + np.array([0.0, 0.0, -0.01])
    return cam


def render_episode(env, controller, *, seed: int, command, out_gif: Path, label: str,
                   seconds: float = SECONDS) -> dict:
    env.rng = np.random.default_rng(seed)
    obs = env.reset(command=np.asarray(command, dtype=np.float64))
    controller.reset(env)
    m, d = env.model, env.data
    renderer = mujoco.Renderer(m, height=H, width=W)
    every = max(1, int(round(env.cfg.control_hz / FPS)))
    frames = []
    steps = int(seconds * env.cfg.control_hz)
    x0 = float(d.qpos[0])
    fell = False
    t = 0
    for t in range(steps):
        kind, val = controller.act(obs, env, t)
        obs, r, done, info = env.step(np.zeros(8), target_deg=val) if kind == "target" else env.step(val)
        if t % every == 0:
            renderer.update_scene(d, camera=_camera(d, m))
            frames.append(Image.fromarray(renderer.render()))
        if done:
            fell = True
            break
    renderer.close()
    dist = float(d.qpos[0] - x0)
    _annotate(frames, f"{label}  {dist:+.2f} m / {min(seconds, (t + 1) / env.cfg.control_hz):.0f} s" + ("  FELL" if fell else ""))
    q = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out_gif, save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    return {"distance_m": round(dist, 3), "fell": fell, "frames": len(frames), "gif": out_gif.name, "seed": seed}


def _annotate(frames: list[Image.Image], text: str) -> None:
    from PIL import ImageDraw

    for im in frames:
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 16], fill=(20, 20, 20))
        dr.text((4, 2), text, fill=(240, 240, 240))


def best_episode_seed(report: dict) -> int | None:
    """The benchmark's best attempt: the primary scene's longest non-fallen episode (else the longest)."""
    eps = report.get("episodes") or []
    primary = (report.get("protocol") or {}).get("scene", "")
    pool = [e for e in eps if (e.get("scene") or "") == (primary or "")] or eps
    if not pool:
        return None
    ok = [e for e in pool if not e.get("fell")] or pool
    best = max(ok, key=lambda e: float(e.get("distance_x", -1e9)))
    return int(best["seed"])


def render_best_clip(store, run_id: str, report: dict, *, label: str | None = None) -> dict | None:
    """Write ``best.gif`` next to the report for the report's best episode (same seed = same episode
    on the CPU env). Returns the clip summary, or None when there is nothing to render."""
    from ..config import TrainConfig
    from ..policy.mlp import NumpyPolicy
    from .evaluator import PolicyController, nominal_env, protocol_kwargs

    seed = best_episode_seed(report)
    if seed is None:
        return None
    cfg = TrainConfig.model_validate(store.config(run_id)).resolved()
    proto = report.get("protocol") or {}
    kw = protocol_kwargs(proto)
    env = nominal_env(cfg, **kw)
    ctrl = PolicyController(NumpyPolicy.load(store.policy_path(run_id)))
    out = store.benchmark_dir(run_id, report["suite"], str(report["suite_version"])) / "best.gif"
    name = label or store.state(run_id).get("name") or run_id
    info = render_episode(env, ctrl, seed=seed, command=proto.get("command", [0.12, 0.0, 0.0]), out_gif=out,
                          label=f"{name} best ({report['suite']})", seconds=float(proto.get("episode_seconds", 10.0)))
    return info
