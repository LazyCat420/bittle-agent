"""Render the figures for documentation/rl-training.md (GIFs, stills, plots).

    MUJOCO_GL=osmesa .venv-trainer/bin/python trainer/scripts/render_docs.py \
        --runs-dir runs --out documentation/media/rl-training [--extra-policy name=path.npz]

Headless rendering uses OSMesa (EGL is not available under WSL2). GIFs are
5 s at 12 fps, 300x225, 64 colours, so the whole set stays a few MB in git.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from trainer.config import TrainConfig  # noqa: E402
from trainer.eval.evaluator import GaitController, PolicyController, StandController, nominal_env  # noqa: E402
from trainer.policy.mlp import NumpyPolicy  # noqa: E402

W, H, FPS, SECONDS = 300, 225, 12, 5.0


def _camera(d: mujoco.MjData, m: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.42
    cam.azimuth = 150
    cam.elevation = -18
    cam.lookat[:] = d.xpos[m.body("torso").id] + np.array([0.0, 0.0, -0.01])
    return cam


def render_episode(env, controller, *, seed: int, command, out_gif: Path, label: str) -> dict:
    env.rng = np.random.default_rng(seed)
    obs = env.reset(command=np.asarray(command, dtype=np.float64))
    controller.reset(env)
    m, d = env.model, env.data
    renderer = mujoco.Renderer(m, height=H, width=W)
    every = max(1, int(round(env.cfg.control_hz / FPS)))
    frames = []
    steps = int(SECONDS * env.cfg.control_hz)
    x0 = float(d.qpos[0])
    fell = False
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
    _annotate(frames, f"{label}  {dist:+.2f} m / {min(SECONDS, (t + 1) / env.cfg.control_hz):.0f} s" + ("  FELL" if fell else ""))
    q = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    q[0].save(out_gif, save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    return {"distance_m": round(dist, 3), "fell": fell, "frames": len(frames), "gif": out_gif.name}


def _annotate(frames: list[Image.Image], text: str) -> None:
    from PIL import ImageDraw

    for im in frames:
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 16], fill=(20, 20, 20))
        dr.text((4, 2), text, fill=(240, 240, 240))


def render_model_stills(out: Path) -> None:
    for variant, groups, title in (("cpu", {0: 1, 1: 0, 4: 0}, "bittle_cpu.xml — visual meshes"),
                                   ("cpu", {0: 0, 1: 1, 4: 1}, "bittle_cpu.xml — mesh collision geoms + foot spheres"),
                                   ("gpu", {0: 0, 1: 0, 4: 1}, "bittle_gpu.xml — primitive collision model")):
        m = mujoco.MjModel.from_xml_path(f"trainer/assets/generated/bittle_{variant}.xml")
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for _ in range(300):
            mujoco.mj_step(m, d)
        opt = mujoco.MjvOption()
        for g, v in groups.items():
            opt.geomgroup[g] = v
        r = mujoco.Renderer(m, height=360, width=480)
        cam = _camera(d, m)
        cam.distance = 0.38
        r.update_scene(d, camera=cam, scene_option=opt)
        im = Image.fromarray(r.render())
        r.close()
        name = title.split(" — ")[1].replace(" ", "_").replace("+", "and")
        im.save(out / f"model_{variant}_{name}.png")


def plot_curves(runs_dir: Path, run_ids: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for rid in run_ids:
        pts = [json.loads(l) for l in (runs_dir / rid / "curves.jsonl").read_text().splitlines() if l.strip()]
        name = json.loads((runs_dir / rid / "state.json").read_text()).get("name", rid)
        axes[0].plot([p["step"] / 1e6 for p in pts], [p["reward"] for p in pts], marker="o", label=name)
        axes[1].plot([p["step"] / 1e6 for p in pts], [p.get("distance_x", float("nan")) for p in pts], marker="o", label=name)
    axes[0].set_xlabel("env steps (M)"); axes[0].set_ylabel("eval episode reward"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("env steps (M)"); axes[1].set_ylabel("eval distance walked (m / 10 s)"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=110)


def plot_gates(runs_dir: Path, run_ids: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = {}
    for rid in run_ids:
        p = sorted((runs_dir / rid / "benchmark").glob("flat_v1@*/report.json"))[-1]
        reports[rid] = json.loads(p.read_text())
    gates = [g for g in reports[run_ids[0]]["gates"] if g["pass"] is not None]
    names = [g["gate"] for g in gates]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(run_ids)
    for i, rid in enumerate(run_ids):
        byname = {g["gate"]: g for g in reports[rid]["gates"]}
        ratios, colors = [], []
        for n in names:
            g = byname[n]
            thr = g["threshold"] if not isinstance(g["threshold"], list) else g["threshold"][1]
            v = g["value"] if g["value"] is not None else 0.0
            ratio = (v / thr) if thr not in (0, None) else (1.0 if v <= 0 else 2.0)
            if g["op"] == ">=":
                ratio = (thr / v) if v else 2.0  # >= gates: threshold/value so <1 still means pass
            ratios.append(min(ratio, 2.0))
            colors.append("#2a9d8f" if g["pass"] else "#e76f51")
        xs = np.arange(len(names)) + (i - (len(run_ids) - 1) / 2) * width
        ax.bar(xs, ratios, width=width, color=colors, alpha=0.85, label=reports[rid].get("run_id", rid))
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("value / threshold  (below 1 = pass)"); ax.set_ylim(0, 2.1); ax.grid(axis="y", alpha=0.3)
    ax.set_title("flat_v1 gates: green pass, red fail (dashed line = threshold)")
    fig.tight_layout()
    fig.savefig(out / "gates.png", dpi=110)


def plot_gait(runs_dir: Path, rid: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = sorted((runs_dir / rid / "benchmark").glob("flat_v1@*/rollouts/seed_0.json"))[-1]
    ro = json.loads(p.read_text())
    t = np.array([f["t"] for f in ro["frames"]])
    c = np.array([f["contacts"] for f in ro["frames"]])
    x = np.array([f["base_pos_m"][0] for f in ro["frames"]])
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    legs = ["RF", "LF", "RR", "LR"]
    for i in range(4):
        on = c[:, i] > 0
        axes[0].fill_between(t, i + 0.1, i + 0.9, where=on, step="post", color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261"][i])
    axes[0].set_yticks(np.arange(4) + 0.5); axes[0].set_yticklabels(legs); axes[0].set_ylabel("foot on floor")
    axes[0].set_title(f"gait diagram, run {rid} seed 0 (diagonal pairs = trot)")
    axes[0].set_xlim(2.0, 5.0)
    axes[1].plot(t, x); axes[1].set_ylabel("base x (m)"); axes[1].set_xlabel("time (s)"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "gait_diagram.png", dpi=110)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="documentation/media/rl-training")
    ap.add_argument("--runs", nargs="*", default=None, help="run ids (default: all done runs, oldest first)")
    ap.add_argument("--extra-policy", action="append", default=[], help="label=path/to/policy.npz")
    a = ap.parse_args()
    runs_dir, out = Path(a.runs_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = json.loads((runs_dir / "index.json").read_text())["runs"]
    run_ids = a.runs or sorted(r["run_id"] for r in idx if r["status"] == "done")
    cfg = TrainConfig()
    summary = {}
    cmd = [0.12, 0.0, 0.0]
    summary["opencat_trF"] = render_episode(nominal_env(cfg, envelope_tier="tested"), GaitController.from_opencat("trF"),
                                            seed=0, command=cmd, out_gif=out / "baseline_opencat_trot.gif", label="OpenCat trF baseline")
    for spec in a.extra_policy:
        label, path = spec.split("=", 1)
        pol = PolicyController(NumpyPolicy.load(path))
        summary[label] = render_episode(nominal_env(cfg), pol, seed=0, command=cmd, out_gif=out / f"policy_{label}.gif", label=label)
    for rid in run_ids:
        name = json.loads((runs_dir / rid / "state.json").read_text()).get("name", rid)
        pol = PolicyController(NumpyPolicy.load(runs_dir / rid / "policy" / "policy.npz"))
        summary[rid] = render_episode(nominal_env(cfg), pol, seed=0, command=cmd, out_gif=out / f"policy_{name}.gif", label=name)
    render_model_stills(out)
    plot_curves(runs_dir, run_ids, out)
    plot_gates(runs_dir, run_ids, out)
    plot_gait(runs_dir, run_ids[0], out)
    (out / "render_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
