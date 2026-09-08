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
from trainer.eval.evaluator import GaitController, PolicyController, StandController, nominal_env, protocol_kwargs  # noqa: E402
from trainer.eval.gates import load_suite, suite_protocol, suite_scenes  # noqa: E402
from trainer.policy.mlp import NumpyPolicy  # noqa: E402

W, H, FPS, SECONDS = 300, 225, 12, 5.0


def _scan_runs(runs_dir: Path) -> list[dict]:
    """Read every run's state.json directly; index.json is only refreshed by the service.
    Runs recorded before suites existed have no suite key and are flat_v1 / flat_walk."""
    out = []
    for d in runs_dir.iterdir():
        if d.is_dir() and (d / "state.json").is_file():
            st = json.loads((d / "state.json").read_text())
            out.append({"run_id": d.name, "status": st.get("status"), "created": st.get("created"),
                        "suite": st.get("suite") or "flat_v1", "task": st.get("task") or "flat_walk",
                        "name": st.get("name", d.name)})
    return out


def _suite_env(cfg: TrainConfig, suite: str, scene: str | None = None, **kw):
    """The env a run is JUDGED in: the suite's fixed terrain, not whatever the run trained on —
    so a GIF and its gate report always show the same ground. ``scene`` picks one scene of a
    scenes suite (default: the primary one)."""
    s = load_suite(suite)
    proto = suite_protocol(s)
    if scene:
        proto = dict(suite_scenes(s))[scene]
    return nominal_env(cfg, **protocol_kwargs(proto), **kw), [float(x) for x in proto["command"]]


def _scene_names(suite: str) -> list[str]:
    """Named scenes of a scenes suite; [] for a classic one."""
    return [n for n, _ in suite_scenes(load_suite(suite)) if n]


def _camera(d: mujoco.MjData, m: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.42 if m.nbody < 40 else 0.55  # the terrain variants carry 32 box bodies: frame wider
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


def _report(runs_dir: Path, rid: str, suite: str) -> dict | None:
    reps = sorted((runs_dir / rid / "benchmark").glob(f"{suite}@*/report.json")) if (runs_dir / rid / "benchmark").is_dir() else []
    return json.loads(reps[-1].read_text()) if reps else None


def plot_gates(runs_dir: Path, run_ids: list[str], out: Path, suite: str = "flat_v1") -> None:
    """One figure per suite: bars from different suites are not comparable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = {}
    for rid in run_ids:
        rep = _report(runs_dir, rid, suite)
        if rep:
            reports[rid] = rep
    run_ids = list(reports)
    if not run_ids:
        return
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
    ax.set_title(f"{suite} gates: green pass, red fail (dashed line = threshold)")
    fig.tight_layout()
    fig.savefig(out / ("gates.png" if suite == "flat_v1" else f"gates_{suite}.png"), dpi=110)


def plot_scenes(runs_dir: Path, run_ids: list[str], out: Path, suite: str) -> None:
    """Scenes suite: per-scene fall rate and progress along the commanded direction for every run on
    the suite, next to the firmware trot replayed on the same scenes (runs/baselines)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trainer.eval.gates import suite_scenes

    s = load_suite(suite)
    scenes = suite_scenes(s)
    names = [n for n, _ in scenes]
    sign = {n: (1.0 if float(p["command"][0]) > 0 else -1.0 if float(p["command"][0]) < 0 else 0.0) for n, p in scenes}
    series: list[tuple[str, dict]] = []
    base = runs_dir / "baselines" / "opencat_trF" / suite / "report.json"
    if base.is_file():
        series.append(("firmware trot", json.loads(base.read_text())["metrics"]))
    for rid in run_ids:
        rep = _report(runs_dir, rid, suite)
        if rep:
            series.append((json.loads((runs_dir / rid / "state.json").read_text()).get("name", rid), rep["metrics"]))
    if not series:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    width = 0.8 / len(series)
    xs = np.arange(len(names))
    for i, (label, m) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * width
        falls = [float(m.get(f"{n}/fall_rate", float("nan"))) for n in names]
        def _f(key: str) -> float:
            v = m.get(key)
            return float(v) if v is not None else float("nan")

        prog = [_f(f"{n}/progress_ratio") * sign[n] if sign[n] else _f(f"{n}/centre_drift_m") for n in names]
        axes[0].bar(xs + off, falls, width=width, label=label, alpha=0.85)
        axes[1].bar(xs + off, prog, width=width, label=label, alpha=0.85)
    for ax, title in zip(axes, ("fall rate per scene (lower is better)", "progress along the command (ratio; statue = drift m)")):
        ax.set_xticks(xs); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8); ax.grid(axis="y", alpha=0.3); ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{suite}: every run on the suite, scene by scene")
    fig.tight_layout()
    fig.savefig(out / f"scenes_{suite}.png", dpi=110)


def plot_gait(runs_dir: Path, rid: str, out: Path, suite: str = "flat_v1") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = sorted((runs_dir / rid / "benchmark").glob(f"{suite}@*/rollouts/seed_0.json"))[-1]
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


# ── run ledger chapter ──────────────────────────────────────────────────────

LEDGER_HEAD = """---
part: RL walking trainer
status: shipped
updated: {date}
---

# Training run ledger

Every policy training run recorded by the trainer, grouped by the gate suite it is judged on and
newest last within a suite. Generated from `runs/` by `trainer/scripts/render_docs.py --ledger`;
rerun it after training and commit the result. Score = gates passed + 0.5·min(distance, 1) − fall
rate, and it is only comparable WITHIN a suite. Each suite's protocol line comes from its yaml.

{sections}
"""

LEDGER_SECTION = """## {suite} — {n} runs

{protocol}

| # | run | task | started | parent | steps | envs | wall-clock | gates | distance p50 | falls | score | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{rows}

"""


def ledger(runs_dir: Path, out_md: Path, media_rel: str = "media/rl-training") -> None:
    import time

    from trainer.config import TrainConfig, config_diff

    runs = sorted(_scan_runs(runs_dir), key=lambda r: r["created"] or "")
    defaults = TrainConfig().model_dump(mode="json")
    rows_by_suite: dict[str, list[str]] = {}
    details = []
    for i, r in enumerate(runs, 1):
        rid = r["run_id"]
        suite = r["suite"]
        st = json.loads((runs_dir / rid / "state.json").read_text())
        cfg = json.loads((runs_dir / rid / "config.json").read_text())
        met = {}
        if (runs_dir / rid / "metrics.json").is_file():
            met = json.loads((runs_dir / rid / "metrics.json").read_text())
        rep = _report(runs_dir, rid, suite) or {}
        gif = f"policy_{st.get('name', rid)}.gif"
        clip = f"[gif]({media_rel}/{gif})" if (out_md.parent.parent / "media" / "rl-training" / gif).is_file() else "—"
        wall = f"{met.get('elapsed_s', 0) / 60:.1f} min" if met.get("elapsed_s") else "—"
        rows_by_suite.setdefault(suite, []).append(
            f"| {i} | `{rid}`<br>{st.get('name', '')} | {r['task']} | {st.get('created', '')[:16]} | {st.get('parent') or '—'} | "
            f"{cfg.get('ppo', {}).get('num_timesteps', 0) / 1e6:.0f}M | {cfg.get('ppo', {}).get('num_envs', '')} | {wall} | "
            f"{rep.get('gates_passed', '—')}/{rep.get('gates_total', '—')} | "
            f"{rep.get('metrics', {}).get('forward_distance_p50', float('nan')):.2f} m | "
            f"{rep.get('metrics', {}).get('fall_rate', float('nan')):.2f} | {rep.get('score', '—')} | {clip} |")
        base = json.loads((runs_dir / st["parent"] / "config.json").read_text()) if st.get("parent") and (runs_dir / st["parent"] / "config.json").is_file() else defaults
        diff = config_diff(base, cfg)
        diff_s = "\n".join(f"- `{d['path']}`: {d['from']} → {d['to']}" for d in diff) or "- (defaults)"
        failing = [g for g in rep.get("gates", []) if g["pass"] is False]
        fail_s = ", ".join(f"{g['gate']} ({g['value']:.3f} {g['op']} {g['threshold']})" for g in failing) or "none"
        warm = st.get("warm_start") or {}
        warm_s = ("warm start from " + warm["parent"]) if warm.get("used") else (f"cold start ({warm.get('reason')})" if warm.get("parent") else "from scratch")
        details.append(f"""### Run {i}: {st.get('name', rid)} (`{rid}`)

- **Task / suite:** {r['task']} / {suite}@{rep.get('suite_version', '—')}; {warm_s}
- **Status:** {st.get('status')}{(' — ' + st['error']) if st.get('error') else ''}
- **Hypothesis / notes:** {st.get('notes') or '—'}
- **Config vs {'parent ' + st['parent'] if st.get('parent') else 'defaults'}:**
{diff_s}
- **Training:** {met.get('num_timesteps', 0) / 1e6:.0f}M steps, {met.get('num_envs', '')} envs, {met.get('elapsed_s', 0):.0f} s, {met.get('steps_per_s_mean', 0):,.0f} steps/s ({met.get('impl', '')}); eval reward {met.get('reward_first', 0):.0f} → {met.get('reward_final', 0):.0f}
- **Gates:** {rep.get('gates_passed', '—')}/{rep.get('gates_total', '—')}; failing: {fail_s}
- **Reflection:** {rep.get('reflection', '—')}
""")
    sections = []
    for suite, rows in rows_by_suite.items():
        s = load_suite(suite)
        proto = suite_protocol(s)
        if proto.get("scenes"):
            protocol = (f"Protocol `{suite}@{s.get('version')}`: {len(proto['scenes'])} scenes ({', '.join(proto['scenes'])}), "
                        f"{proto['n_episodes']} seeded episodes each, {proto['episode_seconds']:.0f} s, CPU MuJoCo on the mesh model; "
                        f"{len(s['gates'])} gates incl. the servo-safety fragment (read on the worst scene).")
            sections.append(LEDGER_SECTION.format(suite=suite, n=len(rows), protocol=protocol, rows="\n".join(rows)))
            continue
        t = proto["terrain"]
        ground = {"flat": "flat ground", "slope": f"a {t.get('slope_deg', 0)} deg incline (tilted world)",
                  "rough": f"{t.get('n_boxes', 0)} x {1000 * float(t.get('box_height_m', 0)):.0f} mm boxes (seed {t.get('field_seed', 0)}, spawn jitter {t.get('spawn_jitter_m', 0)} m)",
                  "rough_slope": "boxes on an incline"}[t.get("kind", "flat")]
        protocol = (f"Protocol `{suite}@{s.get('version')}`: {proto['n_episodes']} seeded episodes, {proto['episode_seconds']:.0f} s, "
                    f"command {proto['command']}, {ground}, CPU MuJoCo on the mesh model; "
                    f"{len(s['gates'])} gates incl. the servo-safety fragment.")
        sections.append(LEDGER_SECTION.format(suite=suite, n=len(rows), protocol=protocol, rows="\n".join(rows)))
    text = LEDGER_HEAD.format(date=time.strftime("%Y-%m-%d"), sections="".join(sections)) + "\n".join(details)
    text += """
## Runs outside the store (development smoke tests, 2026-09-07)

| run | steps | envs | result | lesson |
|---|---|---|---|---|
| smoke1 | 0.3M | 256 | reward 344 → 324, no learning | first end-to-end PPO run; too short to judge |
| crippled 40M (cancelled) | — | 2048 | cancelled after 80 s | servos could not move under the torque cap (kp 40, damping 1.5): trot replay 0.000 m |
| smoke2 | 3M | 1024 | reward 58 → 452, stands, falls when asked to walk (clip on the main page) | pipeline learns; 3M steps is ~6 episodes per env |

The first two ran on the original servo parameters, which is why they are not in the store: the model was rebuilt
with a torque-limited servo (kp 10, ±0.25 N·m, damping 0.05) before run 1.
"""
    out_md.write_text(text)
    print(f"ledger: {len(runs)} runs -> {out_md}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=None, help="write the run-ledger chapter to this path and exit")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="documentation/media/rl-training")
    ap.add_argument("--runs", nargs="*", default=None, help="run ids (default: all done runs, oldest first)")
    ap.add_argument("--extra-policy", action="append", default=[], help="label=path/to/policy.npz")
    a = ap.parse_args()
    runs_dir, out = Path(a.runs_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.ledger:
        ledger(runs_dir, Path(a.ledger))
        return 0
    scanned = {r["run_id"]: r for r in _scan_runs(runs_dir)}
    run_ids = a.runs or [r["run_id"] for r in sorted(scanned.values(), key=lambda r: r["created"] or "") if r["status"] == "done"]
    cfg = TrainConfig()
    summary = {}
    suites = sorted({scanned[r]["suite"] for r in run_ids if r in scanned} | {"flat_v1"})
    for suite in suites:
        env, cmd = _suite_env(cfg, suite, envelope_tier="tested")
        gif = "baseline_opencat_trot.gif" if suite == "flat_v1" else f"baseline_opencat_trot_{suite}.gif"
        summary[f"opencat_trF@{suite}"] = render_episode(env, GaitController.from_opencat("trF"), seed=0, command=cmd,
                                                         out_gif=out / gif, label=f"OpenCat trF on {suite}")
    for spec in a.extra_policy:
        label, path = spec.split("=", 1)
        env, cmd = _suite_env(cfg, "flat_v1")
        summary[label] = render_episode(env, PolicyController(NumpyPolicy.load(path)), seed=0, command=cmd,
                                        out_gif=out / f"policy_{label}.gif", label=label)
    # scene clips (nine GIFs a run) only for the BEST run on each scenes suite: the ledger clip is enough for the rest
    best_on_suite: dict[str, str] = {}
    for rid in run_ids:
        suite = scanned[rid]["suite"]
        if not _scene_names(suite):
            continue
        rep = _report(runs_dir, rid, suite) or {}
        key = (rep.get("gates_passed") or 0, rep.get("score") or 0.0)
        cur = best_on_suite.get(suite)
        if cur is None or key > cur[1]:
            best_on_suite[suite] = (rid, key)
    best_ids = {rid for rid, _ in best_on_suite.values()}
    for rid in run_ids:
        r = scanned[rid]
        # the run's OWN config (control rate, action scale) in its OWN suite's terrain
        run_cfg = TrainConfig.model_validate(json.loads((runs_dir / rid / "config.json").read_text()))
        env, cmd = _suite_env(run_cfg, r["suite"])
        pol = PolicyController(NumpyPolicy.load(runs_dir / rid / "policy" / "policy.npz"))
        summary[rid] = render_episode(env, pol, seed=0, command=cmd, out_gif=out / f"policy_{r['name']}.gif",
                                      label=f"{r['name']} ({r['suite']})")
        # a scenes suite: one clip per scene, on that scene's ground, with its command and pushes
        for scene in (_scene_names(r["suite"]) if rid in best_ids else []):
            env, cmd = _suite_env(run_cfg, r["suite"], scene=scene)
            summary[f"{rid}@{scene}"] = render_episode(env, pol, seed=0, command=cmd,
                                                       out_gif=out / f"policy_{r['name']}_{scene}.gif",
                                                       label=f"{r['name']} / {scene}")
    for suite in suites:
        for scene in _scene_names(suite):
            env, cmd = _suite_env(cfg, suite, scene=scene, envelope_tier="tested")
            summary[f"opencat_trF@{suite}/{scene}"] = render_episode(env, GaitController.from_opencat("trF"), seed=0, command=cmd,
                                                                     out_gif=out / f"baseline_opencat_trot_{suite}_{scene}.gif",
                                                                     label=f"OpenCat trF / {scene}")
    render_model_stills(out)
    plot_curves(runs_dir, run_ids, out)
    for suite in suites:
        plot_gates(runs_dir, [r for r in run_ids if scanned[r]["suite"] == suite], out, suite=suite)
        if _scene_names(suite):
            plot_scenes(runs_dir, [r for r in run_ids if scanned[r]["suite"] == suite], out, suite=suite)
    flat_runs = [r for r in run_ids if scanned[r]["suite"] == "flat_v1"]
    if flat_runs:
        plot_gait(runs_dir, flat_runs[0], out)
    (out / "render_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
