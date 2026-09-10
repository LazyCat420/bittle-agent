"""Gate evaluation: metrics dict -> structured pass/fail report + reflection.

Suites live in ``suites/<name>.yaml``; a suite may ``include:`` fragments
(``_servo_safety``) whose gates are appended unless the suite re-declares a
gate of the same name. Every gate names the reward ``term`` that moves it, so
the reflection's weight-share advice is data, not a Python map.

A suite may carry ``scenes:`` instead of one protocol: a list of named
protocols (each merged over the suite-level ``protocol`` defaults) that the
benchmark runs one after another. Per-scene metrics are published as
``<scene>/<metric>`` next to the combined top-level ones, so a gate can bar
one scene ("rough/fall_rate") or the whole house ("fall_rate_max").

A gate whose metric is absent from the metrics dict is reported as
``pass: None`` ("not evaluated") and does not count towards ``gates_total``.
Gates with ``requires_stage`` above the run's curriculum stage are skipped
the same way, so a stage-0 run is not punished for turning it never trained.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

SUITES_DIR = Path(__file__).resolve().parent / "suites"

_OPS = {
    "<=": lambda v, t: v <= t,
    ">=": lambda v, t: v >= t,
    "<": lambda v, t: v < t,
    ">": lambda v, t: v > t,
    "==": lambda v, t: v == t,
}


def list_suites() -> list[str]:
    """Suite names on disk (fragments start with ``_`` and are not suites)."""
    return sorted(p.stem for p in SUITES_DIR.glob("*.yaml") if not p.stem.startswith("_"))


def load_suite(name: str = "flat_v1") -> dict[str, Any]:
    path = SUITES_DIR / f"{name}.yaml"
    if name.startswith("_") or "/" in name or not path.is_file():
        raise ValueError(f"unknown gate suite {name!r}; available: {list_suites()}")
    suite = yaml.safe_load(path.read_text())
    if suite.get("suite") != name:
        raise ValueError(f"{path.name} declares suite {suite.get('suite')!r}, not {name!r}")
    return _merge_includes(suite)


def _merge_includes(suite: dict[str, Any]) -> dict[str, Any]:
    gates = list(suite.get("gates", []))
    local = {g["name"] for g in gates}
    for inc in suite.get("include", []) or []:
        frag_path = SUITES_DIR / f"{inc}.yaml"
        if not frag_path.is_file():
            raise ValueError(f"suite {suite.get('suite')!r} includes unknown fragment {inc!r}")
        frag = yaml.safe_load(frag_path.read_text())
        for g in frag.get("gates", []):
            if g["name"] not in local:
                gates.append(dict(g, included_from=inc))
    suite["gates"] = gates
    return suite


def _norm_protocol(proto: dict[str, Any]) -> dict[str, Any]:
    proto = dict(proto)
    if "command" not in proto:
        proto["command"] = [float(proto.get("command_vx", 0.12)), 0.0, 0.0]
    proto.setdefault("terrain", {"kind": "flat"})
    return proto


def suite_scenes(suite: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``[(scene_name, protocol)]``. A single-protocol suite is one unnamed scene (``""``), so
    every consumer loops the same way and a classic suite's metrics keep their bare names."""
    scenes = suite.get("scenes")
    if not scenes:
        return [("", suite_protocol(suite))]
    base = dict(suite.get("protocol", {}))
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for sc in scenes:
        name = str(sc.get("name", ""))
        if not name or not name.replace("_", "").isalnum() or name in seen:
            raise ValueError(f"suite {suite.get('suite')!r}: scene names must be unique identifiers, got {name!r}")
        seen.add(name)
        merged = dict(base)
        merged.update({k: v for k, v in sc.items() if k != "name"})
        out.append((name, _norm_protocol(merged)))
    return out


def suite_protocol(suite: dict[str, Any]) -> dict[str, Any]:
    """The protocol with its command as a 3-vector (older suites carried ``command_vx``).
    For a scenes suite this is the FIRST scene's protocol (the primary scene: what the
    multi-command / DR-sweep sub-protocols and the ledger line use) with ``scenes`` = the names."""
    if suite.get("scenes"):
        name, proto = suite_scenes(suite)[0]
        proto["scene"] = name
        proto["scenes"] = [n for n, _ in suite_scenes(suite)]
        return proto
    return _norm_protocol(suite.get("protocol", {}))


def suite_hash(suite: dict[str, Any]) -> str:
    canon = json.dumps(suite.get("gates", []), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def _check(op: str, value: float, threshold: Any) -> bool:
    if op == "within":
        lo, hi = float(threshold[0]), float(threshold[1])
        return lo <= value <= hi
    if op not in _OPS:
        raise ValueError(f"unknown gate op {op!r}")
    return bool(_OPS[op](value, float(threshold)))


def evaluate_gates(
    metrics: dict[str, Any],
    suite: dict[str, Any],
    *,
    curriculum_stage: int = 0,
    groups_enabled: set[str] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the gate report body (without run_id / seeds; caller adds those).

    ``context`` (optional) carries what the reflection needs to be actionable:
    ``baseline_metrics``, ``parent_metrics`` (same metric names) and
    ``reward_breakdown`` ({term: episode contribution} from the last training eval).
    """
    groups_enabled = groups_enabled or set()
    context = context or {}
    rows: list[dict[str, Any]] = []
    for g in suite.get("gates", []):
        row: dict[str, Any] = {
            "gate": g["name"],
            "metric": g["metric"],
            "op": g["op"],
            "threshold": g["threshold"],
            "unit": g.get("unit", ""),
            "term": g.get("term"),
            "value": None,
            "pass": None,
            "note": "",
        }
        req_stage = int(g.get("requires_stage", 0))
        group = g.get("group")
        if req_stage > curriculum_stage:
            row["note"] = f"skipped: requires curriculum_stage >= {req_stage}"
        elif group and group not in groups_enabled:
            row["note"] = f"skipped: group '{group}' not requested"
        elif g["metric"] not in metrics or metrics[g["metric"]] is None:
            row["note"] = "not evaluated: metric missing"
        else:
            value = float(metrics[g["metric"]])
            row["value"] = value
            row["pass"] = _check(g["op"], value, g["threshold"])
            if not row["pass"]:
                row["note"] = g.get("hint", "")
        rows.append(row)

    evaluated = [r for r in rows if r["pass"] is not None]
    passed_n = sum(1 for r in evaluated if r["pass"])
    total_n = len(evaluated)
    dist = float(metrics.get("forward_distance_p50") or 0.0)
    fall = float(metrics.get("fall_rate") or 0.0)
    score = passed_n + 0.5 * min(dist / 1.0, 1.0) - fall
    report = {
        "suite": suite.get("suite"),
        "suite_version": str(suite.get("version")),
        "gates_hash": suite_hash(suite),
        "gates": rows,
        "gates_passed": passed_n,
        "gates_total": total_n,
        "passed": total_n > 0 and passed_n == total_n,
        "score": round(score, 4),
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
    }
    if context:
        report["context"] = _context_view(context, report)
    report["reflection"] = build_reflection(report, metrics, context)
    return report


def _context_view(context: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Per-gate baseline/parent values and the reward-term shares, for tools and the reflection."""
    base = context.get("baseline_metrics") or {}
    parent = context.get("parent_metrics") or {}
    rb = context.get("reward_breakdown") or {}
    total = sum(abs(float(v)) for v in rb.values()) or 1.0
    shares = {k: round(100.0 * abs(float(v)) / total, 3) for k, v in rb.items()}
    per_gate = {}
    for g in report["gates"]:
        if g["value"] is None:
            continue
        per_gate[g["gate"]] = {
            "value": g["value"],
            "baseline_trot": base.get(g["metric"]),
            "parent": parent.get(g["metric"]),
            "term": g.get("term"),
            "term_share_pct": shares.get(g.get("term") or "", None),
        }
    out = {"per_gate": per_gate, "reward_breakdown": rb, "reward_shares_pct": shares,
           "parent_run_id": context.get("parent_run_id"), "baseline": context.get("baseline_name"),
           "baseline_note": context.get("baseline_note"), "parent_note": context.get("parent_note")}
    # train-vs-bench: what the training curve said next to what the suite says (2026-09-10)
    for key in ("train_distance_x", "train_bar_distance_x", "train_bar_distance_p50", "train_bar_fall_rate"):
        if context.get(key) is not None:
            out[key] = float(context[key])
    if context.get("terrain_gap"):
        out["terrain_gap"] = context["terrain_gap"]
    if context.get("reward_weights"):
        out["reward_weights"] = context["reward_weights"]
    return out


_DISTANCE_GATES = ("forward_distance_p50", "progress_ratio", "beats_baseline_trot_distance", "rough_progress",
                   "progress_worst_scene", "rough_slope_progress")


def _train_vs_bench_lines(ctx: dict[str, Any], metrics: dict[str, Any], failed: list[dict[str, Any]]) -> list[str]:
    """The lines that tell "the classroom is easier than the exam" from "the policy is stuck": the training
    curve's distance next to the suite's, the bar eval, the terrain gap, and the stuck diagnostics."""
    out: list[str] = []
    bench = metrics.get("forward_distance_p50")
    train_d = ctx.get("train_distance_x")
    bar_d = ctx.get("train_bar_distance_x")
    gap = ctx.get("terrain_gap") or {}
    distance_failed = any(r["gate"] in _DISTANCE_GATES for r in failed)
    if bench is not None and train_d is not None and distance_failed and float(train_d) > 1.5 * max(float(bench), 1e-6):
        line = (f"TRAIN/BENCH MISMATCH: the training curve walked {float(train_d):.2f} m per episode (random commands on the "
                f"run's own terrain) but the suite reads {float(bench):.2f} m")
        if gap.get("easier_than_protocol"):
            if "protocol_box_height_m" in gap:
                lo, hi = gap["train_box_height_m"]
                line += (f"; the run trained on {gap['train_n_boxes']} boxes of {1000 * lo:.0f}-{1000 * hi:.0f} mm "
                         f"({100 * gap['train_share_at_or_above_protocol_height']:.0f}% at or above the protocol's "
                         f"{1000 * gap['protocol_box_height_m']:.0f} mm) vs the protocol's {gap['protocol_n_boxes']} boxes "
                         f"all at {1000 * gap['protocol_box_height_m']:.0f} mm")
            if "protocol_slope_deg" in gap:
                line += (f"; the run trained on {gap['train_slope_deg'][0]:.0f}-{gap['train_slope_deg'][1]:.0f} deg vs the "
                         f"protocol's {gap['protocol_slope_deg']:.0f} deg")
            line += (". The optimiser solved an easier field than the exam: raise terrain.box_height_m / terrain.n_boxes "
                     "(or terrain.slope_deg) to cover the protocol BEFORE touching reward weights.")
        else:
            line += ("; the training terrain covers the protocol, so the gap is the fixed command / nominal DR / the CPU "
                     "sim, not the field -- read the bar eval and stuck diagnostics below.")
        out.append(line)
    if bar_d is not None:
        line = f"Bar eval (the suite protocol's terrain and command, on the GPU, at the end of training): {float(bar_d):.2f} m"
        if ctx.get("train_bar_fall_rate") is not None:
            line += f", falls {100 * float(ctx['train_bar_fall_rate']):.0f}%"
        if bench is not None:
            ratio = float(bar_d) / max(float(bench), 1e-6)
            if ratio > 1.5:
                line += (f" vs {float(bench):.2f} m here: the GPU and CPU engines disagree on this terrain (ratio {ratio:.2f}) -- "
                         "a sim gap, not a policy gap; check dual_sim before tuning anything")
            elif ratio < 0.67:
                line += f" vs {float(bench):.2f} m here (ratio {ratio:.2f}): the CPU benchmark is KINDER than the GPU protocol"
            else:
                line += f" vs {float(bench):.2f} m here: the engines agree; what the benchmark shows is what the policy learned"
        out.append(line + ".")
    if metrics.get("stuck_seconds_p50") is not None and metrics.get("stuck_episode_rate"):
        rate = float(metrics["stuck_episode_rate"])
        if rate > 0:
            line = (f"STUCK: {100 * rate:.0f}% of episodes stalled (|vx| < 0.02 m/s for >= 0.5 s while commanded), median "
                    f"{float(metrics['stuck_seconds_p50']):.1f} s per episode (worst {float(metrics['stuck_seconds_max']):.1f} s)")
            if metrics.get("stuck_x_p50") is not None:
                line += f", first stall at x = {float(metrics['stuck_x_p50']):.2f} m"
            share = metrics.get("stuck_limb_share")
            if share is not None:
                line += (f"; a shank/thigh was on the ground for {100 * float(share):.0f}% of the stalled steps"
                         + (" -- the legs catch the edges (foot_clearance / stumble)" if share >= 0.3 else
                            " -- the feet are NOT catching edges: it drags without lifting (foot_clearance / feet_air_time), "
                            "or the tracking term is saturated (tracking_sigma)"))
            out.append(line + ".")
    return out


def build_reflection(report: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any] | None = None) -> str:
    """Prose feedback for the LLM, same pattern as simulate_stair_climb_episode."""
    ctx = report.get("context") or {}
    per_gate = ctx.get("per_gate", {})
    failed = [r for r in report["gates"] if r["pass"] is False]
    base_label = str(ctx.get("baseline") or "opencat_trF").replace("opencat_trF", "firmware trot").replace("opencat_", "firmware ").replace("stand", "stand controller")
    head = "BENCHMARK PASSED" if report["passed"] else "BENCHMARK FAILED"
    parts = [f"{head} {report['gates_passed']}/{report['gates_total']} gates (score {report['score']:.2f})."]
    n_ep = metrics.get("n_episodes")
    if "fall_rate" in metrics and n_ep:
        falls = int(round(float(metrics["fall_rate"]) * int(n_ep)))
        parts.append(f"Fell in {falls}/{n_ep} episodes.")
    if "forward_distance_p50" in metrics:
        parts.append(f"Median distance {float(metrics['forward_distance_p50']):.2f} m in {metrics.get('episode_seconds', 10)} s.")
    if "vel_tracking_rmse" in metrics:
        parts.append(f"Velocity tracking RMSE {float(metrics['vel_tracking_rmse']):.3f} m/s.")
    if metrics.get("first_fall_time_mean") is not None:
        parts.append(f"Falls happen on average at t={float(metrics['first_fall_time_mean']):.1f} s.")
    for r in failed[:4]:
        thr = r["threshold"]
        thr_s = f"[{thr[0]}, {thr[1]}]" if isinstance(thr, (list, tuple)) else f"{thr}"
        line = f"FAIL {r['gate']}: {r['value']:.3f} {r['op']} {thr_s} {r['unit']}".rstrip()
        pg = per_gate.get(r["gate"], {})
        cmp = []
        if pg.get("parent") is not None:
            cmp.append(f"parent {pg['parent']:.3f}")
        if pg.get("baseline_trot") is not None:
            cmp.append(f"{base_label} {pg['baseline_trot']:.3f}")
        if cmp:
            line += " (" + ", ".join(cmp) + ")"
        if r["note"]:
            line += f" -> {r['note']}"
        share = pg.get("term_share_pct")
        term = pg.get("term")
        weight = (ctx.get("reward_weights") or {}).get(term) if term else None
        parent_v = pg.get("parent")
        flat_vs_parent = (parent_v is not None and r["value"] is not None
                          and abs(float(r["value"]) - float(parent_v)) <= 0.05 * max(abs(float(parent_v)), 1e-9))
        if term and weight is not None and float(weight) == 0.0:
            line += (f". The '{term}' term is switched OFF (reward.weights.{term} = 0.0): the optimiser cannot see this gate "
                     f"at all -- set the weight first (the task's defaults carry one), then judge its share.")
        elif share is not None and term and share < 1.0:
            factor = max(2, int(round(2.0 / max(share, 1e-6))))
            line += (f". The '{term}' term is only {share:.2f}% of the total reward, so the optimiser barely sees it: "
                     f"multiply its weight by ~{min(factor, 1000)}x (to ~2% share), not by 2-10x.")
        elif (r["gate"] == "foot_clearance" and r["value"] is not None and float(r["value"]) < 2.0
              and share is not None and share >= 2.0):
            shares = ctx.get("reward_shares_pct") or {}
            line += (f". The '{term}' term already carries {share:.1f}% of the reward yet the swing stays under 2 mm: the "
                     f"policy is DRAGGING its feet instead of swinging them, and a term that only scores swings cannot "
                     f"pay for a swing that never happens. The terms that make a real swing profitable are invisible "
                     f"(feet_air_time {shares.get('feet_air_time', 0.0):.2f}%, feet_slip {shares.get('feet_slip', 0.0):.2f}%): "
                     f"raise reward.weights.feet_air_time and reward.weights.feet_slip 10-40x, not foot_clearance again.")
        elif share is not None and term and share >= 2.0 and flat_vs_parent:
            line += (f". The '{term}' term already carries {share:.1f}% of the reward and the gate did not move vs the parent: "
                     f"the weight is not the lever -- this needs a longer budget (ppo.num_timesteps 30M: a gait has to be "
                     f"reshaped, not tuned) or a harder training terrain, not another weight change.")
        parts.append(line)
    if len(failed) > 4:
        parts.append(f"(+{len(failed) - 4} more failing gates in the table)")
    parts.extend(_train_vs_bench_lines(ctx, metrics, failed))
    for key in ("parent_note", "baseline_note"):
        if ctx.get(key):
            parts.append(str(ctx[key]))
    skipped = [r for r in report["gates"] if r["pass"] is None]
    if skipped:
        parts.append(f"{len(skipped)} gates not evaluated (stage/group/metric).")
    if not failed and report["gates_total"] > 0:
        parts.append("All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.")
    return " ".join(parts)
