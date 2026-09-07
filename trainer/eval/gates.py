"""Gate evaluation: metrics dict -> structured pass/fail report + reflection.

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

SUITES_DIR = Path(__file__).resolve().parent

_OPS = {
    "<=": lambda v, t: v <= t,
    ">=": lambda v, t: v >= t,
    "<": lambda v, t: v < t,
    ">": lambda v, t: v > t,
    "==": lambda v, t: v == t,
}


def load_suite(name: str = "flat_v1") -> dict[str, Any]:
    path = SUITES_DIR / "gates.yaml"
    with open(path) as fh:
        suite = yaml.safe_load(fh)
    if suite.get("suite") != name:
        raise ValueError(f"unknown gate suite {name!r}; available: {suite.get('suite')!r}")
    return suite


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


#: Which reward term most directly moves each gate (for the reflection's weight-share advice).
GATE_TO_TERM = {
    "energy_proxy": "energy", "energy_vs_baseline": "energy", "action_smoothness": "action_rate",
    "body_stability_tilt": "orientation", "body_stability_vz": "lin_vel_z", "fall_rate": "orientation",
    "joint_saturation_pct": "joint_saturation", "stand_still_drift": "stand_still", "stand_still_falls": "stand_still",
    "heading_drift_yaw": "tracking_ang_vel", "heading_drift_lateral": "tracking_lin_vel",
    "vel_tracking_rmse": "tracking_lin_vel", "forward_distance_p50": "tracking_lin_vel",
    "beats_baseline_trot_distance": "tracking_lin_vel", "multi_command": "tracking_lin_vel",
}


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
            "term": GATE_TO_TERM.get(g["gate"]),
            "term_share_pct": shares.get(GATE_TO_TERM.get(g["gate"], ""), None),
        }
    return {"per_gate": per_gate, "reward_breakdown": rb, "reward_shares_pct": shares,
            "parent_run_id": context.get("parent_run_id"), "baseline": context.get("baseline_name")}


def build_reflection(report: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any] | None = None) -> str:
    """Prose feedback for the LLM, same pattern as simulate_stair_climb_episode."""
    ctx = report.get("context") or {}
    per_gate = ctx.get("per_gate", {})
    failed = [r for r in report["gates"] if r["pass"] is False]
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
            cmp.append(f"firmware trot {pg['baseline_trot']:.3f}")
        if cmp:
            line += " (" + ", ".join(cmp) + ")"
        if r["note"]:
            line += f" -> {r['note']}"
        share = pg.get("term_share_pct")
        if share is not None and pg.get("term") and share < 1.0:
            factor = max(2, int(round(2.0 / max(share, 1e-6))))
            line += (f". The '{pg['term']}' term is only {share:.2f}% of the total reward, so the optimiser barely sees it: "
                     f"multiply its weight by ~{min(factor, 1000)}x (to ~2% share), not by 2-10x.")
        parts.append(line)
    if len(failed) > 4:
        parts.append(f"(+{len(failed) - 4} more failing gates in the table)")
    skipped = [r for r in report["gates"] if r["pass"] is None]
    if skipped:
        parts.append(f"{len(skipped)} gates not evaluated (stage/group/metric).")
    if not failed and report["gates_total"] > 0:
        parts.append("All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.")
    return " ".join(parts)
