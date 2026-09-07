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


def evaluate_gates(
    metrics: dict[str, Any],
    suite: dict[str, Any],
    *,
    curriculum_stage: int = 0,
    groups_enabled: set[str] | None = None,
) -> dict[str, Any]:
    """Return the gate report body (without run_id / seeds; caller adds those)."""
    groups_enabled = groups_enabled or set()
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
    report["reflection"] = build_reflection(report, metrics)
    return report


def build_reflection(report: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Prose feedback for the LLM, same pattern as simulate_stair_climb_episode."""
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
        parts.append(f"FAIL {r['gate']}: {r['value']:.3f} {r['op']} {thr_s} {r['unit']}".rstrip() + (f" -> {r['note']}" if r["note"] else ""))
    if len(failed) > 4:
        parts.append(f"(+{len(failed) - 4} more failing gates in the table)")
    skipped = [r for r in report["gates"] if r["pass"] is None]
    if skipped:
        parts.append(f"{len(skipped)} gates not evaluated (stage/group/metric).")
    if not failed and report["gates_total"] > 0:
        parts.append("All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.")
    return " ".join(parts)
