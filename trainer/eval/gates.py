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
    return {"per_gate": per_gate, "reward_breakdown": rb, "reward_shares_pct": shares,
            "parent_run_id": context.get("parent_run_id"), "baseline": context.get("baseline_name"),
            "baseline_note": context.get("baseline_note"), "parent_note": context.get("parent_note")}


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
        if share is not None and pg.get("term") and share < 1.0:
            factor = max(2, int(round(2.0 / max(share, 1e-6))))
            line += (f". The '{pg['term']}' term is only {share:.2f}% of the total reward, so the optimiser barely sees it: "
                     f"multiply its weight by ~{min(factor, 1000)}x (to ~2% share), not by 2-10x.")
        parts.append(line)
    if len(failed) > 4:
        parts.append(f"(+{len(failed) - 4} more failing gates in the table)")
    for key in ("parent_note", "baseline_note"):
        if ctx.get(key):
            parts.append(str(ctx[key]))
    skipped = [r for r in report["gates"] if r["pass"] is None]
    if skipped:
        parts.append(f"{len(skipped)} gates not evaluated (stage/group/metric).")
    if not failed and report["gates_total"] > 0:
        parts.append("All evaluated gates pass. Consider raising curriculum_stage or requesting dr_sweep.")
    return " ".join(parts)
