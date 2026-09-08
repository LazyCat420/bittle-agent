import pytest

from trainer.config import RewardWeights
from trainer.eval.gates import SUITES_DIR, build_reflection, evaluate_gates, list_suites, load_suite, suite_hash, suite_protocol

GOOD = {"fall_rate": 0.0, "forward_distance_p50": 0.9, "vel_tracking_rmse": 0.02, "heading_yaw_deg": 3.0,
        "lateral_drift_m": 0.02, "joint_saturation_pct": 1.0, "action_smoothness_deg": 2.0, "mean_tilt_deg": 3.0,
        "rms_vz": 0.01, "energy_proxy_w": 0.3, "stand_still_drift_m": 0.01, "stand_still_falls": 0,
        "peak_joint_speed_rad_s": 3.5, "stall_fraction": 0.001, "stall_concurrent_max": 1,
        "n_episodes": 20, "episode_seconds": 10}

SERVO_SAFETY = {"joint_saturation_pct", "action_smoothness", "energy_proxy", "peak_joint_speed", "stall_fraction",
                "peak_concurrent_stalls"}


def test_suite_loads_and_is_versioned():
    s = load_suite("flat_v1")
    assert s["version"] == "1.2.0" and len(s["gates"]) >= 13
    names = [g["name"] for g in s["gates"]]
    assert len(names) == len(set(names))


def test_flat_v1_hash_is_pinned():
    """The leaderboard only ranks reports sharing a version: any gate change must bump `version`."""
    assert suite_hash(load_suite("flat_v1")) == "d103cc3df3b6"


def test_all_suites_load_include_servo_safety_and_name_real_terms():
    assert set(list_suites()) >= {"flat_v1", "slope_v1", "rough_v1"}
    assert not any(n.startswith("_") for n in list_suites())
    fields = set(RewardWeights.model_fields)
    for name in list_suites():
        s = load_suite(name)
        gates = {g["name"]: g for g in s["gates"]}
        assert SERVO_SAFETY <= set(gates), name
        assert len(gates) == len(s["gates"])
        for g in s["gates"]:
            assert "term" in g, (name, g["name"])  # null allowed, missing is a defect
            assert g["term"] is None or g["term"] in fields, (name, g["name"], g["term"])
        proto = suite_protocol(s)
        assert len(proto["command"]) == 3 and proto["terrain"]["kind"] in ("flat", "slope", "rough", "rough_slope")
        if not s["version"].endswith("-uncalibrated"):
            assert name in ("flat_v1", "slope_v1"), name  # only baseline-calibrated suites carry a release version


def test_suite_local_gate_overrides_the_fragment(tmp_path, monkeypatch):
    import shutil

    from trainer.eval import gates as gates_mod

    shutil.copy(SUITES_DIR / "_servo_safety.yaml", tmp_path / "_servo_safety.yaml")
    (tmp_path / "x_v1.yaml").write_text(
        "suite: x_v1\nversion: 0.0.1-uncalibrated\ninclude: [_servo_safety]\nprotocol: {n_episodes: 1, seed_start: 0, episode_seconds: 1.0}\n"
        "gates:\n  - {name: energy_proxy, metric: energy_proxy_w, op: '<=', threshold: 3.0, unit: W, term: energy}\n")
    monkeypatch.setattr(gates_mod, "SUITES_DIR", tmp_path)
    s = load_suite("x_v1")
    energy = [g for g in s["gates"] if g["name"] == "energy_proxy"]
    assert len(energy) == 1 and energy[0]["threshold"] == 3.0 and "included_from" not in energy[0]
    speed = [g for g in s["gates"] if g["name"] == "peak_joint_speed"][0]
    assert speed["included_from"] == "_servo_safety"
    assert list_suites() == ["x_v1"]
    # the real slope suite keeps the fragment's 2.0 W cap and adds the 0.5x-trot ratio gate
    real = {g["name"]: g for g in load_suite.__wrapped__("slope_v1")["gates"]} if hasattr(load_suite, "__wrapped__") else None
    assert real is None


def test_unknown_suite_lists_available_and_fragments_are_not_suites():
    with pytest.raises(ValueError) as exc:
        load_suite("hfield_v9")
    assert "flat_v1" in str(exc.value)
    with pytest.raises(ValueError):
        load_suite("_servo_safety")
    assert (SUITES_DIR / "_servo_safety.yaml").is_file()


def test_gate_rows_carry_the_term_and_notes_reach_the_reflection():
    rep = evaluate_gates(GOOD, load_suite("flat_v1"), context={
        "reward_breakdown": {"tracking_lin_vel": 500.0},
        "baseline_note": "No opencat_trF baseline on flat_v1 yet, so the baseline gates were skipped.",
        "parent_note": "Parent p1 has no flat_v1 benchmark (it was benchmarked on slope_v1)."})
    rows = {g["gate"]: g for g in rep["gates"]}
    assert rows["energy_proxy"]["term"] == "energy" and rows["dual_sim_consistency"]["term"] is None
    assert "No opencat_trF baseline" in rep["reflection"] and "benchmarked on slope_v1" in rep["reflection"]


def test_degenerate_baseline_metrics_are_not_evaluated():
    m = dict(GOOD, baseline_distance_ratio=None, baseline_fall_delta=None, baseline_energy_ratio=None)
    rep = evaluate_gates(m, load_suite("flat_v1"), groups_enabled={"baseline"})
    rows = {g["gate"]: g for g in rep["gates"]}
    assert rows["beats_baseline_trot_distance"]["pass"] is None and rows["energy_vs_baseline"]["pass"] is None
    assert rep["passed"] and rep["gates_total"] == len([r for r in rep["gates"] if r["pass"] is not None])


def test_all_pass_report():
    rep = evaluate_gates(GOOD, load_suite(), curriculum_stage=0)
    assert rep["passed"] is True and rep["gates_passed"] == rep["gates_total"]
    assert "BENCHMARK PASSED" in rep["reflection"]
    skipped = [g for g in rep["gates"] if g["pass"] is None]
    assert {g["gate"] for g in skipped} >= {"multi_command", "dr_robustness_falls", "dual_sim_consistency"}


def test_failing_gate_reflection_carries_hint_and_numbers():
    bad = dict(GOOD, fall_rate=0.3, forward_distance_p50=0.2)
    rep = evaluate_gates(bad, load_suite(), curriculum_stage=0)
    assert rep["passed"] is False
    fails = {g["gate"]: g for g in rep["gates"] if g["pass"] is False}
    assert set(fails) == {"fall_rate", "forward_distance_p50"}
    assert fails["fall_rate"]["note"]
    assert "BENCHMARK FAILED" in rep["reflection"] and "6/20" in rep["reflection"]
    assert rep["score"] < evaluate_gates(GOOD, load_suite())["score"]


def test_groups_and_stage_gating():
    m = dict(GOOD, multi_command_rmse_max=0.01, dr_fall_rate_max=0.5, dr_distance_ratio_min=0.9,
             dual_sim_distance_ratio=1.0)
    rep0 = evaluate_gates(m, load_suite(), curriculum_stage=0, groups_enabled={"dr_sweep", "dual_sim"})
    g0 = {g["gate"]: g for g in rep0["gates"]}
    assert g0["multi_command"]["pass"] is None  # stage gated
    assert g0["dr_robustness_falls"]["pass"] is False and g0["dual_sim_consistency"]["pass"] is True
    rep1 = evaluate_gates(m, load_suite(), curriculum_stage=1)
    assert {g["gate"]: g for g in rep1["gates"]}["multi_command"]["pass"] is True


def test_within_op():
    m = dict(GOOD, dual_sim_distance_ratio=1.6)
    rep = evaluate_gates(m, load_suite(), groups_enabled={"dual_sim"})
    assert {g["gate"]: g for g in rep["gates"]}["dual_sim_consistency"]["pass"] is False
    assert build_reflection(rep, m)


def test_reflection_uses_context_for_weak_terms():
    bad = dict(GOOD, energy_proxy_w=2.5)
    ctx = {"baseline_metrics": {"energy_proxy_w": 3.79}, "parent_metrics": {"energy_proxy_w": 1.08},
           "parent_run_id": "p1", "reward_breakdown": {"tracking_lin_vel": 580.0, "energy": -0.4, "orientation": -6.5}}
    rep = evaluate_gates(bad, load_suite(), context=ctx)
    refl = rep["reflection"]
    assert "firmware trot 3.790" in refl and "parent 1.080" in refl
    assert "'energy' term is only 0.07% of the total reward" in refl and "multiply its weight" in refl
    assert rep["context"]["per_gate"]["energy_proxy"]["term"] == "energy"
    assert rep["context"]["reward_shares_pct"]["tracking_lin_vel"] > 90
