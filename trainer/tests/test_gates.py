from trainer.eval.gates import build_reflection, evaluate_gates, load_suite

GOOD = {"fall_rate": 0.0, "forward_distance_p50": 0.9, "vel_tracking_rmse": 0.02, "heading_yaw_deg": 3.0,
        "lateral_drift_m": 0.02, "joint_saturation_pct": 1.0, "action_smoothness_deg": 2.0, "mean_tilt_deg": 3.0,
        "rms_vz": 0.01, "energy_proxy_w": 0.3, "stand_still_drift_m": 0.01, "stand_still_falls": 0,
        "n_episodes": 20, "episode_seconds": 10}


def test_suite_loads_and_is_versioned():
    s = load_suite("flat_v1")
    assert s["version"] == "1.0.0" and len(s["gates"]) >= 12
    names = [g["name"] for g in s["gates"]]
    assert len(names) == len(set(names))


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
