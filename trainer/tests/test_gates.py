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
            assert name in ("flat_v1", "slope_v1", "statue_v1", "backward_v1"), name  # only baseline-calibrated suites carry a release version


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


def _aggregate_keys() -> set[str]:
    from trainer.eval.evaluator import EpisodeStats, aggregate

    st = EpisodeStats(seed=0, seconds=1.0, steps=50, cmd=[0.1, 0.0, 0.0], distance_x=0.1, lateral_y=0.0, yaw_deg=0.0,
                      fell=False, fall_time=None, vel_rmse=0.0, saturation_pct=0.0, smoothness_deg=0.0, tilt_deg=0.0,
                      rms_vz=0.0, energy_w=0.0, mean_reward=0.0)
    return set(aggregate([st], 1.0)) | {"stand_still_drift_m", "stand_still_falls"}


def test_house_v1_scenes_are_distinct_and_every_gate_names_a_metric_that_exists():
    """A scenes suite must not lie either: every gate reads a top-level aggregate, a benchmark extra,
    or <scene>/<aggregate> for a scene that is actually in the suite."""
    from trainer.eval.gates import suite_scenes

    s = load_suite("house_v1")
    scenes = suite_scenes(s)
    names = [n for n, _ in scenes]
    assert len(names) == 9 and len(set(names)) == 9 and names[0] == "flat"
    seeds = [p["seed_start"] for _, p in scenes]
    assert len(set(seeds)) == 9 and min(abs(a - b) for a in seeds for b in seeds if a != b) >= 400  # sub-protocols use +100..+300
    for _, p in scenes:
        assert len(p["command"]) == 3 and p["n_episodes"] == 12 and p["terrain"]["kind"] in ("flat", "slope", "rough", "rough_slope")
    primary = suite_protocol(s)
    assert primary["scene"] == "flat" and primary["scenes"] == names and primary["terrain"]["kind"] == "flat"
    top = _aggregate_keys() | {"fall_rate_max", "progress_ratio_min", "baseline_distance_ratio", "baseline_fall_delta",
                               "baseline_energy_ratio", "dr_fall_rate_max", "dr_distance_ratio_min", "multi_command_rmse_max",
                               "dual_sim_distance_ratio"}
    for g in s["gates"]:
        m = g["metric"]
        if "/" in m:
            scene, key = m.split("/", 1)
            assert scene in names and key in _aggregate_keys(), g["name"]
        else:
            assert m in top, g["name"]
    # classic suites are one unnamed scene and keep their bare protocol
    assert suite_scenes(load_suite("flat_v1")) == [("", suite_protocol(load_suite("flat_v1")))]


def test_scene_gates_evaluate_against_prefixed_metrics():
    s = load_suite("house_v1")
    metrics = dict(GOOD, fall_rate_max=0.0, progress_ratio_min=0.8, stand_still_drift_m=0.01)
    for name in ("flat", "slope_up", "slope_down", "rough", "rough_slope", "pushed", "turn", "backward_rough", "statue_rough"):
        metrics.update({f"{name}/fall_rate": 0.0, f"{name}/forward_distance_p50": 0.9 if name != "backward_rough" else -0.9,
                        f"{name}/progress_ratio": 0.8, f"{name}/ang_vel_rmse": 0.1, f"{name}/centre_drift_m": 0.02})
    rep = evaluate_gates(metrics, s, curriculum_stage=2)
    assert rep["passed"] and rep["gates_total"] == len(s["gates"]) - 1  # the baseline group is not requested
    metrics["rough/fall_rate"] = 0.5
    rep = evaluate_gates(metrics, s, curriculum_stage=2)
    failed = {g["gate"] for g in rep["gates"] if g["pass"] is False}
    assert failed == {"rough_falls"}


def test_scene_names_must_be_unique_identifiers(tmp_path, monkeypatch):
    import shutil

    from trainer.eval import gates as gates_mod

    shutil.copy(SUITES_DIR / "_servo_safety.yaml", tmp_path / "_servo_safety.yaml")
    (tmp_path / "y_v1.yaml").write_text(
        "suite: y_v1\nversion: 0.0.1-uncalibrated\ninclude: [_servo_safety]\nprotocol: {n_episodes: 1, episode_seconds: 1.0}\n"
        "scenes:\n  - {name: a, seed_start: 0, command: [0.1, 0, 0]}\n  - {name: a, seed_start: 1000, command: [0.1, 0, 0]}\ngates: []\n")
    monkeypatch.setattr(gates_mod, "SUITES_DIR", tmp_path)
    with pytest.raises(ValueError):
        gates_mod.suite_scenes(load_suite("y_v1"))


# ── 2026-09-10: the reflection tells "easier classroom than exam" from "the policy is stuck" ──

def _rough_report(metrics_over: dict, ctx: dict):
    from trainer.eval.gates import evaluate_gates, load_suite

    base = {"fall_rate": 0.0, "forward_distance_p50": 0.31, "progress_ratio": 0.26, "vel_tracking_rmse": 0.12,
            "stumble_rate": 0.0, "foot_clearance_p50_mm": 1.2, "body_clearance_min_mm": 30.0, "lateral_drift_m": 0.01,
            "joint_saturation_pct": 0.1, "action_smoothness_deg": 1.1, "energy_proxy_w": 1.1, "peak_joint_speed_rad_s": 3.3,
            "stall_fraction": 0.002, "stall_concurrent_max": 2, "n_episodes": 20, "episode_seconds": 10.0}
    base.update(metrics_over)
    return evaluate_gates(base, load_suite("rough_v1"), context=ctx)


def test_reflection_names_the_train_bench_mismatch_and_the_terrain_gap():
    ctx = {"reward_breakdown": {"tracking_lin_vel": 700.0, "foot_clearance": -1.2},
           "reward_weights": {"foot_clearance": -10.0, "tracking_lin_vel": 1.5},
           "train_distance_x": 0.90, "train_bar_distance_x": 0.30, "train_bar_fall_rate": 0.0,
           "terrain_gap": {"train_kind": "rough", "protocol_kind": "rough", "protocol_box_height_m": 0.012,
                           "train_box_height_m": [0.003, 0.012], "protocol_n_boxes": 24, "train_n_boxes": 20,
                           "protocol_box_spacing_m": 0.10, "train_box_spacing_m": 0.12,
                           "train_share_at_or_above_protocol_height": 0.0, "easier_than_protocol": True}}
    refl = _rough_report({}, ctx)["reflection"]
    assert "TRAIN/BENCH MISMATCH" in refl and "0.90 m" in refl and "0.31 m" in refl
    assert "20 boxes of 3-12 mm" in refl and "raise terrain.box_height_m" in refl
    assert "Bar eval" in refl and "the engines agree" in refl
    # the terrain advice comes BEFORE any weight advice would be acted on: it is in the same reflection
    assert refl.index("TRAIN/BENCH MISMATCH") > refl.index("FAIL forward_distance_p50")


def test_reflection_says_a_zero_weight_term_is_off_not_weak():
    ctx = {"reward_breakdown": {"tracking_lin_vel": 700.0}, "reward_weights": {"foot_clearance": 0.0, "stumble": 0.0}}
    rep = _rough_report({}, ctx)
    refl = rep["reflection"]
    assert "'foot_clearance' term is switched OFF" in refl and "reward.weights.foot_clearance = 0.0" in refl
    assert "multiply its weight" not in refl.split("switched OFF")[1].split("FAIL")[0]


def test_reflection_calls_a_flat_gate_with_a_heavy_term_a_budget_problem():
    ctx = {"reward_breakdown": {"tracking_lin_vel": 700.0, "foot_clearance": -40.0},
           "reward_weights": {"foot_clearance": -2.0}, "parent_run_id": "p",
           "parent_metrics": {"foot_clearance_p50_mm": 5.1}}
    # a 5 mm swing is a swing (the dragging rule needs < 2 mm); it just did not move vs the parent
    refl = _rough_report({"foot_clearance_p50_mm": 5.0}, ctx)["reflection"]
    assert "already carries 5.4% of the reward" in refl and "ppo.num_timesteps 30M" in refl


def test_reflection_bar_eval_flags_an_engine_disagreement():
    ctx = {"reward_breakdown": {"tracking_lin_vel": 700.0}, "reward_weights": {},
           "train_distance_x": 0.9, "train_bar_distance_x": 0.85,
           "terrain_gap": {"train_kind": "rough", "protocol_kind": "rough", "easier_than_protocol": False}}
    refl = _rough_report({}, ctx)["reflection"]
    assert "the GPU and CPU engines disagree" in refl and "check dual_sim" in refl
    assert "the training terrain covers the protocol" in refl


def test_reflection_reports_where_it_got_stuck_and_whether_the_legs_caught():
    ctx = {"reward_breakdown": {"tracking_lin_vel": 700.0}, "reward_weights": {}}
    m = {"stuck_episode_rate": 0.8, "stuck_seconds_p50": 6.5, "stuck_seconds_max": 9.0, "stuck_x_p50": 0.31, "stuck_limb_share": 0.05}
    refl = _rough_report(m, ctx)["reflection"]
    assert "STUCK: 80% of episodes stalled" in refl and "first stall at x = 0.31 m" in refl and "NOT catching edges" in refl
    refl2 = _rough_report(dict(m, stuck_limb_share=0.6), ctx)["reflection"]
    assert "the legs catch the edges" in refl2


def test_reflection_names_dragging_when_a_heavy_clearance_term_still_leaves_a_sub_2mm_swing():
    """r11 (2026-09-10): foot_clearance at 17% share, swing 0.66 mm -- the policy stopped swinging; the swing
    rewards (feet_air_time, feet_slip) were < 0.2%. The advice must move to THOSE terms, not the clearance weight."""
    ctx = {"reward_breakdown": {"tracking_lin_vel": 400.0, "foot_clearance": -150.0, "feet_air_time": 1.0, "feet_slip": -0.2},
           "reward_weights": {"foot_clearance": -2.0, "feet_air_time": 0.3, "feet_slip": -0.05}}
    refl = _rough_report({"foot_clearance_p50_mm": 0.66}, ctx)["reflection"]
    assert "DRAGGING its feet" in refl and "raise reward.weights.feet_air_time and reward.weights.feet_slip" in refl
    assert "feet_air_time 0.18%" in refl


def test_reflection_stops_the_swing_loop_when_the_swing_weights_are_at_their_bounds():
    """r13 (GLM cycle, 2026-09-10): feet_air_time 10 and feet_slip -10 (the bounds), swing 0.57 mm. Telling the
    LLM to raise them again would be a loop; the reflection must say the gate needs a design change."""
    ctx = {"reward_breakdown": {"tracking_lin_vel": 400.0, "foot_clearance": -100.0, "feet_air_time": 45.0, "feet_slip": -22.0},
           "reward_weights": {"foot_clearance": -2.0, "feet_air_time": 10.0, "feet_slip": -10.0}}
    refl = _rough_report({"foot_clearance_p50_mm": 0.57}, ctx)["reflection"]
    assert "AT THEIR BOUNDS" in refl and "STOP spending cycles on this gate" in refl
    assert "raise reward.weights.feet_air_time" not in refl


def test_best_episode_seed_prefers_the_longest_non_fallen_primary_episode():
    from trainer.eval.render import best_episode_seed

    rep = {"protocol": {"scene": ""}, "episodes": [
        {"seed": 0, "distance_x": 0.9, "fell": True, "scene": ""},
        {"seed": 1, "distance_x": 0.4, "fell": False, "scene": ""},
        {"seed": 2, "distance_x": 0.6, "fell": False, "scene": ""},
        {"seed": 1000, "distance_x": 1.5, "fell": False, "scene": "other"}]}
    assert best_episode_seed(rep) == 2
    assert best_episode_seed({"episodes": []}) is None
    assert best_episode_seed({"episodes": [{"seed": 7, "distance_x": 0.1, "fell": True}]}) == 7
