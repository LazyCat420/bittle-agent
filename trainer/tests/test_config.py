import pytest
from pydantic import ValidationError

from trainer.config import TrainConfig, apply_patch, config_diff, default_config, validation_errors


def test_defaults_resolve_and_hash_is_stable():
    a, b = TrainConfig(), TrainConfig()
    assert a.config_hash() == b.config_hash()
    assert a.episode_steps == 500 and a.control_dt == 0.02


def test_unknown_key_rejected_anywhere():
    with pytest.raises(ValidationError) as exc:
        apply_patch(None, {"reward": {"weights": {"tracking_lin_vell": 1.0}}})
    assert any("tracking_lin_vell" in e for e in validation_errors(exc.value))
    with pytest.raises(ValidationError):
        apply_patch(None, {"bogus": 1})


def test_bounds_enforced():
    with pytest.raises(ValidationError):
        apply_patch(None, {"ppo": {"num_timesteps": 10 ** 12}})
    with pytest.raises(ValidationError):
        apply_patch(None, {"reward": {"weights": {"orientation": 3.0}}})  # penalties must be <= 0
    with pytest.raises(ValidationError):
        apply_patch(None, {"commands": {"vx": [0.3, 0.1]}})  # lo > hi


def test_patch_merges_deeply_and_diff_lists_leaves():
    base = default_config()
    cfg = apply_patch(base, {"reward": {"weights": {"tracking_lin_vel": 2.5}}, "curriculum_stage": 1})
    assert cfg.reward.weights.tracking_lin_vel == 2.5
    assert cfg.reward.weights.orientation == base.reward.weights.orientation
    d = config_diff(base.resolved().model_dump(mode="json"), cfg.resolved().model_dump(mode="json"))
    paths = {x["path"] for x in d}
    assert "reward.weights.tracking_lin_vel" in paths and "curriculum_stage" in paths
    # stage 1 widened the command ranges the caller did not set explicitly
    assert "commands.wz" in paths


def test_curriculum_stage_defaults_do_not_override_explicit_values():
    cfg = apply_patch(None, {"curriculum_stage": 1, "commands": {"wz": [0.0, 0.0]}})
    r = cfg.resolved()
    assert tuple(r.commands.wz) == (0.0, 0.0)
    assert tuple(r.commands.vx) == (0.0, 0.25)
    assert apply_patch(None, {"curriculum_stage": 2}).resolved().dr.push_enabled is True


def test_stage_bump_on_a_base_keeps_explicit_and_moves_defaults():
    s1 = apply_patch(None, {"curriculum_stage": 1, "commands": {"vx": [0.1, 0.22]}})
    s2 = apply_patch(s1, {"curriculum_stage": 2})
    assert tuple(s2.commands.vx) == (0.1, 0.22)      # explicit earlier, kept
    assert tuple(s2.commands.wz) == (-0.5, 0.5)      # stage-1 default carried
    assert tuple(s2.commands.vy) == (-0.05, 0.05) and s2.dr.push_enabled is True
    # a stored (fully explicit) config round-trips through validate without drift
    again = apply_patch(s2.model_dump(mode="json"), {})
    assert again.config_hash() == s2.config_hash()


def test_warm_start_flag_default_and_patchable():
    assert TrainConfig().init_from_parent is True
    assert apply_patch(None, {"init_from_parent": False}).init_from_parent is False


def test_terrain_bounds_and_unknown_keys_rejected():
    for bad in ({"terrain": {"slope_deg": [0.0, 25.0]}}, {"terrain": {"box_height_m": [0.0, 0.05]}},
                {"terrain": {"n_boxes": 40}}, {"terrain": {"kind": "hfield"}}, {"terrain": {"rocks": 3}},
                {"reward": {"weights": {"stumble": 0.5}}}, {"reward": {"weights": {"slope_progress": -1.0}}},
                {"commands": {"vx": [0.0, 0.4]}}, {"task": "Slope Up"}):
        with pytest.raises(ValidationError):
            apply_patch(None, bad)


def test_terrain_level_defaults_are_a_second_curriculum_axis():
    c1 = apply_patch(None, {"terrain": {"level": 1}})
    assert c1.terrain.kind == "slope" and tuple(c1.terrain.slope_deg) == (0.0, 8.0) and c1.terrain.n_boxes == 0
    assert c1.curriculum_stage == 0 and tuple(c1.commands.vx) == (0.05, 0.20)  # the command axis is untouched
    c2 = apply_patch(c1, {"terrain": {"level": 2}})
    assert c2.terrain.kind == "rough" and c2.terrain.n_boxes == 24 and tuple(c2.terrain.slope_deg) == (0.0, 0.0)
    # explicit wins over the level default, and survives the next level bump
    c3 = apply_patch(None, {"terrain": {"level": 2, "n_boxes": 8}})
    assert c3.terrain.n_boxes == 8
    c4 = apply_patch(c3, {"terrain": {"level": 3}})
    assert c4.terrain.n_boxes == 8 and c4.terrain.kind == "rough_slope"
    # both axes in one patch
    c5 = apply_patch(None, {"curriculum_stage": 1, "terrain": {"level": 1}})
    assert tuple(c5.commands.wz) == (-0.5, 0.5) and c5.terrain.kind == "slope"


def test_pre_terrain_configs_resolve_unchanged():
    """A stored config with no terrain/task keys (the three shipped runs) still resolves to flat_walk."""
    old = {"curriculum_stage": 0, "reward": {"weights": {"energy": -0.05}}, "ppo": {"num_timesteps": 10000000}}
    cfg = apply_patch(old, {})
    assert cfg.task == "flat_walk" and cfg.terrain.kind == "flat" and cfg.terrain.level == 0
    assert cfg.reward.weights.stumble == 0.0 and cfg.reward.weights.foot_clearance == 0.0
    assert cfg.reward.weights.stall == 0.0 and cfg.reward.weights.slope_progress == 0.0


def test_house_terrain_level_and_the_way_back_down():
    c4 = apply_patch(None, {"terrain": {"level": 4}})
    assert c4.terrain.kind == "rough_slope" and c4.terrain.slope_share == 0.6 and c4.terrain.box_share == 0.6
    assert tuple(c4.terrain.slope_yaw_deg) == (-180.0, 180.0) and c4.terrain.n_boxes == 28 and c4.terrain.field_start_m == -1.0
    # stepping back to a single-kind level restores every share and the yaw to their full-field values
    c2 = apply_patch(c4, {"terrain": {"level": 2}})
    assert c2.terrain.kind == "rough" and c2.terrain.slope_share == 1.0 and c2.terrain.box_share == 1.0
    assert tuple(c2.terrain.slope_yaw_deg) == (0.0, 0.0)
    # explicit shares survive a level change
    c = apply_patch(apply_patch(None, {"terrain": {"level": 4, "box_share": 0.9}}), {"terrain": {"level": 3}})
    assert c.terrain.box_share == 0.9 and c.terrain.slope_share == 1.0
    with pytest.raises(ValidationError):
        apply_patch(None, {"terrain": {"level": 7}})
    with pytest.raises(ValidationError):
        apply_patch(None, {"terrain": {"box_share": 1.5}})
    # the house task's own patch resolves to level 4 + the full command box + pushes
    from trainer.tasks import TASKS

    h = apply_patch(None, TASKS["house_walk"].config_patch)
    assert h.terrain.level == 4 and h.curriculum_stage == 2 and h.dr.push_enabled and h.commands.vx[0] < 0 < h.commands.wz[1]


def test_level_2_covers_the_rough_v1_bar():
    """The training field must be at least as hard as the exam: every rough_v1 protocol value lies inside (or at
    the edge of) the level-2 sampling range, and a level change back to flat restores the spacing."""
    from trainer.eval.gates import load_suite

    proto = load_suite("rough_v1")["protocol"]["terrain"]
    c2 = apply_patch(None, {"terrain": {"level": 2}})
    lo, hi = c2.terrain.box_height_m
    assert lo <= proto["box_height_m"] <= hi and hi > proto["box_height_m"], "training rocks never reach the 12 mm bar"
    assert c2.terrain.n_boxes >= proto["n_boxes"]
    assert c2.terrain.box_spacing_m <= proto["box_spacing_m"]
    assert c2.terrain.box_size_m[0] <= proto["box_size_m"] <= c2.terrain.box_size_m[1]
    c0 = apply_patch(c2, {"terrain": {"level": 0}})
    assert c0.terrain.box_spacing_m == 0.12 and c0.terrain.n_boxes == 0


def test_stairs_and_rubble_levels_and_their_bounds():
    s = apply_patch(None, {"terrain": {"level": 5}})
    assert s.terrain.kind == "stairs" and s.terrain.n_boxes == 0 and s.terrain.stair_steps == (1, 4)
    assert s.terrain.stair_rise_m == (0.006, 0.022) and s.terrain.stair_profile == "up_down"
    r = apply_patch(None, {"terrain": {"level": 6}})
    assert r.terrain.kind == "rough" and r.terrain.n_boxes == 32 and r.terrain.box_height_m == (0.010, 0.025)
    assert r.terrain.box_yaw_deg == (-90.0, 90.0) and r.terrain.box_spacing_m == 0.07
    # stepping from stairs back to rocks resets every stair knob; from rubble to rough resets the yaw and width
    back = apply_patch(s, {"terrain": {"level": 2}})
    assert back.terrain.kind == "rough" and back.terrain.stair_steps == (3, 3) and back.terrain.stair_rise_m == (0.012, 0.012)
    back2 = apply_patch(r, {"terrain": {"level": 2}})
    assert back2.terrain.box_yaw_deg == (-45.0, 45.0) and back2.terrain.field_width_m == 0.40 and back2.terrain.n_boxes == 24
    # an explicit stair knob survives a level change
    keep = apply_patch(apply_patch(None, {"terrain": {"level": 5, "stair_rise_m": [0.02, 0.03]}}), {"terrain": {"level": 5}})
    assert keep.terrain.stair_rise_m == (0.02, 0.03)
    for bad in ({"stair_steps": [0, 3]}, {"stair_steps": [2, 7]}, {"stair_steps": [4, 2]},
                {"stair_rise_m": [0.0, 0.04]}, {"stair_tread_m": [0.02, 0.08]}, {"stair_profile": "sideways"},
                {"stair_width_m": 0.9}, {"stair_landing_m": 0.5}):
        with pytest.raises(ValidationError):
            apply_patch(None, {"terrain": {"level": 5, **bad}})
