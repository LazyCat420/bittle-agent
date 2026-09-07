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
    s1 = apply_patch(None, {"curriculum_stage": 1, "commands": {"vx": [0.1, 0.3]}})
    s2 = apply_patch(s1, {"curriculum_stage": 2})
    assert tuple(s2.commands.vx) == (0.1, 0.3)       # explicit earlier, kept
    assert tuple(s2.commands.wz) == (-0.5, 0.5)      # stage-1 default carried
    assert tuple(s2.commands.vy) == (-0.05, 0.05) and s2.dr.push_enabled is True
    # a stored (fully explicit) config round-trips through validate without drift
    again = apply_patch(s2.model_dump(mode="json"), {})
    assert again.config_hash() == s2.config_hash()


def test_warm_start_flag_default_and_patchable():
    assert TrainConfig().init_from_parent is True
    assert apply_patch(None, {"init_from_parent": False}).init_from_parent is False
