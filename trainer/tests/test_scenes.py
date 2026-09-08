"""trainer/eval/scenes.py — one benchmark loop for single-protocol and scenes suites."""

import pytest

mujoco = pytest.importorskip("mujoco")

from trainer.config import TrainConfig  # noqa: E402
from trainer.eval.evaluator import StandController  # noqa: E402
from trainer.eval.gates import load_suite, suite_scenes  # noqa: E402
from trainer.eval.scenes import WORST_OVER_SCENES, run_suite_scenes  # noqa: E402


def _short(suite: dict, seconds: float = 0.6) -> dict:
    s = dict(suite)
    if "scenes" in s:
        s["scenes"] = [dict(sc, episode_seconds=seconds) for sc in s["scenes"]]
    else:
        s["protocol"] = dict(s["protocol"], episode_seconds=seconds)
    return s


@pytest.mark.slow
def test_house_scenes_publish_per_scene_and_pooled_metrics():
    suite = _short(load_suite("house_v1"))
    metrics, stats, rollouts, proto = run_suite_scenes(TrainConfig(), StandController(), suite, n_episodes=1, record_n=1,
                                                       source={"t": 1}, stand_still=True)
    names = [n for n, _ in suite_scenes(suite)]
    assert metrics["n_scenes"] == 9 and metrics["scenes"] == names
    assert proto["scene"] == "flat" and proto["scenes"] == names and proto["n_episodes"] == 1
    for n in names:
        assert f"{n}/fall_rate" in metrics and f"{n}/stand_still_falls" in metrics and f"{n}/progress_ratio" in metrics
    assert {s.scene for s in stats} == set(names) and len(stats) == 9
    assert sorted(rollouts) == [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    assert all(rollouts[k]["source"]["scene"] for k in rollouts)
    # pooled = the forward scenes only (seven of nine: turn walks forward too), worst-of for servo safety and falls
    fwd = [s for s in stats if s.cmd[0] > 0]
    assert len(fwd) == 7 and metrics["n_episodes"] == 7
    for k in WORST_OVER_SCENES:
        assert metrics[k] == max(metrics[f"{n}/{k}"] for n in names), k
    assert metrics["fall_rate_max"] == max(metrics[f"{n}/fall_rate"] for n in names)
    assert metrics["stand_still_falls"] == sum(metrics[f"{n}/stand_still_falls"] for n in names)
    assert "progress_ratio_min" in metrics and "stand_still_drift_m" in metrics
    # the stand controller on a 0.6 s episode moves nowhere and falls nowhere
    assert metrics["fall_rate_max"] == 0.0


def test_classic_suite_keeps_bare_metric_names():
    suite = _short(load_suite("flat_v1"))
    metrics, stats, rollouts, proto = run_suite_scenes(TrainConfig(), StandController(), suite, n_episodes=2, stand_still=False)
    assert "fall_rate" in metrics and not any("/" in k for k in metrics) and "n_scenes" not in metrics
    assert proto.get("scene") is None and len(stats) == 2 and stats[0].scene == "" and rollouts == {}
