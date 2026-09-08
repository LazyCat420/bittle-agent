"""trainer/tasks.py — the catalogue must not lie: every suite exists, every config key is real."""

from trainer.config import TrainConfig, apply_patch
from trainer.eval.gates import list_suites, load_suite
from trainer.tasks import TASKS, catalogue, default_suite_for, task_name_for


def _has_path(model: TrainConfig, path: str) -> bool:
    cur = model
    for k in path.split("."):
        if not hasattr(cur, k):
            return False
        cur = getattr(cur, k)
    return True


def test_every_task_names_an_existing_suite_and_real_config_keys():
    cfg = TrainConfig()
    for t in TASKS.values():
        assert t.suite in list_suites(), t.name
        assert load_suite(t.suite)["suite"] == t.suite
        for key in t.config_keys:
            assert _has_path(cfg, key), (t.name, key)
        for p in t.prerequisites:
            assert p in TASKS, (t.name, p)
        # the task's own patch validates and lands on the task
        c = apply_patch(None, t.config_patch)
        assert c.task == t.name


def test_defaults_and_catalogue():
    assert task_name_for({}) == "flat_walk" and default_suite_for("flat_walk") == "flat_v1"
    assert default_suite_for("slope_up") == "slope_v1" and default_suite_for("rough_walk") == "rough_v1"
    cat = catalogue()
    assert [c["task"] for c in cat] == list(TASKS)
    assert all({"goal", "suite", "aliases", "prerequisites", "config_keys"} <= set(c) for c in cat)
    assert "slope" in TASKS["slope_up"].aliases and "rocks" in TASKS["rough_walk"].aliases
