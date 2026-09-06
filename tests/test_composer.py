import pytest
import tempfile
from pathlib import Path

from app.motion.composer import MovesetComposer
from app.motion.primitives import CompositionError


@pytest.fixture
def temp_composer(tmp_path):
    return MovesetComposer(storage_dir=tmp_path)


def test_composer_list_builtin_primitives(temp_composer):
    prims = temp_composer.list_primitives()
    assert len(prims) >= 15
    head_prims = temp_composer.list_primitives(group="head")
    assert len(head_prims) >= 2


def test_composer_compose_sequential(temp_composer):
    moveset = temp_composer.compose_move(
        name="test_seq",
        primitive_names=["head_scan_left", "head_scan_right"],
        mode="sequential",
        description="Sequential head scans",
    )
    assert moveset["name"] == "test_seq"
    assert len(moveset["frames"]) > 0
    assert "metrics" in moveset
    assert moveset["metrics"]["smoothness"] > 0


def test_composer_compose_parallel_disjoint(temp_composer):
    moveset = temp_composer.compose_move(
        name="test_par",
        primitive_names=["head_scan_left", "rear_wiggle"],
        mode="parallel",
    )
    assert moveset["name"] == "test_par"
    assert len(moveset["frames"]) > 0


def test_composer_compose_parallel_collision(temp_composer):
    with pytest.raises(CompositionError):
        temp_composer.compose_move(
            name="test_collision",
            primitive_names=["head_scan_left", "head_scan_right"],  # Both touch joint 0
            mode="parallel",
        )


def test_composer_iterate_delay_and_speed(temp_composer):
    composed = temp_composer.compose_move(
        name="test_iter",
        primitive_names=["head_scan_left", "head_tilt_left"],
    )
    temp_composer.save_moveset("test_iter", composed)

    initial_delay = composed["frames"][0]["delay_ms"]
    iterated = temp_composer.iterate_move("test_iter", {
        "delay_scale": 1.5,
        "speed_scale": 0.5,
    })
    new_delay = iterated["frames"][0]["delay_ms"]
    assert new_delay == int(initial_delay * 1.5)


def test_composer_iterate_angle_offsets(temp_composer):
    composed = temp_composer.compose_move(
        name="test_offsets",
        primitive_names=["head_scan_left"],
    )
    temp_composer.save_moveset("test_offsets", composed)

    orig_head_angle = composed["frames"][0]["angles"][0]
    iterated = temp_composer.iterate_move("test_offsets", {
        "angle_offsets": {"0": 10},
    })
    new_head_angle = iterated["frames"][0]["angles"][0]
    assert new_head_angle == orig_head_angle + 10


def test_composer_iterate_swap_primitive(temp_composer):
    composed = temp_composer.compose_move(
        name="test_swap",
        primitive_names=["head_scan_left", "front_bow"],
    )
    temp_composer.save_moveset("test_swap", composed)

    swapped = temp_composer.iterate_move("test_swap", {
        "swap_primitive": {"old": "head_scan_left", "new": "head_scan_right"},
    })
    assert swapped["source_primitives"] == ["head_scan_right", "front_bow"]


def test_composer_version_history(temp_composer):
    composed = temp_composer.compose_move(
        name="test_versioned",
        primitive_names=["head_scan_left"],
    )
    saved_v1 = temp_composer.save_moveset("test_versioned", composed)
    assert len(saved_v1["version_history"]) == 1

    saved_v2 = temp_composer.save_moveset("test_versioned", saved_v1)
    assert len(saved_v2["version_history"]) == 2


def test_composer_evaluate_move(temp_composer):
    composed = temp_composer.compose_move(
        name="test_eval",
        primitive_names=["head_scan_left", "front_bow"],
    )
    metrics = temp_composer.evaluate_move(frames=composed["frames"])
    assert "smoothness" in metrics
    assert "total_duration_ms" in metrics
    assert metrics["frame_count"] == len(composed["frames"])
