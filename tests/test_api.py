"""API-level tests.

The safety unit tests prove the validator is correct in isolation. These prove
the HTTP layer actually ROUTES through it -- a validator nothing calls is
worthless, and that's a mistake unit tests cannot catch.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, controller


@pytest.fixture
def client():
    with TestClient(app) as c:
        controller.safety.clear_estop()
        controller.safety._limiter._tokens = float(controller.safety._limiter.burst)
        yield c
        controller.safety.clear_estop()
        controller.safety._limiter._tokens = float(controller.safety._limiter.burst)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_default_target_is_sim_and_needs_no_confirmation(client):
    r = client.post("/api/move", json={"angles": {"8": -40}})
    assert r.status_code == 200
    assert r.json()["target"] == "sim"


def test_real_target_is_refused_when_hardware_disabled(client):
    """Default deployment must not be able to reach hardware at all."""
    r = client.post("/api/move", json={"angles": {"8": -40}, "target": "real"})
    assert r.status_code == 403
    assert r.json()["error"] == "target_unavailable"


def test_out_of_range_angle_is_reported_as_clamped_not_silently_applied(client):
    r = client.post("/api/move", json={"angles": {"12": 200}})
    body = r.json()
    assert body["clamped"] is True
    assert body["applied"]["12"] == 127
    assert body["adjustments"][0]["reason"] == "wire_range"


def test_unknown_joint_returns_structured_safety_error(client):
    r = client.post("/api/move", json={"angles": {"99": 0}})
    assert r.status_code == 400
    assert r.json()["reason"] == "unknown_joint"


def test_unknown_skill_returns_structured_safety_error(client):
    r = client.post("/api/skill", json={"skill": "moonwalk"})
    assert r.status_code == 400
    assert r.json()["reason"] == "unknown_skill"


def test_gait_requires_ack_then_succeeds(client):
    blocked = client.post("/api/skill", json={"skill": "wkF"})
    assert blocked.status_code == 400
    assert blocked.json()["reason"] == "locomotion_unacked"

    allowed = client.post("/api/skill", json={"skill": "wkF", "ack_locomotion": True})
    assert allowed.status_code == 200
    assert allowed.json()["token"] == "kwkF"


def test_estop_blocks_subsequent_motion_with_409(client):
    assert client.post("/api/estop", json={"reason": "test"}).status_code == 200
    blocked = client.post("/api/move", json={"angles": {"8": 0}})
    assert blocked.status_code == 409
    assert blocked.json()["reason"] == "estop_engaged"

    client.post("/api/estop/clear")
    assert client.post("/api/move", json={"angles": {"8": 0}}).status_code == 200


def test_estop_works_even_when_rate_limit_is_exhausted(client):
    """An E-stop that can be rate-limited is not an E-stop."""
    for _ in range(controller.safety._limiter.burst + 5):
        client.post("/api/move", json={"angles": {"8": 0}})
    r = client.post("/api/estop", json={"reason": "flooded"})
    assert r.status_code == 200
    assert r.json()["estop"]["engaged"] is True


def test_preview_does_not_actuate(client):
    before = client.get("/api/joints/state").json()["angles"]
    r = client.post("/api/preview", json={"angles": {"8": 30}, "simultaneous": True})
    assert r.json()["wire"] == "i8 30 \n"
    after = client.get("/api/joints/state").json()["angles"]
    assert before == after


def test_joint_metadata_exposes_wire_clipping(client):
    body = client.get("/api/joints").json()
    knee = next(j for j in body["joints"] if j["index"] == 12)
    assert knee["firmware_max"] == 200
    assert knee["max"] == 127
    assert knee["wire_clipped"] is True


def test_unused_joints_are_not_exposed_as_controllable(client):
    indices = {j["index"] for j in client.get("/api/joints").json()["joints"]}
    assert indices.isdisjoint({4, 5, 6, 7})


def test_skills_list_marks_locomoting_gaits(client):
    skills = {s["name"]: s for s in client.get("/api/skills").json()["skills"]}
    assert skills["wkF"]["locomotes"] is True
    assert skills["sit"]["locomotes"] is False


def test_api_profiles_endpoint(client):
    r = client.get("/api/profiles")
    assert r.status_code == 200
    body = r.json()
    assert "active_profile" in body
    assert len(body["profiles"]) >= 3

    # Switch profile
    r_sel = client.post("/api/profiles/select", json={"profile_id": "bittle-tail-installed-nyboard-v1-p1s"})
    assert r_sel.status_code == 200
    assert 1 in r_sel.json()["installed_joints"]

    # Switch back
    r_back = client.post("/api/profiles/select", json={"profile_id": "bittle-standard-biboard-v1-p1s"})
    assert r_back.status_code == 200
    assert 1 not in r_back.json()["installed_joints"]


def test_api_controlled_stop(client):
    r = client.post("/api/stop/controlled", json={"reason": "api_test"})
    assert r.status_code == 200
    assert r.json()["controlled_stop"]["controlled_stop"] is True

    # Blocked while stopped
    blocked = client.post("/api/move", json={"angles": {"8": 0}})
    assert blocked.status_code == 409

    # Clear
    client.post("/api/estop/clear")
    unblocked = client.post("/api/move", json={"angles": {"8": 0}})
    assert unblocked.status_code == 200


def test_api_skill_authoring_pipeline(client):
    # 1. Draft
    draft_req = {
        "skill": {
            "name": "api_test_nod",
            "frames": [
                {"angles_deg": {"0": 10}, "speed_deg_per_step": 4, "delay_ms": 50},
                {"angles_deg": {"0": 0}, "speed_deg_per_step": 4, "delay_ms": 50},
            ]
        }
    }
    r_draft = client.post("/api/skills/draft", json=draft_req)
    assert r_draft.status_code == 200
    ir = r_draft.json()["skill_ir"]

    # 2. Validate
    r_val = client.post("/api/skills/validate", json=ir)
    assert r_val.status_code == 200
    body_val = r_val.json()
    assert body_val["ok"] is True
    m_hash = body_val["manifest_hash"]

    # 2.5 Simulate
    r_sim = client.post("/api/skills/simulate", json={"manifest_hash": m_hash})
    assert r_sim.status_code == 200
    assert r_sim.json()["manifest"]["status"] == "simulated"

    # 3. List
    r_list = client.get("/api/skills/manifests")
    assert r_list.status_code == 200
    hashes = [m["payload_hash"] for m in r_list.json()["manifests"]]
    assert m_hash in hashes

    # 4. Approve & Promote
    r_app = client.post("/api/skills/approve", json={"manifest_hash": m_hash, "approver": "api_admin"})
    assert r_app.status_code == 200
    assert r_app.json()["manifest"]["status"] == "approved"

    r_prom = client.post("/api/skills/promote", json={"manifest_hash": m_hash})
    assert r_prom.status_code == 200
    assert r_prom.json()["manifest"]["status"] == "promoted"

    # 5. Run Approved
    r_run = client.post("/api/skills/run-approved", json={"manifest_hash": m_hash, "target": "sim"})
    assert r_run.status_code == 200
    assert r_run.json()["ok"] is True
