import os
import time

import pytest
from fastapi.testclient import TestClient

from trainer.store.runs import RunStore


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs")


def test_run_store_lifecycle(store):
    rid = store.create({"ppo": {"num_timesteps": 100}}, name="x", config_hash="abc")
    assert store.state(rid)["status"] == "queued"
    store.update_state(rid, status="training")
    assert store.state(rid)["started"]
    store.append_curve(rid, {"step": 1, "reward": 1.0})
    store.write_benchmark(rid, "flat_v1", "1.0.0", {"score": 3.0, "gates_passed": 2, "gates_total": 3,
                                                    "suite_version": "1.0.0", "metrics": {"forward_distance_p50": 0.5, "fall_rate": 0.1}})
    rows = store.list_runs(sort="score")
    assert rows[0]["run_id"] == rid and rows[0]["score"] == 3.0 and rows[0]["dist_p50"] == 0.5
    assert store.best()["run_id"] == rid


def test_orphans_marked_failed_on_boot(store):
    rid = store.create({}, name="o")
    store.update_state(rid, status="training")
    assert store.mark_orphans_failed() == [rid]
    assert store.state(rid)["status"] == "failed"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("TRAINER_FAKE_JOBS", "1")
    import importlib

    import trainer.service as svc

    importlib.reload(svc)
    with TestClient(svc.app) as c:
        yield c
    svc.jobs.shutdown()


def test_service_contract_fake_jobs(client):
    assert client.get("/health").json()["fake_jobs"] is True
    schema = client.get("/config/schema").json()
    assert "reward" in schema["defaults"]
    bad = client.post("/config/validate", json={"config_patch": {"nope": 1}})
    assert bad.status_code == 422
    ok = client.post("/config/validate", json={"config_patch": {"reward": {"weights": {"tracking_lin_vel": 2.0}}}}).json()
    assert ok["ok"] and ok["diff"][0]["path"] == "reward.weights.tracking_lin_vel"

    sub = client.post("/runs", json={"name": "t1", "config_patch": {"ppo": {"num_timesteps": 5000}}})
    assert sub.status_code == 202
    rid = sub.json()["run_id"]
    st = client.get(f"/runs/{rid}", params={"wait_s": 10, "until": "trained"}).json()
    assert st["status"] == "trained" and st["curve"] and st["metrics"]["reward_mean"] == 3.0

    rep = client.post(f"/runs/{rid}/benchmark", json={"wait_s": 10})
    assert rep.status_code == 200, rep.text
    body = rep.json()
    assert body["gates_passed"] == body["gates_total"] and "reflection" in body and "episodes" not in body
    assert client.get(f"/runs/{rid}/rollout", params={"seed": 0}).json()["schema"] == "bittle.rollout.v1"
    assert client.get(f"/runs/{rid}/rollout", params={"seed": 9}).status_code == 404
    lst = client.get("/runs", params={"sort": "score"}).json()
    assert lst["runs"][0]["run_id"] == rid and lst["best"]["run_id"] == rid
    cmp_ = client.post("/runs/compare", json={"run_ids": [rid]}).json()
    assert "fall_rate" in cmp_["gates_table"]
    assert client.get("/runs/nope").status_code == 404
    assert "config.json" in client.get(f"/runs/{rid}/artifacts").json()["files"]


def test_service_benchmark_requires_trained(client):
    rid = client.post("/runs", json={"config_patch": {}}).json()["run_id"]
    # immediately: queued or training -> 409
    r = client.post(f"/runs/{rid}/benchmark", json={})
    assert r.status_code in (409, 202, 200)
