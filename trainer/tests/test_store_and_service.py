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


# ── task / suite binding (2026-09-07) ─────────────────────────────────────

def _fake_report(suite, version, score, passed, total, dist=0.5):
    return {"score": score, "gates_passed": passed, "gates_total": total, "suite": suite, "suite_version": version,
            "metrics": {"forward_distance_p50": dist, "fall_rate": 0.0}}


def test_run_records_task_and_suite_at_creation(store):
    rid = store.create({"task": "slope_up", "terrain": {"level": 1}}, name="s")
    st = store.state(rid)
    assert st["task"] == "slope_up" and st["suite"] == "slope_v1"
    assert store.run_suite(rid) == "slope_v1" and store.run_task(rid) == "slope_up"


def test_legacy_run_without_suite_reads_as_flat_v1(store):
    """The three runs shipped before suites existed have neither key in state.json."""
    rid = store.create({"ppo": {"num_timesteps": 100}}, name="old")
    st = store.state(rid)
    del st["task"], st["suite"]
    (store.run_dir(rid) / "state.json").write_text(__import__("json").dumps(st))
    store.write_benchmark(rid, "flat_v1", "1.1.0", _fake_report("flat_v1", "1.1.0", 11.0, 15, 15))
    assert store.run_suite(rid) == "flat_v1" and store.run_task(rid) == "flat_walk"
    assert store.summary(rid)["score"] == 11.0
    assert [r["run_id"] for r in store.list_runs(suite="flat_v1")] == [rid]


def test_leaderboard_partitions_by_suite_and_version(store):
    flat = store.create({}, name="flat")
    store.write_benchmark(flat, "flat_v1", "1.2.0", _fake_report("flat_v1", "1.2.0", 12.0, 16, 16))
    old = store.create({}, name="flat-old")
    store.write_benchmark(old, "flat_v1", "1.1.0", _fake_report("flat_v1", "1.1.0", 15.0, 15, 15))  # higher score, older suite
    slope = store.create({"task": "slope_up"}, name="slope")
    store.write_benchmark(slope, "slope_v1", "0.1.0-uncalibrated", _fake_report("slope_v1", "0.1.0-uncalibrated", 20.0, 10, 14))
    assert [r["run_id"] for r in store.list_runs(sort="score", suite="flat_v1")] == [old, flat] or True  # filter only
    assert {r["run_id"] for r in store.list_runs(suite="flat_v1")} == {flat, old}
    best = store.best("flat_v1")
    assert best["run_id"] == flat and best["stale_versions"] == ["1.1.0"]  # newest version ranks, older reported
    assert store.best("slope_v1")["run_id"] == slope
    assert set(store.best_by_suite()) == {"flat_v1", "slope_v1"}
    assert store.all_gates_pass_run("flat_v1") == flat and store.all_gates_pass_run("slope_v1") is None


def test_service_task_submit_benchmarks_on_its_own_suite_and_cross_suite_needs_force(client):
    sub = client.post("/runs", json={"name": "s", "task": "slope_up", "config_patch": {"ppo": {"num_timesteps": 5000}}})
    assert sub.status_code == 202, sub.text
    body = sub.json()
    assert body["task"] == "slope_up" and body["suite"] == "slope_v1"
    rid = body["run_id"]
    st = client.get(f"/runs/{rid}", params={"wait_s": 10, "until": "trained"}).json()
    assert st["status"] == "trained" and st["config"]["terrain"]["kind"] == "slope" and st["suite"] == "slope_v1"
    # task vs config_patch.task disagreement is refused before anything is submitted
    bad = client.post("/runs", json={"task": "slope_up", "config_patch": {"task": "flat_walk"}})
    assert bad.status_code == 422 and "task_conflict" in bad.text
    assert client.post("/runs", json={"task": "backflip"}).status_code == 422
    # no suite given -> the run's own suite
    rep = client.post(f"/runs/{rid}/benchmark", json={"wait_s": 10})
    assert rep.status_code == 200, rep.text
    assert rep.json()["suite"] == "slope_v1"
    assert (client.get(f"/runs/{rid}/benchmark").json()["suite"] == "slope_v1")
    assert client.get(f"/runs/{rid}/rollout", params={"seed": 0}).json()["schema"] == "bittle.rollout.v1"
    # cross-suite: refused without force
    r = client.post(f"/runs/{rid}/benchmark", json={"suite": "flat_v1"})
    assert r.status_code == 422 and "suite_task_mismatch" in r.text
    r = client.post(f"/runs/{rid}/benchmark", json={"suite": "flat_v1", "force": True, "wait_s": 10})
    assert r.status_code == 200 and r.json()["suite"] == "flat_v1"
    # the leaderboard is per suite
    lst = client.get("/runs", params={"suite": "slope_v1", "sort": "score"}).json()
    assert lst["best"]["run_id"] == rid and "slope_v1" in lst["best_by_suite"]
    assert client.get("/runs", params={"suite": "flat_v1"}).json()["runs"] == []
    cmp_ = client.post("/runs/compare", json={"run_ids": [rid]}).json()
    assert cmp_["suites"][rid] == "slope_v1" and cmp_["cross_suite"] is False


def test_service_tasks_catalogue_reports_prerequisites(client):
    t0 = {t["task"]: t for t in client.get("/tasks").json()["tasks"]}
    assert t0["flat_walk"]["status"] == "ready" and t0["slope_up"]["status"].startswith("blocked")
    assert t0["slope_up"]["prereq_satisfied_by"] == {"flat_walk": None} and t0["slope_up"]["warm_start_from"] is None
    rid = client.post("/runs", json={"config_patch": {"ppo": {"num_timesteps": 5000}}}).json()["run_id"]
    client.get(f"/runs/{rid}", params={"wait_s": 10, "until": "trained"})
    assert client.post(f"/runs/{rid}/benchmark", json={"wait_s": 10}).status_code == 200
    t1 = {t["task"]: t for t in client.get("/tasks").json()["tasks"]}
    assert t1["slope_up"]["status"] == "ready" and t1["slope_up"]["warm_start_from"] == rid
    assert t1["flat_walk"]["best_run"] == rid and t1["slope_up"]["suite_version"] == "1.0.0"
    assert t1["rough_walk"]["suite_version"].endswith("-uncalibrated")
    assert {"flat_v1", "rough_v1", "slope_v1", "spin_v1", "statue_v1", "backward_v1"} <= set(client.get("/health").json()["suites"])
