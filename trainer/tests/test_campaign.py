"""trainer/campaign/driver.py — warm starts resolve from the store when this campaign has no champion yet."""

import asyncio
import json

from trainer.campaign.driver import run_campaign
from trainer.campaign import strategies


class FakeAgent:
    def __init__(self):
        self.calls = []
        self.n = 0

    async def call(self, tool, args):
        self.calls.append((tool, args))
        if tool == "bittle_list_runs":
            if args["suite"] == "rough_v1":
                return {"runs": [{"run_id": "rough-champ", "score": 10.2}]}
            return {"runs": []}
        self.n += 1
        return {"ok": True, "run_id": f"run-{self.n}", "score": 5.0 + self.n, "suite": "house_v1", "gates_passed": 3,
                "gates_total": 10, "passed": False, "reflection": "x"}


def test_best_of_falls_back_to_the_store_leaderboard(tmp_path, monkeypatch):
    monkeypatch.setitem(strategies.STRATEGIES, "t", [
        {"name": "a", "task": "house_walk", "base": "best_of:rough_walk", "patch": {}, "notes": ""},
        {"name": "b", "task": "house_walk", "base": "best", "patch": {}, "notes": ""},
        {"name": "c", "task": "house_walk", "base": None, "patch": {}, "notes": ""},
    ])
    agent = FakeAgent()
    log = asyncio.run(run_campaign(agent, strategy="t", out=tmp_path / "c.json"))
    trains = [a for t, a in agent.calls if t == "bittle_train_and_benchmark"]
    assert trains[0]["base_run_id"] == "rough-champ"          # from the store: no house run exists yet
    assert trains[1]["base_run_id"] == "run-1"                 # from this campaign: step a was accepted on house_v1
    assert "base_run_id" not in trains[2]                      # base None = from scratch
    assert [t for t, _ in agent.calls if t == "bittle_list_runs"] == ["bittle_list_runs"]  # only the first lookup hit the store
    assert json.loads((tmp_path / "c.json").read_text())["best_by_suite"]["house_v1"]["run_id"] == "run-3"  # every step scored higher than the last


def test_best_with_no_run_anywhere_is_a_cold_start(tmp_path, monkeypatch):
    monkeypatch.setitem(strategies.STRATEGIES, "t2", [
        {"name": "a", "task": "spin", "base": "best", "patch": {}, "notes": ""}])
    agent = FakeAgent()
    asyncio.run(run_campaign(agent, strategy="t2", out=tmp_path / "c.json"))
    assert "base_run_id" not in [a for t, a in agent.calls if t == "bittle_train_and_benchmark"][0]
