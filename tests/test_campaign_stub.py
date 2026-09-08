"""The scripted campaign driver runs the same tool path as the LLM; here with a stub agent."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from trainer.campaign.driver import run_campaign  # noqa: E402
from trainer.campaign.strategies import STRATEGIES  # noqa: E402


class StubAgent:
    """Scores improve for the first two cycles, then regress; the driver must keep the best."""

    def __init__(self):
        self.calls = []
        self.scores = iter([5.0, 7.0, 6.0, 9.5])

    async def call(self, tool, args):
        assert tool == "bittle_train_and_benchmark"
        self.calls.append(args)
        s = next(self.scores)
        suite = {"flat_walk": "flat_v1", "slope_up": "slope_v1", "rough_walk": "rough_v1"}[args["task"]]
        return {"ok": True, "run_id": f"run-{len(self.calls)}", "task": args["task"], "suite": suite, "score": s,
                "gates_passed": int(s), "gates_total": 12, "passed": False, "reflection": f"BENCHMARK FAILED {int(s)}/12 gates."}


def test_campaign_is_greedy_and_writes_log(tmp_path):
    agent = StubAgent()
    out = tmp_path / "campaign.json"
    log = asyncio.run(run_campaign(agent, strategy="scripted_v1", out=out, max_cycles=4))
    assert [c["accepted"] for c in log["cycles"]] == [True, True, False, True]
    assert log["best"] == {"run_id": "run-4", "score": 9.5} and log["best_by_suite"]["flat_v1"]["run_id"] == "run-4"
    # every cycle after the first builds on the best run so far
    assert agent.calls[1]["base_run_id"] == "run-1" and agent.calls[2]["base_run_id"] == "run-2"
    assert agent.calls[3]["base_run_id"] == "run-2"  # cycle 3 regressed, so cycle 4 still bases on run-2
    saved = json.loads(out.read_text())
    assert saved["cycles"][0]["patch"] == STRATEGIES["scripted_v1"][0]["patch"]


def test_campaign_stops_when_the_final_rung_passes(tmp_path):
    class PassAgent(StubAgent):
        async def call(self, tool, args):
            self.calls.append(args)
            return {"ok": True, "run_id": f"r{len(self.calls)}", "task": args["task"], "suite": "flat_v1", "score": 12.5,
                    "gates_passed": 12, "gates_total": 12, "passed": True, "reflection": "BENCHMARK PASSED 12/12 gates."}

    agent = PassAgent()
    log = asyncio.run(run_campaign(agent, strategy="scripted_v1", out=tmp_path / "c.json"))
    # a ladder must not stop at rung 0 just because flat passes: only the step marked stop_on_pass ends it
    assert len(log["cycles"]) == len(STRATEGIES["scripted_v1"]) and len(agent.calls) == len(STRATEGIES["scripted_v1"])


def test_terrain_ladder_warm_starts_each_rung_from_the_previous_tasks_best(tmp_path):
    class LadderAgent:
        def __init__(self):
            self.calls = []
            self.scores = iter([11.0, 12.0, 10.0, 6.0, 7.0, 4.0, 5.0])  # slope/rough scores are LOWER than flat

        async def call(self, tool, args):
            self.calls.append(args)
            s = next(self.scores)
            suite = {"flat_walk": "flat_v1", "slope_up": "slope_v1", "rough_walk": "rough_v1"}[args["task"]]
            return {"ok": True, "run_id": f"run-{len(self.calls)}", "task": args["task"], "suite": suite, "score": s,
                    "gates_passed": int(s), "gates_total": 14, "passed": False, "reflection": "x"}

    agent = LadderAgent()
    log = asyncio.run(run_campaign(agent, strategy="terrain_ladder_v1", out=tmp_path / "l.json"))
    c = agent.calls
    assert c[0].get("base_run_id") is None and c[0]["task"] == "flat_walk"
    assert c[2]["base_run_id"] == "run-2"            # best flat so far
    assert c[3]["base_run_id"] == "run-2"            # slope warm-starts from the FLAT champion (best_of:flat_walk)
    assert c[4]["base_run_id"] == "run-4"            # slope tune builds on the slope run, not the higher-scoring flat one
    assert c[5]["base_run_id"] == "run-5"            # rough warm-starts from the best SLOPE run
    assert c[6]["base_run_id"] == "run-7" or c[6]["base_run_id"] == "run-6"
    assert set(log["best_by_suite"]) == {"flat_v1", "slope_v1", "rough_v1"}
    assert log["best_by_suite"]["flat_v1"]["run_id"] == "run-2" and log["best_by_suite"]["slope_v1"]["run_id"] == "run-5"


def test_every_strategy_step_names_a_known_task_and_base():
    from trainer.tasks import TASKS

    for name, steps in STRATEGIES.items():
        for step in steps:
            assert step["task"] in TASKS, (name, step["name"])
            base = step.get("base", "best")
            assert base is None or base == "best" or base.split(":", 1)[1] in TASKS, (name, step["name"], base)
