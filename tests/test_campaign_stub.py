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
        return {"ok": True, "run_id": f"run-{len(self.calls)}", "score": s, "gates_passed": int(s), "gates_total": 12,
                "passed": False, "reflection": f"BENCHMARK FAILED {int(s)}/12 gates."}


def test_campaign_is_greedy_and_writes_log(tmp_path):
    agent = StubAgent()
    out = tmp_path / "campaign.json"
    log = asyncio.run(run_campaign(agent, strategy="scripted_v1", out=out, max_cycles=4))
    assert [c["accepted"] for c in log["cycles"]] == [True, True, False, True]
    assert log["best"] == {"run_id": "run-4", "score": 9.5}
    # every cycle after the first builds on the best run so far
    assert agent.calls[1]["base_run_id"] == "run-1" and agent.calls[2]["base_run_id"] == "run-2"
    assert agent.calls[3]["base_run_id"] == "run-2"  # cycle 3 regressed, so cycle 4 still bases on run-2
    saved = json.loads(out.read_text())
    assert saved["cycles"][0]["patch"] == STRATEGIES["scripted_v1"][0]["patch"]


def test_campaign_stops_when_all_gates_pass(tmp_path):
    class PassAgent(StubAgent):
        async def call(self, tool, args):
            self.calls.append(args)
            return {"ok": True, "run_id": "r", "score": 12.5, "gates_passed": 12, "gates_total": 12, "passed": True,
                    "reflection": "BENCHMARK PASSED 12/12 gates."}

    agent = PassAgent()
    log = asyncio.run(run_campaign(agent, strategy="scripted_v1", out=tmp_path / "c.json"))
    assert len(log["cycles"]) == 1 and len(agent.calls) == 1
