"""Deterministic campaign driver: the same tools the LLM uses, without the LLM.

    python -m trainer.campaign --agent-url http://nas:8008 --strategy scripted_v1
    python -m trainer.campaign --in-process --fake        # CI / demo

Writes ``campaign.json`` (patches, run ids, reports, best) which is also the
reproduction log format for an LLM session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from .strategies import STRATEGIES


class AgentHttp:
    """Drive tools through bittle-agent's ``POST /api/agent/step``."""

    def __init__(self, url: str, timeout: float = 3600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.url}/api/agent/step", json={"tool": tool, "args": args})
            r.raise_for_status()
            return r.json()


class AgentInProcess:
    """Drive the tool bodies directly (bittle-agent importable, trainer URL configured)."""

    def __init__(self, trainer_url: str):
        from app.config import Settings
        from app.trainer_client import TrainerClient
        from app.training_tools import TrainingTools

        settings = Settings(trainer_url=trainer_url)
        self.tools = TrainingTools(TrainerClient(trainer_url, timeout=30.0), settings)

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        return await self.tools.execute(tool, args)


async def run_campaign(agent, *, strategy: str, out: Path, max_cycles: int | None = None,
                       dr_sweep: bool = False) -> dict[str, Any]:
    steps = STRATEGIES[strategy]
    if max_cycles:
        steps = steps[:max_cycles]
    log: dict[str, Any] = {"strategy": strategy, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "cycles": [], "best": None}
    best_score, best_run = float("-inf"), None
    for i, step in enumerate(steps):
        args = {"name": step["name"], "config_patch": step["patch"], "notes": step["notes"], "dr_sweep": dr_sweep}
        if best_run:
            args["base_run_id"] = best_run
        t0 = time.time()
        res = await agent.call("bittle_train_and_benchmark", args)
        score = res.get("score") if res.get("ok") else None
        accepted = score is not None and score > best_score
        if accepted:
            best_score, best_run = score, res["run_id"]
        entry = {"cycle": i, "name": step["name"], "patch": step["patch"], "base_run_id": args.get("base_run_id"),
                 "run_id": res.get("run_id"), "ok": res.get("ok"), "score": score,
                 "gates_passed": res.get("gates_passed"), "gates_total": res.get("gates_total"),
                 "reflection": res.get("reflection") or res.get("detail"), "accepted": accepted,
                 "elapsed_s": round(time.time() - t0, 1)}
        log["cycles"].append(entry)
        log["best"] = {"run_id": best_run, "score": best_score if best_run else None}
        out.write_text(json.dumps(log, indent=2))
        print(f"[cycle {i}] {step['name']}: ok={res.get('ok')} score={score} accepted={accepted} run={res.get('run_id')}")
        if res.get("ok") and res.get("passed"):
            print("all gates pass; stopping early")
            break
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.write_text(json.dumps(log, indent=2))
    return log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-url", default=None, help="bittle-agent base URL (uses /api/agent/step)")
    ap.add_argument("--in-process", action="store_true", help="import bittle-agent tool bodies directly")
    ap.add_argument("--trainer-url", default="http://127.0.0.1:8009")
    ap.add_argument("--strategy", default="scripted_v1", choices=sorted(STRATEGIES))
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument("--dr-sweep", action="store_true")
    ap.add_argument("--out", default="campaign.json")
    a = ap.parse_args(argv)
    if a.agent_url:
        agent = AgentHttp(a.agent_url)
    elif a.in_process:
        repo = Path(__file__).resolve().parent.parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        agent = AgentInProcess(a.trainer_url)
    else:
        ap.error("give --agent-url or --in-process")
    log = asyncio.run(run_campaign(agent, strategy=a.strategy, out=Path(a.out), max_cycles=a.max_cycles, dr_sweep=a.dr_sweep))
    return 0 if log["best"]["run_id"] else 1


if __name__ == "__main__":
    sys.exit(main())
