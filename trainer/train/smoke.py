"""30-second smoke training: ``python -m trainer.train.smoke [--out DIR]``.

Tiny budget (256 envs, ~300k steps) to prove the pipeline learns: the final
eval reward must beat the first. Prints measured steps/s so the sizing table
in the plan can be replaced with numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from ..config import TrainConfig
from .ppo import train_policy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--num-timesteps", type=int, default=300_000)
    ap.add_argument("--impl", default=None)
    ap.add_argument("--sim-dt", type=float, default=0.002)
    a = ap.parse_args(argv)
    out = Path(a.out or tempfile.mkdtemp(prefix="bittle-smoke-"))
    cfg = TrainConfig(ppo={"num_envs": a.num_envs, "num_timesteps": a.num_timesteps, "num_evals": 4,
                           "num_minibatches": 8, "batch_size": 128})
    m = train_policy(cfg, out, impl=a.impl, sim_dt=a.sim_dt, progress=lambda p: print(json.dumps(p)))
    print(json.dumps({k: v for k, v in m.items() if k != "curve"}, indent=1))
    ok = m["reward_final"] is not None and m["reward_first"] is not None and m["reward_final"] > m["reward_first"]
    print("SMOKE", "PASS" if ok else "FAIL", "policy at", out / "policy" / "policy.npz")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
