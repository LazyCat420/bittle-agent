"""Training-session recorder: what GLM did, cycle by cycle, for every viewer of the dashboard.

The chat SSE stream only reaches the browser that started the session. In training mode the
harness tees every event through here so the Training tab can show "what is happening right
now" to anyone, replay past sessions, and line each `bittle_train_and_benchmark` call up with
the run it produced. Sessions are kept in memory (a bounded ring) and appended to
``storage/training_sessions/<session_id>.jsonl`` so a redeploy does not lose the history.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator

MAX_SESSIONS = 50
MAX_EVENTS = 4000
#: streamed thought/text chunks are coalesced into one event per burst to keep the log small
COALESCE_TYPES = ("thought", "text")


class TrainingSession:
    def __init__(self, session_id: str, prompt: str, store_dir: Path | None):
        self.session_id = session_id
        self.prompt = prompt
        self.started = time.time()
        self.finished: float | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
        self.cycles: list[dict[str, Any]] = []
        self._path = (store_dir / f"{session_id}.jsonl") if store_dir else None
        self._open_calls: dict[str, dict[str, Any]] = {}

    # ── recording ──────────────────────────────────────────────────────
    def record(self, ev: dict[str, Any]) -> dict[str, Any]:
        ev = dict(ev, t=round(time.time() - self.started, 2))
        typ = ev.get("type")
        if typ in COALESCE_TYPES and self.events and self.events[-1].get("type") == typ and "content" in ev:
            self.events[-1]["content"] = (self.events[-1].get("content") or "") + (ev.get("content") or "")
            self.events[-1]["t"] = ev["t"]
            return ev
        if typ == "tool_call":
            call = {"id": ev.get("id"), "tool": ev.get("name"), "args": ev.get("args") or {}, "t_start": ev["t"],
                    "status": "running", "progress": None, "result": None}
            self._open_calls[str(ev.get("id"))] = call
            if ev.get("name") in ("bittle_train_and_benchmark", "bittle_train_policy"):
                self.cycles.append(call)
        elif typ == "progress":
            call = self._open_calls.get(str(ev.get("id")))
            if call is not None:
                call["progress"] = ev.get("detail")
                call["elapsed_s"] = ev.get("elapsed_s")
        elif typ == "tool_result":
            call = self._open_calls.pop(str(ev.get("id")), None)
            if call is not None:
                res = ev.get("result") or {}
                call["status"] = "ok" if res.get("ok") else "error"
                call["t_end"] = ev["t"]
                call["result"] = _compact_result(res)
        elif typ in ("done", "error"):
            self.finished = time.time()
        self.events.append(ev)
        if self._path is not None:
            try:
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(ev) + "\n")
            except OSError:
                pass
        return ev

    def summary(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "prompt": self.prompt, "started": self.started,
                "finished": self.finished, "live": self.finished is None, "n_events": len(self.events),
                "cycles": self.cycles, "run_ids": [c["result"]["run_id"] for c in self.cycles
                                                    if c.get("result") and c["result"].get("run_id")]}

    def to_dict(self) -> dict[str, Any]:
        return dict(self.summary(), events=list(self.events))


def _compact_result(res: dict[str, Any]) -> dict[str, Any]:
    keep = ("ok", "error", "detail", "run_id", "task", "suite", "suite_version", "gates_passed", "gates_total",
            "passed", "score", "reflection", "elapsed_s", "config_diff", "metrics", "status", "advice", "notes")
    out = {k: res[k] for k in keep if k in res}
    if "gates" in res:
        out["gates"] = [{k: g.get(k) for k in ("gate", "value", "op", "threshold", "pass")} for g in res["gates"]]
    return out


class SessionRecorder:
    """Owns the sessions; ``tee`` wraps a chat event stream, ``subscribe`` fans live events out."""

    def __init__(self, store_dir: Path | None = None):
        self.store_dir = store_dir
        if store_dir is not None:
            store_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, TrainingSession] = {}
        self._order: deque[str] = deque(maxlen=MAX_SESSIONS)
        self._subscribers: set[asyncio.Queue] = set()
        self._n = 0
        self._load_existing()

    def _load_existing(self) -> None:
        if self.store_dir is None:
            return
        for p in sorted(self.store_dir.glob("*.jsonl"))[-MAX_SESSIONS:]:
            try:
                s = TrainingSession(p.stem, "", None)
                for line in p.read_text().splitlines():
                    if line.strip():
                        ev = json.loads(line)
                        if ev.get("type") == "session":
                            s.prompt, s.started = ev.get("prompt", ""), float(ev.get("started", s.started))
                            continue
                        s.record({k: v for k, v in ev.items() if k != "t"})
                s.finished = s.finished or s.started
                self.sessions[s.session_id] = s
                self._order.append(s.session_id)
            except (OSError, ValueError):
                continue

    def start(self, prompt: str) -> TrainingSession:
        self._n += 1
        sid = time.strftime("%Y%m%d-%H%M%S") + f"-{self._n:03d}"
        s = TrainingSession(sid, prompt, self.store_dir)
        if s._path is not None:
            try:
                s._path.write_text(json.dumps({"type": "session", "prompt": prompt, "started": s.started}) + "\n")
            except OSError:
                pass
        self.sessions[sid] = s
        if len(self._order) == self._order.maxlen:
            self.sessions.pop(self._order[0], None)
        self._order.append(sid)
        self._broadcast({"type": "session_start", "session": s.summary()})
        return s

    async def tee(self, session: TrainingSession, stream: AsyncIterator[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        try:
            async for ev in stream:
                session.record(ev)
                self._broadcast({"type": "event", "session_id": session.session_id, "event": ev,
                                 "cycles": session.cycles if ev.get("type") in ("tool_call", "progress", "tool_result") else None})
                yield ev
        finally:
            if session.finished is None:
                session.finished = time.time()
            self._broadcast({"type": "session_end", "session": session.summary()})

    def _broadcast(self, msg: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        try:
            yield {"type": "hello", "live": self.live_session_id(), "sessions": self.list()[:10]}
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    msg = {"type": "keepalive", "live": self.live_session_id()}
                yield msg
        finally:
            self._subscribers.discard(q)

    def list(self) -> list[dict[str, Any]]:
        return [self.sessions[s].summary() for s in reversed(self._order) if s in self.sessions]

    def get(self, session_id: str) -> TrainingSession | None:
        return self.sessions.get(session_id)

    def live_session_id(self) -> str | None:
        for sid in reversed(self._order):
            s = self.sessions.get(sid)
            if s and s.finished is None:
                return sid
        return None
