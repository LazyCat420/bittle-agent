"""Thin async HTTP client for the bittle-trainer service (GPU box, :8009).

bittle-agent runs in a 512 MB container with no GPU; everything heavy lives
behind this client. Errors are typed so tools can return structured JSON.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx


class TrainerUnavailable(Exception):
    """Trainer not configured or not reachable."""


class TrainerError(Exception):
    """Trainer answered with an error status."""

    def __init__(self, status: int, detail: Any):
        super().__init__(f"trainer {status}: {detail}")
        self.status = status
        self.detail = detail


class TrainerClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    async def _request(self, method: str, path: str, *, json: Any = None, params: dict[str, Any] | None = None,
                       timeout: float | None = None) -> Any:
        if not self.configured:
            raise TrainerUnavailable("BITTLE_TRAINER_URL is not set")
        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                resp = await client.request(method, f"{self.base_url}{path}", json=json, params=params)
        except httpx.HTTPError as exc:
            raise TrainerUnavailable(f"{self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise TrainerError(resp.status_code, detail)
        return resp.json()

    # ── endpoints ──────────────────────────────────────────────────────
    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health", timeout=10.0)

    async def schema(self) -> dict[str, Any]:
        return await self._request("GET", "/config/schema")

    async def validate(self, config_patch: dict[str, Any], base_run_id: str | None = None) -> dict[str, Any]:
        return await self._request("POST", "/config/validate", json={"config_patch": config_patch, "base_run_id": base_run_id})

    async def submit(self, config_patch: dict[str, Any], *, name: str = "", base_run_id: str | None = None,
                     notes: str = "", task: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"config_patch": config_patch, "name": name, "base_run_id": base_run_id, "notes": notes}
        if task:
            body["task"] = task
        return await self._request("POST", "/runs", json=body)

    async def status(self, run_id: str, *, wait_s: float = 0.0, until: str = "any") -> dict[str, Any]:
        return await self._request("GET", f"/runs/{run_id}", params={"wait_s": wait_s, "until": until},
                                   timeout=wait_s + 30.0)

    async def cancel(self, run_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/runs/{run_id}/cancel")

    async def benchmark(self, run_id: str, *, suite: str | None = None, force: bool = False,
                        n_episodes: int | None = None, dr_sweep: bool = False, dual_sim: bool = False,
                        wait_s: float = 0.0) -> dict[str, Any]:
        """``suite`` None = the run's own suite (its task's); another suite needs ``force``."""
        body: dict[str, Any] = {"n_episodes": n_episodes, "dr_sweep": dr_sweep, "dual_sim": dual_sim, "wait_s": wait_s}
        if suite:
            body["suite"] = suite
            body["force"] = bool(force)
        return await self._request("POST", f"/runs/{run_id}/benchmark", json=body, timeout=wait_s + 30.0)

    async def get_benchmark(self, run_id: str, suite: str | None = None) -> dict[str, Any]:
        return await self._request("GET", f"/runs/{run_id}/benchmark", params={"suite": suite} if suite else None)

    async def list_runs(self, *, sort: str = "score", limit: int = 20, suite: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"sort": sort, "limit": limit}
        if suite:
            params["suite"] = suite
        return await self._request("GET", "/runs", params=params)

    async def tasks(self) -> dict[str, Any]:
        return await self._request("GET", "/tasks")

    async def compare(self, run_ids: list[str]) -> dict[str, Any]:
        return await self._request("POST", "/runs/compare", json={"run_ids": run_ids})

    async def baselines(self, suite: str | None = None) -> dict[str, Any]:
        return await self._request("GET", "/baselines", params={"suite": suite} if suite else None)

    async def compute_baselines(self, names: list[str] | None = None) -> dict[str, Any]:
        return await self._request("POST", "/baselines/compute", json={"names": names}, timeout=1800.0)

    async def rollout_stream(self, path: str) -> AsyncIterator[bytes]:
        """Stream a rollout JSON body (never loaded into memory whole)."""
        if not self.configured:
            raise TrainerUnavailable("BITTLE_TRAINER_URL is not set")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("GET", f"{self.base_url}{path}") as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise TrainerError(resp.status_code, body.decode("utf-8", "replace")[:300])
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except httpx.HTTPError as exc:
            raise TrainerUnavailable(f"{self.base_url}: {exc}") from exc
