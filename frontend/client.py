"""Frontend → backend HTTP client (M4, practice chat UI).

The client is the SEAM for the UI: every network/parsing failure is coerced
to a typed ``BackendError`` the Streamlit app renders instead of swallowing.
The base URL comes ONLY from the ``BACKEND_URL`` environment variable — never
hard-coded (§7 / M4 acceptance).

Async by design: it runs over ``httpx.AsyncClient`` so tests can inject an
``ASGITransport`` pointing at the real FastAPI seam app (no live server), and
production talks real HTTP. The Streamlit boundary wraps each call in
``asyncio.run`` — one synchronous seam, everything else async.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class BackendError(RuntimeError):
    """Any failure talking to the backend — connection, HTTP status, shape."""


class BackendClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise BackendError("no backend URL configured")
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    @classmethod
    def from_env(cls) -> "BackendClient":
        """The ONLY way the URL is obtained in the UI: $BACKEND_URL (acceptance)."""
        return cls(os.environ.get("BACKEND_URL", ""))

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        return await self._request("POST", "/query", json={"question": question, "top_k": top_k})

    async def _request(self, method: str, path: str, **kw) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=self._base,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                resp = await client.request(
                    method, path, headers={"accept": "application/json"}, **kw
                )
        except httpx.ConnectError as exc:
            raise BackendError(f"cannot reach backend at {self._base}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise BackendError(f"backend at {self._base} timed out after {self._timeout}s") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"backend request failed: {exc}") from exc

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text[:200]
            raise BackendError(f"backend error {resp.status_code}: {detail}")
        try:
            return resp.json()
        except ValueError as exc:
            raise BackendError("backend returned non-JSON response") from exc