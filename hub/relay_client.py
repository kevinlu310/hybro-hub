"""Relay client — SSE subscribe + HTTP publish to hybro.ai relay service.

Maintains a persistent SSE connection to the cloud relay and provides
methods to publish events, register the hub, and sync agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any, AsyncIterator

import httpx
from httpx_sse import aconnect_sse

logger = logging.getLogger(__name__)

MAX_RECONNECT_DELAY = 60
MAX_RETRY_QUEUE = 50

# Short timeout for regular HTTP calls (register, sync, publish, status)
_HTTP_TIMEOUT = httpx.Timeout(connect=10, read=30, write=10, pool=10)

# SSE connections are long-lived; read timeout must be None so heartbeat
# gaps don't kill the connection. The backend sends heartbeats every ~30s,
# so any finite read timeout <=30s would race against the heartbeat.
_SSE_TIMEOUT = httpx.Timeout(connect=10, read=None, write=10, pool=10)


class RelayClient:
    """Client for the hybro.ai relay service."""

    def __init__(
        self,
        *,
        gateway_url: str,
        hub_id: str,
        api_key: str,
    ) -> None:
        self._base = gateway_url.rstrip("/")
        self._hub_id = hub_id
        self._api_key = api_key
        self._connection_token: str | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._sse_client: httpx.AsyncClient | None = None
        self._should_stop = False
        self._retry_queue: deque[tuple[str, list[dict]]] = deque(maxlen=MAX_RETRY_QUEUE)

    @property
    def connection_token(self) -> str | None:
        return self._connection_token

    # ──── Lifecycle ────

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        return self._http_client

    async def _get_sse_client(self) -> httpx.AsyncClient:
        if self._sse_client is None or self._sse_client.is_closed:
            self._sse_client = httpx.AsyncClient(timeout=_SSE_TIMEOUT)
        return self._sse_client

    async def close(self) -> None:
        self._should_stop = True
        for client in (self._http_client, self._sse_client):
            if client is not None:
                await client.aclose()
        self._http_client = None
        self._sse_client = None

    # ──── Registration ────

    async def register(self) -> dict:
        """POST /api/v1/relay/hub/register"""
        client = await self._get_http_client()
        resp = await client.post(
            f"{self._base}/api/v1/relay/hub/register",
            json={"hub_id": self._hub_id},
            headers={"X-API-Key": self._api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info("Hub %s registered for user %s", data["hub_id"], data["user_id"])
        return data

    # ──── Agent sync ────

    async def sync_agents(
        self, agents: list[dict], *, prune_missing: bool = True,
    ) -> list[dict]:
        """POST /api/v1/relay/hub/{hub_id}/agents/sync"""
        client = await self._get_http_client()
        resp = await client.post(
            f"{self._base}/api/v1/relay/hub/{self._hub_id}/agents/sync",
            json={"agents": agents, "prune_missing": prune_missing},
            headers={"X-API-Key": self._api_key},
        )
        resp.raise_for_status()
        return resp.json().get("synced", [])

    # ──── Publish ────

    async def _do_publish(self, room_id: str, events: list[dict]) -> bool:
        """Raw publish HTTP call. Returns True on success, False on 403.

        Raises httpx exceptions on other failures. Does NOT touch the
        retry queue — callers decide what to do on failure.
        """
        client = await self._get_http_client()
        resp = await client.post(
            f"{self._base}/api/v1/relay/hub/{self._hub_id}/publish",
            json={"room_id": room_id, "events": events},
            headers={"Authorization": f"Bearer {self._connection_token}"},
        )
        if resp.status_code == 403:
            self._connection_token = None
            return False
        resp.raise_for_status()
        return True

    async def publish(self, room_id: str, events: list[dict]) -> None:
        """POST /api/v1/relay/hub/{hub_id}/publish with Bearer token.

        On 403 (expired token) or missing token, queues the events for
        retry after the SSE stream reconnects and delivers a fresh token.
        """
        if not self._connection_token:
            logger.warning("No connection token — queueing publish for retry")
            self._retry_queue.append((room_id, events))
            return

        try:
            ok = await self._do_publish(room_id, events)
        except Exception:
            logger.exception("Publish failed for room %s — queueing for retry", room_id)
            self._retry_queue.append((room_id, events))
            return

        if not ok:
            logger.warning(
                "Publish got 403 — token expired, queueing %d events for retry",
                len(events),
            )
            self._retry_queue.append((room_id, events))

    async def _flush_retry_queue(self) -> None:
        """Retry queued publishes after receiving a fresh connection token.

        Drains the queue into a local list first so that _do_publish
        (which doesn't touch the queue) can't cause circular appends.
        If the token is lost mid-flush, remaining items are re-queued.
        """
        if not self._retry_queue or not self._connection_token:
            return

        pending = []
        while self._retry_queue:
            pending.append(self._retry_queue.popleft())

        logger.info("Flushing %d queued publishes", len(pending))
        for i, (room_id, events) in enumerate(pending):
            if not self._connection_token:
                for item in pending[i:]:
                    self._retry_queue.append(item)
                logger.warning(
                    "Token lost mid-flush — re-queued %d remaining publishes",
                    len(pending) - i,
                )
                return
            try:
                ok = await self._do_publish(room_id, events)
                if not ok:
                    self._retry_queue.append((room_id, events))
                    for item in pending[i + 1:]:
                        self._retry_queue.append(item)
                    logger.warning(
                        "403 during flush — re-queued %d publishes",
                        len(pending) - i,
                    )
                    return
            except Exception:
                logger.exception(
                    "Retry publish failed for room %s — re-queued %d publishes",
                    room_id,
                    len(pending) - i,
                )
                self._retry_queue.append((room_id, events))
                for item in pending[i + 1:]:
                    self._retry_queue.append(item)
                return

    # ──── SSE subscription ────

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to relay SSE with auto-reconnect.

        Yields user_message events. Handles connection_token and
        heartbeat events internally. Flushes retry queue when a
        new connection token is received.
        """
        delay = 1
        while not self._should_stop:
            try:
                async for event in self._sse_stream():
                    yield event
                    delay = 1
            except (httpx.HTTPError, httpx.StreamError) as exc:
                if self._should_stop:
                    return
                logger.warning(
                    "SSE connection lost: %s — reconnecting in %ds", exc, delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)
            except asyncio.CancelledError:
                return

    async def _sse_stream(self) -> AsyncIterator[dict[str, Any]]:
        """Single SSE connection session."""
        client = await self._get_sse_client()
        url = f"{self._base}/api/v1/relay/hub/{self._hub_id}/events"
        headers = {"X-API-Key": self._api_key}

        async with aconnect_sse(client, "GET", url, headers=headers) as event_source:
            async for sse in event_source.aiter_sse():
                if self._should_stop:
                    return
                try:
                    data = json.loads(sse.data)
                except (json.JSONDecodeError, TypeError):
                    continue

                event_type = data.get("type")

                if event_type == "connection_token":
                    self._connection_token = data.get("connection_token")
                    logger.info("Received connection token")
                    await self._flush_retry_queue()
                    continue

                if event_type == "heartbeat":
                    logger.debug("Heartbeat received")
                    continue

                if event_type == "user_message":
                    yield data
                    continue

                logger.debug("Unknown relay event type: %s", event_type)

    # ──── Status ────

    async def get_status(self) -> dict:
        """GET /api/v1/relay/hub/status"""
        client = await self._get_http_client()
        resp = await client.get(
            f"{self._base}/api/v1/relay/hub/status",
            headers={"X-API-Key": self._api_key},
        )
        resp.raise_for_status()
        return resp.json()
