"""Hub daemon main loop.

Orchestrates: config loading, relay registration, agent discovery,
SSE subscription, event dispatch, and periodic re-sync.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any
from uuid import uuid4

from .agent_registry import AgentRegistry
from .config import HubConfig
from .dispatcher import Dispatcher
from .privacy_router import PrivacyRouter
from .relay_client import RelayClient

logger = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL = 60
RESYNC_INTERVAL = 120


class HubDaemon:
    """The hub daemon orchestrator."""

    def __init__(self, config: HubConfig) -> None:
        self.config = config
        self.relay = RelayClient(
            gateway_url=config.gateway_url,
            hub_id=config.hub_id,
            api_key=config.api_key,
        )
        self.registry = AgentRegistry(config)
        self.dispatcher = Dispatcher()
        self.privacy = PrivacyRouter(
            sensitive_keywords=config.privacy_sensitive_keywords,
            sensitive_patterns=config.privacy_sensitive_patterns,
            default_routing=config.privacy_default_routing,
        )
        self._shutdown_event = asyncio.Event()
        self._last_sync_payload: list[dict] | None = None

    async def run(self) -> None:
        """Main entry point — run the hub daemon."""
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._signal_shutdown)
        except NotImplementedError:
            signal.signal(signal.SIGINT, lambda *_: self._signal_shutdown())

        try:
            await self._startup()
            await self._event_loop()
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    # ──── Startup ────

    async def _startup(self) -> None:
        logger.info("Starting hub daemon (hub_id=%s)", self.config.hub_id)

        # Register with relay
        await self.relay.register()

        # Discover local agents
        agents = await self.registry.discover()
        if not agents:
            logger.warning(
                "No local agents found. Start an A2A agent and it will be "
                "discovered automatically."
            )

        # Sync agents to cloud
        await self._sync_agents()

        logger.info(
            "Hub ready — %d agent(s) synced. Waiting for messages...",
            len(self.registry.get_healthy_agents()),
        )

    # ──── Event loop ────

    async def _event_loop(self) -> None:
        """Subscribe to relay SSE and dispatch events."""
        # Start background tasks
        health_task = asyncio.create_task(self._health_check_loop())
        resync_task = asyncio.create_task(self._resync_loop())

        try:
            async for event in self.relay.subscribe():
                if self._shutdown_event.is_set():
                    break
                try:
                    await self._handle_event(event)
                except Exception:
                    logger.exception("Failed to handle relay event")
        finally:
            health_task.cancel()
            resync_task.cancel()
            await asyncio.gather(health_task, resync_task, return_exceptions=True)

    async def _handle_event(self, event: dict[str, Any]) -> None:
        """Handle a relay event based on its type."""
        event_type = event.get("type", "user_message")

        if event_type == "cancel_task":
            await self._handle_cancel_task(event)
            return

        if event_type == "user_reply":
            await self._handle_user_reply(event)
            return

        # Default: user_message
        await self._handle_user_message(event)

    async def _handle_user_message(self, event: dict[str, Any]) -> None:
        """Handle a user_message relay event."""
        local_agent_id = event.get("local_agent_id")
        room_id = event.get("room_id")
        agent_message_id = event.get("agent_message_id")
        user_message_id = event.get("user_message_id")
        message_dict = event.get("message")

        if not all([local_agent_id, room_id, agent_message_id, message_dict]):
            logger.warning("Incomplete relay event: %s", event)
            if room_id and agent_message_id:
                await self._publish_failure(
                    room_id, agent_message_id,
                    error="Incomplete relay event received by hub",
                    error_type="InvalidEvent",
                    user_message_id=user_message_id,
                )
            return

        agent = self.registry.get_agent(local_agent_id)
        if not agent:
            logger.error("No agent found for local_agent_id=%s", local_agent_id)
            await self._publish_failure(
                room_id, agent_message_id,
                error=f"Agent '{local_agent_id}' is no longer available on this hub",
                error_type="AgentNotFound",
                user_message_id=user_message_id,
            )
            return

        text = _extract_text(message_dict)
        self.privacy.check_and_log(text, agent.name)

        logger.info(
            "Dispatching to %s (room=%s, msg=%s)",
            agent.name, room_id, agent_message_id[:8],
        )

        await self.relay.publish(room_id, [{
            "type": "task_submitted",
            "agent_message_id": agent_message_id,
            "data": {"task_id": uuid4().hex, "agent_name": agent.name},
        }])

        async for batch in self.dispatcher.dispatch(
            agent=agent,
            message_dict=message_dict,
            agent_message_id=agent_message_id,
            user_message_id=user_message_id,
        ):
            await self.relay.publish(room_id, batch)

    async def _handle_cancel_task(self, event: dict[str, Any]) -> None:
        """Best-effort cancellation of an in-flight task on a local agent."""
        local_agent_id = event.get("local_agent_id")
        agent_message_id = event.get("agent_message_id")
        task_id = event.get("task_id")

        agent = self.registry.get_agent(local_agent_id) if local_agent_id else None
        if not agent:
            logger.warning(
                "cancel_task for unknown agent %s (msg=%s)",
                local_agent_id, agent_message_id,
            )
            return

        if not task_id:
            logger.info("cancel_task without task_id — cannot forward to agent")
            return

        logger.info("Forwarding cancel_task to %s (task=%s)", agent.name, task_id)
        try:
            import httpx as _httpx

            async with _httpx.AsyncClient(timeout=10) as client:
                cancel_body = {
                    "jsonrpc": "2.0",
                    "id": str(__import__("uuid").uuid4().hex),
                    "method": "tasks/cancel",
                    "params": {"id": task_id},
                }
                resp = await client.post(
                    agent.url,
                    json=cancel_body,
                    headers={"Content-Type": "application/json"},
                )
                logger.info(
                    "Cancel response from %s: %d", agent.name, resp.status_code
                )
        except Exception:
            logger.warning("Failed to cancel task on %s (best-effort)", agent.name, exc_info=True)

    async def _handle_user_reply(self, event: dict[str, Any]) -> None:
        """Handle a HITL reply by dispatching it to the local agent."""
        local_agent_id = event.get("local_agent_id")
        room_id = event.get("room_id")
        agent_message_id = event.get("agent_message_id")
        reply_text = event.get("reply_text", "")
        task_id = event.get("task_id")
        context_id = event.get("context_id")

        if not all([local_agent_id, room_id, agent_message_id]):
            logger.warning("Incomplete user_reply event: %s", event)
            if room_id and agent_message_id:
                await self._publish_failure(
                    room_id, agent_message_id,
                    error="Incomplete user_reply event received by hub",
                    error_type="InvalidEvent",
                )
            return

        agent = self.registry.get_agent(local_agent_id)
        if not agent:
            logger.error("No agent found for local_agent_id=%s", local_agent_id)
            await self._publish_failure(
                room_id, agent_message_id,
                error=f"Agent '{local_agent_id}' is no longer available on this hub",
                error_type="AgentNotFound",
            )
            return

        reply_message: dict[str, Any] = {
            "role": "user",
            "parts": [{"kind": "text", "text": reply_text}],
            "messageId": uuid4().hex,
        }
        if task_id:
            reply_message["taskId"] = task_id
        if context_id:
            reply_message["contextId"] = context_id
        if task_id:
            reply_message["referenceTaskIds"] = [task_id]

        logger.info(
            "Dispatching HITL reply to %s (room=%s, msg=%s)",
            agent.name, room_id, agent_message_id[:8],
        )

        await self.relay.publish(room_id, [{
            "type": "task_submitted",
            "agent_message_id": agent_message_id,
            "data": {"task_id": uuid4().hex, "agent_name": agent.name},
        }])

        async for batch in self.dispatcher.dispatch(
            agent=agent,
            message_dict=reply_message,
            agent_message_id=agent_message_id,
        ):
            await self.relay.publish(room_id, batch)

    # ──── Failure publishing ────

    async def _publish_failure(
        self,
        room_id: str,
        agent_message_id: str,
        error: str,
        error_type: str = "HubError",
        user_message_id: str | None = None,
    ) -> None:
        """Publish agent_error + processing_status=failed back to the relay."""
        await self.relay.publish(room_id, [
            {
                "type": "agent_error",
                "agent_message_id": agent_message_id,
                "data": {"error": error, "error_type": error_type},
            },
            {
                "type": "processing_status",
                "agent_message_id": agent_message_id,
                "data": {"status": "failed", "user_message_id": user_message_id},
            },
        ])

    # ──── Background tasks ────

    async def _health_check_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            try:
                await self.registry.health_check()
            except Exception:
                logger.exception("Health check failed")

    async def _resync_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(RESYNC_INTERVAL)
            try:
                await self.registry.discover()
                await self._sync_agents()
            except Exception:
                logger.exception("Agent re-sync failed")

    async def _sync_agents(self) -> None:
        payload = self.registry.to_sync_payload()
        if payload == self._last_sync_payload:
            return
        synced = await self.relay.sync_agents(payload)
        self._last_sync_payload = payload
        logger.info("Synced %d agents to cloud", len(synced))

    # ──── Shutdown ────

    def _signal_shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._shutdown_event.set()

    async def _shutdown(self) -> None:
        logger.info("Shutting down hub daemon...")
        await self.relay.close()
        await self.registry.close()
        await self.dispatcher.close()
        logger.info("Hub daemon stopped.")


def _extract_text(message_dict: dict) -> str:
    """Extract text from an A2A Message dict."""
    parts = message_dict.get("parts", [])
    texts = []
    for p in parts:
        root = p.get("root", p)
        if "text" in root:
            texts.append(root["text"])
    return " ".join(texts)
