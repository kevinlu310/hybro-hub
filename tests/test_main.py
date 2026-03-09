"""Tests for hub.main — HubDaemon event handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hub.agent_registry import LocalAgent
from hub.main import HubDaemon


def _make_daemon() -> HubDaemon:
    """Build a HubDaemon with mocked subsystems."""
    daemon = object.__new__(HubDaemon)
    daemon.relay = MagicMock()
    daemon.relay.publish = AsyncMock()
    daemon.registry = MagicMock()
    daemon.dispatcher = MagicMock()
    daemon.privacy = MagicMock()
    return daemon


AGENT = LocalAgent(
    local_agent_id="agent-1",
    name="Test Agent",
    url="http://localhost:9001",
    agent_card={"capabilities": {"streaming": False}},
)


def _full_user_message_event(**overrides) -> dict:
    base = {
        "type": "user_message",
        "local_agent_id": "agent-1",
        "room_id": "room-1",
        "agent_message_id": "amsg-12345678",
        "user_message_id": "umsg-1",
        "message": {"role": "user", "parts": [{"text": "hello"}]},
    }
    base.update(overrides)
    return base


def _full_user_reply_event(**overrides) -> dict:
    base = {
        "type": "user_reply",
        "local_agent_id": "agent-1",
        "room_id": "room-1",
        "agent_message_id": "amsg-12345678",
        "reply_text": "yes",
        "task_id": "task-1",
        "context_id": "ctx-1",
    }
    base.update(overrides)
    return base


# ──── _handle_user_message ────


class TestHandleUserMessageAgentNotFound:
    async def test_publishes_failure_when_agent_missing(self):
        daemon = _make_daemon()
        daemon.registry.get_agent.return_value = None

        await daemon._handle_user_message(_full_user_message_event())

        daemon.relay.publish.assert_called_once()
        room_id, events = daemon.relay.publish.call_args[0]
        assert room_id == "room-1"
        assert len(events) == 2
        assert events[0]["type"] == "agent_error"
        assert "AgentNotFound" == events[0]["data"]["error_type"]
        assert "agent-1" in events[0]["data"]["error"]
        assert events[1]["type"] == "processing_status"
        assert events[1]["data"]["status"] == "failed"
        assert events[1]["data"]["user_message_id"] == "umsg-1"


class TestHandleUserMessageIncompleteEvent:
    async def test_publishes_failure_when_message_missing(self):
        daemon = _make_daemon()
        event = _full_user_message_event(message=None)

        await daemon._handle_user_message(event)

        daemon.relay.publish.assert_called_once()
        _, events = daemon.relay.publish.call_args[0]
        assert events[0]["type"] == "agent_error"
        assert events[0]["data"]["error_type"] == "InvalidEvent"
        assert events[1]["type"] == "processing_status"
        assert events[1]["data"]["status"] == "failed"

    async def test_publishes_failure_when_local_agent_id_missing(self):
        daemon = _make_daemon()
        event = _full_user_message_event(local_agent_id=None)

        await daemon._handle_user_message(event)

        daemon.relay.publish.assert_called_once()
        _, events = daemon.relay.publish.call_args[0]
        assert events[0]["data"]["error_type"] == "InvalidEvent"

    async def test_no_publish_when_room_id_missing(self):
        """When room_id is absent we can't publish anywhere — just log."""
        daemon = _make_daemon()
        event = _full_user_message_event(room_id=None, agent_message_id=None)

        await daemon._handle_user_message(event)

        daemon.relay.publish.assert_not_called()

    async def test_no_publish_when_agent_message_id_missing(self):
        daemon = _make_daemon()
        event = _full_user_message_event(agent_message_id=None)

        await daemon._handle_user_message(event)

        daemon.relay.publish.assert_not_called()


class TestHandleUserMessageHappyPath:
    async def test_dispatches_to_agent(self):
        daemon = _make_daemon()
        daemon.registry.get_agent.return_value = AGENT

        async def _fake_dispatch(**kwargs):
            yield [{"type": "agent_response", "agent_message_id": "amsg-12345678", "data": {"content": "hi"}}]
            yield [{"type": "processing_status", "agent_message_id": "amsg-12345678", "data": {"status": "completed"}}]

        daemon.dispatcher.dispatch = _fake_dispatch

        await daemon._handle_user_message(_full_user_message_event())

        # task_submitted + 2 dispatch batches = 3 publishes
        assert daemon.relay.publish.call_count == 3
        first_call_events = daemon.relay.publish.call_args_list[0][0][1]
        assert first_call_events[0]["type"] == "task_submitted"


# ──── _handle_user_reply ────


class TestHandleUserReplyAgentNotFound:
    async def test_publishes_failure_when_agent_missing(self):
        daemon = _make_daemon()
        daemon.registry.get_agent.return_value = None

        await daemon._handle_user_reply(_full_user_reply_event())

        daemon.relay.publish.assert_called_once()
        room_id, events = daemon.relay.publish.call_args[0]
        assert room_id == "room-1"
        assert events[0]["type"] == "agent_error"
        assert events[0]["data"]["error_type"] == "AgentNotFound"
        assert events[1]["type"] == "processing_status"
        assert events[1]["data"]["status"] == "failed"


class TestHandleUserReplyIncompleteEvent:
    async def test_publishes_failure_when_local_agent_id_missing(self):
        daemon = _make_daemon()
        event = _full_user_reply_event(local_agent_id=None)

        await daemon._handle_user_reply(event)

        daemon.relay.publish.assert_called_once()
        _, events = daemon.relay.publish.call_args[0]
        assert events[0]["data"]["error_type"] == "InvalidEvent"

    async def test_no_publish_when_room_id_and_msg_id_both_missing(self):
        daemon = _make_daemon()
        event = _full_user_reply_event(room_id=None, agent_message_id=None)

        await daemon._handle_user_reply(event)

        daemon.relay.publish.assert_not_called()


class TestHandleUserReplyHappyPath:
    async def test_dispatches_to_agent(self):
        daemon = _make_daemon()
        daemon.registry.get_agent.return_value = AGENT

        async def _fake_dispatch(**kwargs):
            yield [{"type": "agent_response", "agent_message_id": "amsg-12345678", "data": {"content": "ok"}}]

        daemon.dispatcher.dispatch = _fake_dispatch

        await daemon._handle_user_reply(_full_user_reply_event())

        # task_submitted + 1 dispatch batch = 2 publishes
        assert daemon.relay.publish.call_count == 2


# ──── _handle_cancel_task ────


class TestHandleCancelTaskUnknownAgent:
    async def test_does_not_publish_failure(self):
        """cancel_task is best-effort — unknown agent should not publish an error."""
        daemon = _make_daemon()
        daemon.registry.get_agent.return_value = None

        event = {
            "type": "cancel_task",
            "local_agent_id": "gone-agent",
            "agent_message_id": "amsg-99",
            "task_id": "task-1",
        }
        await daemon._handle_cancel_task(event)

        daemon.relay.publish.assert_not_called()


# ──── _handle_event routing ────


class TestHandleEventRouting:
    async def test_routes_cancel_task(self):
        daemon = _make_daemon()
        daemon._handle_cancel_task = AsyncMock()
        event = {"type": "cancel_task"}

        await daemon._handle_event(event)

        daemon._handle_cancel_task.assert_called_once_with(event)

    async def test_routes_user_reply(self):
        daemon = _make_daemon()
        daemon._handle_user_reply = AsyncMock()
        event = {"type": "user_reply"}

        await daemon._handle_event(event)

        daemon._handle_user_reply.assert_called_once_with(event)

    async def test_routes_user_message_by_default(self):
        daemon = _make_daemon()
        daemon._handle_user_message = AsyncMock()
        event = {"type": "user_message"}

        await daemon._handle_event(event)

        daemon._handle_user_message.assert_called_once_with(event)

    async def test_unknown_type_routes_to_user_message(self):
        daemon = _make_daemon()
        daemon._handle_user_message = AsyncMock()
        event = {"type": "something_new"}

        await daemon._handle_event(event)

        daemon._handle_user_message.assert_called_once_with(event)
