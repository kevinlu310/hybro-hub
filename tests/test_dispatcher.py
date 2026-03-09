"""Tests for hub.dispatcher."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from hub.agent_registry import LocalAgent
from hub.dispatcher import Dispatcher


@pytest.fixture
def agent():
    return LocalAgent(
        local_agent_id="test_001",
        name="Test Agent",
        url="http://localhost:9001",
        agent_card={"capabilities": {"streaming": False}},
    )


@pytest.fixture
def streaming_agent():
    return LocalAgent(
        local_agent_id="test_002",
        name="Streaming Agent",
        url="http://localhost:9002",
        agent_card={"capabilities": {"streaming": True}},
    )


SAMPLE_MESSAGE = {
    "messageId": "msg-123",
    "role": "user",
    "parts": [{"text": "Hello agent"}],
}


class TestDispatchSync:
    @pytest.mark.asyncio
    async def test_dispatch_sync_success(self, agent):
        dispatcher = Dispatcher()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "status": {
                    "state": "completed",
                    "message": {
                        "role": "agent",
                        "parts": [{"text": "Hi there!"}],
                    },
                },
            },
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        dispatcher._client = mock_client

        events = []
        async for batch in dispatcher.dispatch(
            agent=agent,
            message_dict=SAMPLE_MESSAGE,
            agent_message_id="am-001",
            user_message_id="um-001",
        ):
            events.extend(batch)

        assert len(events) == 2
        assert events[0]["type"] == "agent_response"
        assert events[0]["data"]["content"] == "Hi there!"
        assert events[1]["type"] == "processing_status"
        assert events[1]["data"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_dispatch_sync_error(self, agent):
        dispatcher = Dispatcher()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection failed"))
        dispatcher._client = mock_client

        events = []
        async for batch in dispatcher.dispatch(
            agent=agent,
            message_dict=SAMPLE_MESSAGE,
            agent_message_id="am-001",
        ):
            events.extend(batch)

        assert len(events) == 2
        assert events[0]["type"] == "agent_error"
        assert "connection failed" in events[0]["data"]["error"]
        assert events[1]["type"] == "processing_status"
        assert events[1]["data"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_dispatch_error_with_empty_str(self, agent):
        """Exceptions whose str() is empty must still emit agent_error, not agent_response."""
        dispatcher = Dispatcher()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=TimeoutError())
        dispatcher._client = mock_client

        events = []
        async for batch in dispatcher.dispatch(
            agent=agent,
            message_dict=SAMPLE_MESSAGE,
            agent_message_id="am-002",
            user_message_id="um-002",
        ):
            events.extend(batch)

        assert events[0]["type"] == "agent_error"
        assert events[0]["data"]["error"]  # must be non-empty
        assert events[1]["type"] == "processing_status"
        assert events[1]["data"]["status"] == "failed"


class TestJsonRpcBuild:
    def test_build_jsonrpc(self):
        body = Dispatcher._build_jsonrpc(SAMPLE_MESSAGE, "message/send")
        assert body["jsonrpc"] == "2.0"
        assert body["method"] == "message/send"
        assert body["params"]["message"] == SAMPLE_MESSAGE
        assert "id" in body


class TestExtractResponseContent:
    """Tests for Dispatcher._extract_response_content (replaces old _extract_text_from_response)."""

    def test_extract_from_status_message(self):
        result = {
            "result": {
                "kind": "task",
                "status": {
                    "state": "completed",
                    "message": {
                        "parts": [{"text": "Response text"}],
                    },
                },
            },
        }
        dr = Dispatcher._extract_response_content(result)
        assert dr.text == "Response text"

    def test_extract_from_artifacts(self):
        result = {
            "result": {
                "kind": "task",
                "status": {"state": "completed"},
                "artifacts": [
                    {"parts": [{"text": "Artifact text"}]},
                ],
            },
        }
        dr = Dispatcher._extract_response_content(result)
        assert dr.text == "Artifact text"

    def test_extract_from_message_kind(self):
        result = {
            "result": {
                "kind": "message",
                "parts": [{"text": "Direct parts"}],
            },
        }
        dr = Dispatcher._extract_response_content(result)
        assert dr.text == "Direct parts"

    def test_extract_with_root_wrapper(self):
        result = {
            "result": {
                "kind": "task",
                "status": {
                    "state": "completed",
                    "message": {
                        "parts": [{"root": {"text": "Wrapped"}}],
                    },
                },
            },
        }
        dr = Dispatcher._extract_response_content(result)
        assert dr.text == "Wrapped"

    def test_fallback_no_kind(self):
        """Pre-spec agents that don't set kind still extract text."""
        result = {
            "result": {
                "status": {
                    "message": {
                        "parts": [{"text": "Fallback text"}],
                    },
                },
            },
        }
        dr = Dispatcher._extract_response_content(result)
        assert dr.text == "Fallback text"

    def test_task_state_extracted(self):
        result = {
            "result": {
                "kind": "task",
                "id": "task-1",
                "status": {
                    "state": "failed",
                    "message": {"parts": [{"text": "oops"}]},
                },
            },
        }
        dr = Dispatcher._extract_response_content(result)
        assert dr.task_state == "failed"
        assert dr.task_id == "task-1"


class TestExtractArtifactText:
    """Tests for Dispatcher._extract_artifact_text (replaces artifact path of old _extract_chunk_text)."""

    def test_artifact_text(self):
        data = {"artifact": {"parts": [{"text": "chunk"}]}}
        assert Dispatcher._extract_artifact_text(data) == "chunk"

    def test_artifact_with_root_wrapper(self):
        data = {"artifact": {"parts": [{"root": {"text": "nested"}}]}}
        assert Dispatcher._extract_artifact_text(data) == "nested"

    def test_empty(self):
        assert Dispatcher._extract_artifact_text({}) == ""


class TestExtractStatusText:
    """Tests for Dispatcher._extract_status_text (replaces status path of old _extract_chunk_text)."""

    def test_status_text(self):
        data = {"status": {"message": {"parts": [{"text": "done"}]}}}
        assert Dispatcher._extract_status_text(data) == "done"

    def test_status_with_root_wrapper(self):
        data = {"status": {"message": {"parts": [{"root": {"text": "nested"}}]}}}
        assert Dispatcher._extract_status_text(data) == "nested"

    def test_empty(self):
        assert Dispatcher._extract_status_text({}) == ""


class TestCollectParts:
    """Tests for Dispatcher._collect_parts — separates text from non-text."""

    def test_text_parts(self):
        text, non_text = Dispatcher._collect_parts([{"text": "hello"}, {"text": " world"}])
        assert text == "hello world"
        assert non_text == []

    def test_mixed_parts(self):
        parts = [{"text": "hello"}, {"kind": "file", "uri": "file://x"}]
        text, non_text = Dispatcher._collect_parts(parts)
        assert text == "hello"
        assert len(non_text) == 1

    def test_root_wrapper(self):
        text, _ = Dispatcher._collect_parts([{"root": {"text": "wrapped"}}])
        assert text == "wrapped"

    def test_empty(self):
        text, non_text = Dispatcher._collect_parts([])
        assert text == ""
        assert non_text == []
