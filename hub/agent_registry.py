"""Agent registry — discover and health-check local A2A agents.

Supports manual configuration (config.yaml) and auto-discovery
(enumerates listening localhost ports via psutil, then probes for
well-known A2A agent card paths).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

import httpx
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    PREV_AGENT_CARD_WELL_KNOWN_PATH,
)

from .config import HubConfig, LocalAgentConfig

logger = logging.getLogger(__name__)

AGENT_CARD_PATHS = [
    AGENT_CARD_WELL_KNOWN_PATH,       # /.well-known/agent-card.json
    PREV_AGENT_CARD_WELL_KNOWN_PATH,  # /.well-known/agent.json
]
DISCOVERY_TIMEOUT = 3

_LOCALHOST_ADDRS = frozenset({"127.0.0.1", "::1", "0.0.0.0", "::", ""})

_CONCURRENCY_LIMIT = 30


def _normalize_url(url: str) -> str:
    """Canonicalize localhost aliases so the same agent always maps to one URL."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname in _LOCALHOST_ADDRS:
        netloc = f"localhost:{parsed.port}" if parsed.port else "localhost"
        parsed = parsed._replace(netloc=netloc)
    return urlunparse(parsed)


@dataclass
class LocalAgent:
    """A discovered local A2A agent."""

    local_agent_id: str
    name: str
    url: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    agent_card: dict = field(default_factory=dict)
    healthy: bool = True


class AgentRegistry:
    """Registry of local A2A agents."""

    def __init__(self, config: HubConfig) -> None:
        self._config = config
        self._agents: dict[str, LocalAgent] = {}
        self._client: httpx.AsyncClient | None = None

    @property
    def agents(self) -> dict[str, LocalAgent]:
        return dict(self._agents)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=DISCOVERY_TIMEOUT)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ──── Discovery ────

    async def discover(self) -> list[LocalAgent]:
        """Run full discovery: manual config + auto-discovery."""
        for ac in self._config.agents:
            await self._probe_and_register(ac.url, ac.name)

        if self._config.auto_discover:
            await self._auto_discover()

        logger.info("Registry has %d agents", len(self._agents))
        return list(self._agents.values())

    async def _auto_discover(self) -> None:
        """Discover A2A agents by enumerating listening localhost ports."""
        ports = _get_listening_ports(
            exclude=set(self._config.auto_discover_exclude_ports),
        )
        if not ports:
            logger.debug("No listening localhost ports found for auto-discovery")
            return

        logger.debug("Auto-discovery: probing %d listening ports", len(ports))
        sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

        async def _bounded_probe(port: int) -> None:
            async with sem:
                await self._probe_and_register(
                    f"http://localhost:{port}", source="auto"
                )

        await asyncio.gather(
            *(_bounded_probe(p) for p in ports),
            return_exceptions=True,
        )

    async def _probe_and_register(
        self, url: str, name: str | None = None, source: str = "config"
    ) -> LocalAgent | None:
        """Try to fetch an agent card from a URL and register it."""
        url = _normalize_url(url.rstrip("/"))
        card = await self._fetch_agent_card(url, source)
        if card is None:
            return None

        agent_name = name or card.get("name", f"Agent@{url}")
        existing = next(
            (a for a in self._agents.values() if a.url == url), None
        )
        if existing:
            existing.agent_card = card
            existing.healthy = True
            existing.name = agent_name
            return existing

        local_id = hashlib.sha256(url.encode()).hexdigest()[:12]
        agent = LocalAgent(
            local_agent_id=local_id,
            name=agent_name,
            url=url,
            description=card.get("description", ""),
            capabilities=_extract_capabilities(card),
            agent_card=card,
            healthy=True,
        )
        self._agents[local_id] = agent
        logger.info("Discovered agent: %s at %s (id=%s)", agent_name, url, local_id)
        return agent

    # ──── Health check ────

    async def health_check(self) -> None:
        """Ping all registered agents and update health status."""
        for agent in list(self._agents.values()):
            card = await self._fetch_agent_card(agent.url)
            agent.healthy = card is not None
            if card is not None:
                agent.agent_card = card
            else:
                logger.debug("Agent %s unhealthy", agent.name)

    # ──── Agent card fetch ────

    async def _fetch_agent_card(self, url: str, source: str = "config") -> dict | None:
        """Try each well-known agent card path and return the first valid card."""
        client = await self._get_client()
        for path in AGENT_CARD_PATHS:
            try:
                resp = await client.get(f"{url}{path}")
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                continue
        if source == "config":
            logger.debug("Agent at %s not reachable", url)
        return None

    # ──── Lookup ────

    def get_agent(self, local_agent_id: str) -> LocalAgent | None:
        return self._agents.get(local_agent_id)

    def get_healthy_agents(self) -> list[LocalAgent]:
        return [a for a in self._agents.values() if a.healthy]

    # ──── Sync payload ────

    def to_sync_payload(self) -> list[dict]:
        """Convert registry to HubAgentSyncRequest format.

        Includes all known agents regardless of health status so that
        a transient health-check failure doesn't cause the cloud to
        prune agents that still exist locally.
        """
        return [
            {
                "local_agent_id": a.local_agent_id,
                "name": a.name,
                "description": a.description,
                "capabilities": a.capabilities,
                "agent_card": a.agent_card,
            }
            for a in self._agents.values()
        ]


# ──── Port enumeration ────

def _get_listening_ports(exclude: set[int] | None = None) -> list[int]:
    """Return sorted list of TCP ports listening on localhost.

    Uses psutil to enumerate connections — works cross-platform
    (macOS, Linux, Windows) without shelling out.
    """
    try:
        import psutil
    except ImportError:
        logger.warning(
            "psutil not installed — auto-discovery disabled. "
            "Install with: pip install psutil"
        )
        return []

    exclude = exclude or set()
    ports: set[int] = set()

    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status != psutil.CONN_LISTEN:
                continue
            addr, port = conn.laddr
            if port in exclude:
                continue
            if addr in _LOCALHOST_ADDRS:
                ports.add(port)
    except (psutil.AccessDenied, PermissionError):
        logger.warning(
            "Insufficient permissions for psutil.net_connections(). "
            "Auto-discovery may be incomplete. Try running with elevated privileges, "
            "or configure agents manually in config.yaml."
        )
        return []

    return sorted(ports)


def _extract_capabilities(card: dict) -> list[str]:
    """Extract capability list from an agent card."""
    caps = []
    abilities = card.get("capabilities", {})
    if abilities.get("streaming"):
        caps.append("streaming")
    if abilities.get("pushNotifications"):
        caps.append("push_notifications")
    skills = card.get("skills", [])
    for s in skills:
        if tags := s.get("tags"):
            caps.extend(tags)
    return caps
