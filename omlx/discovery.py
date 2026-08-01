# SPDX-License-Identifier: Apache-2.0
"""
mDNS-based peer discovery for oMLX distributed inference.

Each node advertises itself via mDNS (_omlx._tcp.local) with its node_id.
Other nodes resolve these advertisements into a live PeerRegistry.

Usage:
    from omlx.discovery import DiscoveryService

    discovery = DiscoveryService()
    await discovery.start()

    # Access discovered peers (for API / admin UI)
    peers = discovery.peer_list()

    # Query peer details via HTTP metadata endpoint
    from omlx.engine.distributed import discover_peer_metadata
    for peer in discovery.peers():
        meta = await discover_peer_metadata(peer)

    await discovery.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from zeroconf import ServiceInfo, ServiceListener, Zeroconf
    from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf
except ImportError:
    Zeroconf = None  # type: ignore[misc,assignment]
    AsyncServiceBrowser = None  # type: ignore[misc,assignment]
    ServiceListener = object  # type: ignore[misc,assignment]
    ServiceInfo = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

_SERVICE_TYPE = "_omlx._tcp.local."


@dataclass
class PeerInfo:
    """Information about a discovered peer node.

    node_id, host, and api_port come from mDNS; ram_total_gb,
    ram_available_gb, chip_model, and compute_weight are populated in the
    background via HTTP metadata requests after discovery.

    is_local indicates this entry represents the local node (discovered
    from its own mDNS advertisement) rather than a remote peer.
    """

    node_id: str
    host: str = ""
    api_port: int = 8000
    is_local: bool = False
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    chip_model: str = ""
    compute_weight: float = 1.0


class _OmlxServiceListener(ServiceListener):
    """Zeroconf service listener for oMLX nodes."""

    def __init__(self, registry: PeerRegistry) -> None:
        self.registry = registry

    def add_service(self, zc: Zeroconf, _type: str, name: str) -> None:
        self.registry._on_discovered(zc, _type, name)

    def remove_service(self, zc: Zeroconf, _type: str, name: str) -> None:
        self.registry._on_removed(zc, _type, name)

    def update_service(self, zc: Zeroconf, _type: str, name: str) -> None:
        self.registry._on_discovered(zc, _type, name)


class PeerRegistry:
    """Thread-safe registry of discovered oMLX peer nodes.

    Maintains the current set of available peers via mDNS advertisements.
    """

    def __init__(self, my_node_id: str) -> None:
        self.my_node_id = my_node_id
        self._peers: dict[str, PeerInfo] = {}
        self._listeners: list[Callable[[PeerRegistry], None]] = []
        self._lock = threading.Lock()

    @property
    def my_node_id(self) -> str:
        return self._my_node_id

    @my_node_id.setter
    def my_node_id(self, value: str) -> None:
        self._my_node_id = value

    def peers(self) -> list[PeerInfo]:
        """Get a snapshot of all discovered nodes (includes self)."""
        with self._lock:
            return list(self._peers.values())

    def count(self) -> int:
        """Number of discovered nodes."""
        return len(self.peers())

    def add_listener(
        self, callback: Callable[[PeerRegistry], None]
    ) -> Callable[[], None]:
        """Register a change-notification callback. Returns unsubscriber."""
        with self._lock:
            self._listeners.append(callback)

        def _unsub() -> None:
            with self._lock:
                if callback in self._listeners:
                    self._listeners.remove(callback)

        return _unsub

    def notify_listeners(self) -> None:
        """Call all registered change-notification callbacks."""
        with self._lock:
            callbacks = list(self._listeners)
        for cb in callbacks:
            try:
                cb(self)
            except Exception as exc:
                logger.debug("Discovery listener callback error: %s", exc)

    def _on_discovered(self, zc: Zeroconf, _type: str, name: str) -> None:
        """Handle a new/updated service advertisement.

        Called from the AsyncServiceBrowser event loop thread. We schedule
        an async service info lookup because get_service_info() raises when
        called from the same event loop.
        """
        logger.info("mDNS discovery: service discovered %s (%s)", name, _type)
        try:
            # Use a background task to resolve service info without blocking
            asyncio.create_task(self._resolve_peer(zc, _type, name))
        except Exception as exc:
            logger.warning("Error scheduling mDNS resolution for %s: %s", name, exc)

    async def _resolve_peer(self, zc: Zeroconf, _type: str, name: str) -> None:
        """Resolve service info and add peer to registry.

        Uses zc.async_get_service_info which checks the cache first (the
        PTR was just received, so zeroconf has partial info), then sends
        a network query for SRV/TXT/A records that may not have arrived.
        """
        try:
            logger.info("mDNS resolving %s ...", name)
            info = await zc.async_get_service_info(_type, name, timeout=2000)
            if info is None:
                logger.warning("mDNS resolution failed for %s", name)
                return
            logger.info("mDNS resolved %s: addresses=%s, txt=%s", name, info.addresses, info.properties)
            self._process_service_info(info, name)
        except asyncio.TimeoutError:
            logger.warning("mDNS timeout resolving %s", name)
        except Exception as exc:
            logger.warning("Error resolving mDNS service %s: %s", name, exc)

    def _process_service_info(self, info: Any, name: str) -> None:
        """Extract peer data from a fully-resolved ServiceInfo."""
        try:
            # Parse TXT properties (node_id and api_port only; other
            # capabilities are fetched via HTTP metadata on demand)
            data: dict[str, str] = {}
            if info.properties:
                for key, val in info.properties.items():
                    if isinstance(val, bytes):
                        try:
                            val = val.decode("utf-8")
                        except UnicodeDecodeError:
                            continue
                    data[str(key)] = val

            node_id = data.get("node_id", name.split(".")[0])
            is_local = node_id == self.my_node_id
            api_port = int(data.get("api_port", "8000"))

            if info.addresses:
                host = socket.inet_ntoa(info.addresses[0])
            else:
                host = ""

            with self._lock:
                self._peers[node_id] = PeerInfo(
                    node_id=node_id,
                    host=host,
                    api_port=api_port,
                    is_local=is_local,
                )

            self.notify_listeners()

            # Background-fetch metadata for remote peers (not local)
            if not is_local:
                try:
                    asyncio.create_task(self._fetch_metadata(self._peers[node_id]))
                except Exception:
                    pass

        except Exception as exc:
            logger.debug("Error parsing mDNS service info for %s: %s", name, exc)

    async def _fetch_metadata(self, peer: PeerInfo) -> None:
        """Fetch metadata for a single peer via HTTP and cache it.

        Called in background after discovery or during periodic refresh.
        """
        url = f"http://{peer.host or peer.node_id}:{peer.api_port}/api/distributed-node-info"
        try:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode())

            with self._lock:
                # Re-read peer in case it was replaced (e.g. on update)
                cached = self._peers.get(peer.node_id)
                if cached is not None:
                    cached.ram_total_gb = float(data.get("ram_total_gb", 0))
                    cached.ram_available_gb = float(data.get("ram_available_gb", 0))
                    cached.chip_model = data.get("chip_model", "Unknown")
                    cached.compute_weight = float(data.get("compute_weight", 1.0))

        except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
            logger.debug("Metadata fetch failed for %s", peer.node_id)

    async def refresh_all_metadata(self) -> None:
        """Fetch metadata for all remote peers concurrently."""
        with self._lock:
            peers_snapshot = [p for p in self._peers.values() if not p.is_local]
        tasks = [self._fetch_metadata(p) for p in peers_snapshot]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed = sum(1 for r in results if isinstance(r, Exception))
            if failed:
                logger.debug("Metadata refresh: %d/%d peers failed", failed, len(tasks))

    def _on_removed(self, zc: Zeroconf, _type: str, name: str) -> None:
        """Handle a service removal."""
        try:
            short_name = name.split(".")[0]
            with self._lock:
                self._peers.pop(short_name, None)
            self.notify_listeners()
        except Exception as exc:
            logger.debug("Error removing mDNS service %s: %s", name, exc)


class DiscoveryService:
    """mDNS discovery service for oMLX nodes.

    Advertises this node and discovers other oMLX nodes on the local network.
    Discovery only starts when enabled via settings; stop() both stops
    advertisement and clears the registry.

    Args:
        api_port: Port for the admin API (advertised so peers can query it).
    """

    def __init__(self, api_port: int = 8000) -> None:
        self.api_port = api_port
        self._lock = threading.Lock()

        # Create registry BEFORE setting my_node_id (setter accesses it)
        self.registry = PeerRegistry("")

        # Load or generate persistent node ID
        stored = _load_persistent_node_id()
        nid = stored if stored else self._default_node_id()
        self._my_node_id = nid  # Set directly, avoiding setter recursion
        self.registry.my_node_id = nid

        if not stored:
            _save_persistent_node_id(nid)

        self._zc: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._refresh_task: asyncio.Task[None] | None = None

    @property
    def my_node_id(self) -> str:
        return self._my_node_id

    @my_node_id.setter
    def my_node_id(self, value: str) -> None:
        self._my_node_id = value
        self.registry.my_node_id = value

    # --- Peer snapshot helpers (for admin UI JSON response) ---

    def peer_list(self) -> list[dict[str, Any]]:
        """Get a JSON-serializable snapshot of peers.

        node_id, host, api_port come from mDNS. Metadata fields
        (ram_total_gb, ram_available_gb, chip_model,
        compute_weight) are populated from background HTTP queries.
        is_local marks this node as the local machine.
        """
        return [
            {
                "node_id": p.node_id,
                "host": p.host,
                "api_port": p.api_port,
                "is_local": p.is_local,
                "ram_total_gb": p.ram_total_gb,
                "ram_available_gb": p.ram_available_gb,
                "chip_model": p.chip_model,
                "compute_weight": p.compute_weight,
            }
            for p in self.peers()
        ]

    async def start_periodic_refresh(self, interval: float = 60.0) -> None:
        """Start background task that refreshes peer metadata periodically.

        Args:
            interval: Seconds between refresh cycles (default 60).
        """
        async def _refresh_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(interval)
                    await self.registry.refresh_all_metadata()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.warning("Periodic metadata refresh error: %s", exc)

        self._refresh_task = asyncio.create_task(_refresh_loop())

    def peers(self) -> list[PeerInfo]:
        """Snapshot of discovered peers (excludes self)."""
        return self.registry.peers()

    def count(self) -> int:
        """Number of discovered peers."""
        return self.registry.count()

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start mDNS advertisement and discovery."""
        if AsyncServiceBrowser is None:
            logger.warning("zeroconf not installed; discovery disabled")
            return

        try:
            from zeroconf import InterfaceChoice

            self._zc = AsyncZeroconf(interfaces=InterfaceChoice.All)
            logger.info("mDNS started: node_id=%s, api_port=%d", self.my_node_id, self.api_port)
            listener = _OmlxServiceListener(self.registry)

            txt_record = {
                b"node_id": self.my_node_id.encode("utf-8"),
                b"api_port": str(self.api_port).encode("utf-8"),
            }

            # Standard POSIX idiom: open UDP socket to an external host,
            # read the local interface IP — this is what zeroconf itself uses.
            local_ip = _get_local_ip()

            service_name = f"{self.my_node_id}.{_SERVICE_TYPE}"
            await self._zc.async_register_service(
                ServiceInfo(
                    type_=_SERVICE_TYPE,
                    name=service_name,
                    port=self.api_port,
                    properties=txt_record,
                    addresses=[socket.inet_aton(local_ip)],
                ),
            )

            self._browser = AsyncServiceBrowser(
                self._zc.zeroconf, _SERVICE_TYPE, listener=listener
            )

            logger.info(
                "Discovery service started: node_id=%s",
                self.my_node_id,
            )

        except Exception as exc:
            logger.warning("Discovery service start failed: %s", exc)
            self._zc = None

    async def stop(self) -> None:
        """Stop mDNS advertisement and discovery."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None

        if self._browser is not None:
            try:
                await self._browser.async_cancel()
            except Exception:
                pass
            self._browser = None

        if self._zc is not None:
            try:
                await self._zc.async_unregister_all_services()
                await self._zc.async_close()
            except Exception:
                pass
            self._zc = None

        with self._lock:
            pass  # No model list to clear

    # --- Static helpers ---

    @staticmethod
    def _default_node_id() -> str:
        """Get the machine's LocalHostName as the node identifier.

        Falls back to a short UUID if scutil is unavailable or returns
        an empty value.
        """
        try:
            import subprocess

            result = subprocess.run(
                ["scutil", "--get", "LocalHostName"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            name = result.stdout.strip()
            if name:
                return name
        except Exception:
            pass

        # Fallback: use socket hostname (may include dots)
        try:
            return socket.gethostname().split(".")[0]
        except Exception:
            pass

        # Final fallback: compact UUID
        return str(uuid.uuid4()).replace("-", "")[:12]


def _node_id_path() -> Path:
    """Get path to persistent node ID file."""
    from .settings import resolve_default_base_path

    base = resolve_default_base_path()
    return base / ".node-id"


def _load_persistent_node_id() -> str | None:
    """Load a persistent node ID from disk, or None."""
    try:
        return _node_id_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _save_persistent_node_id(node_id: str) -> None:
    """Persist a node ID to disk."""
    try:
        _node_id_path().write_text(node_id + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("Failed to save node ID: %s", exc)


def _get_local_ip() -> str:
    """Return this machine's routable IPv4 address.

    Uses the standard POSIX UDP-connect idiom: open a datagram socket
    to an arbitrary remote host, read the local interface address that
    the kernel assigns, then close.  The remote endpoint need not exist
    or be reachable — only the routing table matters.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        # 203.0.113.x is TEST-NET-3 (RFC 5737) — guaranteed unroutable
        # but the kernel will still pick an interface based on the route.
        try:
            s.settimeout(0.5)
            s.connect(("203.0.113.1", 53))
            return s.getsockname()[0]
        except OSError:
            pass
    # Fallback: try 192.0.2.1 (also TEST-NET-1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.settimeout(0.5)
            s.connect(("192.0.2.1", 53))
            return s.getsockname()[0]
        except OSError:
            pass
    # Final fallback — should never reach here on a functional network
    return socket.gethostbyname(socket.gethostname())
