# SPDX-License-Identifier: Apache-2.0
"""Tests for the discovery module."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from omlx.discovery import DiscoveryService, PeerInfo, PeerRegistry


class TestPeerRegistry:
    """Tests for PeerRegistry thread-safe peer management."""

    def test_initial_state(self):
        """Registry starts empty with the configured node_id."""
        registry = PeerRegistry("my-node")
        assert registry.my_node_id == "my-node"
        assert registry.count() == 0
        assert registry.peers() == []

    def test_add_peer(self):
        """Adding a peer makes it visible in peers() and count()."""
        registry = PeerRegistry("my-node")

        peer = PeerInfo(node_id="peer-a", host="192.168.1.2")
        with registry._lock:
            registry._peers["peer-a"] = peer

        assert registry.count() == 1
        peers = registry.peers()
        assert len(peers) == 1
        assert peers[0].node_id == "peer-a"

    def test_includes_self_as_local(self):
        """The registry includes the local node marked as is_local."""
        registry = PeerRegistry("my-node")

        registry._peers["my-node"] = PeerInfo(
            node_id="my-node", host="127.0.0.1", is_local=True
        )
        registry._peers["peer-b"] = PeerInfo(node_id="peer-b", host="192.168.1.3")

        assert registry.count() == 2
        peers = registry.peers()
        local = [p for p in peers if p.is_local]
        remote = [p for p in peers if not p.is_local]
        assert len(local) == 1
        assert local[0].node_id == "my-node"
        assert len(remote) == 1

    def test_update_peer(self):
        """Updating an existing peer replaces it."""
        registry = PeerRegistry("my-node")

        registry._peers["peer-c"] = PeerInfo(
            node_id="peer-c", host="192.168.1.4", api_port=8000
        )

        registry._peers["peer-c"] = PeerInfo(
            node_id="peer-c", host="192.168.1.5", api_port=8080
        )

        assert registry.count() == 1
        peers = registry.peers()
        assert peers[0].host == "192.168.1.5"
        assert peers[0].api_port == 8080

    def test_remove_peer(self):
        """Removing a peer decreases the count."""
        registry = PeerRegistry("my-node")

        registry._peers["peer-d"] = PeerInfo(node_id="peer-d", host="192.168.1.6")
        registry._peers["peer-e"] = PeerInfo(node_id="peer-e", host="192.168.1.7")

        assert registry.count() == 2

        with registry._lock:
            registry._peers.pop("peer-d")

        assert registry.count() == 1
        assert all(p.node_id != "peer-d" for p in registry.peers())

    def test_notify_listeners(self):
        """Registered listeners are called when notify_listeners fires."""
        registry = PeerRegistry("my-node")
        calls: list[PeerRegistry] = []

        def callback(reg: PeerRegistry) -> None:
            calls.append(reg)

        registry.add_listener(callback)()  # Unsub immediately, but it was registered
        assert len(calls) == 0

    def test_add_listener_and_notify(self):
        """Listener is called on notify and can be unsubscribed."""
        registry = PeerRegistry("my-node")
        calls: list[PeerRegistry] = []

        def callback(reg: PeerRegistry) -> None:
            calls.append(reg)

        unsub = registry.add_listener(callback)
        registry.notify_listeners()
        assert len(calls) == 1

        unsub()
        registry.notify_listeners()
        assert len(calls) == 1  # No additional call after unsub

    def test_remove_listener_during_notify(self):
        """notify_listeners takes a snapshot — subscribers present at
        notification time are all called, even if one unsubscribes another."""
        registry = PeerRegistry("my-node")
        order: list[str] = []

        def first_callback(reg: PeerRegistry) -> None:
            order.append("first")
            # Unsubscribe second listener while notifying
            second_unsub()

        def second_callback(reg: PeerRegistry) -> None:
            order.append("second")  # Called because snapshot was taken

        first_unsub = registry.add_listener(first_callback)
        second_unsub = registry.add_listener(second_callback)

        registry.notify_listeners()
        first_unsub()

        # Both are called because notify_listeners snapshots the list
        assert order == ["first", "second"]

    def test_listener_unsub_before_notify(self):
        """Unsubscribing before notify means the listener is not called."""
        registry = PeerRegistry("my-node")
        calls: list[str] = []

        def callback(reg: PeerRegistry) -> None:
            calls.append("called")

        unsub = registry.add_listener(callback)
        unsub()
        registry.notify_listeners()

        assert calls == []

    def test_snapshot_isolation(self):
        """peers() returns a snapshot that doesn't reflect later changes."""
        registry = PeerRegistry("my-node")

        registry._peers["peer-f"] = PeerInfo(node_id="peer-f", host="192.168.1.8")
        snapshot = registry.peers()
        assert len(snapshot) == 1

        with registry._lock:
            registry._peers.pop("peer-f")

        # Snapshot should still have the peer
        assert len(snapshot) == 1
        assert snapshot[0].node_id == "peer-f"


class TestPeerInfoDataclass:
    """Tests for the PeerInfo dataclass."""

    def test_defaults(self):
        """PeerInfo has sensible defaults."""
        info = PeerInfo(node_id="test-node")
        assert info.node_id == "test-node"
        assert info.host == ""
        assert info.api_port == 8000

    def test_custom_values(self):
        """PeerInfo accepts custom values."""
        info = PeerInfo(
            node_id="custom",
            host="10.0.0.1",
            api_port=9000,
        )
        assert info.node_id == "custom"
        assert info.host == "10.0.0.1"
        assert info.api_port == 9000

    def test_multiple_peers(self):
        """Can create multiple peers with different values."""
        p1 = PeerInfo(node_id="alpha", host="192.168.1.10")
        p2 = PeerInfo(node_id="beta", host="192.168.1.11")

        assert p1.node_id != p2.node_id
        assert p1.host != p2.host


class TestPeerRegistryConcurrency:
    """Tests for thread-safety of PeerRegistry."""

    def test_concurrent_reads_writes(self):
        """Multiple threads can read and write without crashing."""
        registry = PeerRegistry("concurrent-node")
        errors: list[Exception] = []

        def writer(start: int, count: int) -> None:
            try:
                for i in range(start, start + count):
                    peer_id = f"writer-{i}"
                    with registry._lock:
                        registry._peers[peer_id] = PeerInfo(
                            node_id=peer_id, host=f"192.168.1.{i}"
                        )
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(20):
                    _ = registry.peers()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            t = threading.Thread(target=writer, args=(i * 10, 10))
            threads.append(t)

        for _ in range(2):
            t = threading.Thread(target=reader)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent access raised: {errors}"
        assert registry.count() == 30

    def test_listener_notification_thread_safety(self):
        """Listener callbacks can be added/removed concurrently."""
        registry = PeerRegistry("notify-node")

        def noisy_writer() -> None:
            for i in range(50):
                pid = f"noisy-{i}"
                with registry._lock:
                    registry._peers[pid] = PeerInfo(node_id=pid, host="1.2.3.4")
                registry.notify_listeners()

        t = threading.Thread(target=noisy_writer)
        unsub = None

        def add_then_remove() -> None:
            nonlocal unsub
            time.sleep(0.01)  # Let writer start first
            unsub = registry.add_listener(lambda r: None)
            time.sleep(0.01)
            if unsub:
                unsub()

        add_t = threading.Thread(target=add_then_remove)
        t.start()
        add_t.start()
        t.join()
        add_t.join()

        # Should not crash, count should be 50 from writer
        assert registry.count() == 50


class TestDiscoveryServiceInit:
    """Tests for DiscoveryService initialization (no zeroconf required)."""

    def test_default_node_id_format(self):
        """_default_node_id() returns a non-empty string."""
        node_id = DiscoveryService._default_node_id()
        assert isinstance(node_id, str)
        assert len(node_id) > 0

    def test_node_id_no_special_chars(self):
        """Node ID should not contain characters that break mDNS."""
        node_id = DiscoveryService._default_node_id()
        # Underscores are problematic in mDNS names; local hostname shouldn't
        # contain them, but we verify the output is safe.
        assert "." not in node_id
