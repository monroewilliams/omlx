# SPDX-License-Identifier: Apache-2.0
"""
MLX distributed pipeline parallel initialization utilities.

Creates an MLX ring group from a list of node endpoints and provides
synchronization primitives for distributed inference.

Hostfile format (JSON) — matches MLX's launch_ring output:
    ["192.168.1.1:5000", "192.168.1.2:5000"]

Each entry maps to one rank. The ring topology matches the layer order:
rank 0 -> 1 -> ... -> N-1. The back-edge (N-1 -> 0) is unused during inference.
"""

from __future__ import annotations

import io
import json
import logging
import os
import socket
import sys
import tempfile
import threading
from typing import Any, List, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def is_coordinator(local_rank: int) -> bool:
    """Check if this rank is the coordinator (rank 0)."""
    return local_rank == 0


def init_mlx_distributed(
    endpoints: List[str],
    local_rank: int,
    ring_port: Optional[int] = None,
) -> mx.distributed.Group | None:
    """Initialize MLX ring distributed group.

    Creates a hostfile from the endpoint list, sets required environment
    variables, and calls mx.distributed.init(). This blocks until all
    ranks connect (strict=True).

    MLX binds to the exact port specified in the hostfile entry for own rank.
    The ring_port is reserved via the trigger-worker protocol before this call —
    the placeholder socket is closed and MLX binds to that port.

    Args:
        endpoints: List of "ip:port" strings for all ranks (from ring list).
        local_rank: This node's rank in the ring (0 = coordinator).
        ring_port: Port this rank should listen on. If None, uses the port
                   from endpoints[local_rank].

    Returns:
        The MLX distributed group if initialized successfully, None otherwise.

    Raises:
        ValueError: If local_rank is out of range or endpoints is empty.
    """
    if not endpoints:
        raise ValueError("endpoints list cannot be empty")

    num_ranks = len(endpoints)
    if not (0 <= local_rank < num_ranks):
        raise ValueError(
            f"local_rank {local_rank} out of range [0, {num_ranks})"
        )

    # Parse endpoints to get IPs and ports
    def _parse_endpoint(ep: str) -> tuple[str, int]:
        ep = ep.strip()
        if ":" in ep:
            ip, port_str = ep.rsplit(":", 1)
            return ip, int(port_str)
        return ep, ring_port if ring_port is not None else 5000

    parsed = [_parse_endpoint(ep) for ep in endpoints]
    own_ip, _own_port = parsed[local_rank]

    # Build hostfile JSON: 2D array of ip:port per rank
    # Own rank listens on 0.0.0.0:<ring_port>; others connect via their IPs
    own_port = ring_port if ring_port is not None else _own_port
    hostfile_data = []
    for i, (ip, port) in enumerate(parsed):
        if i == local_rank:
            hostfile_data.append([f"0.0.0.0:{own_port}"])
        else:
            hostfile_data.append([f"{ip}:{port}"])

    # Write hostfile to a temp file visible to all ranks.
    hostfile_path = _write_hostfile(hostfile_data, local_rank)
    if hostfile_path is None:
        logger.warning("Failed to write hostfile; distributed mode disabled")
        return None

    env_vars = {
        "MLX_HOSTFILE": hostfile_path,
        "MLX_RANK": str(local_rank),
        "MLX_RING_VERBOSE": "1",
    }
    for key, value in env_vars.items():
        logger.info("Setting %s=%s", key, value)
        os.environ[key] = value

    logger.info(
        "Initializing MLX ring: %d ranks, local_rank=%d, own_addr=0.0.0.0:%d",
        num_ranks,
        local_rank,
        own_port,
    )

    # Capture MLX's C++ stderr (verbose ring logs go to std::cerr)
    old_stderr = sys.stderr
    stderr_capture = io.StringIO()

    def _flush_ring_logs():
        for line in stderr_capture.getvalue().splitlines():
            if "[ring]" in line:
                logger.info("%s (rank %d)", line.strip(), local_rank)

    try:
        sys.stderr = stderr_capture
        group = mx.distributed.init(backend="ring", strict=True)
    except Exception as exc:
        _flush_ring_logs()
        logger.error("MLX ring init failed (rank %d): %s", local_rank, exc)
        raise
    finally:
        sys.stderr = old_stderr
        _flush_ring_logs()

    rank = group.rank()
    size = group.size()
    logger.info(
        "MLX ring initialized: rank=%d/%d", rank, size
    )
    return group


def barrier(group: mx.distributed.Group | None = None) -> None:
    """Synchronize all ranks via an all-reduce noop.

    Blocks until all ranks have reached the barrier. Uses mx.distributed
    primitives since MLX has no built-in collective_barrier for ring backend.

    Args:
        group: The MLX distributed group. After init, collectives work with
            the implicit global group, so passing None is fine.
    """
    # Simple barrier: each rank contributes a value to an all-sum,
    # and every rank waits until the sum equals size.
    value = mx.array(1.0)

    # Get group and size if provided; otherwise use implicit global group.
    g = group
    size: int | None = group.size() if group is not None else None

    result = mx.distributed.all_sum(value, group=g)

    # If we don't have the size (no explicit group), loop until we can get it
    # from the all_sum result, or assume size 2 (coordinator + worker).
    if size is None:
        # Try to infer size — the barrier will timeout after a few iterations.
        for _ in range(20):
            mx.eval(result)
            current_sum = mx.array(result).item()
            if current_sum > 0:
                size = int(current_sum)
                break
            mx.eval(result)
            result = mx.distributed.all_sum(value, group=g)
        if size is None:
            size = 2  # fallback

    # Block until all ranks contributed
    while mx.array(result).item() != size:
        mx.eval(result)
        result = mx.distributed.all_sum(value, group=g)

    # Final evaluation to ensure all pending ops complete
    mx.eval(result)


def create_hostfile_json(
    endpoints: List[str],
) -> list:
    """Create a hostfile JSON structure from endpoints.

    Returns a 2D list of "ip:port" strings per rank, matching MLX's
    ring backend hostfile format. Each rank gets its own inner array
    (for multiple connections per peer).

    Args:
        endpoints: List of "ip:port" strings (ports must be present).

    Returns:
        List of ["ip:port"] arrays ready to serialize as JSON.
    """

    resolved = []
    for ep in endpoints:
        ep = ep.strip()
        if ":" in ep:
            ip, port_str = ep.rsplit(":", 1)
            resolved.append((ip, int(port_str)))

    return [[f"{ip}:{port}"] for ip, port in resolved]


def _write_hostfile(hostfile_data: list, local_rank: int) -> str | None:
    """Write hostfile to disk in a location all ranks can find.

    For local testing, writes to /tmp/omlx_distributed_rank_<rank>.json.
    For multi-node, returns None so callers can use a shared path instead.

    Args:
        hostfile_data: List of "ip:port" strings for the hostfile.
        local_rank: This node's rank.

    Returns:
        Path to the written hostfile, or None if writing failed.
    """
    try:
        path = f"/tmp/omlx_distributed_rank_{local_rank}.json"
        with open(path, "w") as f:
            json.dump(hostfile_data, f)
        return path
    except OSError as exc:
        logger.error("Failed to write hostfile: %s", exc)
        return None
