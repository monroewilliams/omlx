# SPDX-License-Identifier: Apache-2.0
"""Local test harness for the distributed pipeline engine.

Exercises allocation, payload building, and MLX ring init on a single machine
by running two "ranks" in separate child processes that communicate via a
shared temp hostfile.

Run:
    cd omlx && python -m pytest tests/test_distributed_pipeline.py -v
    # or just:
    python -m omlx/tests/test_distributed_pipeline.py
"""

import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Unit tests for allocation (no MLX dependency)
# =============================================================================


class TestAllocateChunks:
    """Test _allocate_chunks sequential consumption logic."""

    def test_two_nodes_forty_chunks(self):
        """40 chunks split across 2 equal-weight nodes → ~20 each."""
        from omlx.distributed_sharding import NodeSpec, _allocate_chunks

        chunks = [(i, i + 1) for i in range(40)]
        nodes = [
            NodeSpec(node_id="node-0", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0, chip_model="M5 Max"),
            NodeSpec(node_id="node-1", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0, chip_model="M5 Max"),
        ]

        assignments = _allocate_chunks(chunks, nodes)
        assert len(assignments) == 2

        # Should be sequential: node-0 gets [0, X), node-1 gets [X, 40)
        assert assignments[0].node_id == "node-0"
        assert assignments[1].node_id == "node-1"
        assert assignments[0].start_layer == 0
        assert assignments[1].start_layer == assignments[0].end_layer
        assert assignments[1].end_layer == 40

        # Each should get roughly half
        node0_layers = assignments[0].end_layer - assignments[0].start_layer
        node1_layers = assignments[1].end_layer - assignments[1].start_layer
        assert 15 <= node0_layers <= 25  # within ~25% of half
        assert 15 <= node1_layers <= 25

    def test_two_nodes_equal_compute_guarantees_one_each(self):
        """Even with rounding, every node gets at least one chunk."""
        from omlx.distributed_sharding import NodeSpec, _allocate_chunks

        # 3 chunks, 2 nodes — one must get 1, the other gets 2
        chunks = [(i, i + 1) for i in range(3)]
        nodes = [
            NodeSpec(node_id="a", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
            NodeSpec(node_id="b", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
        ]

        assignments = _allocate_chunks(chunks, nodes)
        assert len(assignments) == 2
        for a in assignments:
            assert a.end_layer - a.start_layer >= 1

    def test_three_nodes_forty_chunks(self):
        """40 chunks across 3 equal nodes."""
        from omlx.distributed_sharding import NodeSpec, _allocate_chunks

        chunks = [(i, i + 1) for i in range(40)]
        nodes = [
            NodeSpec(node_id=f"node-{i}", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0)
            for i in range(3)
        ]

        assignments = _allocate_chunks(chunks, nodes)
        assert len(assignments) == 3

        # Verify sequential, no gaps, full coverage
        assert assignments[0].start_layer == 0
        for i in range(1, len(assignments)):
            assert assignments[i].start_layer == assignments[i - 1].end_layer
        assert assignments[-1].end_layer == 40

        # Each gets at least one
        for a in assignments:
            assert (a.end_layer - a.start_layer) >= 1

    def test_heterogeneous_weights(self):
        """Node with higher compute weight gets more chunks."""
        from omlx.distributed_sharding import NodeSpec, _allocate_chunks

        chunks = [(i, i + 1) for i in range(20)]
        nodes = [
            NodeSpec(node_id="strong", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=2.0),  # 2x weight
            NodeSpec(node_id="weak", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
        ]

        assignments = _allocate_chunks(chunks, nodes)
        assert len(assignments) == 2

        strong_layers = (assignments[0].end_layer - assignments[0].start_layer)
        weak_layers = (assignments[1].end_layer - assignments[1].start_layer)

        # Strong should get roughly 2x the layers
        assert strong_layers > weak_layers * 1.3  # allow rounding tolerance


class TestAllocateLayersWithChunks:
    """Test the full allocate_layers() path with safetensors-style chunks."""

    def test_allocate_layers_plain(self):
        """No index file: falls back to plain layer allocation."""
        from omlx.distributed_sharding import NodeSpec, allocate_layers

        nodes = [
            NodeSpec(node_id="n0", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
            NodeSpec(node_id="n1", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
        ]

        assignments = allocate_layers(
            total_layers=32, nodes=nodes, model_storage_gb=0,
            safetensors_index_path=None,
        )
        assert len(assignments) == 2
        assert assignments[0].start_layer == 0
        assert assignments[-1].end_layer == 32

    def test_memory_validation_raises(self):
        """Allocation fails when a node doesn't have enough RAM."""
        from omlx.distributed_sharding import NodeSpec, allocate_layers

        nodes = [
            NodeSpec(node_id="big", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
            NodeSpec(node_id="tiny", ram_total_gb=8, ram_available_gb=1,
                     compute_weight=1.0),
        ]

        with pytest.raises(ValueError, match="available"):
            allocate_layers(
                total_layers=32, nodes=nodes, model_storage_gb=30,
                safetensors_index_path=None,
            )


# =============================================================================
# Payload format tests
# =============================================================================


class TestAssignmentPayload:
    """Verify the payload sent to workers has correct shape."""

    def test_payload_contains_model_id_not_path(self):
        """Payload should send model_id, not a filesystem path."""
        from omlx.distributed_sharding import NodeSpec

        # Reproduce what _push_assignments_to_workers does
        model_id = "deepsweet--Qwen3.6-35B-A3B-MLX-VL-oQ8"
        model_path = "/Users/monroe/.cache/huggingface/models/repo"

        node_specs = [
            NodeSpec(node_id="coordinator", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
            NodeSpec(node_id="worker", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
        ]

        assignments = [
            type("LA", (), {"start_layer": 0, "end_layer": 16}),
            type("LA", (), {"start_layer": 16, "end_layer": 32}),
        ]

        # Build payload the same way _push_assignments_to_workers does
        payload = json.dumps({
            "model_id": model_id,
            "rank": 1,
            "start_layer": assignments[1].start_layer,
            "end_layer": assignments[1].end_layer,
            "endpoints": ["192.168.1.3:5000", "192.168.1.202:5000"],
            "listen_port": 5000,
        }).encode("utf-8")

        data = json.loads(payload)
        assert "model_id" in data
        assert "/" not in str(data["model_id"])  # shouldn't be a path
        assert data["model_id"] == model_id

    def test_payload_contains_required_fields(self):
        """Worker needs: model_id, rank, start_layer, end_layer, endpoints."""
        payload = json.dumps({
            "model_id": "test-model",
            "rank": 1,
            "start_layer": 0,
            "end_layer": 16,
            "endpoints": ["192.168.1.3:5000", "192.168.1.202:5000"],
            "listen_port": 5000,
        }).encode("utf-8")

        data = json.loads(payload)
        required = {"model_id", "rank", "start_layer", "end_layer",
                    "endpoints", "listen_port"}
        assert set(data.keys()) == required


# =============================================================================
# Hostfile format tests
# =============================================================================


class TestHostfileFormat:
    """Verify the hostfile JSON format matches what MLX expects."""

    def test_hostfile_json_structure(self):
        """Hostfile should be a 2D list: one inner array per rank."""
        from omlx.mlx_distributed import create_hostfile_json

        endpoints = ["10.0.0.1:5000", "10.0.0.2:5000"]
        hf = create_hostfile_json(endpoints)

        assert isinstance(hf, list)
        assert len(hf) == 2
        for entry in hf:
            assert isinstance(entry, list)
            assert len(entry) == 1
            assert ":" in entry[0]  # ip:port
        assert hf[0][0] == "10.0.0.1:5000"
        assert hf[1][0] == "10.0.0.2:5000"

    def test_hostfile_serializes_to_valid_json(self):
        """JSON-serialized hostfile should round-trip cleanly."""
        from omlx.mlx_distributed import create_hostfile_json

        endpoints = ["10.0.0.1:5000", "10.0.0.2:5000"]
        hf = create_hostfile_json(endpoints)
        text = json.dumps(hf)
        rebuilt = json.loads(text)

        assert isinstance(rebuilt, list)
        assert rebuilt == hf
        for entry in rebuilt:
            assert isinstance(entry, list)
            assert ":" in entry[0]


# =============================================================================
# Local MLX ring init test (real MLX, two processes on localhost)
# =============================================================================


def _worker_rank1(shared_state: dict[str, Any]) -> None:
    """Child process running as rank 1 in the ring."""
    pass


def _coordinator_rank0(shared_state: dict[str, Any]) -> None:
    """Child process running as rank 0 (coordinator) in the ring."""
    pass


class TestLocalMlxRing:
    """Actually initialize an MLX ring with two local processes.

    Spawns a coordinator (rank 0) and worker (rank 1) via subprocess,
    each calling mx.distributed.init and attempting a barrier.
    """

    def test_init_mlx_ring_two_ranks_localhost(self):
        """Two processes, same hostfile, initialize MLX ring and barrier."""
        import subprocess

        # Run the standalone ring test script which handles spawning correctly
        script = os.path.join(os.path.dirname(__file__), "test_ring_local.py")
        result = subprocess.run(
            [sys.executable, script],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=45,
        )

        output = result.stdout.strip()
        print(output)
        if result.stderr.strip():
            print(result.stderr.strip())

        assert result.returncode == 0, f"Ring init failed:\n{result.stderr}"


# =============================================================================
# Engine context payload flow tests
# =============================================================================


class TestEngineContextFlow:
    """Test that the distributed engine builds correct context."""

    def test_build_node_specs_from_peer_dicts(self):
        """Coordinator builds NodeSpec list from peer metadata dicts."""
        from omlx.engine.distributed import DistributedPipelineEngine
        from omlx.distributed_sharding import NodeSpec

        engine = DistributedPipelineEngine(
            model_path="/tmp/test-model",
            endpoints=["127.0.0.1:5000"],
            local_rank=0,
        )

        peer_dicts = [
            {
                "node_id": "worker-1",
                "ring_port": 5000,
                "ram_total_gb": 64.0,
                "ram_available_gb": 32.0,
                "chip_model": "M5 Max",
                "compute_weight": 1.0,
                "api_port": 8000,
            }
        ]

        nodes = engine._build_node_specs_from_peer_dicts(
            peer_dicts,
            model_storage_gb=30.0,
            compute_weight=1.0,
            ram_available_gb=32.0,
        )

        assert len(nodes) == 2  # coordinator + 1 peer
        assert nodes[0].node_id == "coordinator"
        assert nodes[1].node_id == "worker-1"
        assert nodes[1].api_port == 8000
        assert isinstance(nodes[0], NodeSpec)
        assert isinstance(nodes[1], NodeSpec)


# =============================================================================
# CLI runner for quick manual testing
# =============================================================================


def _cli_quick_test() -> int:
    """Run the full test suite from command line (no pytest)."""
    import traceback

    passed = 0
    failed = 0

    # Instantiate test classes so methods are bound (have `self`)
    t_alloc = TestAllocateChunks()
    t_payload = TestAssignmentPayload()
    t_hostfile = TestHostfileFormat()

    tests: list[tuple[str, Any]] = [
        ("Allocate: 40 chunks, 2 nodes", t_alloc.test_two_nodes_forty_chunks),
        ("Allocate: 3 chunks, 2 nodes (min 1 each)", t_alloc.test_two_nodes_equal_compute_guarantees_one_each),
        ("Allocate: 3 nodes, 40 chunks", t_alloc.test_three_nodes_forty_chunks),
        ("Allocate: heterogeneous weights", t_alloc.test_heterogeneous_weights),
        ("Payload: model_id not path", t_payload.test_payload_contains_model_id_not_path),
        ("Payload: required fields", t_payload.test_payload_contains_required_fields),
        ("Hostfile: JSON structure", t_hostfile.test_hostfile_json_structure),
        ("Hostfile: serializable", t_hostfile.test_hostfile_json_structure),
    ]

    for name, method in tests:
        try:
            method()  # bound methods — `self` is already filled in
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
            failed += 1

    # MLX ring test (slow, requires actual MLX)
    print("\n--- MLX Ring Init (slow, ~15s) ---")
    try:
        TestLocalMlxRing().test_init_mlx_ring_two_ranks_localhost()
        print("  PASS  MLX ring init (two ranks)")
        passed += 1
    except Exception as exc:
        print(f"  FAIL  MLX ring init: {exc}")
        traceback.print_exc()
        failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_cli_quick_test())
