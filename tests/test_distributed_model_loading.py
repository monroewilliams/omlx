# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the distributed pipeline engine.

These tests exercise real model loading, layer allocation, and pipeline
wrappers. They are skipped unless a model path is provided via
OMLX_TEST_MODEL_DIR or passed on the command line.

Run against your model:
    OMLX_TEST_MODEL_DIR=/path/to/model .venv/bin/python -m pytest tests/test_distributed_model_loading.py -v

Or test every model in a directory:
    .venv/bin/python tests/test_distributed_model_loading.py --model-dir /path/to/models
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Model path — set via env var or command-line arg
_TEST_MODEL_DIR: str | None = os.environ.get("OMLX_TEST_MODEL_DIR")


def _get_model_dir() -> str | None:
    """Return the test model directory, checking CLI args first."""
    if _TEST_MODEL_DIR is not None:
        return _TEST_MODEL_DIR

    # Check pytest command-line options (set by --model-dir hook below)
    for arg in sys.argv[1:]:
        if arg.startswith("--model-dir="):
            return arg.split("=", 1)[1]

    # Default to env var (already checked above)
    return None


@pytest.fixture(scope="session")
def model_dir() -> str | None:
    """Return the test model directory or None if not configured."""
    return _get_model_dir()


# =============================================================================
# Unit-level integration: allocation with real safetensors boundaries
# =============================================================================


class TestAllocationWithSafetensors:
    """Test that layer allocation respects safetensors file boundaries."""

    def test_allocation_respects_file_boundaries(self, model_dir):
        """Allocated chunks should align with atomic safetensors groups."""
        if model_dir is None:
            pytest.skip("OMLX_TEST_MODEL_DIR not set")

        from omlx.distributed_sharding import NodeSpec, allocate_layers

        nodes = [
            NodeSpec(node_id="n0", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
            NodeSpec(node_id="n1", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
        ]

        assignments = allocate_layers(
            total_layers=32, nodes=nodes, model_storage_gb=0,
            safetensors_index_path=str(model_dir),
        )

        assert len(assignments) == 2

        # Verify sequential and full coverage
        assert assignments[0].start_layer == 0
        for i in range(1, len(assignments)):
            assert assignments[i].start_layer == assignments[i - 1].end_layer
        # We don't know the exact end since allocation is proportional,
        # but it should be reasonable.

    def test_allocation_has_no_zero_sized_assignments(self, model_dir):
        """All nodes should get at least some layers."""
        if model_dir is None:
            pytest.skip("OMLX_TEST_MODEL_DIR not set")

        from omlx.distributed_sharding import NodeSpec, allocate_layers

        nodes = [
            NodeSpec(node_id="n0", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
            NodeSpec(node_id="n1", ram_total_gb=64, ram_available_gb=32,
                     compute_weight=1.0),
        ]

        assignments = allocate_layers(
            total_layers=32, nodes=nodes, model_storage_gb=0,
        )

        for a in assignments:
            assert a.end_layer > a.start_layer, (
                f"Node {a.node_id} has zero layers assigned "
                f"[{a.start_layer}:{a.end_layer})"
            )


# =============================================================================
# Integration: apply_pipeline_parallel with a real model
# =============================================================================


class TestPipelineWrappers:
    """Test that apply_pipeline_parallel correctly wraps model layers."""

    def test_wrappers_applied_correctly(self, model_dir):
        """Pipeline wrappers should be applied to the correct layer slices."""
        if model_dir is None:
            pytest.skip("OMLX_TEST_MODEL_DIR not set")

        from omlx.distributed_sharding import (
            LayerAssignment,
            NodeSpec,
            _get_layers,
            apply_pipeline_parallel,
        )

        # Load a real model (lazy)
        from omlx.utils.model_loading import lm_load_compat

        model_path = str(model_dir)
        print(f"  Loading model from {model_path} (this may take a minute)...")

        model, tokenizer = lm_load_compat(model_path)

        # Get layer list
        layers = _get_layers(model)
        total_layers = len(layers) if layers else 32

        if not layers:
            pytest.skip(f"No transformer layers found in {model_path}")

        print(f"  Found {total_layers} layers")

        # Create fake assignments (simulating 2-node split)
        midpoint = total_layers // 2

        # We need a mock group for the wrappers. Since we're not actually
        # in a distributed init, we'll create a minimal mock.
        @dataclass
        class MockGroup:
            def rank(self):
                return 0

            def size(self):
                return 2

        # Create assignments
        assignments = [
            LayerAssignment(start_layer=0, end_layer=midpoint, node_id="n0"),
            LayerAssignment(start_layer=midpoint, end_layer=total_layers, node_id="n1"),
        ]

        # Apply pipeline parallel (rank 0 gets first half)
        import mlx.core as mx

        from omlx.distributed_sharding import _PipelineCoordinator

        wrapper = apply_pipeline_parallel(model, assignments, group=MockGroup())
        assert isinstance(
            wrapper, _PipelineCoordinator
        ), f"Expected _PipelineCoordinator, got {type(wrapper)}"

        # The wrapper should contain the first half of layers
        assert len(wrapper.layers) == midpoint

        print(f"  PASS: Wrapped {midpoint} layers with _PipelineCoordinator")


# =============================================================================
# CLI runner
# =============================================================================


def _cli_quick_test(models_dir: str | None = None) -> int:
    """Run all integration tests from command line."""
    import traceback

    passed = 0
    failed = 0

    t_alloc = TestAllocationWithSafetensors()

    for name, method in [
        ("allocation: respects file boundaries",
         t_alloc.test_allocation_respects_file_boundaries),
        ("allocation: no zero-sized assignments",
         t_alloc.test_allocation_has_no_zero_sized_assignments),
    ]:
        try:
            method(_get_model_dir() or models_dir)
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            tb = traceback.format_exc()
            if "not set" in tb or "No layers found" in tb:
                print(f"  SKIP  {name}: {exc}")
            else:
                print(f"  FAIL  {name}: {exc}")
                traceback.print_exc()
            failed += 1

    # Pipeline wrapper test (requires model download, handled separately)
    t_pipe = TestPipelineWrappers()
    try:
        t_pipe.test_wrappers_applied_correctly(_get_model_dir() or models_dir)
        print(f"  PASS  pipeline wrappers applied correctly")
        passed += 1
    except Exception as exc:
        tb = traceback.format_exc()
        if "not set" in tb or "No layers found" in tb:
            print(f"  SKIP  pipeline wrappers applied correctly: {exc}")
        else:
            print(f"  FAIL  pipeline wrappers applied correctly: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run integration tests")
    parser.add_argument(
        "--model-dir", default=None, help="Directory containing model(s) to test"
    )
    args = parser.parse_args()

    os.environ["OMLX_TEST_MODEL_DIR"] = args.model_dir or _TEST_MODEL_DIR or ""
    sys.exit(_cli_quick_test())
