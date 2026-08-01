# SPDX-License-Identifier: Apache-2.0
"""
Distributed pipeline parallel sharding for MLX models.

Provides:
- Compute-weighted layer allocation across heterogeneous nodes
- Pipeline boundary wrappers (recv/send) for MLX nn.Module trees
- Micro-batched prefill support via chunked forwarding

Pipeline topology (ring): rank 0 → 1 → … → N-1 → 0
Data flows forward (0→1→…), then the last rank sends back via dst=-1,
which unblocks rank 0's recv_like from src=-1.
"""

from __future__ import annotations

import json
import logging
import math
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx.nn.layers import transformer as _transf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline execution context (thread-local, safe for async/threading)
# ---------------------------------------------------------------------------

_PipelineState: ContextVar[dict[str, object] | None] = ContextVar(
    "_pipeline_ctx", default=None,
)


class pipeline_context:
    """Set pipeline execution parameters for the current forward pass.

    Used by PipelineLastLayer to decide whether to request logits back
    from the pipeline. Decode always wants logits; prefill chunks do not.

    Usage::

        with pipeline_context(want_logits=True):
            logits = model.decode(batch)
    """

    def __init__(
        self,
        *,
        want_logits: bool = False,
        start_pos: int = 0,
    ) -> None:
        self._want_logits = want_logits
        self._start_pos = start_pos
        self._token: object | None = None

    def __enter__(self):
        state = {"want_logits": self._want_logits, "start_pos": self._start_pos}
        self._token = _PipelineState.set(state)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _PipelineState.reset(self._token)


def get_pipeline_state() -> dict[str, object] | None:
    """Return current pipeline context (or None)."""
    return _PipelineState.get()


def pipeline_want_logits() -> bool:
    """Return True if the current forward pass wants logits from the last rank."""
    state = _PipelineState.get()
    return bool(state) if isinstance(state, dict) else False


def pipeline_start_pos() -> int:
    """Return start position from current pipeline context."""
    state = _PipelineState.get()
    return int(state.get("start_pos", 0)) if isinstance(state, dict) else 0


def _pipeline_dst_rank(rank: int, world_size: int) -> int:
    """Next rank in the ring."""
    return (rank + 1) % world_size


def _pipeline_src_rank(rank: int, world_size: int) -> int:
    """Previous rank in the ring (who sends to us)."""
    return (rank - 1) % world_size


def _pipeline_last_rank(world_size: int) -> int:
    """Last rank (computes logits on decode)."""
    return world_size - 1

# =============================================================================
# Layer allocation (compute-weighted)
# =============================================================================

@dataclass
class NodeSpec:
    """Describes one participating node."""

    node_id: str  # IP for identification
    ram_total_gb: float  # From psutil.virtual_memory().total
    ram_available_gb: float  # Available at allocation time (minus system overhead)
    compute_weight: float  # Normalized compute score (0.0-1.0)
    chip_model: str = ""  # e.g., "M5 Max", "M3 Max"
    api_port: int = 8000  # HTTP API port for coordinator→worker communication


@dataclass
class LayerAssignment:
    """Assigns a contiguous range of layers to one node."""

    start_layer: int
    end_layer: int  # exclusive
    node_id: str


def allocate_layers(
    total_layers: int,
    nodes: list[NodeSpec],
    model_storage_gb: float = 0.0,
    model_overhead_factor: float = 1.3,
    safetensors_index_path: str | None = None,
) -> list[LayerAssignment]:
    """Allocate layers to nodes proportionally by available RAM.

    Strategy: each node gets layers proportional to its share of total RAM.
    Compute weights are recorded but not used for allocation — RAM is the
    dominant constraint. We may tune compute weighting later for mixed-chip
    clusters (e.g., M5 + M1).

    Memory validation: required_ram = model_storage * overhead_factor *
    (node_layers / total_layers). Warn if node.ram_available < required_ram.

    Args:
        total_layers: Total number of transformer layers in the model.
        nodes: List of participating node specifications.
        model_storage_gb: Model checkpoint size on disk (GB).
        model_overhead_factor: Multiplier for memory overhead. Default 1.3.
        safetensors_index_path: Unused (legacy symlink sharding removed).

    Returns:
        List of LayerAssignment in rank order (same as nodes input).
    """
    if not nodes:
        raise ValueError("nodes list cannot be empty")

    if total_layers <= 0:
        raise ValueError(f"total_layers must be positive, got {total_layers}")

    # Allocate layers proportionally by available RAM using largest-remainder method.
    total_ram = sum(n.ram_available_gb for n in nodes)
    if total_ram <= 0:
        raise ValueError("Sum of available RAM across nodes must be positive")

    raw_shares = [(n.ram_available_gb / total_ram) * total_layers for n in nodes]
    floored = [int(math.floor(s)) for s in raw_shares]
    remainders = [(s - floored[i], i) for i, s in enumerate(raw_shares)]
    allocated = total_layers - sum(floored)
    if allocated < 0:
        allocated = 0

    remainders.sort(key=lambda x: -x[0])
    for i in range(allocated):
        floored[remainders[i][1]] += 1

    assignments: list[LayerAssignment] = []
    start = 0
    for node, count in zip(nodes, floored):
        if count > 0:
            assignments.append(
                LayerAssignment(
                    start_layer=start,
                    end_layer=start + count,
                    node_id=node.node_id,
                )
            )
        elif start < total_layers and not assignments:
            # First node with 0 layers gets everything (single-node fallback)
            assignments.append(
                LayerAssignment(
                    start_layer=0,
                    end_layer=total_layers,
                    node_id=node.node_id,
                )
            )
        start += count

    # Memory validation (warning only)
    if model_storage_gb > 0:
        for i, assignment in enumerate(assignments):
            node_layers = assignment.end_layer - assignment.start_layer
            required_ram = model_storage_gb * model_overhead_factor * (node_layers / total_layers)
            if nodes[i].ram_available_gb < required_ram:
                logger.warning(
                    "Node %s: %.0fGB RAM available, ~%.0fGB estimated required (%d/%d layers)",
                    nodes[i].node_id, nodes[i].ram_available_gb,
                    required_ram, node_layers, total_layers,
                )

    return assignments

def get_inner_model(model: nn.Module) -> nn.Module:
    """Find the inner model object (model.model, model.transformer, etc.).

    Matches exo's get_inner_model.
    """
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "transformer", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "language_model", None)
    if isinstance(inner, nn.Module):
        inner_inner = getattr(inner, "model", None)
        if isinstance(inner_inner, nn.Module):
            return inner_inner

    inner = getattr(model, "backbone", None)
    if isinstance(inner, nn.Module):
        return inner

    # Fallback: try common attr names for VLM variants.
    for attr in ("vision_model", "text_model", "llm"):
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module):
            return inner

    raise ValueError(
        f"Cannot find inner model in {model.__class__.__name__}. "
        "Expected 'model', 'transformer', 'language_model', or 'backbone'."
    )


def _fix_layer_indices(model: nn.Module, layers: list) -> None:
    """Fix model-specific layer indices after slicing.

    Models like Qwen3.5/Next use hybrid attention (full attn + SSM) and store
    indices into the layer list. After slicing, these indices are stale.
    Matches exo's pipeline_auto_parallel index fixes.
    """
    inner = get_inner_model(model)

    # Qwen3.5 / Qwen3Next: fa_idx (full attention), ssm_idx (linear/SSM)
    try:
        from mlx_lm.models.qwen3_5 import Qwen3_5TextModel
        from mlx_lm.models.qwen3_next import Qwen3NextModel

        if isinstance(inner, (Qwen3_5TextModel, Qwen3NextModel)):
            full_attn = [
                i for i, l in enumerate(layers)
                if not getattr(l, "is_linear", True)
            ]
            linear = [
                i for i, l in enumerate(layers)
                if getattr(l, "is_linear", False)
            ]
            inner.fa_idx = full_attn[0] if full_attn else 0
            inner.ssm_idx = linear[0] if linear else 0
            if not full_attn or not linear:
                _patch_hybrid_cache(
                    model, inner.fa_idx, bool(full_attn),
                    inner.ssm_idx, bool(linear),
                )
    except ImportError:
        pass

    # Step 3.5: num_layers, _swa_idx (sliding), _full_idx
    try:
        from mlx_lm.models.step3p5 import Step3p5Model

        if isinstance(inner, Step3p5Model):
            inner.num_layers = len(layers)
            sliding = [
                i for i, l in enumerate(layers)
                if getattr(l, "is_sliding", False)
            ]
            full = [
                i for i, l in enumerate(layers)
                if not getattr(l, "is_sliding", True)
            ]
            inner._swa_idx = 0 if not sliding else sliding[0]
            inner._full_idx = 0 if not full else full[0]
    except ImportError:
        pass

    # GptOssMoE: layer_types, swa_idx, ga_idx
    try:
        from mlx_lm.models.gpt_oss import GptOssMoeModel

        if isinstance(inner, GptOssMoeModel):
            inner.layer_types = inner.layer_types[
                0:len(layers)
            ] if hasattr(inner, "layer_types") else []
            inner.swa_idx = (
                0 if not getattr(inner, "layer_types", [])
                or "sliding_attention" not in inner.layer_types
                else inner.layer_types.index("sliding_attention")
            )
            inner.ga_idx = (
                0 if not getattr(inner, "layer_types", [])
                or "full_attention" not in inner.layer_types
                else inner.layer_types.index("full_attention")
            )
    except ImportError:
        pass

    # NemotronH: fa_idx, ssm_idx (complex block_type counting)
    try:
        from mlx_lm.models.nemotron_h import NemotronHModel

        if isinstance(inner, NemotronHModel):
            cache_idx = 0
            fa_idx: int | None = None
            ssm_idx: int | None = None
            for layer in layers:
                block_type = getattr(layer, "block_type", None)
                if block_type == "*":
                    if fa_idx is None:
                        fa_idx = cache_idx
                    cache_idx += 1
                elif block_type == "M":
                    if ssm_idx is None:
                        ssm_idx = cache_idx
                    cache_idx += 1
            inner.fa_idx = fa_idx if fa_idx is not None else 0
            inner.ssm_idx = ssm_idx if ssm_idx is not None else 0
            if fa_idx is None or ssm_idx is None:
                _patch_hybrid_cache(
                    model, inner.fa_idx, fa_idx is not None,
                    inner.ssm_idx, ssm_idx is not None,
                )
    except ImportError:
        pass


def _patch_hybrid_cache(
    model: nn.Module,
    fa_idx: int,
    has_full_attn: bool,
    ssm_idx: int,
    has_linear: bool,
) -> None:
    """Patch model.make_cache to handle missing attention types in shard.

    When a shard only has full-attention or only has SSM layers, the missing
    cache entry needs its make_mask patched to ignore unexpected kwargs.
    """
    original_make_cache = model.make_cache

    def patched():
        cache = original_make_cache()
        if not has_full_attn:
            entry = cache[fa_idx]
            orig_mm = entry.make_mask
            entry.make_mask = lambda n, **_kw: orig_mm(n)  # type: ignore
        if not has_linear:
            entry = cache[ssm_idx]
            orig_mm = entry.make_mask
            def _ssm_mask(n: int, **kw):
                return orig_mm(n, **kw) if kw else None  # type: ignore
            entry.make_mask = _ssm_mask  # type: ignore
        return cache

    model.make_cache = patched



def apply_pipeline_parallel(
    model: nn.Module,
    assignments: list[LayerAssignment],
    group: mx.distributed.Group | None = None,
) -> nn.Module:
    """Slice model layers for this rank and wrap with pipeline comm.

    Each rank's layer list is wrapped:
    - First layer: ``PipelineFirstLayer`` (recv hidden states from prev rank)
    - Middle layers: ``_EvalLayer`` (break lazy graph per-layer)
    - Last layer: ``PipelineLastLayer`` (send hidden states to next rank,
      optionally recv logits from last rank)

    Worker loop (and coordinator's standard Scheduler) handles all inference.
    """
    if group is not None:
        rank = group.rank()
    else:
        rank = 0

    if not assignments:
        logger.warning("No layer assignments; model unchanged")
        return model

    own = None
    for i, a in enumerate(assignments):
        if i == rank:
            own = a
            break

    if own is None:
        raise ValueError(
            f"Rank {rank} has no assignment in {len(assignments)} assignments"
        )

    logger.info(
        "Rank %d: layers [%d:%d] of %d total",
        rank,
        own.start_layer,
        own.end_layer,
        own.end_layer - own.start_layer,
    )

    layer_list = _get_layers(model)
    if not layer_list:
        raise ValueError("Could not find transformer layers in model.")

    if own.end_layer - own.start_layer <= 0:
        logger.warning("Rank %d has no layers assigned", rank)
        return model

    # Slice to this rank's layers.
    our_layers = layer_list[own.start_layer : own.end_layer]
    world_size = group.size() if group is not None else 1

    # Build wrapped layers: PipelineFirst + EvalLayer* + PipelineLast
    wrapped_layers = _wrap_pipeline_layers(
        our_layers, rank=rank, world_size=world_size,
        group=group, model=model,
    )

    # Update model-specific layer indices after slicing (matches exo pattern).
    _fix_layer_indices(model, wrapped_layers)

    _set_layers(model, wrapped_layers)

    logger.info(
        "Rank %d: layers sliced and wrapped (layers %d-%d)",
        rank, own.start_layer, own.end_layer - 1,
    )

    return model


def _wrap_pipeline_layers(
    layers: list, rank: int, world_size: int,
    group: mx.distributed.Group | None,
    model: nn.Module | None = None,
) -> list:
    """Wrap layers with per-layer eval (no pipeline comm — handled by caller).

    Per-layer eval breaks MLX's lazy graph so Metal only holds one layer's
    intermediates at a time. Pipeline recv/send stays in the worker loop
    and coordinator scheduler, not inside layer wrappers.
    """
    class _EvalLayer:
        """Wrapper that calls original layer then mx.eval(output) to break lazy graph."""
        __slots__ = ("_orig",)

        def __init__(self, orig):
            self._orig = orig

        def __call__(self, x, mask=None, cache=None):
            out = self._orig(x, mask=mask, cache=cache)
            mx.eval(out)
            return out

        def __getattr__(self, name):
            return getattr(self._orig, name)

        @property
        def is_linear(self):
            return getattr(self._orig, "is_linear", False)

    # All layers get _EvalLayer wrapper only. Pipeline comm handled by caller.
    return [_EvalLayer(layer) for layer in layers]


def _get_layers(model: nn.Module) -> list:
    """Extract the list of transformer layers from a model.

    Handles common attribute names: model.layers, model.model.layers
    (VLM), model.blocks, etc.

    Returns:
        List of layer modules or empty list if not found.
    """
    # Direct layers attribute
    if hasattr(model, "layers"):
        val = model.layers
        if isinstance(val, (list, tuple)):
            return list(val)

    # VLM pattern: language_model.model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            val = lm.model.layers
            if isinstance(val, (list, tuple)):
                return list(val)

    # blocks pattern
    if hasattr(model, "blocks"):
        val = model.blocks
        if isinstance(val, (list, tuple)):
            return list(val)

    return []


def _set_layers(model: nn.Module, layers: list) -> None:
    """Replace the model's transformer layer list.

    Follows exo's pattern: get_inner_model() → inner.layers = layers.
    Handles model.model, model.transformer, model.backbone, language_model.model,
    and fallback to direct layers/blocks.
    """
    # Exo-style inner model resolution
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        _try_set_layers_attr(inner, layers)
        return

    inner = getattr(model, "transformer", None)
    if isinstance(inner, nn.Module):
        _try_set_layers_attr(inner, layers)
        return

    inner = getattr(model, "backbone", None)
    if isinstance(inner, nn.Module):
        _try_set_layers_attr(inner, layers)
        return

    inner = getattr(model, "language_model", None)
    if isinstance(inner, nn.Module):
        inner_inner = getattr(inner, "model", None)
        if isinstance(inner_inner, nn.Module):
            _try_set_layers_attr(inner_inner, layers)
            return
        # language_model itself might have layers
        _try_set_layers_attr(inner, layers)
        return

    # Direct fallback: model itself has layers/blocks
    _try_set_layers_attr(model, layers)


def _try_set_layers_attr(obj: nn.Module, layers: list) -> None:
    """Try to set .layers or .blocks on an object. Raises if neither exists."""
    if hasattr(obj, "layers"):
        obj.layers = layers
        return
    if hasattr(obj, "blocks"):
        obj.blocks = layers
        return
    raise ValueError(
        f"Could not find settable layer attribute (layers/blocks) on {obj.__class__.__name__}"
    )
