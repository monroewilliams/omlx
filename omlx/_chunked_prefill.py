# SPDX-License-Identifier: Apache-2.0
"""
Shared chunked prefill utility — used by both distributed and non-distributed paths.

Mirrors the scheduler's chunked prefill pattern:
- Splits input into step_size token chunks
- Processes each chunk through layers with KV cache
- Evals cache states per chunk to release Metal intermediates

This avoids accumulating all layer intermediates in Metal's buffer cache,
which would exceed the max buffer size for long sequences.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mlx.core as mx

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import mlx.nn as nn


def chunked_prefill(
    model_or_layers: Any,
    x: mx.array,
    cache: list[Any] | None = None,
    step_size: int = 2048,
    **kwargs: Any,
) -> mx.array:
    """Process input through model/layers in chunks with KV cache.

    This mirrors the scheduler's chunked prefill loop from
    Scheduler._do_external_prefill.

    Args:
        model_or_layers: Callable that processes x through layers/model.
            Signature: callable(x_chunk, cache=cache_slice, **kwargs).
        x: Input embeddings (1, seq_len, hidden_dim) or token IDs (1, seq_len).
        cache: List of KVCache objects — one per layer. If None, no caching
            is used (single forward pass). The caller may slice a full
            model cache to the assigned layer range.
        step_size: Maximum tokens per chunk (default 2048).
        **kwargs: Additional arguments forwarded to the model call.

    Returns:
        Concatenated hidden states for all tokens, shape (1, seq_len, hidden).
    """
    seq_len = x.shape[1]

    if seq_len <= step_size or cache is None:
        # Short enough for a single forward pass, or no cache available.
        return model_or_layers(x, cache=cache, **kwargs)

    # Process in chunks
    output_parts: list[mx.array] = []
    processed = 0

    while processed < seq_len:
        remaining = seq_len - processed
        chunk_size = min(step_size, remaining)

        x_chunk = x[:, processed : processed + chunk_size]
        out_chunk = model_or_layers(x_chunk, cache=cache, **kwargs)

        # Eval cache states to materialize this chunk and release intermediates.
        if cache is not None:
            state_arrays = _collect_cache_states(cache)
            if state_arrays:
                mx.eval(state_arrays)

        output_parts.append(out_chunk)
        processed += chunk_size

    # Concatenate chunk outputs across sequence dimension
    return mx.concatenate(output_parts, axis=1)


def _collect_cache_states(cache: list[Any]) -> list[mx.array]:
    """Extract .state tensors from a cache list for evaluation."""
    states: list[mx.array] = []
    for entry in cache:
        state = getattr(entry, "state", None)
        if isinstance(state, list):
            # Some cache types have multiple state arrays (e.g. KV pairs)
            for s in state:
                if isinstance(s, mx.array):
                    states.append(s)
        elif isinstance(state, mx.array):
            states.append(state)
    return states
