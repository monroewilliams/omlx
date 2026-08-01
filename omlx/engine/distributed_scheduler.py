# SPDX-License-Identifier: Apache-2.0
"""
DistributedScheduler inheriting from standard oMLX Scheduler.

All scheduler infrastructure (chunked prefill, continuous batching, memory
tracking, eviction) comes from the parent. Pipeline communication is handled
by wrapping ``self.model`` so every forward pass goes through the ring.

The wrapper determines decode vs prefill by input sequence length:
- Decode (seq_len == 1): send hidden states, recv logits from last rank
- Prefill (seq_len > 1): send hidden states, no logit recv
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Callable

import mlx.core as mx

from .distributed_ring import recv_header_and_data, send_header_and_data
from ..scheduler import Scheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prefill tracking — distinguishes prefill chunks from decode tokens
# ---------------------------------------------------------------------------

_PrefillActive: ContextVar[bool] = ContextVar("_prefill_active", default=False)


def is_prefill_active() -> bool:
    """Return True if we're inside a prefill forward pass."""
    return _PrefillActive.get()


class _prefill_context:
    """Mark a forward pass as prefill (not decode)."""

    def __enter__(self):
        self._token = _PrefillActive.set(True)
        return self

    def __exit__(self, *exc: object) -> None:
        _PrefillActive.reset(self._token)


# ---------------------------------------------------------------------------
# Model wrapper — pipeline round-trip for every forward pass
# ---------------------------------------------------------------------------

class _PipelineModel:
    """Wrap the sharded model so every forward pass does pipeline send/recv.

    Determines decode vs prefill by input size:
    - Decode: inputs.shape[1] == 1 → send hidden states, recv logits
    - Prefill: inputs.shape[1] > 1 → send hidden states, no logit recv
    """

    __slots__ = ("_model", "_group", "_rank", "_dst", "_start_pos")

    def __init__(self, model: Any) -> None:
        self._model = model
        self._group = None  # Set later when ring is ready
        self._rank = 0
        self._dst = -1
        self._start_pos = 0  # Track cumulative token position for RoPE

    def set_group(self, group: Any) -> None:
        """Set ring group after initialization."""
        self._group = group
        if group is not None:
            self._rank = group.rank()
            self._dst = (self._rank + 1) % group.size()

    def __call__(self, inputs: mx.array, cache: Any = None) -> Any:
        # Forward through local transformer layers (skip lm_head).
        # Pipeline needs hidden states, not logits.
        from ..distributed_sharding import get_inner_model
        inner = get_inner_model(self._model)
        output = inner(inputs, cache=cache)

        if self._group is None:
            return output  # No ring yet, skip pipeline

        seq_len = inputs.shape[1] if inputs.ndim > 1 else 1
        # Decode wants logits, prefill never does (even if last chunk is 1 token).
        want_logits = not _PrefillActive.get() and seq_len == 1

        # Use tracked start_pos for RoPE offset. Advance after each forward.
        start_pos = self._start_pos
        self._start_pos += seq_len

        send_header_and_data(
            output, dst=self._dst, group=self._group,
            want_logits=want_logits, start_pos=start_pos,
        )

        if want_logits:
            last = self._group.size() - 1
            logits, _ = recv_header_and_data(src=last, group=self._group)
            mx.eval(logits)
            return logits
        return output

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# DistributedScheduler — inherits everything from standard Scheduler
# ---------------------------------------------------------------------------

class DistributedScheduler(Scheduler):
    """Scheduler subclass that adds pipeline communication for distributed inference.

    Wraps ``self.model`` so every forward pass (prefill + decode) goes through
    the pipeline. ``_PipelineModel`` handles ring send/recv based on input size.
    """

    _group: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Wrap model for pipeline communication on every forward pass.
        if self.model is not None:
            self.model = _PipelineModel(self.model)

    @property
    def group(self) -> Any:
        return self._group

    @group.setter
    def group(self, value: Any) -> None:
        self._group = value
        if isinstance(self.model, _PipelineModel):
            self.model.set_group(value)

    def _do_external_prefill(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        """Override: mark prefill context so wrapper doesn't request logits."""
        with _prefill_context():
            return super()._do_external_prefill(request, *args, **kwargs)

    def _step_prefill_chunk(self, state: Any) -> bool:
        """Override: mark prefill context so wrapper doesn't request logits."""
        with _prefill_context():
            return super()._step_prefill_chunk(state)
