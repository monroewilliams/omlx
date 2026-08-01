# SPDX-License-Identifier: Apache-2.0
"""
Ring communication primitives for distributed pipeline parallel inference.

Shared by DistributedPipelineEngine and DistributedScheduler.
"""

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Dtype encoding for ring header protocol (int32[5])
_FP32, _FP16, _BF16 = 0, 1, 2
_DTYPE_MAP = {_FP32: mx.float32, _FP16: mx.float16, _BF16: mx.bfloat16}
_DTYPE_REV = {v: k for k, v in _DTYPE_MAP.items()}


def send_header_and_data(
    data: mx.array,
    dst: int,
    group: mx.distributed.Group,
    want_logits: bool = False,
    start_pos: int = 0,
) -> None:
    """Send header + data tensor over the ring.

    Args:
        data: Hidden states or logits to send. Shape [1, seq_len, dim].
        dst: Destination rank.
        group: MLX distributed group.
        want_logits: Whether coordinator wants logits back (for decode).
        start_pos: Token position offset for RoPE embeddings.
    """
    seq_len = data.shape[1] if data.ndim >= 2 else 1
    inner_dim = data.shape[-1]
    dtype_code = _DTYPE_REV.get(data.dtype, _FP32)

    header = mx.array(
        [int(seq_len), int(inner_dim), dtype_code, int(want_logits), start_pos],
        dtype=mx.int32,
    )

    logger.info(
        "Rank %d: send header=%s to rank %d | data shape=%s dtype=%s",
        group.rank(), header.tolist(), dst, list(data.shape), data.dtype,
    )

    try:
        sent_hdr = mx.distributed.send(header, dst=dst, group=group)
        mx.eval(sent_hdr)
    except Exception as e:
        logger.warning("Rank %d: mx.eval(sent_hdr) failed: %s", group.rank(), e)
        raise
    try:
        logger.info("Rank %d: send data to rank %d (shape=%s)", group.rank(), dst, list(data.shape))
        mx.eval(data)  # Materialize to break lazy graph.
        logger.info("Rank %d: eval done, queueing send to rank %d", group.rank(), dst)
        sent_data = mx.distributed.send(data, dst=dst, group=group)
        mx.async_eval(sent_data)  # Queue send, don't block — overlaps with next chunk.
    except Exception as e:
        logger.warning("Rank %d: mx.eval(sent_data) failed: %s", group.rank(), e)
        raise


def recv_header_and_data(
    src: int,
    group: mx.distributed.Group,
) -> tuple[mx.array, dict[str, Any]]:
    """Receive header + data tensor over the ring.

    Args:
        src: Source rank.
        group: MLX distributed group.

    Returns:
        Tuple of (data tensor, header dict with want_logits/start_pos).
    """
    header = mx.distributed.recv([5], mx.int32, src=src, group=group)
    mx.eval(header)
    h = header.tolist()
    seq_len = int(h[0])
    inner_dim = int(h[1])
    dtype_code = int(h[2])
    recv_dtype = _DTYPE_MAP.get(dtype_code, mx.float32)
    shape = [1, seq_len, inner_dim]
    data = mx.distributed.recv(shape, recv_dtype, src=src, group=group)
    mx.eval(data)
    return data, {
        'want_logits': bool(int(h[3])),
        'start_pos': int(h[4]),
    }
