# SPDX-License-Identifier: Apache-2.0
"""
Distributed pipeline parallel engine for oMLX (per-model lifecycle).

Architecture:
  Coordinator (rank 0): API server + standard Scheduler + first pipeline stage
  Workers (rank > 0):    EngineCore for memory tracking, model driven by pipeline layers

Pipeline communication is handled by layer wrappers (PipelineFirstLayer /
PipelineLastLayer) inside the sharded model — transparent to Scheduler.

Per-model lifecycle:
  1. Coordinator receives model load request via API
  2. Gathers peer capabilities via HTTP metadata requests
  3. Initializes MLX ring (blocks until all ranks connect)
  4. Computes layer allocation and distributes to workers
  5. Each node loads its assigned layers and applies pipeline wrappers
  6. Serves generation requests (coordinator path only)
  7. Tears down ring and cleans up on model unload
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import BaseEngine, GenerationOutput

import mlx.core as mx

from ..distributed_sharding import (
    LayerAssignment,
    NodeSpec,
    allocate_layers,
    apply_pipeline_parallel,
)
from ..mlx_distributed import (
    barrier,
    is_coordinator,
    init_mlx_distributed,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class EngineContext:
    """Immutable context for a distributed pipeline instance.

    Shared across coordinator and workers after initialization.
    """

    model_path: str
    assignments: list[LayerAssignment]
    local_assignment: LayerAssignment | None
    group: Any  # mx.distributed.Group
    node_specs: list[NodeSpec]
    total_layers: int = 0
    shard_fraction: float = 1.0  # this rank's layer count / total layers


class DistributedPipelineEngine:
    """Per-model distributed pipeline parallel engine.

    Each model load creates a new instance. The ring is initialized at
    model-load time, not server start. All ranks participate; the
    coordinator (rank 0) also handles API requests.

    Port reservation uses an ephemeral port picked during the trigger
    handshake, not a static configured value.

    Args:
        model_path: Path or repo ID to the model.
        endpoints: All node "ip:port" strings for ring init (reserved ports).
        local_rank: This node's rank (0 = coordinator).
    """

    def __init__(
        self,
        model_path: str,
        endpoints: list[str],
        local_rank: int,
        model_id: str | None = None,
    ):
        self.model_path = model_path
        self.model_id = model_id or model_path  # Coordinator uses real model ID, workers fall back to path
        self.endpoints = endpoints
        self.local_rank = local_rank

        self._model: Any = None
        self._tokenizer: Any = None
        self._ctx: EngineContext | None = None
        self._cache = None  # Shared KV cache across prefill and decode
        self._loaded = False
        self._local_assignment: LayerAssignment | None = None  # Set by worker HTTP trigger

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def is_coordinator(self) -> bool:
        return is_coordinator(self.local_rank)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def init_ring(self) -> Any:
        """Initialize MLX ring distributed group.

        Blocks until all ranks connect (strict=True). Must be called
        simultaneously on all ranks. Workers should have already loaded
        their shard (via HTTP trigger) before calling this.

        The ring port was reserved during the trigger phase and delivered
        via the ring-list phase of the trigger-worker endpoint.

        Returns:
            The MLX distributed group, or raises on failure.

        Raises:
            RuntimeError: If ring init fails.
        """
        ring_port = None
        if not self.is_coordinator:
            # Workers get ring info from _distributed_pending_rings (set by ring-list phase)
            try:
                from ..server import _distributed_pending_rings
                ring_info = _distributed_pending_rings.get(self.local_rank)
                logger.info(
                    "Rank %d: _distributed_pending_rings[%d] = %s",
                    self.local_rank, self.local_rank, ring_info,
                )
                if ring_info:
                    ring_port = ring_info["ring_port"]
            except Exception as exc:
                logger.warning(
                    "Worker rank %d: failed to read pending ring info: %s",
                    self.local_rank, exc,
                )

        logger.info(
            "DistributedPipelineEngine init_ring: rank=%d, endpoints=%s, ring_port=%s",
            self.local_rank, self.endpoints, ring_port,
        )

        group = init_mlx_distributed(
            endpoints=self.endpoints,
            local_rank=self.local_rank,
            ring_port=ring_port,
        )
        if group is None:
            raise RuntimeError("Failed to initialize MLX distributed ring")

        self._group = group  # Store for later use in load_model

        rank = group.rank()
        size = group.size()
        logger.info(
            "DistributedPipelineEngine ring ready: rank %d/%d", rank, size
        )
        return group

    async def load_and_init(self) -> EngineContext:
        """Convenience: init_ring + load_model in sequence.

        For workers that received their shard via HTTP trigger before
        this is called, the ring will block until all ranks are ready.
        """
        await self.init_ring()
        return await self.load_model()

    async def load_model(
        self,
        total_layers: int | None = None,
        model_storage_gb: float = 0.0,
        trust_remote_code: bool = False,
        tokenizer_config: dict | None = None,
        peer_specs: list[dict[str, Any]] | None = None,
        ram_available_gb: float = 0.0,
        compute_weight: float = 1.0,
    ) -> EngineContext:
        """Load model with pipeline sharding across ranks.

        Coordinator path (rank 0):
          1. Builds node_specs from peer_specs (HTTP metadata) + self
          2. Computes layer allocation via allocate_layers() on itself only
          3. Pushes each worker's assignment to that worker via HTTP
          4. Loads full model from local disk
          5. Applies pipeline wrappers for own assigned layer range
          6. Calls barrier() to synchronize

        Worker path (rank > 0):
          1. Loads full model from local disk
          2. Receives pre-computed local_assignment (passed in or pushed)
          3. Applies pipeline wrappers for its assigned layer range
          4. Calls barrier() to synchronize

        Args:
            total_layers: Total transformer layers in the model.
            model_storage_gb: Model checkpoint size on disk (GB).
            trust_remote_code: Whether to trust remote code.
            tokenizer_config: Tokenizer configuration dict.
            peer_specs: Coordinator-provided list of peer dicts from
                HTTP metadata requests. On the coordinator this builds the
                full node_specs for allocation. On workers this contains
                pre-computed per-rank assignments under the key "assignments".
            ram_available_gb: This node's available RAM in GB.
            compute_weight: This node's compute weight for allocation.

        Returns:
            EngineContext with assignment and group info.

        Raises:
            RuntimeError: If ring not initialized or loading fails.
        """
        if self._ctx is not None:
            raise RuntimeError("Model already loaded; create a new engine")

        # ---- Step 1: Build node_specs and compute assignments (coordinator) ----
        node_specs: list[NodeSpec] | None = None
        assignments: list[LayerAssignment] | None = None

        if self.is_coordinator and peer_specs:
            # Coordinator has full picture from HTTP metadata requests.
            # Build NodeSpec list from peer_specs + self, compute allocation.
            node_specs = self._build_node_specs_from_peer_dicts(
                peer_specs,
                model_storage_gb,
                compute_weight,
                ram_available_gb,
            )

            assignments = allocate_layers(
                total_layers=total_layers,
                nodes=node_specs,
                model_storage_gb=model_storage_gb,
                safetensors_index_path=self.model_path,
            )

            # Push each worker's assignment to that worker via HTTP.
            await self._push_assignments_to_workers(
                node_specs, assignments, total_layers, model_storage_gb,
            )

        if node_specs is None:
            # Worker path: use pre-computed local assignment from HTTP trigger.
            if self._local_assignment is None:
                raise RuntimeError("Coordinator must provide peer_specs or worker must set _local_assignment")
            own = self._local_assignment
        else:
            if assignments is None:
                raise RuntimeError("Layer assignments not computed")

        # ---- Step 2: Initialize distributed ring (blocks until all ranks connect) ----
        # Coordinator pushes assignments BEFORE this call so workers are ready.
        logger.info(
            "Rank %d: calling init_ring() — _local_assignment=%s, is_coordinator=%s",
            self.local_rank, self._local_assignment, self.is_coordinator,
        )
        await self.init_ring()
        logger.info(
            "Rank %d: init_ring() returned, _group=%s",
            self.local_rank, self._group is not None,
        )

        # ---- Expand assignments list for worker path ----
        if assignments is None and self._group is not None:
            # Worker path: create placeholder list from own assignment.
            size = self._group.size()
            expanded = [None] * size
            own_idx = self.local_rank
            if 0 <= own_idx < size:
                expanded[own_idx] = own
            assignments = expanded
        elif assignments is not None and len(assignments) == 1:
            # Coordinator that only has its own local assignment.
            size = self._group.size()
            expanded = [None] * size
            own_idx = self.local_rank
            if 0 <= own_idx < size:
                expanded[own_idx] = own
            assignments = expanded

        # ---- Step 2.5: Barrier to synchronize all ranks ----
        barrier(self._group)

        # ---- Step 3: Load model from disk (all ranks) ----
        self._load_model_from_disk(
            total_layers,
            trust_remote_code=trust_remote_code,
            tokenizer_config=tokenizer_config,
        )

        # ---- Step 4: Apply pipeline wrappers for this rank's assignment ----
        own_idx = self.local_rank
        if own_idx >= len(assignments):
            raise RuntimeError(
                f"Rank {self.local_rank} out of range for "
                f"{len(assignments)} assignments"
            )

        own = assignments[own_idx]
        if total_layers is not None:
            logger.info(
                "Rank %d: assigned layers [%d:%d] of %d total",
                self.local_rank,
                own.start_layer,
                own.end_layer,
                total_layers,
            )
        else:
            logger.info(
                "Rank %d: assigned layers [%d:%d]",
                self.local_rank,
                own.start_layer,
                own.end_layer,
            )

        if own.end_layer - own.start_layer > 0:
            apply_pipeline_parallel(self._model, assignments, self._group)

        # Selectively evaluate only our assigned layer weights.
        # Unassigned layers stay as lazy mmap'd pointers (virtual address only,
        # no physical RAM). RoPE/freq buffers and output layers are always eval'd.
        self._eval_assigned_layers(own.start_layer, own.end_layer)

        barrier()

        self._ctx = EngineContext(
            model_path=self.model_path,
            assignments=assignments,
            local_assignment=own,
            group=self._group,
            node_specs=node_specs,
            total_layers=total_layers or 0,
            shard_fraction=(own.end_layer - own.start_layer) / max(total_layers or 1, 1),
        )
        self._loaded = True

        logger.info(
            "Rank %d: model loaded and sharded (layers %d-%d)",
            self.local_rank,
            own.start_layer,
            own.end_layer - 1 if own.end_layer > own.start_layer else own.start_layer,
        )

        return self._ctx

    async def _push_assignments_to_workers(
        self,
        node_specs: list[NodeSpec],
        assignments: list[LayerAssignment],
        total_layers: int,
        model_storage_gb: float,
    ) -> None:
        """Push ring list and assignments to worker nodes via HTTP.

        Two-phase protocol (coordinator only):
        Phase 1: Trigger each worker → ephemeral port reservation + cookie
        Phase 2: Push ring-list (endpoints + port) to each worker
        Phase 3: Push assignment details (layer ranges) for model load

        Uses requests.Session per worker — one TCP connection carries all
        phases. The session also carries HMAC-based mutual auth when
        cluster_key is configured.

        Called by the coordinator after computing allocations, before
        init_ring() blocks.
        """
        logger.info(
            "Pushing assignments to workers: %d ranks, endpoints=%s",
            len(assignments), self.endpoints,
        )

        import hashlib
        import hmac as hmac_module
        import requests

        # Build per-rank HTTP API port map from node_specs and peer metadata
        api_ports: dict[int, int] = {}
        for spec in node_specs[1:]:  # Skip coordinator (rank 0)
            if hasattr(spec, "api_port"):
                for i, ns in enumerate(node_specs[1:], 1):
                    if ns.node_id == spec.node_id:
                        api_ports[i] = spec.api_port
                        break

        size = len(assignments)

        # Phase 1: Reserve ports on all workers (collect cookies first)
        triggers: dict[int, dict] = {}  # rank → {host, port, cookie}
        for rank in range(1, size):
            if rank >= len(self.endpoints):
                continue

            endpoint = self.endpoints[rank]
            parts = endpoint.rsplit(":", 1)
            host = parts[0] if len(parts) == 2 else endpoint
            api_port = api_ports.get(rank, 8000)

            try:
                resp = requests.post(
                    f"http://{host}:{api_port}/api/distributed/trigger-worker",
                    json={"model_id": self.model_id, "rank": rank},
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.warning(
                        "Worker rank %d trigger returned %d: %s",
                        rank, resp.status_code, resp.text[:200],
                    )
                    continue
                data = resp.json()
            except Exception as exc:
                logger.warning(
                    "Failed to trigger worker rank %d (%s): %s",
                    rank, host, exc,
                )
                continue

            triggers[rank] = {
                "host": host,
                "port": data["port"],
                "cookie": data["cookie"],
            }

            # Handle challenge-response auth (if cluster_key configured)
            server_auth = data.get("server_auth", "")
            if "challenge" in data and server_auth:
                nonce = data["challenge"]

                # Verify server knows the cluster key
                gs = self._get_cluster_key()
                if gs:
                    expected = hmac_module.new(
                        gs.encode(), f"server_nonce|{nonce}".encode(), hashlib.md5
                    ).hexdigest()
                    if not hmac_module.compare_digest(expected, server_auth):
                        logger.warning(
                            "Server auth failed for rank %d (%s), continuing without auth",
                            rank, host,
                        )
                    else:
                        # Server authenticated — send our HMAC on next requests
                        triggers[rank]["auth"] = hmac_module.new(
                            gs.encode(),
                            f"ring-list|{nonce}".encode(),
                            hashlib.md5,
                        ).hexdigest()

        # Update endpoints with real reserved ports for all ranks
        final_endpoints = list(self.endpoints)  # Copy: rank 0 already has coordinator port
        for rank, info in triggers.items():
            self.endpoints[rank] = f"{info['host']}:{info['port']}"
            final_endpoints[rank] = f"{info['host']}:{info['port']}"

        # Phase 2+3: Push ring-list then assignment on same session per worker
        for rank, info in triggers.items():
            own = assignments[rank]
            host = info["host"]
            api_port = api_ports.get(rank, 8000)

            session = requests.Session()
            try:
                headers: dict[str, str] = {"Content-Type": "application/json"}
                if "auth" in info:
                    headers["X-Auth"] = info["auth"]

                # Phase 2: Ring list
                session.post(
                    f"http://{host}:{api_port}/api/distributed/trigger-worker",
                    json={
                        "endpoints": final_endpoints,
                        "ring_port": info["port"],
                        "cookie": info["cookie"],
                    },
                    headers=headers,
                    timeout=10,
                )

                # Phase 3: Assignment
                resp = session.post(
                    f"http://{host}:{api_port}/api/distributed/trigger-worker",
                    json={
                        "model_id": self.model_id,
                        "rank": rank,
                        "start_layer": own.start_layer,
                        "end_layer": own.end_layer,
                    },
                    headers=headers,
                    timeout=10,
                )
                resp.json()

                logger.info(
                    "Assigned rank %d: layers [%d:%d] at %s",
                    rank, own.start_layer, own.end_layer, final_endpoints[rank],
                )
            except (requests.RequestException, OSError) as exc:
                logger.warning(
                    "Failed to push ring-list+assignment to rank %d (%s): %s",
                    rank, host, exc,
                )
            finally:
                session.close()

        logger.info(
            "Assignment push complete: %d workers configured", len(triggers),
        )

    def _get_cluster_key(self) -> str | None:
        """Get configured cluster key, or None if not set."""
        try:
            gs = None
            from ..server import _server_state
            if getattr(_server_state, "global_settings", None):
                gs = _server_state.global_settings
            if gs and getattr(gs, "distributed", None):
                key = getattr(gs.distributed, "cluster_key", "") or ""
                return key if key else None
        except Exception:
            pass
        return None

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 0,
    ) -> Any:
        """Generate text tokens via the pipeline.

        Coordinator only. Workers run _worker_loop() started at load time.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded; call load_model() first")

        ctx = self._ctx
        if ctx is None:
            raise RuntimeError("No engine context available")

        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer not available")

        # Coordinator: tokenize and drive generation.
        tokenized = tokenizer.encode(prompt)
        tokens = mx.array(tokenized, dtype=mx.int32)

        if tokens.shape[0] == 0:
            return

        # Reset cache for new request.
        self._cache = None

        # Prefill (populates KV cache on both ranks, returns logits for first decode step).
        prefill_logits = self._prefill(tokens)

        # Decode: autoregressive token-by-token with send/recv pipelining.
        start_pos = len(tokens) - 1
        # Use logits from prefill's last chunk — no reprocessing needed.
        logits = prefill_logits

        # Collect stop token IDs from tokenizer.
        stop_ids: set[int] = set()
        if hasattr(tokenizer, 'eos_token_id'):
            eos = tokenizer.eos_token_id
            if isinstance(eos, int):
                stop_ids.add(eos)
            elif isinstance(eos, list):
                stop_ids.update(eos)

        # First token — already have logits from prefill.
        last_token = self._sample(logits, temperature, top_p, top_k)
        token_val = int(last_token.item())

        if token_val not in stop_ids:
            try:
                yield tokenizer.decode([token_val])
            except Exception:
                yield str(token_val)

        # Send first decode token through ring.
        start_pos += 1
        self._decode_send(mx.array([token_val]), start_pos=start_pos)

        for i in range(1, max_tokens):
            # Recv logits for the token we sent last iteration.
            logits = self._decode_recv()

            # Sample next token.
            last_token = self._sample(logits, temperature, top_p, top_k)
            token_val = int(last_token.item())

            if token_val in stop_ids:
                break

            try:
                yield tokenizer.decode([token_val])
            except Exception:
                yield str(token_val)

            # Send next token before blocking on recv (overlaps network I/O).
            start_pos += 1
            self._decode_send(mx.array([token_val]), start_pos=start_pos)

    async def generate(self, prompt: str) -> str:
        """Non-streaming generation (shim over stream_generate).

        Args:
            prompt: Input text.

        Returns:
            Generated text string.
        """
        result = []
        async for token in self.stream_generate(prompt, max_tokens=512):
            result.append(token)
        return "".join(result)

    async def stop(self) -> None:
        """Cleanup distributed resources."""
        self._loaded = False

        # Close the MLX ring group by dropping references.
        # MLX has no close() API — GC triggers C++ RingGroup destructor
        # which shuts down sockets, unblocking peer recv ops.
        if self._group is not None:
            import gc
            del self._group
            gc.collect()
            logger.info(
                "Rank %d: ring group destroyed, sockets closed", self.local_rank
            )

        self._model = None
        self._tokenizer = None
        self._ctx = None
        logger.info(
            "DistributedPipelineEngine stopped on rank %d", self.local_rank
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_node_specs_from_peer_dicts(
        self,
        peer_dicts: list[dict[str, Any]],
        model_storage_gb: float,
        compute_weight: float,
        ram_available_gb: float,
    ) -> list[NodeSpec]:
        """Build NodeSpec list from HTTP metadata request results.

        Called by coordinator only. peer_dicts come from
        ``discover_peer_metadata()`` calls to each peer's ring port.

        Args:
            peer_dicts: List of dicts with keys like node_id, ram_available_gb,
                chip_model, compute_weight from metadata endpoints.
            model_storage_gb: Model storage size for allocation validation.
            compute_weight: This node's compute weight.
            ram_available_gb: This node's available RAM in GB.

        Returns:
            Ordered list of NodeSpec (coordinator first, then peers).
        """
        nodes: list[NodeSpec] = []

        # Add self (coordinator, rank 0)
        nodes.append(
            NodeSpec(
                node_id="coordinator",
                ram_total_gb=ram_available_gb * 1.2,  # Estimate total from available
                ram_available_gb=ram_available_gb,
                compute_weight=compute_weight,
                chip_model="",
            )
        )

        for peer in peer_dicts:
            nodes.append(
                NodeSpec(
                    node_id=peer.get("node_id", "unknown"),
                    ram_total_gb=float(peer.get("ram_total_gb", 0)),
                    ram_available_gb=float(peer.get("ram_available_gb", 0)),
                    compute_weight=float(peer.get("compute_weight", 1.0)),
                    chip_model=peer.get("chip_model", ""),
                    api_port=int(peer.get("api_port", 8000)),
                )
            )

        return nodes

    def _gather_node_specs_distributed(
        self,
        compute_weight: float,
        ram_available_gb: float,
        peer_specs: list[dict[str, Any]] | None = None,
    ) -> list[NodeSpec]:
        """Gather NodeSpec from all ranks using MLX collectives.

        Each rank packs its specs into a fixed-size float array, all ranks
        collect them via mx.distributed.all_gather, then unpack.

        Falls back to peer_specs (coordinator gossip) if all-gather
        produces incomplete results.

        Args:
            compute_weight: This node's compute weight.
            ram_available_gb: This node's available RAM in GB.
            peer_specs: Optional coordinator-provided specs as fallback.

        Returns:
            List of NodeSpec from all ranks in order.
        """
        size = self._group.size()

        # Pack this node's spec into a float array: [compute, ram_total_est, ram_avail, chip_code, node_id_len]
        # We use a simplified format: compute_weight, ram_available_gb, and node_id as padded string
        this_data = self._pack_node_spec(
            compute_weight=compute_weight,
            ram_available_gb=ram_available_gb,
        )

        # All-gather: each rank sends its blob to all others
        blob_size = len(this_data)
        gathered = mx.zeros((size * blob_size,), dtype=mx.float32)

        # Build send buffer: local data at our rank's offset
        send_buf = mx.zeros((size * blob_size,), dtype=mx.float32)
        send_buf[self.local_rank * blob_size : (self.local_rank + 1) * blob_size] = mx.array(this_data)

        result = mx.distributed.all_gather(send_buf, axis=0)
        mx.eval(result)

        node_specs = []
        for i in range(size):
            chunk = result[i * blob_size : (i + 1) * blob_size]
            mx.eval(chunk)
            try:
                vals = chunk.tolist()
                cw, ram_avail = vals[0], vals[1] if len(vals) > 1 else ram_available_gb
                node_specs.append(
                    NodeSpec(
                        node_id=f"rank_{i}",
                        ram_total_gb=ram_avail * 1.2,
                        ram_available_gb=ram_avail,
                        compute_weight=cw,
                        chip_model=f"rank_{i}",
                    )
                )
            except Exception:
                node_specs.append(
                    NodeSpec(
                        node_id=f"rank_{i}",
                        ram_total_gb=ram_available_gb,
                        ram_available_gb=ram_available_gb,
                        compute_weight=compute_weight,
                        chip_model=f"rank_{i}",
                    )
                )

        # Coordinator: override with peer_specs if all-gather was incomplete
        if self.is_coordinator and peer_specs and len(node_specs) < len(peer_specs) + 1:
            node_specs = self._build_node_specs_from_peer_dicts(
                peer_specs, 0.0, compute_weight, ram_available_gb
            )

        return node_specs

    def _pack_node_spec(
        self, compute_weight: float, ram_available_gb: float
    ) -> list[float]:
        """Pack node spec into a fixed-size float array for all-gather.

        Format: [compute_weight, ram_available_gb, pad...]
        """
        blob = [compute_weight, ram_available_gb]
        # Pad to 8 floats for alignment
        while len(blob) < 8:
            blob.append(0.0)
        return blob

    def _load_model_from_disk(
        self,
        total_layers: int,
        trust_remote_code: bool = False,
        tokenizer_config: dict | None = None,
    ) -> None:
        """Load model from disk with lazy weight loading.

        All ranks load the full model directory with lazy=True. Only
        assigned layers are accessed during forward pass; unassigned
        layers stay as lazy file-mapped arrays and never allocate RAM.

        Args:
            total_layers: Total transformer layers in the model.
            trust_remote_code: Whether to trust remote code.
            tokenizer_config: Tokenizer configuration dict.
        """
        logger.info(
            "Rank %d: loading model from %s (lazy)",
            self.local_rank, self.model_path,
        )

        from ..utils.model_loading import lm_load_compat

        self._model, self._tokenizer = lm_load_compat(
            self.model_path,
            tokenizer_config=tokenizer_config,
            trust_remote_code=trust_remote_code,
            lazy=True,
        )

        logger.info("Rank %d: model loaded from disk", self.local_rank)

    def _eval_assigned_layers(
        self, start_layer: int, end_layer: int
    ) -> None:
        """Evaluate only weights for assigned layers; leave others lazy.

        MLX's lazy evaluation is fine-grained -- mx.eval() on a subset of
        arrays only materializes those specific arrays and their dependencies.
        Unassigned layer weights stay as lazy file-mapped pointers that
        consume virtual address space but not physical RAM.

        Always evaluates RoPE/frequency buffers and output layers (lm_head,
        embed_tokens) regardless of assignment -- these are small tensors
        needed for the forward pass on every rank.

        Args:
            start_layer: First layer index (inclusive).
            end_layer: Last layer index (exclusive).
        """
        from mlx.utils import tree_flatten

        layer_pattern = re.compile(r"\.layers\.(\d+)\.")
        lazy_arrays: list[tuple[str, mx.array]] = []

        for param_path, param_val in tree_flatten(self._model.parameters()):
            if not isinstance(param_val, mx.array):
                continue

            m = layer_pattern.search(param_path)
            if m:
                layer_num = int(m.group(1))
                if start_layer <= layer_num < end_layer:
                    lazy_arrays.append((param_path, param_val))
            # Always evaluate these -- needed for forward pass, small size
            elif any(
                k in param_path.lower()
                for k in ("freqs", "rope", "freqs_complex", "pos_ids")
            ):
                lazy_arrays.append((param_path, param_val))
            elif "lm_head" in param_path or "embed_tokens" in param_path:
                lazy_arrays.append((param_path, param_val))

        if lazy_arrays:
            logger.info(
                "Rank %d: evaluating %d arrays for layers [%d:%d]",
                self.local_rank, len(lazy_arrays), start_layer, end_layer,
            )
            for path, arr in lazy_arrays:
                logger.debug(
                    "  Eval: %s (dtype=%s, shape=%s)", path, arr.dtype, arr.shape
                )
            try:
                mx.eval([a for _, a in lazy_arrays])
            except Exception as exc:
                logger.warning(
                    "Selective eval failed (%s), falling back to full eval", exc
                )
                mx.eval(self._model.parameters())

    @staticmethod
    def _find_embed_tokens(model: Any) -> Any | None:
        """Find embed_tokens module at any depth in the model tree.

        For standard LLMs it's at model.embed_tokens.
        For VLMs (e.g. Qwen2/3-VL) it's at model.language_model.model.embed_tokens.

        Args:
            model: The loaded MLX model.

        Returns:
            The embed_tokens module if found, None otherwise.
        """
        if hasattr(model, "embed_tokens"):
            return model.embed_tokens
        # VLM pattern: language_model.model.embed_tokens
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model"):
                if hasattr(lm.model, "embed_tokens"):
                    return lm.model.embed_tokens
        return None

    @staticmethod
    def _find_lm_head(model: Any) -> Any | None:
        """Find lm_head module at any depth in the model tree.

        For standard LLMs it's at model.lm_head.
        For VLMs (e.g. Qwen2/3-VL) it's at model.language_model.lm_head.

        Args:
            model: The loaded MLX model.

        Returns:
            The lm_head module if found, None otherwise.
        """
        if hasattr(model, "lm_head"):
            return model.lm_head
        # VLM pattern: language_model.lm_head
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "lm_head"):
                return lm.lm_head
        return None

    def _prefill(self, tokens: mx.array) -> Optional[mx.array]:
        """Coordinator prefill: embed chunks through local layers, send output.

        Calls the inner transformer (not the full model with lm_head) so
        we get hidden states, not logits.
        """
        group = self._group
        if group is None:
            return

        rank = self.local_rank
        world_size = group.size()

        # KV cache.
        cache = None
        try:
            from mlx_lm.models.cache import make_prompt_cache
            cache = self._cache or make_prompt_cache(self._model)
            if not self._cache:
                self._cache = cache
        except Exception:
            logger.warning("Rank %d: no KV cache", rank)
            cache = None

        # Find inner transformer (no lm_head/output projection).
        from ..distributed_sharding import get_inner_model
        inner_model = get_inner_model(self._model)

        # Find embed_tokens on the full model (VLMs have it on outer wrapper).
        embed = self._find_embed_tokens(self._model)

        # Chunk tokens.
        step_size = 2048 // min(4, world_size)
        step_size = max(step_size, 256)
        total = len(tokens)

        if total <= step_size:
            token_chunk = tokens.reshape(1, -1)
            output = inner_model(token_chunk, cache=cache,
                                 input_embeddings=embed(token_chunk) if embed else None)
            send_header_and_data(output, dst=1, group=group,
                                  want_logits=True, start_pos=0)
            return

        logger.info("Rank %d: chunked prefill %d tokens (step=%d)", rank, total, step_size)

        # Process in chunks. Last chunk requests logits from last rank.
        processed = 0
        chunk_num = 0
        while processed < total:
            remaining = total - processed
            chunk_size = min(step_size, remaining)

            token_chunk = tokens[processed : processed + chunk_size].reshape(1, -1)
            output = inner_model(token_chunk, cache=cache,
                                 input_embeddings=embed(token_chunk) if embed else None)
            processed += chunk_size

            # Last chunk: want logits back (feeds into decode).
            is_last = processed >= total
            send_header_and_data(output, dst=1, group=group,
                                  want_logits=is_last, start_pos=processed)


        # Flush any pending async sends.
        mx.async_eval()

        # Recv logits from last rank for the first decode step.
        prefill_logits, _ = recv_header_and_data(src=1, group=group)
        logger.info(
            "Rank %d: prefill done (%d tokens), recv prefill logits shape=%s",
            rank, processed, str(list(prefill_logits.shape)),
        )

        return prefill_logits

    def _decode_send(self, token: mx.array, start_pos: int = 0) -> None:
        """Coordinator decode: embed token, send through pipeline, recv logits."""
        group = self._group

        # Find inner transformer and embed_tokens.
        from ..distributed_sharding import get_inner_model
        inner_model = get_inner_model(self._model)
        embed = self._find_embed_tokens(self._model)

        t = token.reshape(1, -1)
        output = inner_model(t, cache=self._cache,
                             input_embeddings=embed(t) if embed else None)
        logger.info(
            "Rank 0: decode output_shape=%s cache_type=%s",
            str(list(output.shape)), type(self._cache).__name__,
        )

        # Send hidden states to rank 1, recv logits back.
        send_header_and_data(output, dst=1, group=group, want_logits=True,
                              start_pos=start_pos)

    def _decode_recv(self) -> mx.array:
        group = self._group
        
        # Recv logits from last rank (header + data protocol).
        logits, _ = recv_header_and_data(src=group.size() - 1, group=group)
        return logits

    def _worker_loop(self) -> None:
        """Worker inference loop: recv hidden states, process, forward.

        Pipeline recv/send is explicit in this loop (not inside layer wrappers).
        Layer wrappers provide per-layer eval only (_EvalLayer).
        """
        group = self._group
        rank = self.local_rank
        world_size = group.size()
        src = (rank - 1) % world_size
        dst = (rank + 1) % world_size

        logger.info("Rank %d: worker loop starting (recv from %d, send to %d)",
                     rank, src, dst)

        from ..distributed_sharding import get_inner_model
        from .distributed_ring import recv_header_and_data, send_header_and_data
        inner_model = get_inner_model(self._model)

        cache: Any = None  # Lazily created on new request
        iteration = 0

        while True:
            try:
                hidden_states, header_dict = recv_header_and_data(
                    src=src, group=group,
                )
            except Exception as e:
                logger.error(
                    "Rank %d: recv failed, coordinator likely gone (%s). Exiting.",
                    rank, e,
                )
                return

            seq_len = hidden_states.shape[1]
            want_logits = header_dict["want_logits"]
            start_pos = header_dict["start_pos"]
            logger.info(
                "Rank %d: recv done, shape=%s seq=%d start_pos=%d",
                rank, str(list(hidden_states.shape)), seq_len, start_pos,
            )

            # Reset cache at start of each new request.
            if start_pos == 0:
                try:
                    from mlx_lm.models.cache import make_prompt_cache
                    cache = make_prompt_cache(self._model)
                except Exception:
                    logger.warning("Rank %d: failed to reset KV cache", rank)

            # Build position-aware dummy tokens for correct RoPE.
            dummy_tokens = mx.arange(start_pos, start_pos + seq_len).reshape(1, -1)

            # Process through local model.
            if rank == world_size - 1 and want_logits:
                # Last rank + needs logits (decode): call full model including lm_head.
                output = self._model(dummy_tokens, cache=cache,
                                     input_embeddings=hidden_states)
            else:
                # Skip lm_head during prefill.
                output = inner_model(dummy_tokens, cache=cache,
                                     input_embeddings=hidden_states)

            mx.clear_cache()
            import gc; gc.collect()

            iteration += 1

            # Last rank: only send back if coordinator wants logits (decode).
            if dst == 0 and not want_logits:
                logger.info("Rank %d: skip send (no logits wanted)", rank)
            else:
                send_header_and_data(output, dst=dst, group=group,
                                      want_logits=want_logits, start_pos=start_pos)

    def _get_layer_list(self) -> list:
        """Extract layer list from the model.

        Uses _get_layers() from distributed_sharding to handle different
        model architectures.
        """
        from ..distributed_sharding import _get_layers

        return _get_layers(self._model)

    def _sample(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 0,
    ) -> mx.array:
        """Sample a token from logits.

        Args:
            logits: Logit vector.
            temperature: Sampling temperature.
            top_p: Top-p parameter.
            top_k: Top-k parameter.

        Returns:
            Sampled token ID array. Shape is (1,) — last position only.
        """
        if temperature != 1.0:
            logits = logits / temperature

        # Only sample the last position (autoregressive generation)
        if logits.ndim == 3 and logits.shape[1] > 0:
            last_logits = logits[:, -1, :]
        elif logits.ndim == 2 and logits.shape[0] > 0:
            last_logits = logits[:, -1]
        elif logits.ndim == 3 and logits.shape[1] == 0:
            # Empty sequence — return dummy zeros for fallback sampling
            last_logits = mx.zeros((1, logits.shape[-1]), dtype=logits.dtype)
        elif logits.ndim == 2 and logits.shape[0] == 0:
            last_logits = mx.zeros((logits.shape[-1],), dtype=logits.dtype)
        else:
            last_logits = logits

        # Apply top-p (nucleus) sampling
        if top_p > 0 and top_p < 1.0:
            from ..utils.sampling import apply_top_p
            last_logits = apply_top_p(last_logits, top_p)

        # Apply top-k sampling
        if top_k > 0:
            from ..utils.sampling import apply_top_k
            last_logits = apply_top_k(last_logits, top_k)

        return mx.random.categorical(last_logits)


# =============================================================================
# Module-level helpers for coordinator workflow
# =============================================================================

from .distributed_ring import (
    recv_header_and_data,
    send_header_and_data,
)

def local_ip() -> str:
    """Get this node's local IPv4 address for hostfile entries.

    Tries psutil first, falls back to socket-based discovery.

    Returns:
        Local IPv4 address string, or "127.0.0.1" as last resort.
    """
    try:
        from ..utils.network import _local_ipv4_addresses

        addrs = _local_ipv4_addresses()
        if addrs:
            return addrs[0]
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass

    return "127.0.0.1"

def select_distributed_peers(peers: list[Any]) -> list[tuple[Any, dict[str, Any]]]:
    """Return all discovered peers ready for metadata queries.

    Called by the coordinator before querying peer capabilities.
    Peers are populated with metadata results after discovery.

    Args:
        peers: List of PeerInfo from discovery service.

    Returns:
        List of (peer, {}) tuples for all discovered peers.
    """
    return [(peer, {}) for peer in peers]


async def discover_peer_metadata(
    peer: Any, timeout: float = 2.0
) -> dict[str, Any]:
    """Query a peer's /api/distributed-node-info for capabilities.

    Uses the peer's API port (from mDNS TXT record) to reach the existing
    FastAPI endpoint.

    Args:
        peer: PeerInfo object from discovery service.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: node_id, ram_total_gb,
            ram_available_gb, chip_model, compute_weight.
    """
    import urllib.error
    import urllib.request

    # Query the peer's admin API
    url = f"http://{peer.host or peer.node_id}:{peer.api_port}/api/distributed-node-info"

    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())

        return {
            "node_id": data.get("node_id", peer.node_id),
            "ram_total_gb": float(data.get("ram_total_gb", 0)),
            "ram_available_gb": float(data.get("ram_available_gb", 0)),
            "chip_model": data.get("chip_model", "Unknown"),
            "compute_weight": float(data.get("compute_weight", 1.0)),
        }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        logger.info("Metadata request failed for %s: %s", peer.node_id, exc)
        return {
            "node_id": peer.node_id,
            "ram_total_gb": 0,
            "ram_available_gb": 0,
            "chip_model": "Unknown",
            "compute_weight": 1.0,
        }


async def build_distributed_endpoints(
    peers: list[Any],  # list[PeerInfo] from discovery
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build endpoint list and peer metadata for distributed loading.

    Coordinator workflow:
    1. Get all discovered peers from PeerRegistry
    2. Use cached metadata when available (populated by background fetch)
    3. Fall back to network query if metadata is stale/default
    4. Build hostfile entries from resolved ring_port + metadata

    Args:
        peers: Discovered PeerInfo list from discovery service.

    Returns:
        (endpoints, peer_metadata) where endpoints is a list of
        "ip:port" strings (resolved ring ports) and peer_metadata is
        the metadata dict for each participating peer.
    """
    candidates = select_distributed_peers(peers)

    endpoints: list[str] = []
    peer_metadata: list[dict[str, Any]] = []

    for peer, _ in candidates:
        # Check if we have non-default cached metadata (0.0 for ram means
        # metadata hasn't been fetched yet)
        if peer.ram_available_gb > 0.0:
            meta = {
                "node_id": peer.node_id,
                "ram_total_gb": peer.ram_total_gb,
                "ram_available_gb": peer.ram_available_gb,
                "chip_model": peer.chip_model,
                "compute_weight": peer.compute_weight,
            }
        else:
            # Fallback: fetch from network
            meta = await discover_peer_metadata(peer)

        # Include API port so coordinator can reach workers via HTTP
        meta["api_port"] = peer.api_port

        # Port is a placeholder; real ports are assigned via trigger protocol
        endpoint = f"{peer.host or peer.node_id}:0"
        endpoints.append(endpoint)
        peer_metadata.append(meta)

    return endpoints, peer_metadata


# =============================================================================
# BaseEngine-compatible wrapper using the standard EngineCore chain
# =============================================================================


import concurrent.futures as _cf
import asyncio as _asyncio

class _LoaderThreadExecutor:
    """ThreadPoolExecutor-compatible dispatcher that runs tasks on the loader thread."""

    def __init__(self, loader_loop):
        self._loop = loader_loop

    def submit(self, fn, *args, **kwargs):
        future = _cf.Future()
        async def _run_async():
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        _asyncio.run_coroutine_threadsafe(_run_async(), self._loop)
        return future

    def shutdown(self, wait=True):
        pass

class DistributedEngineWrapper(BaseEngine):
    """BaseEngine-compatible wrapper using the standard EngineCore chain.

    Creates AsyncEngineCore → EngineCore → DistributedScheduler (inherits from
    standard Scheduler) so the distributed path gets continuous batching, request
    lifecycle management, output collectors, and memory tracking for free.

    Pipeline communication is handled by layer wrappers (PipelineFirstLayer /
    PipelineLastLayer) inside the sharded model, transparent to the scheduler.

    Args:
        pipeline_engine: Loaded DistributedPipelineEngine (ring + model).
        tokenizer: Tokenizer for the model.
    """

    def __init__(
        self,
        pipeline_engine: Any,
        tokenizer: Any = None,
        loader_loop: Any = None,
    ) -> None:
        self._pipeline_engine = pipeline_engine
        self._tokenizer = tokenizer
        self._loader_loop = loader_loop  # event loop on inference thread
        self._loader_thread: threading.Thread | None = None  # set by caller

        # Create the EngineCore chain with DistributedScheduler.
        # DistributedScheduler inherits from standard Scheduler — all chunked
        # prefill, continuous batching, memory tracking, eviction come from parent.
        # Pipeline send/recv happens inside wrapped model layers, not the scheduler.
        from ..engine_core import AsyncEngineCore, EngineConfig
        from .distributed_scheduler import DistributedScheduler
        from ..scheduler import SchedulerConfig

        config = EngineConfig(
            scheduler_class=DistributedScheduler,
            scheduler_config=SchedulerConfig(),
        )
        self._engine = AsyncEngineCore(
            model=pipeline_engine._model,
            tokenizer=tokenizer,
            config=config,
        )

        # Replace EngineCore's executor with a loader-thread dispatcher.
        # step() must run on the loader thread (same as model loading) to use
        # matching stream encoders.
        self._engine.engine._mlx_executor = _LoaderThreadExecutor(loader_loop)

        # Wire the ring group into the scheduler for pipeline communication.
        scheduler = self._engine.engine.scheduler
        scheduler.group = pipeline_engine._group

    @property
    def model_name(self) -> str:
        return self._pipeline_engine.model_path

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_type(self) -> Optional[str]:
        """Get the model type from config (e.g., 'qwen2_vl', 'llama')."""
        model = getattr(self._engine.engine, "model", None)
        if model is None:
            return None
        try:
            config = getattr(model, "config", None)
            if config is not None:
                mt = getattr(config, "model_type", None)
                return mt if isinstance(mt, str) else None
            args = getattr(model, "args", None)
            if args is not None:
                mt = getattr(args, "model_type", None)
                return mt if isinstance(mt, str) else None
        except Exception:
            pass
        return None

    async def start(self) -> None:
        """Start the engine loop (runs scheduler step loop in background)."""
        await self._engine.start()

    async def stop(self) -> None:
        """Stop the engine and clean up."""
        await self._engine.stop()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> "GenerationOutput":
        """Non-streaming generation via EngineCore chain."""
        from ..request import SamplingParams

        result = []
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop or [],
            **kwargs,
        ):
            if not output.finished:
                result.append(output.new_text)
        return GenerationOutput(
            text="".join(result),
            prompt_tokens=0,
            completion_tokens=len(result),
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator["GenerationOutput"]:
        """Stream generation via EngineCore chain (add_request → stream_outputs)."""
        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [],
            xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            thinking_budget=kwargs.get("thinking_budget", None),
            compiled_grammar=kwargs.get("compiled_grammar", None),
            seed=kwargs.get("seed", None),
        )

        # Ensure engine loop is running (lazy start, like BatchedEngine).
        # AsyncEngineCore.start() is non-blocking — schedules the engine loop
        # as an asyncio task in the current event loop.
        if not self._engine.engine.is_running():
            # EngineCore.start() runs on the main (HTTP) thread's event loop.
            # But we need step() to run on the loader thread. Instead of using
            # EngineCore's executor, we dispatch step() to the loader loop.
            self._engine.start()

        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        full_text = ""
        try:
            async for output in self._engine.stream_outputs(request_id):
                text = output.output_text or ""
                new_text = output.new_text or ""
                full_text = text

                yield GenerationOutput(
                    text=text,
                    new_text=new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                )
        except Exception as e:
            error_msg = str(e)
            # Detect ring communication failures — peer likely gone.
            if any(kw in error_msg.lower() for kw in
                   ["distributed", "ring", "send", "recv", "connection",
                    "broken pipe", "socket"]):
                logger.error(
                    "DistributedEngineWrapper: ring error, unloading model. "
                    "Will reload fresh on next request. Error: %s",
                    error_msg,
                )
                await self._engine.stop()
                self.close()  # Calls EngineCore close + Metal cleanup on loader thread
                # Clear from engine_pool so next request triggers full reload
                # (fresh ring init, fresh worker load).
                import gc
                try:
                    from ..server import get_engine_pool
                    pool = get_engine_pool()
                except Exception:
                    pool = None  # Fallback if import fails
                if pool is not None:
                    entry = pool._entries.get(self.model_name)
                    if entry is not None:
                        logger.info("DistributedEngineWrapper: clearing pool entry for %s", self.model_name)
                        entry.engine = None
                await asyncio.sleep(0)
                gc.collect()
            else:
                logger.exception(
                    "DistributedEngineWrapper: stream_generate failed for %s",
                    request_id,
                )
            yield GenerationOutput(
                text=full_text,
                finished=True,
            )
        finally:
            # Clean up collector even on client disconnect
            try:
                await self._engine.abort_request(request_id)
            except Exception:
                pass

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> "GenerationOutput":
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            template_kwargs["tools"] = tools
        if kwargs.get("chat_template_kwargs"):
            template_kwargs.update(kwargs["chat_template_kwargs"])

        prompt = self._tokenizer.apply_chat_template(
            messages, **template_kwargs
        ) if self._tokenizer else ""

        return await self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> AsyncIterator["GenerationOutput"]:
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            template_kwargs["tools"] = tools
        if kwargs.get("chat_template_kwargs"):
            template_kwargs.update(kwargs["chat_template_kwargs"])

        prompt = self._tokenizer.apply_chat_template(
            messages, **template_kwargs
        ) if self._tokenizer else ""

        async for output in self.stream_generate(prompt, max_tokens=max_tokens, temperature=temperature):
            yield output

    def has_active_requests(self) -> bool:
        return self._engine.engine.is_running() and self._engine.engine.scheduler.has_requests()

    def get_stats(self) -> dict[str, Any]:
        stats = {
            "engine_type": "distributed",
            "model_name": self._pipeline_engine.model_path,
            "loaded": self._pipeline_engine._loaded,
        }
        group = getattr(self._pipeline_engine, "_group", None)
        if group is not None:
            stats["rank"] = group.rank()
            stats["size"] = group.size()
        ctx = getattr(self._pipeline_engine, "_ctx", None)
        if ctx is not None and ctx.local_assignment is not None:
            stats["layers"] = (
                ctx.local_assignment.start_layer,
                ctx.local_assignment.end_layer,
            )
        sched = self._engine.engine.scheduler
        if hasattr(sched, "get_stats"):
            stats.update(sched.get_stats())
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        sched = self._engine.engine.scheduler
        return getattr(sched, "get_cache_stats", lambda: None)()

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> int:
        """Count prompt tokens for chat messages after applying chat template."""
        from ..api.tool_calling import convert_tools_for_template
        from ..api.utils import detect_and_strip_partial

        if self._tokenizer is None:
            return 0
        template_tools = (
            convert_tools_for_template(tools) if tools else None
        )
        if is_partial is None:
            is_partial = detect_and_strip_partial(messages)
        else:
            for msg in messages:
                msg.pop("partial", None)
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": not is_partial,
        }
        if is_partial:
            template_kwargs["continue_final_message"] = True
        if template_tools:
            template_kwargs["tools"] = template_tools
        template_kwargs["enable_thinking"] = False
        if chat_template_kwargs:
            template_kwargs.update(chat_template_kwargs)

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        except TypeError:
            for key in list(template_kwargs):
                if key not in ("tokenize", "add_generation_prompt"):
                    template_kwargs.pop(key, None)
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, **template_kwargs
                )
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        return len(self._tokenizer.encode(prompt))

    async def stop(self) -> None:
        """Stop the engine and clean up."""
        await self._engine.stop()
        # Also close inner EngineCore (AsyncEngineCore has no close()).
        # Matches BatchedEngine.stop() pattern.
        inner = getattr(getattr(self._engine, 'engine', None), 'close', None)
        if callable(inner):
            inner()

    async def _cleanup_metal(self):
        """Run Metal cleanup on the loader thread's event loop."""
        import gc
        mx.synchronize()
        mx.clear_cache()
        mx.clear_streams()
        gc.collect()

    def close(self) -> None:
        """Shutdown the engine and stop the loader thread."""
        # Close EngineCore chain — AsyncEngineCore has no close(), but inner EngineCore does.
        if hasattr(self._engine, 'close'):
            self._engine.close()
        inner = getattr(getattr(self._engine, 'engine', None), 'close', None)
        if callable(inner):
            inner()

        # Close the ring to unblock worker recv on its thread.
        self._pipeline_engine._group = None
        import gc; gc.collect()
        logger.info("DistributedEngineWrapper: ring group cleared")

        # Run Metal cleanup on the loader thread where MLX ops ran.
        if self._loader_loop is not None and not self._loader_loop.is_closed():
            try:
                logger.info("DistributedEngineWrapper: dispatching Metal cleanup to loader thread")
                fut = asyncio.run_coroutine_threadsafe(
                    self._cleanup_metal(), self._loader_loop
                )
                fut.result(timeout=10)  # Block until cleanup completes
                logger.info("DistributedEngineWrapper: Metal cleanup done")
            except Exception as e:
                logger.warning("DistributedEngineWrapper: Metal cleanup failed: %s", e)
            finally:
                # Stop the loader thread's event loop (blocks until current step finishes)
                try:
                    self._loader_loop.call_soon_threadsafe(self._loader_loop.stop)
                except Exception:
                    pass
                self._loader_loop = None
        else:
            # Loop already closed or not running — cleanup from this thread.
            logger.info("DistributedEngineWrapper: Metal cleanup from event loop thread")
            try:
                mx.synchronize()
                mx.clear_cache()
                import gc; gc.collect()
            except Exception as e:
                logger.warning("DistributedEngineWrapper: Metal cleanup failed: %s", e)


class DistributedWorkerEngine:
    """Minimal EngineCore-based wrapper for distributed workers.

    Workers don't schedule requests independently — the coordinator drives
    all work through pipeline layer wrappers (PipelineFirstLayer/LastLayer).
    This class exists for:

    - Memory tracking (ProcessMemoryEnforcer walks engine_pool → scheduler)
    - Stream setup (per-engine thread-local streams)
    - Model lifecycle management via EngineCore patterns

    The worker's actual inference loop calls the sharded model directly.
    Pipeline recv/send is handled inside layer wrappers, not by this class.

    Args:
        model: Loaded + sharded model (pipeline wrappers applied).
        tokenizer: Tokenizer for the model.
    """

    __slots__ = ("_engine", "_model", "_tokenizer")

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
    ) -> None:
        from ..engine_core import AsyncEngineCore, EngineConfig
        from .distributed_scheduler import DistributedScheduler
        from ..scheduler import SchedulerConfig

        # Standard EngineCore chain — scheduler is idle on the worker
        # (no queued requests), but provides memory tracking infrastructure.
        config = EngineConfig(
            scheduler_class=DistributedScheduler,
            scheduler_config=SchedulerConfig(),
        )
        self._engine = AsyncEngineCore(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        self._model = model
        self._tokenizer = tokenizer

    @property
    def scheduler(self) -> Any:
        """Expose scheduler for ProcessMemoryEnforcer resolution."""
        return self._engine.engine.scheduler

    @property
    def model_name(self) -> str:
        return getattr(self._model, "config", None).model_type if hasattr(getattr(self._model, "config", None), "model_type") else "distributed_worker"

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_type(self) -> Optional[str]:
        return None

    async def start(self) -> None:
        """No-op — worker has no step loop. Coordinator drives all work."""
        pass

    async def generate(
        self,
        prompt: str,
        stream: bool = False,
        sampling_params: Any = None,
        request_id: str | None = None,
    ) -> Any:
        raise NotImplementedError("Workers do not accept requests directly")

    async def generate_images(
        self,
        prompt: str,
        image_size: tuple[int, int] | None = None,
        num_images: int = 1,
        request_id: str | None = None,
    ) -> list:
        raise NotImplementedError("Workers do not accept image generation")

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        if self._tokenizer is None:
            return 0
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        return len(self._tokenizer.encode(prompt))

    def close(self) -> None:
        if hasattr(self._engine, 'close'):
            self._engine.close()

        # Clean up Metal state so next load starts fresh.
        try:
            mx.synchronize()
            mx.clear_cache()
            import gc; gc.collect()
        except Exception:
            pass

    async def stop(self) -> None:
        """Stop the engine (called by engine_pool on shutdown)."""
        # Close inner EngineCore — AsyncEngineCore has no close().
        inner = getattr(getattr(self._engine, 'engine', None), 'close', None)
        if callable(inner):
            inner()
        self.close()
