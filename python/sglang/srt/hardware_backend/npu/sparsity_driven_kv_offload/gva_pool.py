"""GVA-based shared DRAM pool for sparsity-driven KV offload (paged layout).

When ``SGLANG_NPU_USE_GVA_KV_POOL=1``, host KV buffers are allocated from a
per-DP-group shared DRAM pool using memfabric_hybrid's GVA (Global Virtual
Address) mechanism.  Only TP rank 0 allocates physical DRAM; all TP ranks
share the same virtual address via IOMMU mapping.  This saves N*memory for
MLA models where ``num_kv_heads=1`` means KV is identical across TP ranks.

Pool layout is **paged**, matching the prefill side's ``k_buffer`` /
``v_buffer`` and the D2RH test case
(``test_transfer_engine_d2rh_shared.py``):

    k_buffer[layer]:  [num_blocks, page_size, 1, kv_cache_dim]
    v_buffer[layer]:  [num_blocks, page_size, 1, v_cache_dim]

When ``dsa_kv_cache_store_fp8`` is True, V is packed into K and only K
buffers are allocated.

The flat ``host_kv_buffer`` list follows the same entry order as the
prefill side's ``_get_disagg_buffer_entries``:
    [k_layer_0, ..., k_layer_{L-1}, v_layer_0, ..., v_layer_{L-1}]

so that PD-transfer infrastructure (``get_contiguous_buf_infos``,
``send_kvcache_d2rh``, etc.) works transparently.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import socket
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# memfabric_hybrid.offload import (optional dependency)
# ---------------------------------------------------------------------------
_offload = None
_offload_import_error: Optional[Exception] = None

try:
    from memfabric_hybrid import offload as _offload  # type: ignore
except Exception as exc:  # pragma: no cover - depends on optional package
    _offload_import_error = exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024  # 2 MiB
_OFFLOAD_STORE_PORT_BASE = 8500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _probe_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _restore_tensor_from_ptr(
    ptr: int,
    shape: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a writable CPU tensor view backed by a raw GVA virtual address.

    Used by TP rank > 0 to obtain a PyTorch tensor that points to the same
    GVA memory allocated by rank 0.  The ctypes buffer reference is stored on
    the returned tensor to prevent premature garbage collection.
    """
    ptr = int(ptr)
    elem_size = torch.empty((), dtype=dtype).element_size()
    numel = 1
    for d in shape:
        numel *= int(d)
    total_bytes = numel * elem_size
    buf = (ctypes.c_byte * total_bytes).from_address(ptr)
    tensor = torch.frombuffer(buf, dtype=dtype, requires_grad=False)
    tensor = tensor.reshape(*shape)
    tensor._gva_ctypes_buffer = buf
    return tensor


# ---------------------------------------------------------------------------
# Per-buffer descriptor
# ---------------------------------------------------------------------------
class _BufferSpec:
    """Describes one paged buffer (K or V) for GVA allocation."""

    __slots__ = ("shape", "dtype", "bytes", "tensor", "host_ptr", "dev_ptr")

    def __init__(self, shape: List[int], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype
        elem_size = torch.empty((), dtype=dtype).element_size()
        numel = 1
        for d in shape:
            numel *= int(d)
        self.bytes = numel * elem_size
        self.tensor: Optional[torch.Tensor] = None
        self.host_ptr: int = 0
        self.dev_ptr: int = 0


# ---------------------------------------------------------------------------
# GVAKVPoolManager
# ---------------------------------------------------------------------------
class GVAKVPoolManager:
    """Per-DP-group shared DRAM pool for paged host KV buffers using GVA.

    Layout is paged to match the prefill side's ``k_buffer`` / ``v_buffer``,
    enabling direct page-to-page D2RH transfer without layout conversion.

    Attributes:
        layer_num: number of KV cache layers.
        page_size: tokens per page (block).
        num_blocks: number of pages per layer.
        k_buffer: list of K CPU tensors per layer (paged).
        v_buffer: list of V CPU tensors per layer (paged, empty if fp8).
        host_kv_buffer: flat list [k_0..k_{L-1}, v_0..v_{L-1}] matching
            prefill ``_get_disagg_buffer_entries`` order.
        host_ptr_list: flat GVA addresses matching ``host_kv_buffer``.
        dev_ptr_list: flat NPU-visible addresses (== host_ptr under GVA).
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        page_size: int,
        kv_cache_dim: int,
        v_cache_dim: int,
        layer_num: int,
        store_dtype: torch.dtype,
        device_id: int,
        dram_size_gb: int = 32,
    ) -> None:
        if _offload is None:
            raise ImportError(
                "SGLANG_NPU_USE_GVA_KV_POOL requires the memfabric_hybrid "
                f"package. Import error: {_offload_import_error}"
            )

        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if getattr(parallel, "attn_dp_enabled", False) and parallel.attn_dp_size > 1:
            self.tp_rank: int = parallel.attn_tp_rank
            self.tp_size: int = parallel.attn_tp_size
        else:
            self.tp_rank: int = parallel.tp_rank
            self.tp_size: int = parallel.tp_size
        self.tp_group = get_tp_group()
        self.device_id: int = device_id
        self.layer_num: int = layer_num
        self.page_size: int = page_size
        self.store_dtype: torch.dtype = store_dtype

        # Derive num_blocks from DRAM budget, NOT from full token capacity.
        # Per-block bytes = page_size * (kv_cache_dim + v_cache_dim) * dtype_size
        elem_size = torch.empty((), dtype=store_dtype).element_size()
        per_block_bytes = page_size * (kv_cache_dim + v_cache_dim) * elem_size
        dram_budget_bytes = dram_size_gb * (1 << 30)
        max_blocks_by_budget = dram_budget_bytes // (
            per_block_bytes * layer_num
        )
        # Also cap by the logical maximum (size * max_context_len / page_size)
        max_blocks_logical = (
            size * max_context_len + page_size - 1
        ) // page_size
        self.num_blocks: int = int(min(max_blocks_by_budget, max_blocks_logical))
        if self.num_blocks <= 0:
            raise RuntimeError(
                f"GVAKVPoolManager DRAM budget too small: {dram_size_gb} GiB "
                f"cannot fit even 1 block per layer (per_block={per_block_bytes}, "
                f"layers={layer_num})"
            )

        # Build buffer specs: K per layer, then V per layer (if v_cache_dim > 0)
        self._specs: List[_BufferSpec] = []
        k_shape = [self.num_blocks, page_size, 1, kv_cache_dim]
        for _ in range(layer_num):
            self._specs.append(_BufferSpec(k_shape, store_dtype))
        self._has_v = v_cache_dim > 0
        v_shape = [self.num_blocks, page_size, 1, v_cache_dim]
        if self._has_v:
            for _ in range(layer_num):
                self._specs.append(_BufferSpec(v_shape, store_dtype))

        num_buffers = len(self._specs)

        # Total pool size: all buffers + 2 MiB alignment slack per buffer
        total_pool_bytes = sum(s.bytes for s in self._specs) + num_buffers * _CPU_CACHE_ALIGNMENT
        self._pool_size = total_pool_bytes

        dp_index = parallel.dp_rank if parallel.dp_rank is not None else 0
        logger.info(
            "GVAKVPoolManager init: tp_rank=%d tp_size=%d dp_index=%d "
            "dram_budget=%d GiB pool=%.2f GiB layers=%d num_blocks=%d "
            "page_size=%d kv_dim=%d v_dim=%d buffers=%d",
            self.tp_rank,
            self.tp_size,
            dp_index,
            dram_size_gb,
            total_pool_bytes / (1 << 30),
            layer_num,
            self.num_blocks,
            page_size,
            kv_cache_dim,
            v_cache_dim,
            num_buffers,
        )

        store_url = self._resolve_store_url(dp_index)

        config = _offload.OffloadConfig()
        config.device_id = device_id
        config.reserve_size = total_pool_bytes
        config.alloc_size = total_pool_bytes if self.tp_rank == 0 else 0
        config.world_size = self.tp_size
        config.rank_id = self.tp_rank
        config.scene = _offload.Scene.SHARED
        config.store_url = store_url
        ret = _offload.initialize(config)
        if ret != 0:
            raise RuntimeError(
                f"GVAKVPoolManager offload.initialize failed ret={ret} "
                f"tp_rank={self.tp_rank} store_url={store_url}"
            )
        self._store_url = store_url
        logger.info(
            "GVAKVPoolManager pool ready tp_rank=%d store_url=%s",
            self.tp_rank,
            store_url,
        )

        self.tp_group.barrier()

        # Allocate buffers
        if self.tp_size == 1:
            for spec in self._specs:
                self._alloc_one(spec)
        else:
            gva_bases = torch.zeros(
                num_buffers, dtype=torch.int64, device=f"npu:{device_id}"
            )
            if self.tp_rank == 0:
                for i, spec in enumerate(self._specs):
                    self._alloc_one(spec)
                    gva_bases[i] = spec.host_ptr
                logger.info(
                    "GVAKVPoolManager rank0 allocated %d buffers first_gva=%#x",
                    num_buffers,
                    self._specs[0].host_ptr,
                )

            self.tp_group.broadcast(gva_bases, src=0)

            if self.tp_rank != 0:
                for i, spec in enumerate(self._specs):
                    ptr = int(gva_bases[i].item())
                    spec.tensor = _restore_tensor_from_ptr(
                        ptr, spec.shape, spec.dtype
                    )
                    spec.host_ptr = ptr
                    spec.dev_ptr = ptr
                logger.info(
                    "GVAKVPoolManager rank%d restored %d buffers first_gva=%#x",
                    self.tp_rank,
                    num_buffers,
                    self._specs[0].host_ptr,
                )

        self.tp_group.barrier()

        # Expose flat lists matching prefill _get_disagg_buffer_entries order
        self.k_buffer: List[torch.Tensor] = [
            s.tensor for s in self._specs[:layer_num]
        ]
        self.v_buffer: List[torch.Tensor] = (
            [s.tensor for s in self._specs[layer_num:]] if self._has_v else []
        )
        self.host_kv_buffer: List[torch.Tensor] = [s.tensor for s in self._specs]
        self.host_ptr_list: List[int] = [s.host_ptr for s in self._specs]
        self.dev_ptr_list: List[int] = [s.dev_ptr for s in self._specs]

        logger.info(
            "[GVA] decode host DRAM KV pool is allocated from the "
            "memfabric_hybrid (mf) GVA shared DRAM pool: rank0 allocates, "
            "all %d TP ranks share the same virtual address "
            "(NPU-visible via IOMMU, host_ptr == dev_ptr). "
            "pool=%.2f GiB, layout: %d K buffers %s + %d V buffers %s "
            "(paged %d blocks x %d tokens), first_gva=%#x, store_url=%s",
            self.tp_size,
            total_pool_bytes / (1 << 30),
            layer_num,
            k_shape,
            len(self.v_buffer),
            v_shape if self._has_v else [],
            self.num_blocks,
            self.page_size,
            self.host_ptr_list[0],
            self._store_url,
        )

        atexit.register(self.cleanup)

    # ------------------------------------------------------------------
    def _resolve_store_url(self, dp_index: int) -> str:
        port = _OFFLOAD_STORE_PORT_BASE + dp_index
        return f"tcp://127.0.0.1:{port}"

    def _alloc_one(self, spec: _BufferSpec) -> None:
        """Allocate one 2 MiB-aligned paged buffer from the GVA pool.

        Called only on TP rank 0 (the rank with ``alloc_size > 0``).
        """
        backing = _offload.empty(
            [spec.bytes + _CPU_CACHE_ALIGNMENT],
            dtype=torch.int8,
            pin_memory=True,
        )
        offset = (-int(backing.data_ptr())) % _CPU_CACHE_ALIGNMENT
        tensor = (
            backing[offset : offset + spec.bytes]
            .view(spec.dtype)
            .reshape(spec.shape)
        )
        if int(tensor.data_ptr()) % _CPU_CACHE_ALIGNMENT != 0:
            raise RuntimeError(
                f"GVAKVPoolManager alignment failed: data_ptr="
                f"{tensor.data_ptr()} alignment={_CPU_CACHE_ALIGNMENT}"
            )
        spec.tensor = tensor
        spec.host_ptr = int(tensor.data_ptr())
        spec.dev_ptr = spec.host_ptr  # GVA: same VA is NPU-visible via IOMMU
        tensor._gva_backing = backing

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Release the GVA pool (idempotent)."""
        if _offload is not None:
            try:
                _offload.uninitialize()
                logger.info("GVAKVPoolManager pool uninitialized")
            except Exception as exc:
                logger.warning("GVAKVPoolManager cleanup failed: %s", exc)
