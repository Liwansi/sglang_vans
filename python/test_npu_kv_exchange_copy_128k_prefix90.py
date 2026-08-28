"""A5 integration test for a 128K request with a 90% HiCache host hit.

This intentionally calls ``memfabric_hybrid.offload.kv_exchange_copy``
directly with the same metadata ABI used by MLATokenToKVPoolHost.  It models
the first two-layer transfer group of PP stage 0 for GLM-5.2 DSA:

* page_size=128
* packed MLA K width=656 bytes (512 FP8 + 64 BF16 RoPE + 4 FP32 scales)
* physical Indexer layers 0 and 1, width=128 FP8 + one FP32 scale
* 128K input tokens, floor(90% of 1024 pages)=921 host-resident pages

Run on an Atlas A5 worker after sourcing the same memfabric environment as
``python/run128K_16p_103_lc.sh``::

    export PYTHONPATH=/path/to/sglang_vans/python:$PYTHONPATH
    export SGLANG_HICACHE_HOST_MEM=hybm
    export SGLANG_HICACHE_IO_ASCENDC=1
    python python/test_npu_kv_exchange_copy_128k_prefix90.py -v
"""

import unittest

import torch

from sglang.srt.hardware_backend.npu.attention.fp8_contracts import (
    get_dsa_fp8_packed_cache_dim,
)
from sglang.srt.mem_cache.pool_host.common import (
    alloc_with_hybm,
    ensure_hybm_capacity,
    track_pinned_staging,
)


REQUEST_TOKENS = 128 * 1024
PAGE_SIZE = 128
REQUEST_PAGES = REQUEST_TOKENS // PAGE_SIZE
PREFIX_PAGES = int(REQUEST_PAGES * 0.90)
PREFIX_TOKENS = PREFIX_PAGES * PAGE_SIZE
GROUP_LAYERS = 2
INDEXER_LAYERS = 2  # GLM-5.2 layers 0 and 1 both own physical Indexers.
INDEX_HEAD_DIM = 128
PACKED_KV_DIM = get_dsa_fp8_packed_cache_dim(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
)


def _component_meta(device_tensor, host_tensor, layer_lo, layer_hi):
    """Build one component descriptor consumed by kv_exchange_copy."""
    itemsize = device_tensor.dtype.itemsize
    width = 1
    for dim in device_tensor.shape[2:]:
        width *= dim
    return (
        device_tensor.data_ptr(),
        host_tensor.data_ptr(),
        device_tensor.stride(0) * itemsize,
        device_tensor.stride(1) * itemsize,
        host_tensor.stride(0) * itemsize,
        host_tensor.stride(1) * itemsize,
        width * itemsize,
        layer_lo,
        layer_hi,
    )


def _hybm_empty(shape, dtype):
    return alloc_with_hybm(
        tuple(shape),
        dtype=dtype,
        device="cpu",
        pin_memory=True,
        allocator=None,
    )


class TestNPUKVExchangeCopy128KPrefix90(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise unittest.SkipTest("This integration test requires an Ascend NPU")
        try:
            from memfabric_hybrid import offload  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"memfabric_hybrid is unavailable: {exc}")

    def test_h2d_packed_kv_and_indexer_for_90_percent_prefix(self):
        from memfabric_hybrid import offload
        from sgl_kernel_npu.kvcacheio import TransferDirection

        device = torch.device("npu", torch.npu.current_device())
        num_pages = PREFIX_PAGES

        # Host is page-first, matching MLATokenToKVPoolHost:
        #   [page, layer/indexer_slot, page_size, 1, width]
        host_k_shape = (num_pages, GROUP_LAYERS, PAGE_SIZE, 1, PACKED_KV_DIM)
        host_ik_shape = (num_pages, INDEXER_LAYERS, PAGE_SIZE, 1, INDEX_HEAD_DIM)
        host_scale_shape = (num_pages, INDEXER_LAYERS, PAGE_SIZE, 1)
        host_bytes = (
            torch.Size(host_k_shape).numel()
            + torch.Size(host_ik_shape).numel()
            + torch.Size(host_scale_shape).numel() * torch.float32.itemsize
        )
        ensure_hybm_capacity(host_bytes, torch.npu.current_device())
        host_k = _hybm_empty(host_k_shape, torch.float8_e4m3fn)
        host_index_k = _hybm_empty(host_ik_shape, torch.float8_e4m3fn)
        host_index_scale = _hybm_empty(host_scale_shape, torch.float32)

        # Fill via byte views because the packed FP8 tensor contains opaque
        # FP8/BF16/FP32 byte regions. Different values make layer mixups visible.
        host_k[:, 0].view(torch.uint8).fill_(0x11)
        host_k[:, 1].view(torch.uint8).fill_(0x22)
        host_index_k[:, 0].view(torch.uint8).fill_(0x33)
        host_index_k[:, 1].view(torch.uint8).fill_(0x44)
        host_index_scale[:, 0].fill_(1.25)
        host_index_scale[:, 1].fill_(2.5)

        # Device is layer-first. Page 0 and the last page are guards; requested
        # host pages [0, num_pages) are scattered to device pages [1, num_pages+1).
        device_pages = num_pages + 2
        device_k = torch.zeros(
            (GROUP_LAYERS, device_pages, PAGE_SIZE, 1, PACKED_KV_DIM),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        device_index_k = torch.zeros(
            (INDEXER_LAYERS, device_pages, PAGE_SIZE, 1, INDEX_HEAD_DIM),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        device_index_scale = torch.zeros(
            (INDEXER_LAYERS, device_pages, PAGE_SIZE, 1),
            dtype=torch.float32,
            device=device,
        )
        # kv_exchange_copy consumes flattened token-slot indices, not page ids.
        # Source pages are [0, num_pages); destination starts at page 1 so page
        # 0 can be checked as an untouched guard page.
        host_indices = torch.arange(PREFIX_TOKENS, dtype=torch.int64, device=device)
        device_indices = torch.arange(
            PAGE_SIZE,
            PAGE_SIZE + PREFIX_TOKENS,
            dtype=torch.int64,
            device=device,
        )

        components = [
            _component_meta(device_k, host_k, 0, GROUP_LAYERS),
            _component_meta(device_index_k, host_index_k, 0, INDEXER_LAYERS),
            _component_meta(
                device_index_scale, host_index_scale, 0, INDEXER_LAYERS
            ),
        ]
        vals = [
            len(components),
            num_pages,
            PAGE_SIZE,
            TransferDirection.H2D.value,
            device_indices.data_ptr(),
            host_indices.data_ptr(),
        ]
        for component in components:
            vals.extend(component)

        pinned_meta = torch.tensor(vals, dtype=torch.int64, pin_memory=True)
        meta = torch.empty_like(pinned_meta, device=device)
        meta.copy_(pinned_meta, non_blocking=True)
        track_pinned_staging(pinned_meta)

        ret = offload.kv_exchange_copy(meta, device)
        self.assertEqual(ret, 0)
        torch.npu.synchronize()

        # Sample the beginning, middle and end of the 90% prefix. Comparing
        # byte views validates every byte of each packed entry without trying
        # to interpret the mixed packed representation as FP8 numbers.
        sampled_pages = torch.tensor(
            [1, num_pages // 2 + 1, num_pages], dtype=torch.long, device=device
        )
        for layer, expected in ((0, 0x11), (1, 0x22)):
            actual = device_k[layer, sampled_pages].view(torch.uint8)
            self.assertTrue(torch.all(actual == expected).item())
        for layer, expected in ((0, 0x33), (1, 0x44)):
            actual = device_index_k[layer, sampled_pages].view(torch.uint8)
            self.assertTrue(torch.all(actual == expected).item())
        self.assertTrue(
            torch.all(device_index_scale[0, sampled_pages] == 1.25).item()
        )
        self.assertTrue(
            torch.all(device_index_scale[1, sampled_pages] == 2.5).item()
        )

        # Pages not listed in device_indices must remain untouched.
        self.assertTrue(torch.all(device_k[:, 0].view(torch.uint8) == 0).item())
        self.assertTrue(
            torch.all(device_k[:, -1].view(torch.uint8) == 0).item()
        )

        transferred_bytes = PREFIX_TOKENS * (
            GROUP_LAYERS * PACKED_KV_DIM
            + INDEXER_LAYERS * INDEX_HEAD_DIM
            + INDEXER_LAYERS * torch.float32.itemsize
        )
        print(
            f"kv_exchange_copy H2D passed: request={REQUEST_TOKENS} tokens, "
            f"host_prefix={PREFIX_TOKENS} tokens ({PREFIX_TOKENS / REQUEST_TOKENS:.2%}), "
            f"pages={num_pages}/{REQUEST_PAGES}, page_size={PAGE_SIZE}, "
            f"bytes={transferred_bytes} "
            f"({transferred_bytes / 2**20:.2f} MiB)"
        )


if __name__ == "__main__":
    unittest.main()
