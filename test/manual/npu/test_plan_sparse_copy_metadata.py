"""Run on Ascend950 to validate the Plan -> MemFabric descriptor ABI."""

import unittest

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_plan_sparse_copy_metadata import (
    prepare_plan_sparse_copy_metadata,
)
from sglang.test.test_utils import CustomTestCase


class TestPlanSparseCopyMetadata(CustomTestCase):
    def test_compacts_mtp_misses_and_maps_physical_tokens(self):
        device = "npu:0"
        miss_tokens = torch.tensor(
            [[2, 3, -1, -1], [4, -1, -1, -1], [8, -1, -1, -1], [-1] * 4],
            dtype=torch.int32,
            device=device,
        )
        miss_slots = torch.tensor(
            [[1, 5, -1, -1], [6, -1, -1, -1], [9, -1, -1, -1], [-1] * 4],
            dtype=torch.int32,
            device=device,
        )
        counts = torch.tensor([2, 1, 1, 0], dtype=torch.int32, device=device)
        req_to_token = torch.zeros((2, 16), dtype=torch.int32, device=device)
        req_to_token[1, 2:5] = torch.tensor([7, 9, 11], device=device)
        sources = torch.empty(16, dtype=torch.int64, device=device)
        destinations = torch.empty_like(sources)
        lengths = torch.empty_like(sources)
        count = torch.zeros(1, dtype=torch.int32, device=device)
        host_base, device_base, stride, row_bytes = 0x100000, 0x200000, 4224, 656
        prepare_plan_sparse_copy_metadata(
            miss_tokens=miss_tokens,
            miss_slots=miss_slots,
            miss_counts=counts,
            request_slots=torch.tensor([1, 0], dtype=torch.int32, device=device),
            resident_slots=torch.tensor([2, 99], dtype=torch.int32, device=device),
            valid_rows=torch.tensor([True, False], device=device),
            req_to_token=req_to_token,
            sources=sources,
            destinations=destinations,
            lengths=lengths,
            count=count,
            host_base=host_base,
            device_base=device_base,
            query_width=2,
            cache_stride=stride,
            cache_capacity=4096,
            host_capacity=2048,
            row_bytes=row_bytes,
        )
        torch.npu.synchronize()
        self.assertEqual(count.item(), 3)
        self.assertEqual(
            sources[:3].cpu().tolist(),
            [host_base + physical * row_bytes for physical in (7, 9, 11)],
        )
        self.assertEqual(
            destinations[:3].cpu().tolist(),
            [device_base + (stride + slot) * row_bytes for slot in (1, 5, 6)],
        )
        self.assertEqual(lengths[:3].cpu().tolist(), [row_bytes] * 3)


if __name__ == "__main__":
    unittest.main()
