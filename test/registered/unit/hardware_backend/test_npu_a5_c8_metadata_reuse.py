"""C8 resident slots and graph padding across sparse_copy and SFA."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


def _load_manager(prepare_descriptors):
    path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/"
        "c8_manager.py"
    )
    package = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload"
    dependencies = {
        "sgl_kernel_npu.sparsity_driven_kv_offload": Mock(),
        "sglang.kernels.ops.attention.npu_sparse_kv_plan": SimpleNamespace(
            SparseKvPlanOp=Mock()
        ),
        "sglang.kernels.ops.kvcache.npu_sparse_kv": Mock(),
        "sglang.srt.runtime_context": Mock(),
        f"{package}.fused_budget": SimpleNamespace(C8_ROW_BYTES=656),
        f"{package}.fused_lim": SimpleNamespace(
            LIMDecodeAdapter=object, sparse_query_width=Mock()
        ),
        f"{package}.fused_manager": SimpleNamespace(FusedSparseKVCacheManager=object),
        f"{package}.layer_layout": SimpleNamespace(SparseKVLayerLayout=Mock()),
        f"{package}.memfabric_pool": SimpleNamespace(MemFabricKVPool=object),
        f"{package}.npu_plan_sparse_copy_metadata": SimpleNamespace(
            prepare_plan_sparse_copy_metadata=prepare_descriptors
        ),
    }
    spec = importlib.util.spec_from_file_location("a5_c8_manager_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**dependencies, spec.name: module}):
        spec.loader.exec_module(module)
    return module.C8SparseKVCacheManager, module.C8PlanSelection


def _load_attention(sfa):
    path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/"
        "c8_full_attention.py"
    )
    spec = importlib.util.spec_from_file_location("a5_c8_attention_test", path)
    module = importlib.util.module_from_spec(spec)
    torch_npu_stub = SimpleNamespace(npu_kv_quant_sparse_flash_attention=sfa)
    with patch.dict(sys.modules, {"torch_npu": torch_npu_stub}):
        spec.loader.exec_module(module)
    return module


class TestC8SparseCopySFA(CustomTestCase):
    def test_shared_layer_copies_before_sfa_and_masks_padding(self):
        events = []

        def prepare_descriptors(**kwargs):
            events.append("descriptors")
            kwargs["count"].fill_(2)

        manager_type, selection_type = _load_manager(prepare_descriptors)
        manager = object.__new__(manager_type)
        manager.start_layer = 0
        manager.layer_layout = SimpleNamespace(group_for=lambda _: (0, 1, 2, 3))
        manager.budget = SimpleNamespace(copy_sfa_query_heads=1)
        manager.page_size = 128
        manager.cache_capacity = 4096
        manager.host_token_capacity = 2048
        manager.sparse_topk = 3
        manager.device_kv_buffer = [
            None,
            torch.zeros((2, 4224, 1, 656), dtype=torch.int8),
        ]
        manager.host_kv_buffer = [None, torch.zeros((10, 128, 1, 656))]
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((4, 32), dtype=torch.int32)
        )
        manager.sparse_copy_sources = torch.empty(12, dtype=torch.int64)
        manager.sparse_copy_destinations = torch.empty(12, dtype=torch.int64)
        manager.sparse_copy_lengths = torch.empty(12, dtype=torch.int64)
        manager.sparse_copy_count = torch.zeros(1, dtype=torch.int32)
        manager.lim_adapter = SimpleNamespace(
            query_ends=torch.tensor([[1, 2], [2, 4]], dtype=torch.int32)
        )

        def sparse_copy(*args):
            events.append("sparse_copy")
            self.assertEqual(args[3].item(), 2)
            return 0

        manager.host_pool = SimpleNamespace(
            offload=SimpleNamespace(sparse_copy=sparse_copy)
        )
        plan = selection_type(
            sparse_indices=torch.tensor(
                [[1, 3, -1], [0, 2, -1], [0, 0, 0], [0, 0, 0]],
                dtype=torch.int32,
            ),
            sfa_block_table=torch.tensor(
                [[33 + i for i in range(33)], [i for i in range(33)]],
                dtype=torch.int32,
            ),
            miss_tokens=torch.zeros((4, 3), dtype=torch.int32),
            miss_slots=torch.zeros((4, 3), dtype=torch.int32),
            miss_counts=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
            request_slots=torch.tensor([2, 0], dtype=torch.int32),
            resident_slots=torch.tensor([2, 99], dtype=torch.int32),
            valid_rows=torch.tensor([True, False]),
            query_width=2,
            owner_layer=0,
        )

        def sfa(**kwargs):
            events.append("sfa")
            self.assertEqual(
                kwargs["sparse_indices"].tolist(),
                [[[1, 3, -1]], [[0, 2, -1]], [[0, 0, 0]], [[0, 0, 0]]],
            )
            self.assertEqual(kwargs["actual_seq_lengths_kv"].tolist(), [4097, 1])
            self.assertEqual(kwargs["block_table"][1, 0].item(), 0)
            self.assertEqual(kwargs["key"].shape, (66, 128, 1, 656))
            return torch.ones((4, 1, 512))

        layer = SimpleNamespace(layer_id=1, scaling=1.0)
        cache = manager.copy_shared_kv(layer, plan)
        output = _load_attention(sfa).forward_shared_c8(
            SimpleNamespace(sparse_kv_manager=manager),
            torch.zeros((4, 1, 512)),
            torch.zeros((4, 1, 64)),
            layer,
            plan,
            cache,
        )
        self.assertEqual(events, ["descriptors", "sparse_copy", "sfa"])
        self.assertEqual(output.shape, (4, 512))
        self.assertTrue(output[:2].eq(1).all())
        self.assertTrue(output[2:].eq(0).all())

    def test_full_layer_uses_native_pool_and_forward_sparse(self):
        manager = SimpleNamespace(
            budget=SimpleNamespace(copy_sfa_query_heads=8),
            sparse_topk=3,
        )
        calls = []

        def forward_sparse(q, k, v, layer, batch, **kwargs):
            calls.append((q.shape, k.shape, kwargs))
            self.assertEqual(q.shape, (2, 8, 512))
            self.assertEqual(kwargs["q_rope"].shape, (2, 8, 64))
            self.assertEqual(kwargs["topk_indices"].tolist(), [[1, 2, 3]] * 2)
            self.assertTrue(kwargs["save_kv_cache"])
            return torch.ones((2, 8, 512), dtype=q.dtype)

        backend = SimpleNamespace(
            sparse_kv_manager=manager, forward_sparse=forward_sparse
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
        )
        output = _load_attention(Mock()).forward_full_c8(
            backend,
            torch.zeros((2, 1, 512), dtype=torch.bfloat16),
            torch.zeros((2, 1, 512), dtype=torch.bfloat16),
            torch.zeros((2, 1, 64), dtype=torch.bfloat16),
            torch.zeros((2, 1, 64), dtype=torch.bfloat16),
            SimpleNamespace(tp_q_head_num=1),
            batch,
            torch.tensor([[1, 2, 3]] * 2, dtype=torch.int32),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(output.shape, (2, 1, 512))
        self.assertTrue(output.eq(1).all())


if __name__ == "__main__":
    unittest.main()
