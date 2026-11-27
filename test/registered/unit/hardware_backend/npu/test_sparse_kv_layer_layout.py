import unittest

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.layer_layout import (
    SparseKVLayerLayout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSparseKVLayerLayout(CustomTestCase):
    def test_shared_layers_follow_their_indexer_with_partial_tail(self):
        layout = SparseKVLayerLayout.build(4, 11, (4, 8))
        self.assertEqual(layout.full_layers, (4, 8))
        self.assertEqual(layout.shared_layers, (5, 6, 7, 9, 10))
        self.assertEqual(layout.group_for(6), (4, 5, 6, 7))
        self.assertEqual(layout.group_for(10), (8, 9, 10))

    def test_rejects_missing_group_owner(self):
        with self.assertRaisesRegex(ValueError, "first sparse KV layer"):
            SparseKVLayerLayout.build(4, 8, (5,))

    def test_pd_buffers_keep_model_layer_order(self):
        layout = SparseKVLayerLayout.build(4, 11, (4, 8))
        buffers = layout.transfer_main_buffers(
            ("hbm4", "hbm8"),
            ("host5", "host6", "host7", "host9", "host10"),
        )
        self.assertEqual(
            buffers,
            ("hbm4", "host5", "host6", "host7", "hbm8", "host9", "host10"),
        )
        with self.assertRaisesRegex(ValueError, "buffer count"):
            layout.transfer_main_buffers(("hbm4",), ("host5",))


if __name__ == "__main__":
    unittest.main()
