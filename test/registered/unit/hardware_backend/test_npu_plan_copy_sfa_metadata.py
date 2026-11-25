"""Copy-SFA needs miss-first TopK even though SparseKvPlan consumes ordered TopK."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_plan_copy_sfa_metadata import (
    plan_to_copy_sfa,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPlanToCopySfa(CustomTestCase):
    def test_shared_miss_is_in_each_query_prefix_but_copied_once(self):
        # The Plan's request descriptor contains 4 and 5 once each; token 5
        # appears in both query rows and must enter both Copy-SFA miss prefixes.
        source_ids = torch.tensor(
            [[2, 4, 5, -1], [2, 5, 8, -1], [1, 4, 7, -1], [2, 5, 8, -1]],
            dtype=torch.int32,
        )
        current_slots = torch.tensor(
            [[0, 1, 2, -1], [0, 2, 3, -1], [0, 1, 2, -1], [0, 1, 2, -1]],
            dtype=torch.int32,
        )
        miss_count = torch.tensor([2, 0, 0, 0], dtype=torch.int32)
        miss_tokens = torch.tensor(
            [[4, 5, -1, -1], [-1] * 4, [-1] * 4, [-1] * 4], dtype=torch.int32
        )
        miss_slots = torch.tensor(
            [[1, 2, -1, -1], [-1] * 4, [-1] * 4, [-1] * 4], dtype=torch.int32
        )

        sources, slots, topk_misses, misses, destinations, count = plan_to_copy_sfa(
            source_ids, current_slots, miss_count, miss_tokens, miss_slots, width=2
        )

        torch.testing.assert_close(
            sources,
            torch.tensor(
                [[4, 5, 2, -1], [5, 2, 8, -1], [1, 4, 7, -1], [2, 5, 8, -1]],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(
            slots,
            torch.tensor(
                [[1, 2, 0, -1], [2, 0, 3, -1], [0, 1, 2, -1], [0, 1, 2, -1]],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(
            topk_misses, torch.tensor([2, 1, 0, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(count, torch.tensor([2, 0], dtype=torch.int32))
        torch.testing.assert_close(misses[0, :2], torch.tensor([4, 5], dtype=torch.int32))
        torch.testing.assert_close(
            destinations[0, :2], torch.tensor([1, 2], dtype=torch.int32)
        )
        self.assertTrue(torch.all(misses[1] == -1).item())


if __name__ == "__main__":
    unittest.main()
