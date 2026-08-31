from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.npu.utils import is_ascend_a5
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.utils import get_num_new_pages, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class NPUPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.roundup = page_size - 1
        # Scratch for the fused free() kernel (unique + cat in one Triton
        # launch): [cnt, done, seen(num_pages + 1)]
        self._free_scratch = torch.zeros(
            self.num_pages + 3, dtype=torch.int32, device=device
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        if num_new_pages is None:
            num_new_pages_tensor = (
                (seq_lens_cpu + self.roundup) // self.page_size
                - (prefix_lens_cpu + self.roundup) // self.page_size
            ).sum()
            num_new_pages_item = num_new_pages_tensor.item()
        else:
            num_new_pages_item = num_new_pages
        if self.need_sort and num_new_pages_item > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages_item > len(self.free_pages):
            return None

        if num_new_pages_item < 200:
            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            max_num_extend_tokens = next_power_of_2(extend_num_tokens)
            bs = prefix_lens.shape[0]
            if is_ascend_a5():
                from sglang.kernels.ops.memory.allocator_npu import alloc_extend_npu

                alloc_extend_npu(
                    prefix_lens=prefix_lens,
                    seq_lens=seq_lens,
                    last_loc=last_loc,
                    free_pages=self.free_pages,
                    out_indices=out_indices,
                    page_size=self.page_size,
                    max_num_extend_tokens=max_num_extend_tokens,
                )
            else:
                from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel

                alloc_extend_kernel[(bs,)](
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.free_pages,
                    out_indices,
                    next_power_of_2(bs),
                    self.page_size,
                    max_num_extend_tokens,
                )

        else:
            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=torch.int32,
                device=self.device,
            )
            alloc_extend_naive(
                prefix_lens,
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                self.page_size,
                self.device,
            )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages_item:]
        return out_indices.int()

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )

        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()

        if num_new_pages > len(self.free_pages):
            return None

        need_new_pages = (seq_lens % self.page_size == 1).int()
        end_new_pages = torch.cumsum(need_new_pages, 0)
        start_new_pages = end_new_pages - need_new_pages
        if num_new_pages == 0:
            out_indices = last_loc + 1
        else:
            out_indices = (last_loc + 1) * (1 - need_new_pages) + self.free_pages[
                start_new_pages
            ] * self.page_size * need_new_pages

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices.int()

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            from sglang.kernels.ops.memory.allocator_npu import (
                free_pages_unique_cat_npu,
            )

            if self.need_sort:
                self.release_pages = free_pages_unique_cat_npu(
                    free_index=free_index,
                    old_pages=self.release_pages,
                    page_size=self.page_size,
                    scratch=self._free_scratch,
                )
            else:
                self.free_pages = free_pages_unique_cat_npu(
                    free_index=free_index,
                    old_pages=self.free_pages,
                    page_size=self.page_size,
                    scratch=self._free_scratch,
                )
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)
