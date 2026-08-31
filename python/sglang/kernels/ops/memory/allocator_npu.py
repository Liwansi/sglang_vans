# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def alloc_extend_npu_kernel(
    prefix_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    BS_UPPER: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    max_num_extend_tokens,
    BLOCK_SIZE: tl.constexpr = 2048,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BS_UPPER)
    seq_lens = tl.load(seq_lens_ptr + offsets, mask=offsets <= pid, other=0)
    prefix_lens = tl.load(prefix_lens_ptr + offsets, mask=offsets <= pid, other=0)
    extend_lens = seq_lens - prefix_lens
    seq_len = tl.load(seq_lens_ptr + pid)
    prefix_len = tl.load(prefix_lens_ptr + pid)
    extend_len = seq_len - prefix_len
    output_start = tl.sum(extend_lens) - extend_len

    pages_after = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
    pages_before = (prefix_lens + PAGE_SIZE - 1) // PAGE_SIZE
    num_new_pages = pages_after - pages_before
    num_pages_for_row = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE - (
        prefix_len + PAGE_SIZE - 1
    ) // PAGE_SIZE
    new_page_start = tl.sum(num_new_pages) - num_pages_for_row

    last_loc = tl.load(last_loc_ptr + pid).to(tl.int64)
    num_part1 = (
        min(seq_len, (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE) - prefix_len
    )
    page_offsets = tl.arange(0, PAGE_SIZE)
    tl.store(
        out_indices + output_start + page_offsets,
        last_loc + 1 + page_offsets,
        mask=page_offsets < num_part1,
    )
    if prefix_len + num_part1 == seq_len:
        return

    num_part2 = (
        seq_len // PAGE_SIZE * PAGE_SIZE
        - (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
    )
    block_offsets = tl.arange(0, BLOCK_SIZE)
    for block_idx in range(tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)):
        current = block_offsets + block_idx * BLOCK_SIZE
        page_start = tl.load(
            free_page_ptr + new_page_start + current // PAGE_SIZE,
            mask=current < num_part2,
        )
        tl.store(
            out_indices + output_start + num_part1 + current,
            page_start * PAGE_SIZE + current % PAGE_SIZE,
            mask=current < num_part2,
        )
    if prefix_len + num_part1 + num_part2 == seq_len:
        return

    num_part3 = seq_len - seq_len // PAGE_SIZE * PAGE_SIZE
    start_loc = tl.load(free_page_ptr + new_page_start + num_pages_for_row - 1)
    tl.store(
        out_indices + output_start + num_part1 + num_part2 + page_offsets,
        start_loc * PAGE_SIZE + page_offsets,
        mask=page_offsets < num_part3,
    )


_npu_aiv_core_count_cache: int = 0


def _npu_vector_core_count() -> int:
    """Number of physical vector (AIV) cores on the current NPU device.

    On Ascend, the Triton grid maps 1:1 onto physical cores (grid size must
    not exceed the total AI Core count; each core executes a block exactly
    once).  Query the count through triton-ascend's runtime
    (``triton.runtime.driver.active.utils.get_device_properties``), then
    torch_npu, and finally an env override; result is cached.
    """
    global _npu_aiv_core_count_cache
    if _npu_aiv_core_count_cache:
        return _npu_aiv_core_count_cache

    count = 0
    # 1) triton-ascend runtime (the canonical source for AIV core count)
    try:
        props = triton.runtime.driver.active.utils.get_device_properties()
        for key in ("core_count", "vector_core_count", "aicore_count", "aiv_count"):
            count = int(props.get(key, 0)) if isinstance(props, dict) else int(
                getattr(props, key, 0)
            )
            if count:
                break
    except Exception:
        pass
    # 2) torch_npu device properties
    if not count:
        try:
            props = torch.npu.get_device_properties(torch.npu.current_device())
            for key in ("vector_core_count", "core_count", "aicore_count"):
                count = int(getattr(props, key, 0) or 0)
                if count:
                    break
        except Exception:
            pass
    # 3) explicit override / fallback
    if not count:
        count = int(os.environ.get("SGLANG_NPU_AIV_COUNT", "0"))
    if not count:
        count = 48  # conservative fallback (Atlas A2-class AIV count)

    _npu_aiv_core_count_cache = count
    return count


@triton.jit
def free_pages_unique_cat_npu_kernel(
    free_index_ptr,  # [N] int64, token indices being freed
    old_pages_ptr,  # [M] int64, current page pool (free_pages/release_pages)
    out_ptr,  # [N + M] int64, output = [unique_new_pages | old_pages]
    seen_ptr,  # [num_pages + 1] int32, dedup marks, zeroed before launch
    cnt_ptr,  # [1] int32, unique new-page counter, zeroed before launch
    done_ptr,  # [1] int32, finished-program counter, zeroed before launch
    N,
    M,
    page_size,
    COPY_BLOCK: tl.constexpr = 2048,
):
    # Persistent grid-stride over tokens: the grid is sized to the number
    # of physical vector cores (one program per core) and each core walks
    # the token array with stride = num_programs.  The grid does NOT scale
    # with N (batch size); parallelism saturates the physical cores exactly
    # once, and each element is handled scalar (no lane masking needed).
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Phase 1: token index -> page id. The first core to touch a page
    # (atomic mark in `seen`) claims the next output slot via `cnt`.
    for i in range(pid, N, num_programs):
        idx = tl.load(free_index_ptr + i)
        page = (idx // page_size).to(tl.int32)
        if tl.atomic_add(seen_ptr + page, 1) == 0:
            pos = tl.atomic_add(cnt_ptr, 1)
            tl.store(out_ptr + pos, page.to(out_ptr.dtype.element_ty))

    # Phase 2: the last program to finish phase 1 appends the old pool at
    # out[cnt : cnt + M]. No spin-wait, so the kernel is deadlock-free
    # under any program scheduling order.
    if tl.atomic_add(done_ptr, 1) == num_programs - 1:
        cnt = tl.atomic_add(cnt_ptr, 0)
        for i in range(0, tl.cdiv(M, COPY_BLOCK)):
            offs = i * COPY_BLOCK + tl.arange(0, COPY_BLOCK)
            m = offs < M
            old = tl.load(old_pages_ptr + offs, mask=m, other=0)
            tl.store(out_ptr + cnt + offs, old, mask=m)


def free_pages_unique_cat_npu(
    *,
    free_index: torch.Tensor,
    old_pages: torch.Tensor,
    page_size: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    """Fused replacement for
    `torch.cat((torch.unique(free_index // page_size), old_pages))`.

    scratch: int32 tensor of length >= num_pages + 3, zeroed on every call:
      scratch[0] = cnt, scratch[1] = done, scratch[2:] = dedup marks.

    Notes:
    - The unique new pages are NOT sorted (unlike torch.unique); the pool
      is consumed as a set (alloc pops from the front, need_sort re-sorts
      in merge_and_sort_free), so this is semantically equivalent.
    - `scratch[0].item()` is the single D2H sync needed to learn the
      data-dependent output length; torch.unique syncs internally as well.
    """
    N = free_index.numel()
    M = old_pages.numel()
    out = torch.empty(N + M, dtype=old_pages.dtype, device=free_index.device)
    scratch.zero_()
    # Grid = number of physical vector cores (persistent); each core
    # strides over the N tokens. Never sized from N/batch.
    grid = (_npu_vector_core_count(),)
    free_pages_unique_cat_npu_kernel[grid](
        free_index,
        old_pages,
        out,
        scratch[2:],
        scratch[0],
        scratch[1],
        N,
        M,
        page_size,
    )
    num_new = int(scratch[0].item())
    return out[: num_new + M]


def alloc_extend_npu(
    *,
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    page_size: int,
    max_num_extend_tokens: int,
) -> None:
    batch_size = int(prefix_lens.shape[0])
    if batch_size == 0:
        return
    alloc_extend_npu_kernel[(batch_size,)](
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        BS_UPPER=triton.next_power_of_2(batch_size),
        PAGE_SIZE=page_size,
        max_num_extend_tokens=max_num_extend_tokens,
    )
