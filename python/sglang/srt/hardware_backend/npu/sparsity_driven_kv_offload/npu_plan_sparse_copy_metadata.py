"""Compact SparseKvPlan misses into MemFabric sparse_copy descriptors."""

import torch
import triton
import triton.language as tl


@triton.jit
def _plan_sparse_copy_metadata_kernel(
    miss_tokens_ptr,
    miss_slots_ptr,
    miss_counts_ptr,
    request_slots_ptr,
    resident_slots_ptr,
    valid_rows_ptr,
    req_to_token_ptr,
    sources_ptr,
    destinations_ptr,
    lengths_ptr,
    count_ptr,
    host_base,
    device_base,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    CONTEXT: tl.constexpr,
    HOST_CAPACITY: tl.constexpr,
    CACHE_CAPACITY: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    COUNT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    previous = tl.arange(0, COUNT_BLOCK)
    counts = tl.load(miss_counts_ptr + previous, previous < ROWS, other=0)
    request_valid = tl.load(
        valid_rows_ptr + previous // WIDTH, previous < ROWS, other=0
    ) != 0
    counts = tl.where(request_valid, counts, 0)
    offset = tl.sum(tl.where(previous < row, counts, 0), 0)
    if row == 0:
        tl.store(count_ptr, tl.sum(counts, 0))

    batch_row = row // WIDTH
    valid_request = tl.load(valid_rows_ptr + batch_row) != 0
    request = tl.load(request_slots_ptr + batch_row).to(tl.int64)
    resident = tl.load(resident_slots_ptr + batch_row).to(tl.int64) - 1
    nmiss = tl.load(miss_counts_ptr + row)
    col = tl.arange(0, TOPK_BLOCK)
    active = (col < nmiss) & (col < TOPK) & valid_request
    logical = tl.load(miss_tokens_ptr + row * TOPK + col, active, other=0)
    slot = tl.load(miss_slots_ptr + row * TOPK + col, active, other=0)
    physical = tl.load(
        req_to_token_ptr
        + request * REQUEST_STRIDE
        + logical.to(tl.int64) * POSITION_STRIDE,
        active & (logical >= 0) & (logical < CONTEXT),
        other=-1,
    )
    src = host_base + tl.maximum(physical.to(tl.int64), 0) * ROW_BYTES
    dst = device_base + tl.maximum(
        resident * CACHE_STRIDE + slot.to(tl.int64), 0
    ) * ROW_BYTES
    output = offset + col
    tl.store(sources_ptr + output, src, active)
    tl.store(destinations_ptr + output, dst, active)
    tl.store(
        lengths_ptr + output,
        tl.where(
            (physical >= 0)
            & (physical < HOST_CAPACITY)
            & (slot >= 0)
            & (slot < CACHE_CAPACITY),
            ROW_BYTES,
            0,
        ),
        active,
    )


def prepare_plan_sparse_copy_metadata(
    *,
    miss_tokens: torch.Tensor,
    miss_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    request_slots: torch.Tensor,
    resident_slots: torch.Tensor,
    valid_rows: torch.Tensor,
    req_to_token: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    lengths: torch.Tensor,
    count: torch.Tensor,
    host_base: int,
    device_base: int,
    query_width: int,
    cache_stride: int,
    cache_capacity: int,
    host_capacity: int,
    row_bytes: int,
) -> None:
    rows, topk = miss_tokens.shape
    _plan_sparse_copy_metadata_kernel[(rows,)](
        miss_tokens,
        miss_slots,
        miss_counts,
        request_slots,
        resident_slots,
        valid_rows,
        req_to_token,
        sources,
        destinations,
        lengths,
        count,
        host_base,
        device_base,
        rows,
        query_width,
        topk,
        triton.next_power_of_2(topk),
        req_to_token.shape[1],
        host_capacity,
        cache_capacity,
        req_to_token.stride(0),
        req_to_token.stride(1),
        cache_stride,
        row_bytes,
        triton.next_power_of_2(rows),
    )
