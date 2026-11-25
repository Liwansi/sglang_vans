"""Translate ordered Plan outputs to Copy-SFA's miss-prefix ABI."""

import torch


def plan_to_copy_sfa(source_ids, current_slots, miss_count, miss_tokens, miss_slots, width):
    batch = source_ids.shape[0] // width
    topk = source_ids.shape[1]
    # Plan writes a request's deduplicated misses across its query rows.
    tokens = miss_tokens.reshape(batch, width * topk)
    slots = miss_slots.reshape(batch, width * topk)
    sentinel = torch.iinfo(torch.int32).max
    sortable = torch.where(tokens >= 0, tokens, sentinel)
    sorted_tokens, order = torch.sort(sortable, dim=1)
    sorted_slots = torch.gather(slots, 1, order)
    request_misses = miss_count.reshape(batch, width).sum(dim=1, dtype=torch.int32)

    positions = torch.searchsorted(
        sorted_tokens.contiguous(), source_ids.reshape(batch, -1).contiguous()
    ).clamp_max(width * topk - 1)
    matched = torch.gather(sorted_tokens, 1, positions.long()).reshape(-1, topk)
    is_miss = (source_ids >= 0) & (source_ids == matched)
    index = torch.arange(topk, device=source_ids.device).expand_as(source_ids)
    # Copy-SFA's steady-state kernel treats the first topk_misses entries as
    # misses. Preserve logical order within the miss and hit partitions.
    permutation = torch.argsort(torch.where(is_miss, index, index + topk), dim=1)
    ordered_sources = torch.gather(source_ids, 1, permutation)
    ordered_slots = torch.gather(current_slots, 1, permutation)
    topk_misses = is_miss.sum(dim=1, dtype=torch.int32)
    return (
        ordered_sources,
        ordered_slots,
        topk_misses,
        torch.where(sorted_tokens < sentinel, sorted_tokens, -1),
        sorted_slots,
        request_misses,
    )
