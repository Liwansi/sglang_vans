"""Sparse attention over full and resident C8 MLA caches in HBM."""

import torch
import torch_npu


def _run_c8_sfa(
    query, cache, sparse_indices, query_ends, kv_lengths, block_table, layer
):
    result = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query.contiguous(),
        key=cache,
        value=cache,
        sparse_indices=sparse_indices,
        scale_value=layer.scaling,
        key_quant_mode=2,
        value_quant_mode=2,
        key_dequant_scale=None,
        value_dequant_scale=None,
        actual_seq_lengths_query=query_ends.to(device=query.device, dtype=torch.int32),
        actual_seq_lengths_kv=kv_lengths.to(device=query.device, dtype=torch.int32),
        block_table=block_table,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    return result[0] if isinstance(result, tuple) else result


def forward_full_c8(backend, q, k, q_rope, k_rope, layer, batch, topk_indices):
    manager = backend.sparse_kv_manager
    prefill = batch.forward_mode.is_extend_without_speculative()
    if prefill:
        rows = q.shape[0]
    else:
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
            sparse_query_width,
        )

        rows = batch.batch_size * sparse_query_width(batch)
    heads = layer.tp_q_head_num
    padded_heads = manager.budget.copy_sfa_query_heads
    query_nope = q[:rows].reshape(rows, heads, 512)
    query_rope = q_rope[:rows].reshape(rows, heads, 64)
    if heads < padded_heads:
        padding = (0, 0, 0, padded_heads - heads)
        query_nope = torch.nn.functional.pad(query_nope, padding)
        query_rope = torch.nn.functional.pad(query_rope, padding)
    output = backend.forward_sparse(
        query_nope,
        k,
        None,
        layer,
        batch,
        save_kv_cache=True,
        q_rope=query_rope,
        k_rope=k_rope,
        topk_indices=topk_indices[:rows].reshape(rows, manager.sparse_topk),
    )
    if isinstance(output, tuple):
        output = output[0]
    output = output[:, :heads].contiguous()
    if not prefill:
        width = sparse_query_width(batch)
        valid = batch.seq_lens[: batch.batch_size] > 0
        output = torch.where(
            valid.repeat_interleave(width)[:, None, None],
            output,
            torch.zeros_like(output),
        )
        output = output.reshape(rows, -1)
    if q.shape[0] > rows:
        output = torch.nn.functional.pad(output, (0, 0, 0, q.shape[0] - rows))
    return output


def forward_shared_c8(backend, query_nope, query_rope, layer, plan, cache):
    """Attend to resident slots after the caller copies Plan misses into HBM."""
    manager = backend.sparse_kv_manager
    batch, width = plan.valid_rows.numel(), plan.query_width
    rows, heads = batch * width, query_nope.shape[1]
    padded_heads = manager.budget.copy_sfa_query_heads
    query = torch.cat((query_nope[:rows], query_rope[:rows]), dim=-1)
    if heads < padded_heads:
        query = torch.nn.functional.pad(query, (0, 0, 0, padded_heads - heads))
    kv_lengths = torch.where(
        plan.valid_rows, manager.cache_capacity + width - 1, 1
    ).to(torch.int32)
    output = _run_c8_sfa(
        query,
        cache.view(-1, manager.page_size, 1, cache.shape[-1]).view(
            torch.float8_e4m3fn
        ),
        plan.sparse_indices.unsqueeze(1),
        manager.lim_adapter.query_ends[width - 1, :batch],
        kv_lengths,
        plan.sfa_block_table,
        layer,
    )
    output = torch.where(
        plan.valid_rows.repeat_interleave(width)[:, None, None],
        output,
        torch.zeros_like(output),
    )
    return output[:, :heads].reshape(rows, -1)
