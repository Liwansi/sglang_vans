"""Sparse attention over a full, page-addressed C8 MLA cache in HBM."""

import torch
import torch_npu


def forward_full_c8(backend, q, q_rope, layer, batch, topk_indices):
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
    query = torch.cat(
        (
            q[:rows].reshape(rows, heads, 512),
            q_rope[:rows].reshape(rows, heads, 64),
        ),
        dim=-1,
    )
    if heads < padded_heads:
        query = torch.nn.functional.pad(query, (0, 0, 0, padded_heads - heads))
    if prefill:
        query_ends = backend.forward_metadata.actual_seq_lengths_q
        if query_ends is None:
            query_ends = batch.extend_seq_lens.cumsum(0)
    else:
        width = sparse_query_width(batch)
        query_ends = manager.lim_adapter.query_ends[width - 1, : batch.batch_size]
    valid = batch.seq_lens[: batch.batch_size] > 0
    sparse_indices = topk_indices[:rows].reshape(rows, 1, manager.sparse_topk)
    if not prefill:
        sparse_indices = torch.where(
            valid.repeat_interleave(width)[:, None, None],
            sparse_indices,
            torch.zeros_like(sparse_indices),
        )
    cache = manager.main_kv_buffer(layer.layer_id).view(torch.float8_e4m3fn)
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
        actual_seq_lengths_query=query_ends.to(device=q.device, dtype=torch.int32),
        actual_seq_lengths_kv=batch.seq_lens[: batch.batch_size]
        .clamp_min(1)
        .to(device=q.device, dtype=torch.int32),
        block_table=backend.forward_metadata.block_tables[: batch.batch_size].to(
            torch.int32
        ),
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    output = result[0] if isinstance(result, tuple) else result
    output = output[:, :heads].contiguous()
    if not prefill:
        output = torch.where(
            valid.repeat_interleave(width)[:, None, None],
            output,
            torch.zeros_like(output),
        )
        output = output.reshape(rows, -1)
    if q.shape[0] > rows:
        output = torch.nn.functional.pad(output, (0, 0, 0, q.shape[0] - rows))
    return output
