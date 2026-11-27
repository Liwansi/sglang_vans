"""Ascend950 packed payload, ordered indexer and sparse KV plan adapters."""

from importlib import import_module

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

from sglang.kernels.ops.attention.npu_sparse_kv_plan import SparseKvPlanOp
from sglang.kernels.ops.kvcache.npu_sparse_kv import prepare_sparse_kv_rows
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    C8_ROW_BYTES,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
    LIMDecodeAdapter,
    sparse_query_width,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.layer_layout import (
    SparseKVLayerLayout,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    FusedSparseKVCacheManager,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.memfabric_pool import (
    MemFabricKVPool,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_copy_sfa_metadata import (
    prepare_copy_sfa_metadata,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_plan_copy_sfa_metadata import (
    plan_to_copy_sfa,
)
from sglang.srt.runtime_context import get_parallel


def _copy_packed(source, destination, src, dst, valid):
    unidex_copy_inplace(
        source.view(-1, C8_ROW_BYTES),
        destination.view(-1, C8_ROW_BYTES),
        src.reshape(-1).long(),
        dst.reshape(-1).long(),
        valid.reshape(-1),
        1,
        1,
        block_dim=48,
        src_ptr=source.data_ptr() if source.is_cpu else None,
        dst_ptr=destination.data_ptr() if destination.is_cpu else None,
    )


class PackedMemFabricKVPool(MemFabricKVPool):
    def copy_rows(self, sources, destinations, source_rows, destination_rows, valid):
        (source,), (destination,) = sources, destinations
        if source_rows.numel():
            _copy_packed(source, destination, source_rows, destination_rows, valid)


class C8DecodeAdapter(LIMDecodeAdapter):
    """Convert the ordered indexer/Plan outputs to the existing Copy-SFA ABI."""

    def __init__(self, manager):
        super().__init__(manager)
        self.plan_operator = SparseKvPlanOp()
        self._states = {}
        # Device scalars are updated inside captured graphs as well as eager
        # forwards. A width change invalidates the other width's cache state.
        self._width_epochs = torch.zeros(
            (manager.layer_num,), dtype=torch.int64, device=manager.device
        )
        self._last_widths = torch.zeros(
            (manager.layer_num,), dtype=torch.int32, device=manager.device
        )

    @staticmethod
    def _load_operator():
        for package in (
            "cann_ops_transformer_sparsekvplan",
            "cann_ops_transformer_qliorder",
        ):
            try:
                module = import_module(
                    f"{package}.ops.attention.quant_lightning_indexer_ordered"
                )
            except ImportError:
                continue
            return module.quant_lightning_indexer_ordered
        raise RuntimeError(
            "A5 C8 Ordered+Plan requires the QuantLightningIndexerOrdered Torch "
            "wheel (cann_ops_transformer_sparsekvplan or cann_ops_transformer_qliorder)."
        )

    def _state(self, manager, layer, width):
        key = (layer, width)
        if key not in self._states:
            capacity = 8192 if width == 4 else 4096
            # Batch rows are gathered by stable resident slot before each Plan
            # launch. Extra rows give graph padding a unique scratch identity.
            count = (manager.budget.resident_rows + manager.batch_capacity) * width
            device = manager.device
            self._states[key] = (
                torch.full((count,), -1, dtype=torch.int64, device=device),
                torch.full((count, capacity), -1, dtype=torch.int32, device=device),
                torch.arange(capacity, dtype=torch.int32, device=device)
                .expand(count, -1)
                .clone(),
                torch.zeros((count,), dtype=torch.int32, device=device),
            )
        return self._states[key]

    def reset_requests(self, resident_slots):
        super().reset_requests(resident_slots)
        for layer in {layer for layer, _ in self._states}:
            self._reset_layer_state(layer, resident_slots)

    def reset_layer_requests(self, layer, resident_slots):
        self.last_prefix[layer].index_fill_(0, resident_slots, -1)
        self._reset_layer_state(layer, resident_slots)

    def _reset_layer_state(self, layer, resident_slots):
        for (state_layer, width), (
            last_ids,
            slot_to_token,
            lru_slots,
            row_modes,
        ) in self._states.items():
            if state_layer != layer:
                continue
            indices = (
                resident_slots.long()[:, None] * width
                + torch.arange(width, device=resident_slots.device)[None, :]
            ).reshape(-1)
            last_ids.index_fill_(0, indices, -1)
            slot_to_token.index_fill_(0, indices, -1)
            lru_slots.index_copy_(
                0,
                indices,
                torch.arange(
                    lru_slots.shape[1],
                    device=resident_slots.device,
                    dtype=torch.int32,
                ).expand(indices.numel(), -1),
            )
            row_modes.index_fill_(0, indices, 0)

    def select(
        self,
        manager,
        *,
        query,
        weights,
        key,
        block_table,
        forward_batch,
        layer_id,
        query_scale,
        key_scale,
    ):
        layer = layer_id - manager.start_layer
        batch, width = self.prepare(manager, block_table, forward_batch, layer)
        if width not in (1, 2, 4):
            raise ValueError("SparseKvPlan supports only 1, 2 or 4 query rows/request")
        rows = batch * width
        capacity = 8192 if width == 4 else 4096
        if manager.cache_capacity < capacity:
            raise ValueError(
                "The C8 resident KV allocation is smaller than SparseKvPlan requires"
            )
        # A max-width-4 pool has 8192 physical slots, but decode and width-2
        # Plan runs manage only the first 4096. Keep Copy-SFA's per-request
        # cache size and sparse/short decision consistent with that Plan run.
        sparse = self._pre_valid[:batch] & (self._pre_lengths[:batch] >= capacity)
        self._pre_sparse[:batch].copy_(sparse)
        self._pre_cache_sizes[:batch].copy_(
            torch.where(sparse, capacity, 0).to(torch.int32)
        )
        self._pre_next_prefix[:batch].copy_(
            torch.where(sparse, self._pre_lengths[:batch], -1)
        )
        index_lengths = torch.where(
            self._pre_valid[:batch], self._pre_lengths[:batch], 1
        ).to(torch.int32)
        self.block_table[:batch].masked_fill_(
            ~self._pre_valid[:batch, None], 0
        )
        ordered = self.operator(
            query=query[:rows].contiguous(),
            key=key,
            weights=weights[:rows].contiguous(),
            query_dequant_scale=query_scale[:rows].contiguous(),
            key_dequant_scale=key_scale,
            actual_seq_lengths_query=self.query_ends[width - 1, :batch],
            actual_seq_lengths_key=index_lengths,
            block_table=self.block_table[:batch],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=manager.sparse_topk,
            sparse_mode=3,
            query_quant_mode=0,
            key_quant_mode=0,
        )
        source_ids = ordered.squeeze(1).contiguous()
        source_ids = torch.where(
            self._pre_valid[:batch].repeat_interleave(width)[:, None],
            source_ids,
            -1,
        )
        epoch = self._width_epochs[layer] + (self._last_widths[layer] != width).long()
        self._width_epochs[layer].copy_(epoch)
        self._last_widths[layer].fill_(width)
        persistent = self._state(manager, layer, width)
        safe_resident = torch.where(
            self._pre_valid[:batch],
            self.resident_safe_slots[:batch],
            torch.arange(batch, device=query.device, dtype=torch.int32)
            + manager.budget.resident_rows,
        ).long()
        state_rows = (
            safe_resident[:, None] * width
            + torch.arange(width, device=query.device)[None, :]
        ).reshape(-1)
        last_ids, slots, lru, modes = (
            value.index_select(0, state_rows) for value in persistent
        )
        old_prefix = self.last_prefix[layer].index_select(
            0, self.resident_safe_slots[:batch].long()
        )
        visible = torch.where(
            self._pre_valid[:batch].repeat_interleave(width),
            forward_batch.positions[:rows].to(torch.int32) + 1,
            0,
        ).contiguous()
        if width > 1:
            first_position = forward_batch.positions[:rows:width].to(torch.int32)
            old_prefix = old_prefix.minimum(first_position)
        stable = (
            old_prefix.clamp_min(0)
            .repeat_interleave(width)
            .minimum(visible)
            .contiguous()
        )
        current_slots = torch.empty_like(source_ids)
        miss_count = torch.empty((rows,), dtype=torch.int32, device=query.device)
        miss_tokens = torch.empty_like(source_ids)
        miss_slots = torch.empty_like(source_ids)
        status = torch.full(
            (rows * 1024,),
            torch.iinfo(torch.int32).max,
            dtype=torch.int32,
            device=query.device,
        )
        resident_rows = torch.empty((rows,), dtype=torch.int32, device=query.device)
        resident_table = torch.empty(
            (rows, capacity // manager.page_size), dtype=torch.int32, device=query.device
        )
        req_ids = self.request_safe_slots[:batch].long() + epoch * manager.size
        self.plan_operator(
            (
                req_ids.contiguous(),
                source_ids,
                stable,
                visible,
                torch.arange(batch, device=query.device, dtype=torch.int32)
                .repeat_interleave(width)
                .contiguous(),
                torch.full((1,), rows, dtype=torch.int32, device=query.device),
            ),
            (
                last_ids,
                slots,
                lru,
                current_slots,
                miss_count,
                miss_tokens,
                miss_slots,
                status,
                modes,
                resident_rows,
                resident_table,
            ),
            capacity=capacity,
            max_token=manager.logical_max_context_len,
            block_size=manager.page_size,
        )
        # Keep device-side diagnostics available without a per-token D2H sync.
        self.last_plan_status = status[:rows]
        self.last_plan_modes = modes
        for destination, value in zip(persistent, (last_ids, slots, lru, modes)):
            destination.index_copy_(0, state_rows, value)
        (
            sorted_sources,
            sorted_slots,
            topk_misses,
            misses,
            miss_destinations,
            miss_counts,
        ) = plan_to_copy_sfa(
            source_ids, current_slots, miss_count, miss_tokens, miss_slots, width
        )
        self.source_ids[:rows].copy_(sorted_sources)
        self.device_slots[:rows].copy_(sorted_slots)
        self.topk_misses[:rows].copy_(topk_misses)
        self.miss_sources[:batch].fill_(-1)
        self.miss_slots[:batch].fill_(-1)
        self.miss_sources[:batch, : width * manager.sparse_topk].copy_(misses)
        self.miss_slots[:batch, : width * manager.sparse_topk].copy_(
            miss_destinations
        )
        self.miss_counts[:batch].copy_(miss_counts)
        prepare_copy_sfa_metadata(
            source_ids=self.source_ids[:rows],
            valid_rows=self._pre_valid[:batch],
            sparse_rows=self._pre_sparse[:batch],
            request_slots=self.resident_safe_slots[:batch],
            actual_seq_lengths_kv=self._pre_lengths[:batch],
            hbm_block_table=manager.planner_hbm_block_table[:batch],
            capacity=capacity,
            query_width=width,
            hbm_row_offset=1,
        )
        # Preserve the raw Ordered result for the full HBM SFA.  The converted
        # Copy-SFA source IDs have a different miss-first order.
        return self.finish(layer, batch, width), source_ids


class C8SparseKVCacheManager(FusedSparseKVCacheManager):
    packed_c8 = True

    def __init__(self, req_to_token_pool, sparse_topk, *, paged_kv_cache):
        self.layer_layout = SparseKVLayerLayout.build(
            paged_kv_cache.start_layer,
            paged_kv_cache.start_layer + paged_kv_cache.layer_num,
            tuple(paged_kv_cache.indexer_layer_ids),
        )
        if not self.layer_layout.shared_layers:
            raise ValueError("C8 HiSparse V2 requires at least one shared MLA layer")
        budget = paged_kv_cache.sparse_kv_budget_plan
        if (
            budget.cache_layers != len(self.layer_layout.shared_layers)
            or budget.plan_layers != len(self.layer_layout.full_layers)
        ):
            raise ValueError("C8 sparse layer budget does not match the KV pool")
        super().__init__(
            req_to_token_pool, sparse_topk, paged_kv_cache=paged_kv_cache
        )
        shared_buffers = self.device_kv_buffer
        self.device_kv_buffer = [None] * self.layer_num
        for layer_id, buffer in zip(
            self.layer_layout.shared_layers, shared_buffers, strict=True
        ):
            self.device_kv_buffer[layer_id - self.start_layer] = buffer
        # Same physical token IDs as the indexer K pool and PD prefill source.
        self.full_layer_to_slot = {
            layer_id: slot
            for slot, layer_id in enumerate(self.layer_layout.full_layers)
        }
        self.full_hbm_kv_buffer = torch.empty(
            (
                len(self.full_layer_to_slot),
                self.host_token_capacity // self.page_size,
                self.page_size,
                1,
                C8_ROW_BYTES,
            ),
            dtype=torch.int8,
            device=self.device,
        )
        # Graph padding addresses page zero.  Keep its data finite before the
        # invalid query outputs are masked in the full-HBM attention path.
        self.full_hbm_kv_buffer[:, 0].zero_()

    def main_kv_buffer(self, layer_id):
        slot = self.full_layer_to_slot.get(layer_id)
        if slot is not None:
            return self.full_hbm_kv_buffer[slot]
        return self.host_kv_buffer[layer_id - self.start_layer]

    def pd_main_buffers(self):
        return self.layer_layout.transfer_main_buffers(
            tuple(self.full_hbm_kv_buffer),
            tuple(
                self.host_kv_buffer[layer_id - self.start_layer]
                for layer_id in self.layer_layout.shared_layers
            ),
        )

    @staticmethod
    def _load_copy_sfa_operator():
        if not (
            hasattr(torch.ops.npu, "a5_copy_sfa_capabilities")
            and torch.ops.npu.a5_copy_sfa_capabilities().get("resident_source_gather")
            == 1
        ):
            raise RuntimeError(
                "A5 C8 requires the paired native-LIM source-gather CopySFA wheel."
            )
        return torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2

    def _make_decode_adapter(self):
        return C8DecodeAdapter(self)

    def _init_host_pool(self):
        self.host_pool = self.mf_host_pool = PackedMemFabricKVPool(
            tokens=self.host_token_capacity,
            layers=len(self.layer_layout.shared_layers),
            page_size=self.page_size,
            head_num=1,
            widths=(C8_ROW_BYTES,),
            dtype=torch.int8,
            device=self.device,
            group=get_parallel().attn_tp_group,
        )
        (shared_buffers,) = self.host_pool.buffers
        self.host_kv_buffer = [None] * self.layer_num
        self.dev_ptr_list = [0] * self.layer_num
        for layer_id, buffer in zip(
            self.layer_layout.shared_layers, shared_buffers, strict=True
        ):
            layer_index = layer_id - self.start_layer
            self.host_kv_buffer[layer_index] = buffer
            self.dev_ptr_list[layer_index] = buffer.data_ptr()

    def invalidate_packed_request(self, request_id):
        resident_id = (
            request_id
            if self.request_to_resident is None
            else self._resident_rows_by_request[request_id]
        )
        if resident_id:
            self.reset_requests([request_id], [resident_id])

    def prepare_publication_rows(self, forward_batch, tokens, device):
        """Build layer-independent row addresses for this target forward."""
        prefill = forward_batch.forward_mode.is_extend_without_speculative()
        return prepare_sparse_kv_rows(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            None if prefill else forward_batch.positions,
            forward_batch.out_cache_loc,
            tokens=tokens,
            device=device,
            request_rows=self.size,
            context_len=self.logical_max_context_len,
            host_capacity=self.host_token_capacity,
            query_width=1 if prefill else sparse_query_width(forward_batch),
            extend_seq_lens=forward_batch.extend_seq_lens if prefill else None,
            extend_prefix_lens=forward_batch.extend_prefix_lens if prefill else None,
            compact_prefill=prefill
            and sum(forward_batch.extend_seq_lens_cpu[: forward_batch.batch_size])
            == tokens,
        )

    def offload_v2(self, k, k_rope, layer, forward_batch, publication_rows=None):
        layer_index = layer.layer_id - self.start_layer
        current = self.paged_kv_cache._pack_dsa_fp8_kv_cache(k, k_rope).view(torch.int8)
        prefill = forward_batch.forward_mode.is_extend_without_speculative()
        if publication_rows is None:
            publication_rows = self.prepare_publication_rows(
                forward_batch, current.shape[0], current.device
            )
        src, dst, positions, valid = publication_rows
        if layer.layer_id in self.full_layer_to_slot:
            _copy_packed(current, self.main_kv_buffer(layer.layer_id), src, dst, valid)
        else:
            if self.host_pool.rank == 0:
                _copy_packed(current, self.host_kv_buffer[layer_index], src, dst, valid)
            self.host_pool.publish()
        if prefill:
            requests = forward_batch.req_pool_indices.long()
            self.resident_end[layer_index].index_fill_(0, requests, -1)
            if self.request_to_resident is None:
                self.lim_adapter.reset_layer_requests(layer_index, requests)
        return current, positions, valid

    def copy_sfa_attention(self, layer, query_nope, query_rope, plan, current_inputs):
        layer_index = layer.layer_id - self.start_layer
        owner_layer = self.layer_layout.group_for(layer.layer_id)[0]
        if plan.owner_layer != owner_layer - self.start_layer:
            raise RuntimeError("Copy-SFA received a plan from another indexer group")
        batch, width = plan.valid_rows.numel(), plan.query_width
        rows, heads = batch * width, query_nope.shape[1]
        padded_heads = self.budget.copy_sfa_query_heads
        lengths = self.lim_adapter._pre_lengths[:batch]
        current, positions, valid = current_inputs
        hbm_table = self.planner_hbm_block_table[:batch]
        if heads < padded_heads:
            padding = (0, 0, 0, padded_heads - heads)
            query_nope = torch.nn.functional.pad(query_nope, padding)
            query_rope = torch.nn.functional.pad(query_rope, padding)
        output = query_nope.new_empty((rows, padded_heads, 512))
        self._copy_sfa_operator(
            query_nope.contiguous(),
            query_rope.contiguous(),
            self.device_kv_buffer[layer_index].view(
                -1, self.page_size, 1, C8_ROW_BYTES
            ),
            self.host_kv_buffer[layer_index],
            self.dev_ptr_list[layer_index],
            hbm_table,
            self.lim_adapter.block_table[:batch],
            plan.device_slots.unsqueeze(1),
            plan.source_ids.unsqueeze(1),
            plan.topk_misses,
            self.lim_adapter.query_ends[width - 1, :batch],
            lengths,
            self.lim_adapter._pre_cache_sizes[:batch],
            plan.miss_sources,
            plan.miss_slots,
            plan.miss_counts,
            layer.scaling,
            output,
            current=current[:rows],
            positions=positions[:rows],
            current_valid=valid[:rows],
            request_slots=plan.request_safe_slots,
            req_to_token=self.req_to_token_pool.req_to_token,
            resident_end=self.resident_end[layer_index, : self.actual_request_rows],
            native_lim_plan=True,
        )
        # Invalid requests have positive TND query width and zero KV length.
        # CopySFA initializes their output before skipping them in attention.
        return output[:, :heads].reshape(rows, -1)
