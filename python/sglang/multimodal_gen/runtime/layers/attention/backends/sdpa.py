# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.common import is_npu


logger = init_logger(__name__)

_is_npu = is_npu()


class SDPABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SDPA"

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


class SDPAImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # transpose to bs, heads, seq_len, head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_kwargs = {
            "attn_mask": None,
            "dropout_p": self.dropout,
            "is_causal": self.causal,
            "scale": self.softmax_scale,
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, **attn_kwargs
        )
        output = output.transpose(1, 2).flatten(2, 3)
        return output

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu
        bs, tokens, q_head_num, q_head_dim = query.shape
        _, _, kv_head_num, kv_head_dim = key.shape
        query = query.reshape(-1, q_head_num, q_head_dim)
        key = key.reshape(-1, kv_head_num, kv_head_dim)
        value = value.reshape(-1, kv_head_num, kv_head_dim)
        actual_seq_lengths = torch.tensor([tokens], dtype=torch.int32)

        attn_output = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key,
            value,
            num_query_heads=q_head_num,
            num_key_value_heads=kv_head_num,
            input_layout="TND",
            softmax_scale=self.softmax_scale,
            atten_mask=None,
            block_table=None,
            actual_seq_qlen=actual_seq_lengths,
            actual_seq_kvlen=actual_seq_lengths,
        )[0]
        output = attn_output.view(-1, q_head_num * q_head_dim).unsqueeze(0)
        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if not _is_npu:
            return self.forward_native(query, key, value)
        else:
            return self.forward_npu(query, key, value)
        #return self.forward_native(query, key, value)
