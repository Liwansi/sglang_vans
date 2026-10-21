import logging
import os
from typing import List

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.utils.network import NetworkAddress

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)

_DEFAULT_PROTOCOL = "sdma"


class AscendTransferEngine(MooncakeTransferEngine):
    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
    ):
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs/docs/advanced_features/pd_disaggregation.mdx"
            )
            raise import_error

        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id

        # Centralized storage address of the AscendTransferEngine
        self.store_url = os.getenv("ASCEND_MF_STORE_URL")
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        rpc_port = self.engine.get_rpc_port()
        self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()
        self.initialize()
        if rpc_port == 0:
            rpc_port = self.engine.get_rpc_port()
            self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()

    def initialize(self) -> None:
        from sglang.srt.runtime_context import get_parallel

        transfer_protocol = self._get_transfer_protocol()
        if transfer_protocol == "device_rdma":
            # with device RDMA for PD transfer: initialize hccl in advance
            # through all_gather to avoid conflicts with rdma initialization.
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor)
                for _ in range(get_parallel().launch_world_size)
            ]
            torch.distributed.all_gather(
                output_tensor_list,
                tmp_tensor,
                group=get_parallel().world_group.device_group,
            )

        trans_op_type = self._resolve_trans_op_type(transfer_protocol)
        """Initialize the ascend transfer instance."""
        ret_value = self.engine.initialize(
            self.store_url, self.session_id, self.role, self.npu_id, trans_op_type
        )
        if ret_value != 0:
            logger.error("Ascend Transfer Engine initialization failed.")
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark register as failed
            ret_value = -1
        if ret_value != 0:
            logger.debug(f"Ascend memory registration for ptr {ptrs} failed.")

    def batch_transfer_sync_write(
        self,
        session_id: str,
        src_ptrs: List[int],
        dst_ptrs: List[int],
        lengths: List[int],
    ) -> int:
        """Push-mode batch transfer: write local source buffers to remote
        destinations registered under *session_id*.

        Used by the D2RH path where Prefill pushes main KV to Decode's GVA
        DRAM pool and indexer to Decode's HBM in a single batch.  Falls back
        to ``batch_transfer_sync`` when the memfabric build does not expose
        the explicit write API.
        """
        write_fn = getattr(self.engine, "batch_transfer_sync_write", None)
        if write_fn is not None:
            try:
                return write_fn(session_id, src_ptrs, dst_ptrs, lengths)
            except Exception:
                return -1
        return self.engine.batch_transfer_sync(
            session_id, src_ptrs, dst_ptrs, lengths
        )

    @staticmethod
    def _get_transfer_protocol() -> str:
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        return protocol.strip().lower() if protocol else _DEFAULT_PROTOCOL

    @staticmethod
    def _resolve_trans_op_type(protocol: str):
        op_type = getattr(TransferEngine.TransDataOpType, protocol.upper(), None)
        if op_type is None:
            logger.warning(
                "Transfer protocol %r is not supported by the installed "
                "memfabric_hybrid, falling back to %r.",
                protocol,
                _DEFAULT_PROTOCOL,
            )
            op_type = TransferEngine.TransDataOpType.SDMA
        return op_type
