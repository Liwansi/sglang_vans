"""ACLNN entry for the Ascend950 ordered sparse KV planner.

The vendor package currently exposes an ACLNN entry but no Torch binding.  Keep
the descriptor lifetime local to the launch and submit work on PyTorch's current
NPU stream; synchronizing here would defeat decode/graph execution.
"""

from __future__ import annotations

import ctypes
import glob
import os

import torch
import torch_npu  # noqa: F401


def _find_vendor_opapi():
    """Locate the sparsekvplan vendor's libcust_opapi.so by full path.

    deep_ep preloads its own libcust_opapi.so on import, so a bare-name
    ctypes.CDLL will hit that cached copy.  Search ASCEND_CUSTOM_OPP_PATH
    for the sparsekvplan vendor and load its .so explicitly.
    """
    opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    for entry in opp_path.split(":"):
        if not entry or "sparsekvplan" not in entry:
            continue
        so_path = os.path.join(entry, "op_api", "lib", "libcust_opapi.so")
        if os.path.isfile(so_path):
            return so_path
    raise OSError("sparsekvplan libcust_opapi.so not found in ASCEND_CUSTOM_OPP_PATH")


class SparseKvPlanOp:
    def __init__(self):
        try:
            acl = ctypes.CDLL("libascendcl.so")
            nnopbase = ctypes.CDLL("libnnopbase.so")
            vendor = ctypes.CDLL(_find_vendor_opapi())
        except OSError as exc:
            raise RuntimeError(
                "SparseKvPlan requires the installed Ascend950 CANN vendor package "
                "and its set_env.bash in the process environment."
            ) from exc

        self._create = nnopbase.aclCreateTensor
        self._create.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self._create.restype = ctypes.c_void_p
        self._destroy = nnopbase.aclDestroyTensor
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = None
        acl.aclGetRecentErrMsg.restype = ctypes.c_char_p
        self._recent_error = acl.aclGetRecentErrMsg
        self._query = vendor.aclnnSparseKvPlanGetWorkspaceSize
        self._query.argtypes = (
            [ctypes.c_void_p] * 6
            + [ctypes.c_int64] * 3
            + [ctypes.c_void_p] * 11
            + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
        )
        self._query.restype = ctypes.c_int
        self._execute = vendor.aclnnSparseKvPlan
        self._execute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._execute.restype = ctypes.c_int

    def __call__(self, inputs, outputs, *, capacity: int, max_token: int, block_size: int):
        descriptors = []
        try:
            for tensor in (*inputs, *outputs):
                if not tensor.is_contiguous() or tensor.dtype not in (
                    torch.int32,
                    torch.int64,
                ):
                    raise ValueError("SparseKvPlan requires contiguous INT32/INT64 tensors")
                shape = (ctypes.c_int64 * tensor.ndim)(*tensor.shape)
                strides = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
                descriptor = self._create(
                    shape,
                    tensor.ndim,
                    9 if tensor.dtype == torch.int64 else 3,
                    strides,
                    0,
                    2,  # ACL_FORMAT_ND
                    shape,
                    tensor.ndim,
                    tensor.data_ptr(),
                )
                if not descriptor:
                    raise RuntimeError(
                        f"SparseKvPlan aclCreateTensor: {self._recent_error()!r}"
                    )
                descriptors.append(descriptor)
            workspace_size = ctypes.c_uint64()
            executor = ctypes.c_void_p()
            rc = self._query(
                *descriptors[:6],
                capacity,
                max_token,
                block_size,
                *descriptors[6:],
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if rc:
                raise RuntimeError(
                    f"SparseKvPlan GetWorkspaceSize failed ({rc}): "
                    f"{self._recent_error()!r}"
                )
            workspace = torch.empty(
                max(workspace_size.value, 1), dtype=torch.uint8, device=inputs[0].device
            )
            rc = self._execute(
                workspace.data_ptr(),
                workspace_size.value,
                executor,
                torch.npu.current_stream().npu_stream,
            )
            if rc:
                raise RuntimeError(
                    f"SparseKvPlan launch failed ({rc}): {self._recent_error()!r}"
                )
            # Retain the ACLNN workspace until the submitted work has completed.
            workspace.record_stream(torch.npu.current_stream())
        finally:
            for descriptor in descriptors:
                self._destroy(descriptor)
