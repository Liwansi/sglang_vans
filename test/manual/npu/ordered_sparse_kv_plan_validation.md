# Ascend950 Ordered QLI + SparseKvPlan + Copy-SFA 验证

适用提交：`00fb42fd69`（`lws/integrate/a5-c8-fully-visible-newop`）。这一改动仅替换 packed C8 decode 侧的 LIM 选路与槽位规划；Copy-SFA 算子仍使用原有版本。本文件中的命令在 Ascend950 Linux 环境执行。

## 1. 构建并验证两个新算子

先按机器上的安装方式初始化 CANN，再在 `ops-transformer` 的 `hjh_sparse_offload` 分支执行：

```bash
cd /path/to/ops-transformer
git fetch origin hjh_sparse_offload
git switch --detach FETCH_HEAD

bash build.sh --pkg --soc=ascend950 \
  --ops=sparse_kv_plan,quant_lightning_indexer_ordered \
  --vendor_name=sparsekvplan

SPARSE_PLAN_INSTALL=$(mktemp -d)
bash build_out/cann-ops-transformer-sparsekvplan*.run \
  --install-path="$SPARSE_PLAN_INSTALL"
source "$SPARSE_PLAN_INSTALL/vendors/sparsekvplan_transformer/bin/set_env.bash"

bash build.sh --torch_extension \
  --ops=quant_lightning_indexer_ordered \
  --vendor_name=sparsekvplan
python3 -m pip install --no-deps --force-reinstall \
  build_out/cann_ops_transformer_sparsekvplan-*.whl

python3 attention/sparse_kv_plan/tests/check_sparse_kv_plan.py \
  --device 0 \
  --opapi "$SPARSE_PLAN_INSTALL/vendors/sparsekvplan_transformer/op_api/lib/libcust_opapi.so"

python3 attention/sparse_kv_plan/tests/check_ordered_to_plan.py \
  --device 0 \
  --package cann_ops_transformer_sparsekvplan \
  --opapi "$SPARSE_PLAN_INSTALL/vendors/sparsekvplan_transformer/op_api/lib/libcust_opapi.so"
```

两个脚本都应成功；第二个脚本应打印 `PASS Ordered V1 -> Plan`。服务进程也必须在加载上述 `set_env.bash` 的环境中启动，否则可能找不到 `libcust_opapi.so` 或算子注册。

## 2. 拉取 SGLang 并测试元数据转换

在 SGLang 仓库执行：

```bash
git fetch lws integrate/a5-c8-fully-visible-newop
git switch --detach FETCH_HEAD
git rev-parse --short=10 HEAD
python3 test/registered/unit/hardware_backend/test_npu_plan_copy_sfa_metadata.py
```

提交号应为 `00fb42fd69` 或包含它的后续提交；单测应通过。单测覆盖多 query 共用一个 miss、每条 query 的 miss 前缀以及不同请求之间的隔离。

## 3. 验证服务输出

沿用原来的 Ascend950 packed C8 LIM + Copy-SFA 启动命令，并确认：

- `SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1` 和 `SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA=1`。
- KV 缓存 dtype 为 `fp8_e4m3`，模型为支持 DSA 的 MLA 模型，且设置了 `--max-running-requests`。
- `SGLANG_NPU_SPARSE_KV_CACHE_SIZE` 先不设置。若显式设置，query 宽度 1/2 要求 4096，宽度 4 要求 8192。
- 当前实现仍要求 PP=1、DCP 关闭。先以 decode 宽度 1 测通，再测 MTP；PD 场景先单独验证 decode 节点。

先在原命令中加入 `--disable-cuda-graph` 测 eager；成功后去掉该参数再测图模式。两轮都使用固定 prompt、`temperature=0`、相同的输出长度，比较旧提交 `dd1615c49d` 与新提交的 token 输出。建议依次覆盖：

1. 短上下文：少于 4096 token。
2. 长上下文：超过 4096 token；宽度 4 的 MTP 再测超过 8192 token。
3. 单请求、并发请求、请求槽复用，以及 decode/MTP 交替。
4. 如使用 PD 分离，再重复 decode 节点的长上下文与并发用例。

先比对输出，再运行 `bench serving` 比较 TTFT 和吞吐。性能对比应使用相同的模型、请求集、并发、缓存容量和图模式。旧提交可显式设为与新提交相同的缓存容量，避免缓存容量差异干扰比较。

Profiler 中，C8 decode 路径应出现 `QuantLightningIndexerOrdered`、`SparseKvPlan` 和原有 Copy-SFA；不应再出现 `fused_quant_lightning_indexer_manage_full_visible`。若失败，保存第一处完整 traceback、启动参数、是否启用 MTP/图模式，以及能复现差异的固定 prompt。新路径在设备侧使用 `sort/searchsorted` 整理 miss，正确性通过后还需重点测这部分开销。
