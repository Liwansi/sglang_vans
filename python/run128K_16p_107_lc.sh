unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

pkill -9 python | pkill -9 sglang | pkill -9 VLLM
pkill -9 python | pkill -9 sglang | pkill -9 VLLM

#export PYTHONPATH=/home/lws/sglang/python:$PYTHONPATH
#export PYTHONPATH=/mnt/share/lws/sglang_opt/python:$PYTHONPATH
#export PYTHONPATH=/mnt/share/l00890003/codes/sglang/python:$PYTHONPATH
#export PYTHONPATH=/mnt/share/l00519189/sglang/python:$PYTHONPATH
#export PYTHONPATH=/mnt/share/f00932606/sglang-ljx/python:$PYTHONPATH

export PYTHONPATH=/mnt/share/luochen/a5_glm52/sglang_vans/python:$PYTHONPATH

#export DEEPEP_HCCL_BUFFSIZE=2048
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE=""AIV""
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export SGLANG_HICACHE_HYBM_RESERVE_GB=30
export SGLANG_HICACHE_IO_ASCENDC=1
export SGLANG_HICACHE_HOST_MEM=hybm

export ACL_DEVICE_SYNC_TIMEOUT=300

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#export SGLANG_NPU_USE_MULTI_STREAM=1
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=32
# 网卡
#export HCCL_SOCKET_IFNAME=lo
#export GLOO_SOCKET_IFNAME=lo

#MODEL_PATH=/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4
MODEL_PATH=/mnt/share/w00936111/GLM-5.2-W8A8C8-mxfp8
#MODEL_PATH=/mnt/share/lws/glm_weights/
# [FIA]
export ASCEND_USE_FIA=1

# [MLAPO]
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_NPU_GLM_NEXTN_BF16_KV_CACHE=1

source /mnt/share/chenxu/SFA/vendors/custom_transformer/bin/set_env.bash
source /usr/local/memfabric_hybrid/set_env.sh

# [DEEPEP]
# export MOE_ENABLE_TOPK_NEG_ONE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35
# export SGLANG_NPU_DEEPEP_QUANT="MXFP4"
# export SGLANG_DEEPEP_BF16_DISPATCH=1
export DEEP_USE_MODE="allgather"

# [Prefill Delay]
#export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
#export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

# [MTP]
#export SGLANG_ENABLE_SPEC_V2=1
#export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export TRANSFORMERS_VERBOSITY=error

# [双机]
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export GLOO_SOCKET_IFNAME=data0.173
export HCCL_SOCKET_IFNAME=enp35s0f2
#enp34s0f1

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

export SGLANG_PP_LAYER_PARTITION="38,40"
#export SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1

export SGLANG_LOG_HICACHE_LAYER_TIME=1

# zbal
#export HCCL_BUFFSIZE=0
#unset PYTORCH_NPU_ALLOC_CONF
#export SGLANG_ZBAL_LOCAL_MEM_SIZE=80000
#export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
#export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://127.0.0.1:24669"
# zbal if use mix alloc
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
# zbal if support graph
#export ZBAL_ENABLE_GRAPH=1

NODE_IPS=(
  "141.61.94.103"
  "141.61.94.107"
)

MASTER_ADDR="141.61.94.103"
MASTER_PORT="5000"
DIST_INIT_ADDR="${MASTER_ADDR}:${MASTER_PORT}"

NNODES=2
TP_SIZE=16
SERVED_MODEL_NAME=glm52

SERVER_HOST=141.61.94.107
SERVER_PORT=6678

LOCAL_IPS="141.61.94.107"
NODE_RANK=""

for i in "${!NODE_IPS[@]}"; do
  if [[ " ${LOCAL_IPS} " == *" ${NODE_IPS[$i]} "* ]]; then
    NODE_RANK="${i}"
    break
  fi
done

if [[ -z "${NODE_RANK}" ]]; then
  echo "ERROR: local IPs [${LOCAL_IPS}] not found in NODE_IPS=[${NODE_IPS[*]}]"
  exit 1
fi

echo "========================================"
echo "Launching GLM5.2 2 Nodes"
echo "node-rank       : ${NODE_RANK}"
echo "local IPs       : ${LOCAL_IPS}"
echo "dist-init-addr  : ${DIST_INIT_ADDR}"
echo "nnodes          : ${NNODES}"
echo "tp-size         : ${TP_SIZE}"
echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
echo "========================================"

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--served-model-name "${SERVED_MODEL_NAME}" \
--host "${SERVER_HOST}" \
--port "${SERVER_PORT}" \
--nnodes "${NNODES}" \
--node-rank "${NODE_RANK}" \
--dist-init-addr "${DIST_INIT_ADDR}" \
--tp-size 8 \
--pp-size 2 \
--enable-nsa-prefill-context-parallel \
--dsa-prefill-cp-mode in-seq-split \
--attn-cp-size 8 \
--moe-dense-tp-size 1 \
--disable-cuda-graph \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--max-running-requests 36 \
--mem-fraction-static 0.84 \
--quantization modelslim \
--max-prefill-tokens 2048000 \
--chunked-prefill-size 16384 \
--kv-cache-dtype "fp8_e4m3" \
--moe-a2a-backend deepep \
--deepep-mode auto \
--enable-metrics \
--skip-server-warmup \
--max-total-tokens 327680 \
--enable-hierarchical-cache --hicache-io-backend kernel_ascend --enable-cache-report --hicache-ratio 2 \

#curl --location 'http://127.0.0.1:6677/generate' --header 'Content-Type: application/json' --data '{
#    "text": "The capital of France is",
#    "sampling_params": {
#        "temperature": 0,
#        "max_new_tokens": 1
#    }
#}'
 # python3 aisbench_test.py --input_len 131072 --output_len 1024 --data_num 368 --concurrency 92 --dataset_type prefix_cache --repeat_rate 0.9 --dp 16 --prefix_test
