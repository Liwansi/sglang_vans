echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.num_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000


#export ASCEND_USE_FIA=1
export SGLANG_SET_CPU_AFFINITY=1
#export ASCEND_MF_STORE_URL="tcp://172.22.3.181:12345"
cd /home/lws/sglang/sglang
export PYTHONPATH=${PWD}/python:$PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/zcgy26/Qwen3-235B-A22B-Instruct-2507-w8a8
#MODEL_PATH=/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Instruct-2507-W8A8
EAGLE3_PATH=/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100

#export ASCEND_LAUNCH_BLOCKING=1
#export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_BUFFSIZE=400
#export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
#export DEEPEP_NORMAL_LONG_SEQ_ROUND=128
#export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=65536
export SGLANG_NPU_FUSED_MOE_MODE=2

#MODEL_PATH=/root/.cache/modelscope/hub/models/zcgy26/Qwen3-235B-A22B-Instruct-2507-w8a8

python3 -m sglang.launch_server \
   --model-path ${MODEL_PATH} \
   --attention-backend ascend \
   --quantization modelslim \
   --chunked-prefill-size 32768 \
   --device npu \
   --tp 16 \
   --dp-size 8 \
   --enable-dp-attention \
   --enable-dp-lm-head \
   --mem-fraction-static 0.78 \
   --max-running-requests 32 \
   --host 127.0.0.1 \
   --port 7439 \
   --nnodes 1 \
   --node-rank 0 \
   --moe-a2a-backend ascend_fuseep \
   --deepep-mode normal \
   --dtype bfloat16 \
   --max-prefill-tokens 32768 \
   --cuda-graph-bs 1 2 3 4 \
   --context-length 35000 \
   --speculative-algorithm EAGLE3 \
   --speculative-draft-model-path $EAGLE3_PATH \
   --speculative-num-steps 3 \
   --speculative-eagle-topk 1 \
   --speculative-num-draft-tokens 4 \
   --speculative-draft-model-quantization unquant
   #--cuda-graph-bs 1 2


