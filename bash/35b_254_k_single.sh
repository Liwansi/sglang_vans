# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1
cd /home/lws/sglang_vans/
export PYTHONPATH=${PWD}/python:$PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
#export ASCEND_LAUNCH_BLOCKING=1
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1600
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export ASCEND_USE_FIA=1

#export SGLANG_NPU_FUSED_MOE_MODE=1
#export SGLANG_DEEPEP_BF16_DISPATCH=1
#export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
#export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
#export DEEP_NORMAL_MODE_USE_INT8_QUANT=0

python3 -m sglang.launch_server \
        --model-path /home/weights/Qwen3.6-35B-A3B \
        --attention-backend ascend \
        --device npu \
        --tp-size 2 \
        --chunked-prefill-size 131072 --max-prefill-tokens 254000 \
        --disable-radix-cache --base-gpu-id 4 \
        --trust-remote-code \
        --host 127.0.0.1 --max-running-requests 1 --max-mamba-cache-size 6 \
        --mem-fraction-static 0.65 \
        --port 6003 \
        --cuda-graph-bs 1 \
        --enable-multimodal \
        --mm-attention-backend ascend_attn \
        --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
#--moe-a2a-backend deepep \
#--deepep-mode auto \

