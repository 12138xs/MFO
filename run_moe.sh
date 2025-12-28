#!/bin/bash

# ==========================================
# 0. 环境设置 (针对 RTX 40系列优化)
# ==========================================
export PYTHONPATH=$PYTHONPATH:.

# 禁用 P2P 和 IB，解决 4090/3090 通信报错
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# 设置可见的 GPU ID (根据实际情况修改)
export CUDA_VISIBLE_DEVICES=0,3

# ==========================================
# 启动训练 (Accelerate Launch)
# ==========================================
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "Starting Training on $NUM_GPUS GPUs..."
echo "Precision: FP32 (Full Precision)"

# --mixed_precision no: 强制使用全精度 (FP32)
# --main_process_port: 指定端口防止冲突
accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision no \
    --main_process_port 29501 \
    train.py configs/moe.yaml