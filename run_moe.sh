#!/bin/bash

# ==========================================
# 0. 环境设置
# ==========================================
export PYTHONPATH=$PYTHONPATH:.
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_MODE=offline

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
    train.py configs/moe.yaml \
    # --train_small_time_transition True \
    # --output_dir ./results/
    # --resume_from_checkpoint \
    # --overwrite_output_dir
