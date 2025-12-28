#!/bin/bash

# ==========================================
# 环境设置
# ==========================================
export PYTHONPATH=$PYTHONPATH:.

# 禁用 P2P (Peer-to-Peer) 和 IB (InfiniBand)
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# 设置可见的 GPU (根据你的实际情况修改，例如使用 0,1 号卡)
export CUDA_VISIBLE_DEVICES=0,3

# 配置文件路径
CONFIG_FILE="configs/moe.yaml"

# ==========================================
# 使用 Accelerate 启动训练
# ==========================================

# 获取当前机器的 GPU 数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "Starting Poseidon MoE Training with Accelerate on $NUM_GPUS GPUs..."

# 使用 accelerate launch 启动
# --multi_gpu: 开启多卡模式
# --num_processes: GPU 数量
# --mixed_precision: 使用 fp16 混合精度 (与 yaml 中的 fp16: true 配合)
# --main_process_port: 指定主进程端口，防止冲突

accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --main_process_port 29500 \
    train.py \
    --output_dir ./checkpoints/moe_run_v1 \
    --dataset_name fluids.incompressible.Gaussians \
    --data_path /data1/cenjianhuan/UniPDESolver/datasets \
    --img_size 128 \
    --patch_size 4 \
    --embed_dim 128 \
    --num_experts 4 \
    --top_k 2 \
    --fno_modes 16 \
    --aux_loss_weight 0.01 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-4 \
    --num_train_epochs 50 \
    --fp16 \
    --remove_unused_columns False \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --evaluation_strategy steps
