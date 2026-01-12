#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES=3

CKPT_DIR="./checkpoints/moe_poseidon_aligned/checkpoint-72369"
# CKPT_DIR="./checkpoints/moe_poseidon_aligned_multi/checkpoint-98523"
# CKPT_DIR="./checkpoints/moe_poseidon_aligned_single/checkpoint-106450"

# python infer.py --checkpoint_dir ${CKPT_DIR} \
#     --data_path /data1/cenjianhuan/UniPDESolver/datasets/

python infer_and_plot.py --checkpoint_dir ${CKPT_DIR} \
    --data_path /data1/cenjianhuan/UniPDESolver/datasets/
