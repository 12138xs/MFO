#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python inference.py \
    --checkpoint_dir "./checkpoints/moe_run_final/checkpoint-16422" \
    --dataset_name "fluids.incompressible.Gaussians" \
    --data_path "/data1/cenjianhuan/UniPDESolver/datasets/" \
    --split "test" \
    --batch_size 16
