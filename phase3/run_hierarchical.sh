#!/bin/bash
# Phase 3: Standard hierarchical balanced training

python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 3000 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --clip_max_norm 1.0 \
    --use_hierarchical \
    --gamma_s1 0.008 \
    --gamma_s2 0.006 \
    --gamma_s3 0.004 \
    --w_lr 0.025 \
    --scale_weights 0.3 0.4 0.3 \
    --save_path ./outputs/hierarchical \
    --log_dir ./logs/hierarchical \
    --wandb_project HPCM-Phase3 \
    --wandb_name hierarchical_base \
    --cuda
