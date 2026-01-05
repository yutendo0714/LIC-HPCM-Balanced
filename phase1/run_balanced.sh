#!/bin/bash
# Phase 1: Balanced R-D training

python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 3000 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --clip_max_norm 1.0 \
    --use_balanced \
    --gamma 0.003 \
    --w_lr 0.025 \
    --save_path ./outputs/balanced \
    --log_dir ./logs/balanced \
    --cuda
