#!/bin/bash
# Phase 1: Standard Adam training (baseline)

python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 3000 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --clip_max_norm 1.0 \
    --save_path ./outputs/standard \
    --log_dir ./logs/standard \
    --cuda
