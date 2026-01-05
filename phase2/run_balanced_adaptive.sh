#!/bin/bash
# Phase 2: Balanced training with adaptive gamma (HPCM-optimized schedule)

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
    --gamma 0.006 \
    --w_lr 0.025 \
    --adaptive_gamma \
    --gamma_strategy hpcm \
    --save_path ./outputs/balanced_adaptive \
    --log_dir ./logs/balanced_adaptive \
    --cuda
