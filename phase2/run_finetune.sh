#!/bin/bash
# Phase 2: Fine-tuning with Balanced R-D from pre-trained checkpoint

python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 500 \
    --batch-size 16 \
    --clip_max_norm 1.0 \
    --use_balanced \
    --gamma 0.002 \
    --w_lr 0.025 \
    --adaptive_gamma \
    --gamma_strategy cosine \
    --finetune \
    --finetune_lr 1e-5 \
    --checkpoint /path/to/pretrained_model.pth \
    --save_path ./outputs/finetuned \
    --log_dir ./logs/finetuned \
    --cuda
