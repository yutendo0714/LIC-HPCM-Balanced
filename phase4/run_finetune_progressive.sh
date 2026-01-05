#!/bin/bash
# Phase 4: Fine-tuning with progressive unfreezing

python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 500 \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --freeze_entropy \
    --context_lr_ratio 0.1 \
    --progressive_unfreeze \
    --checkpoint /path/to/phase3/best_model.pth \
    --save_path ./outputs/finetune_progressive \
    --log_dir ./logs/finetune_progressive \
    --wandb_project HPCM-Phase4 \
    --wandb_name finetune_progressive \
    --cuda
