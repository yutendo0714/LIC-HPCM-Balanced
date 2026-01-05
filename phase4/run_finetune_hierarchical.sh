#!/bin/bash
# Phase 4: Fine-tuning with hierarchical loss and scale-specific early stopping

python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 500 \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --use_hierarchical \
    --freeze_entropy \
    --context_lr_ratio 0.1 \
    --scale_early_stopping \
    --early_stopping_patience 50 \
    --checkpoint /path/to/phase3/best_model.pth \
    --save_path ./outputs/finetune_hierarchical \
    --log_dir ./logs/finetune_hierarchical \
    --wandb_project HPCM-Phase4 \
    --wandb_name finetune_hierarchical_es \
    --cuda
