#!/bin/bash
# Phase 4: Ablation study - Freeze strategies

# Config 1: No freezing (full fine-tuning)
echo "Running Config 1: No freezing"
python train.py \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 300 \
    --learning-rate 5e-6 \
    --checkpoint /path/to/phase3/best_model.pth \
    --save_path ./outputs/ablation/no_freeze \
    --wandb_name ablation_no_freeze \
    --cuda

# Config 2: Freeze entropy only (DEFAULT)
echo "Running Config 2: Freeze entropy"
python train.py \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 300 \
    --freeze_entropy \
    --context_lr_ratio 0.1 \
    --checkpoint /path/to/phase3/best_model.pth \
    --save_path ./outputs/ablation/freeze_entropy \
    --wandb_name ablation_freeze_entropy \
    --cuda

# Config 3: Progressive unfreezing
echo "Running Config 3: Progressive unfreezing"
python train.py \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 300 \
    --freeze_entropy \
    --context_lr_ratio 0.1 \
    --progressive_unfreeze \
    --checkpoint /path/to/phase3/best_model.pth \
    --save_path ./outputs/ablation/progressive_unfreeze \
    --wandb_name ablation_progressive \
    --cuda

echo "Freeze strategy ablation study complete!"
