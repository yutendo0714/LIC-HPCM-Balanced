#!/bin/bash
# Phase 4: Ablation study - Different context LR ratios

# Config 1: context_lr = 1.0 (same as main network)
echo "Running Config 1: context_lr_ratio=1.0"
python train.py \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 300 \
    --freeze_entropy \
    --context_lr_ratio 1.0 \
    --checkpoint /path/to/phase3/best_model.pth \
    --save_path ./outputs/ablation/context_lr_1.0 \
    --wandb_name ablation_context_lr_1.0 \
    --cuda

# Config 2: context_lr = 0.5
echo "Running Config 2: context_lr_ratio=0.5"
python train.py \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 300 \
    --freeze_entropy \
    --context_lr_ratio 0.5 \
    --checkpoint /path/to/phase3/best_model.pth \
    --save_path ./outputs/ablation/context_lr_0.5 \
    --wandb_name ablation_context_lr_0.5 \
    --cuda

# Config 3: context_lr = 0.1 (DEFAULT)
echo "Running Config 3: context_lr_ratio=0.1"
python train.py \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 300 \
    --freeze_entropy \
    --context_lr_ratio 0.1 \
    --checkpoint /path/to/phase3/best_model.pth \
    --save_path ./outputs/ablation/context_lr_0.1 \
    --wandb_name ablation_context_lr_0.1 \
    --cuda

# Config 4: context_lr = 0.01
echo "Running Config 4: context_lr_ratio=0.01"
python train.py \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 300 \
    --freeze_entropy \
    --context_lr_ratio 0.01 \
    --checkpoint /path/to/phase3/best_model.pth \
    --save_path ./outputs/ablation/context_lr_0.01 \
    --wandb_name ablation_context_lr_0.01 \
    --cuda

echo "Context LR ablation study complete!"
