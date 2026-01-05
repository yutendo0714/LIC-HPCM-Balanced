#!/bin/bash
# Phase 3: Ablation study - Compare scale weight configurations

# Config 1: Equal weights
echo "Running Config 1: Equal scale weights"
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 1000 \
    --use_hierarchical \
    --scale_weights 0.33 0.33 0.34 \
    --save_path ./outputs/ablation/equal_weights \
    --wandb_name ablation_equal_weights \
    --cuda

# Config 2: Emphasize s1 (coarse scale)
echo "Running Config 2: Emphasize s1"
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 1000 \
    --use_hierarchical \
    --scale_weights 0.5 0.3 0.2 \
    --save_path ./outputs/ablation/s1_emphasis \
    --wandb_name ablation_s1_emphasis \
    --cuda

# Config 3: Emphasize s2 (middle scale) - DEFAULT
echo "Running Config 3: Emphasize s2"
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 1000 \
    --use_hierarchical \
    --scale_weights 0.3 0.4 0.3 \
    --save_path ./outputs/ablation/s2_emphasis \
    --wandb_name ablation_s2_emphasis \
    --cuda

# Config 4: Emphasize s3 (fine scale)
echo "Running Config 4: Emphasize s3"
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 1000 \
    --use_hierarchical \
    --scale_weights 0.2 0.3 0.5 \
    --save_path ./outputs/ablation/s3_emphasis \
    --wandb_name ablation_s3_emphasis \
    --cuda

echo "Ablation study complete!"
