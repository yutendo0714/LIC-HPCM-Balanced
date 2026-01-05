#!/bin/bash
# Phase 3: Ablation study - Compare gamma configurations

# Config 1: Uniform gammas
echo "Running Config 1: Uniform gammas"
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 1000 \
    --use_hierarchical \
    --gamma_s1 0.006 \
    --gamma_s2 0.006 \
    --gamma_s3 0.006 \
    --save_path ./outputs/ablation/uniform_gamma \
    --wandb_name ablation_uniform_gamma \
    --cuda

# Config 2: Increasing gammas (s1 < s2 < s3)
echo "Running Config 2: Increasing gammas"
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 1000 \
    --use_hierarchical \
    --gamma_s1 0.004 \
    --gamma_s2 0.006 \
    --gamma_s3 0.008 \
    --save_path ./outputs/ablation/increasing_gamma \
    --wandb_name ablation_increasing_gamma \
    --cuda

# Config 3: Decreasing gammas (s1 > s2 > s3) - DEFAULT
echo "Running Config 3: Decreasing gammas"
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train/dataset \
    --test_dataset /path/to/test/dataset \
    --lambda 0.013 \
    --epochs 1000 \
    --use_hierarchical \
    --gamma_s1 0.008 \
    --gamma_s2 0.006 \
    --gamma_s3 0.004 \
    --save_path ./outputs/ablation/decreasing_gamma \
    --wandb_name ablation_decreasing_gamma \
    --cuda

echo "Gamma ablation study complete!"
