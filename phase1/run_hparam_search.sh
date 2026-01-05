#!/bin/bash
# Phase 1: Hyperparameter search for Balanced R-D
# Grid search over gamma and w_lr parameters

LAMBDAS=(0.013)  # Start with single lambda for validation
GAMMAS=(0.001 0.003 0.005 0.01)
W_LRS=(0.01 0.025 0.05)

for lambda in "${LAMBDAS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for w_lr in "${W_LRS[@]}"; do
            echo "=================================================="
            echo "Training with lambda=$lambda, gamma=$gamma, w_lr=$w_lr"
            echo "=================================================="
            
            python train.py \
                --model_name HPCM_Base \
                --train_dataset /path/to/train/dataset \
                --test_dataset /path/to/test/dataset \
                --lambda $lambda \
                --epochs 1000 \
                --batch-size 16 \
                --learning-rate 5e-5 \
                --clip_max_norm 1.0 \
                --use_balanced \
                --gamma $gamma \
                --w_lr $w_lr \
                --save_path ./outputs/hparam_search/lambda_${lambda}_gamma_${gamma}_wlr_${w_lr} \
                --log_dir ./logs/hparam_search/lambda_${lambda}_gamma_${gamma}_wlr_${w_lr} \
                --cuda
            
            echo "Completed: lambda=$lambda, gamma=$gamma, w_lr=$w_lr"
            echo ""
        done
    done
done

echo "Hyperparameter search completed!"
echo "Results saved in ./outputs/hparam_search/"
