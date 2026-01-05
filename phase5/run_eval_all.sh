#!/bin/bash
# Evaluate on all datasets (Kodak, CLIC, Tecnick)

set -e

echo "=========================================="
echo "Evaluating on All Datasets"
echo "=========================================="

python evaluate.py \
    --checkpoint ../phase4/outputs/best_model.pth \
    --model hpcm_base \
    --method_name "HPCM-Phase4" \
    --datasets kodak clic tecnick \
    --data_root ./datasets \
    --quality_levels 1 2 3 4 5 \
    --device cuda \
    --output_dir ./outputs/evaluation/all

echo ""
echo "Multi-dataset evaluation complete!"
echo "Results saved to ./outputs/evaluation/all"
