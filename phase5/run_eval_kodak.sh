#!/bin/bash
# Evaluate on Kodak dataset

set -e

echo "=========================================="
echo "Evaluating on Kodak Dataset"
echo "=========================================="

python evaluate.py \
    --checkpoint ../phase4/outputs/best_model.pth \
    --model hpcm_base \
    --method_name "HPCM-Phase4" \
    --datasets kodak \
    --data_root ./datasets \
    --quality_levels 1 2 3 4 5 \
    --device cuda \
    --output_dir ./outputs/evaluation/kodak

echo ""
echo "Kodak evaluation complete!"
echo "Results saved to ./outputs/evaluation/kodak"
