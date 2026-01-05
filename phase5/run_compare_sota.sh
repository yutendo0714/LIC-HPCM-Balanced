#!/bin/bash
# Compare with SOTA methods

set -e

echo "=========================================="
echo "Comparing with SOTA Methods"
echo "=========================================="

# Run comparison
python compare_sota.py \
    --results_dir ./outputs/evaluation/all \
    --method_name "HPCM-Phase4" \
    --reference BPG \
    --datasets kodak clic \
    --output_dir ./outputs/comparison

echo ""
echo "Comparison complete!"
echo "Results saved to ./outputs/comparison"
echo ""
echo "Generated files:"
ls -lh ./outputs/comparison/
