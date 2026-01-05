#!/bin/bash
# Phase 2: Analyze hyperparameter search results

RESULTS_DIR="./outputs/hparam_search"
OUTPUT_REPORT="hparam_analysis_report.txt"

echo "Analyzing hyperparameter search results from: $RESULTS_DIR"

python -m src.utils.hparam_analyzer \
    "$RESULTS_DIR" \
    --output "$OUTPUT_REPORT" \
    --plot-heatmap \
    --lambda 0.013

echo ""
echo "Analysis complete!"
echo "  - Report saved to: $OUTPUT_REPORT"
echo "  - Heatmap saved to: heatmap_lambda_0.013.png"
echo "  - Comparison plot saved to: comparison.png"
