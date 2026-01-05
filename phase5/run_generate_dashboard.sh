#!/bin/bash
# Generate visualization dashboard

set -e

echo "=========================================="
echo "Generating Visualization Dashboard"
echo "=========================================="

python scripts/generate_dashboard.py \
    --results_dir ./outputs/evaluation/all \
    --method_name "HPCM-Phase4" \
    --output_dir ./outputs/visualization \
    --output_name dashboard.png

echo ""
echo "Dashboard generation complete!"
echo "Output saved to ./outputs/visualization/dashboard.png"
echo ""
echo "Open with:"
echo "  xdg-open ./outputs/visualization/dashboard.png"
