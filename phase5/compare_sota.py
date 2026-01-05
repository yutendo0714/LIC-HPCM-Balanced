#!/usr/bin/env python3
"""
Compare with SOTA Methods

Compare evaluation results with state-of-the-art compression methods.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import RDCurve, SOTAComparator, RDCurvePlotter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_evaluation_results(results_dir: str, method_name: str) -> dict:
    """Load evaluation results from JSON files."""
    results_path = Path(results_dir)
    
    curves_by_dataset = {}
    
    # Find all result files
    for result_file in results_path.glob('rd_curve_*.json'):
        # Extract dataset name
        dataset = result_file.stem.replace('rd_curve_', '')
        
        # Load curve
        curve = RDCurve.load(str(result_file))
        curve.name = method_name  # Override name
        
        curves_by_dataset[dataset] = {method_name: curve}
        logger.info(f"Loaded {method_name} results for {dataset}")
    
    return curves_by_dataset


def main():
    parser = argparse.ArgumentParser(description='Compare with SOTA methods')
    
    # Input arguments
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing evaluation results')
    parser.add_argument('--method_name', type=str, default='HPCM-Phase4',
                       help='Method name')
    
    # Comparison arguments
    parser.add_argument('--reference', type=str, default='BPG',
                       choices=['VTM', 'BPG', 'JPEG2000'],
                       help='Reference method for BD-rate calculation')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['kodak', 'clic'],
                       help='Datasets to compare')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs/comparison',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation results
    logger.info("Loading evaluation results...")
    curves_by_dataset = load_evaluation_results(args.results_dir, args.method_name)
    
    if not curves_by_dataset:
        logger.error("No evaluation results found!")
        return
    
    # Filter datasets
    filtered_curves = {
        dataset: curves
        for dataset, curves in curves_by_dataset.items()
        if dataset in args.datasets
    }
    
    if not filtered_curves:
        logger.warning(f"No matching datasets found. Available: {list(curves_by_dataset.keys())}")
        return
    
    # Compare with SOTA
    logger.info("\nComparing with SOTA methods...")
    logger.info(f"{'='*80}")
    
    comparator = SOTAComparator()
    comparator.generate_sota_report(
        method_curves_by_dataset=filtered_curves,
        output_dir=str(output_dir),
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparison complete! Results saved to {output_dir}")
    logger.info(f"{'='*80}")
    
    # Print summary
    logger.info("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        logger.info(f"  - {file.name}")


if __name__ == '__main__':
    main()
