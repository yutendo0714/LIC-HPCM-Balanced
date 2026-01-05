#!/usr/bin/env python3
"""
Generate Visualization Dashboard

Create comprehensive visualization dashboard for evaluation results.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import RDCurve, RDCurvePlotter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dashboard(
    curves_by_dataset: dict,
    output_path: str,
    title: str = 'Compression Evaluation Dashboard',
):
    """
    Create comprehensive visualization dashboard.
    
    Args:
        curves_by_dataset: Dict of {dataset: [curves]}
        output_path: Output file path
        title: Dashboard title
    """
    n_datasets = len(curves_by_dataset)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'v', 'D']
    
    # Plot 1: RD curves for all datasets (top left, 2x2)
    ax_rd = fig.add_subplot(gs[0:2, 0:2])
    
    for idx, (dataset, curves) in enumerate(curves_by_dataset.items()):
        for curve in curves:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            
            ax_rd.plot(
                curve.rates,
                curve.psnrs,
                marker=marker,
                linestyle='-',
                color=color,
                linewidth=2,
                markersize=8,
                label=f'{curve.name} ({dataset})',
            )
    
    ax_rd.set_xlabel('Rate (bpp)', fontsize=12, fontweight='bold')
    ax_rd.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax_rd.set_title('Rate-Distortion Curves', fontsize=14, fontweight='bold')
    ax_rd.grid(True, alpha=0.3, linestyle='--')
    ax_rd.legend(loc='lower right', fontsize=9)
    
    # Plot 2: PSNR comparison (top right)
    ax_psnr = fig.add_subplot(gs[0, 2])
    
    dataset_names = []
    psnr_means = []
    psnr_stds = []
    
    for dataset, curves in curves_by_dataset.items():
        for curve in curves:
            dataset_names.append(f'{dataset[:4]}')
            psnr_means.append(np.mean(curve.psnrs))
            psnr_stds.append(np.std(curve.psnrs))
    
    x_pos = np.arange(len(dataset_names))
    ax_psnr.bar(x_pos, psnr_means, yerr=psnr_stds,
               color=colors[:len(dataset_names)], alpha=0.7,
               error_kw={'linewidth': 2, 'ecolor': 'black'})
    ax_psnr.set_xticks(x_pos)
    ax_psnr.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax_psnr.set_ylabel('PSNR (dB)', fontweight='bold')
    ax_psnr.set_title('Average PSNR', fontweight='bold')
    ax_psnr.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: BPP comparison (middle right)
    ax_bpp = fig.add_subplot(gs[1, 2])
    
    bpp_means = []
    bpp_stds = []
    
    for dataset, curves in curves_by_dataset.items():
        for curve in curves:
            bpp_means.append(np.mean(curve.rates))
            bpp_stds.append(np.std(curve.rates))
    
    ax_bpp.bar(x_pos, bpp_means, yerr=bpp_stds,
              color=colors[:len(dataset_names)], alpha=0.7,
              error_kw={'linewidth': 2, 'ecolor': 'black'})
    ax_bpp.set_xticks(x_pos)
    ax_bpp.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax_bpp.set_ylabel('BPP', fontweight='bold')
    ax_bpp.set_title('Average BPP', fontweight='bold')
    ax_bpp.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: PSNR vs BPP scatter (bottom left)
    ax_scatter = fig.add_subplot(gs[2, 0])
    
    for idx, (dataset, curves) in enumerate(curves_by_dataset.items()):
        for curve in curves:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            
            ax_scatter.scatter(
                curve.rates,
                curve.psnrs,
                marker=marker,
                color=color,
                s=100,
                alpha=0.6,
                label=f'{dataset}',
                edgecolors='black',
                linewidth=1,
            )
    
    ax_scatter.set_xlabel('BPP', fontweight='bold')
    ax_scatter.set_ylabel('PSNR (dB)', fontweight='bold')
    ax_scatter.set_title('PSNR vs BPP Scatter', fontweight='bold')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(loc='lower right', fontsize=9)
    
    # Plot 5: Rate distribution (bottom middle)
    ax_rate_dist = fig.add_subplot(gs[2, 1])
    
    all_rates = []
    for dataset, curves in curves_by_dataset.items():
        for curve in curves:
            all_rates.extend(curve.rates.tolist())
    
    ax_rate_dist.hist(all_rates, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax_rate_dist.set_xlabel('BPP', fontweight='bold')
    ax_rate_dist.set_ylabel('Count', fontweight='bold')
    ax_rate_dist.set_title('Rate Distribution', fontweight='bold')
    ax_rate_dist.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: PSNR distribution (bottom right)
    ax_psnr_dist = fig.add_subplot(gs[2, 2])
    
    all_psnrs = []
    for dataset, curves in curves_by_dataset.items():
        for curve in curves:
            all_psnrs.extend(curve.psnrs.tolist())
    
    ax_psnr_dist.hist(all_psnrs, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax_psnr_dist.set_xlabel('PSNR (dB)', fontweight='bold')
    ax_psnr_dist.set_ylabel('Count', fontweight='bold')
    ax_psnr_dist.set_title('PSNR Distribution', fontweight='bold')
    ax_psnr_dist.grid(True, alpha=0.3, axis='y')
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved dashboard to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualization dashboard')
    
    # Input arguments
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing evaluation results')
    parser.add_argument('--method_name', type=str, default='HPCM-Phase4',
                       help='Method name')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs/visualization',
                       help='Output directory')
    parser.add_argument('--output_name', type=str, default='dashboard.png',
                       help='Output filename')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    logger.info("Loading evaluation results...")
    results_path = Path(args.results_dir)
    
    curves_by_dataset = {}
    
    for curve_file in results_path.glob('rd_curve_*.json'):
        dataset = curve_file.stem.replace('rd_curve_', '')
        curve = RDCurve.load(str(curve_file))
        curve.name = args.method_name
        
        curves_by_dataset[dataset] = [curve]
        logger.info(f"Loaded {dataset}: {len(curve)} points")
    
    if not curves_by_dataset:
        logger.error("No RD curves found!")
        return
    
    # Create dashboard
    logger.info("\nGenerating visualization dashboard...")
    
    output_path = output_dir / args.output_name
    create_dashboard(
        curves_by_dataset=curves_by_dataset,
        output_path=str(output_path),
        title=f'{args.method_name} Evaluation Dashboard',
    )
    
    logger.info(f"\nDashboard saved to {output_path}")
    
    # Also create individual plots
    logger.info("\nGenerating individual plots...")
    
    plotter = RDCurvePlotter()
    
    # Plot all datasets together
    all_curves = []
    for curves in curves_by_dataset.values():
        all_curves.extend(curves)
    
    plotter.plot(
        curves=all_curves,
        title='Rate-Distortion Comparison',
        save_path=str(output_dir / 'rd_curves.png'),
    )
    
    # Plot per dataset
    plotter.plot_multi_dataset(
        curves_by_dataset=curves_by_dataset,
        save_path=str(output_dir / 'rd_curves_multi_dataset.png'),
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Visualization complete! Files saved to {output_dir}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
