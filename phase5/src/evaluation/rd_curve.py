"""
Rate-Distortion Curve Generation and Plotting

Generate and visualize rate-distortion curves for compression evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class RDCurve:
    """
    Rate-Distortion curve representation.
    
    Stores rate-distortion points and provides utility methods.
    """
    
    def __init__(
        self,
        name: str,
        rates: List[float],
        psnrs: List[float],
        ms_ssims: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Args:
            name: Curve name (e.g., "HPCM-Phase4")
            rates: Bitrates in bpp
            psnrs: PSNR values in dB
            ms_ssims: MS-SSIM values (optional)
            metadata: Additional metadata (dataset, model config, etc.)
        """
        self.name = name
        self.rates = np.array(rates)
        self.psnrs = np.array(psnrs)
        self.ms_ssims = np.array(ms_ssims) if ms_ssims else None
        self.metadata = metadata or {}
        
        # Sort by rate
        idx = np.argsort(self.rates)
        self.rates = self.rates[idx]
        self.psnrs = self.psnrs[idx]
        if self.ms_ssims is not None:
            self.ms_ssims = self.ms_ssims[idx]
    
    def __len__(self) -> int:
        return len(self.rates)
    
    def __repr__(self) -> str:
        return f"RDCurve(name='{self.name}', points={len(self)})"
    
    def get_point(self, index: int) -> Dict[str, float]:
        """Get a single RD point."""
        point = {
            'rate': float(self.rates[index]),
            'psnr': float(self.psnrs[index]),
        }
        if self.ms_ssims is not None:
            point['ms_ssim'] = float(self.ms_ssims[index])
        return point
    
    def get_rate_range(self) -> Tuple[float, float]:
        """Get min and max bitrate."""
        return float(self.rates.min()), float(self.rates.max())
    
    def get_psnr_range(self) -> Tuple[float, float]:
        """Get min and max PSNR."""
        return float(self.psnrs.min()), float(self.psnrs.max())
    
    def save(self, filepath: str):
        """Save RD curve to JSON file."""
        data = {
            'name': self.name,
            'rates': self.rates.tolist(),
            'psnrs': self.psnrs.tolist(),
            'metadata': self.metadata,
        }
        if self.ms_ssims is not None:
            data['ms_ssims'] = self.ms_ssims.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved RD curve to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RDCurve':
        """Load RD curve from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            name=data['name'],
            rates=data['rates'],
            psnrs=data['psnrs'],
            ms_ssims=data.get('ms_ssims'),
            metadata=data.get('metadata', {}),
        )


class RDCurvePlotter:
    """
    Publication-quality RD curve plotter.
    
    Supports multiple curves, markers, annotations, and export.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 300,
        style: str = 'seaborn-v0_8-paper',
    ):
        """
        Args:
            figsize: Figure size (width, height)
            dpi: Resolution for saved figures
            style: Matplotlib style
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Try to set style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
        
        # Color palette for different methods
        self.colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
        ]
        
        # Marker styles
        self.markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '<', '>']
        
        # Line styles
        self.linestyles = ['-', '--', '-.', ':']
    
    def plot(
        self,
        curves: List[RDCurve],
        metric: str = 'psnr',
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        legend_loc: str = 'lower right',
        grid: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot multiple RD curves.
        
        Args:
            curves: List of RDCurve objects
            metric: Quality metric ('psnr' or 'ms_ssim')
            title: Plot title
            xlabel: X-axis label (default: "Rate (bpp)")
            ylabel: Y-axis label (default: "PSNR (dB)" or "MS-SSIM")
            xlim: X-axis limits
            ylim: Y-axis limits
            legend_loc: Legend location
            grid: Show grid
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for i, curve in enumerate(curves):
            rates = curve.rates
            
            if metric == 'psnr':
                quality = curve.psnrs
                ylabel_default = 'PSNR (dB)'
            elif metric == 'ms_ssim':
                if curve.ms_ssims is None:
                    logger.warning(f"Curve '{curve.name}' has no MS-SSIM data, skipping")
                    continue
                quality = curve.ms_ssims
                ylabel_default = 'MS-SSIM'
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Plot curve
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            
            ax.plot(
                rates,
                quality,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=8,
                label=curve.name,
                markeredgewidth=1.5,
                markeredgecolor='white',
            )
        
        # Set labels
        ax.set_xlabel(xlabel or 'Rate (bpp)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or ylabel_default, fontsize=12, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set limits
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        # Grid
        if grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(loc=legend_loc, fontsize=10, framealpha=0.9)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_with_convex_hull(
        self,
        curves: List[RDCurve],
        highlight_best: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot RD curves with convex hull highlighting best performance.
        
        Args:
            curves: List of RDCurve objects
            highlight_best: Highlight convex hull of best points
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        all_rates = []
        all_psnrs = []
        
        # Plot individual curves
        for i, curve in enumerate(curves):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            ax.plot(
                curve.rates,
                curve.psnrs,
                marker=marker,
                linestyle='--',
                color=color,
                linewidth=1.5,
                markersize=6,
                label=curve.name,
                alpha=0.7,
            )
            
            all_rates.extend(curve.rates)
            all_psnrs.extend(curve.psnrs)
        
        # Find convex hull (best points)
        if highlight_best:
            all_rates = np.array(all_rates)
            all_psnrs = np.array(all_psnrs)
            
            # Sort by rate
            idx = np.argsort(all_rates)
            rates_sorted = all_rates[idx]
            psnrs_sorted = all_psnrs[idx]
            
            # Find convex hull (upper envelope)
            hull_rates = [rates_sorted[0]]
            hull_psnrs = [psnrs_sorted[0]]
            
            for i in range(1, len(rates_sorted)):
                # Only add if PSNR is higher than previous
                if psnrs_sorted[i] > hull_psnrs[-1]:
                    hull_rates.append(rates_sorted[i])
                    hull_psnrs.append(psnrs_sorted[i])
            
            # Plot convex hull
            ax.plot(
                hull_rates,
                hull_psnrs,
                color='black',
                linewidth=3,
                linestyle='-',
                label='Best (Convex Hull)',
                zorder=100,
            )
        
        ax.set_xlabel('Rate (bpp)', fontsize=12, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title('Rate-Distortion Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_comparison_table(
        self,
        curves: List[RDCurve],
        reference_curve: RDCurve,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot RD curves with BD-rate comparison table.
        
        Args:
            curves: List of test curves
            reference_curve: Reference curve
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        from .bd_rate import compute_bd_rate, compute_bd_psnr
        
        fig = plt.figure(figsize=(14, 6), dpi=self.dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
        
        # Left: RD curves
        ax1 = fig.add_subplot(gs[0])
        
        # Plot reference
        ax1.plot(
            reference_curve.rates,
            reference_curve.psnrs,
            marker='o',
            linestyle='-',
            color='black',
            linewidth=2.5,
            markersize=8,
            label=f'{reference_curve.name} (Reference)',
            zorder=100,
        )
        
        # Plot test curves
        for i, curve in enumerate(curves):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            
            ax1.plot(
                curve.rates,
                curve.psnrs,
                marker=marker,
                linestyle='--',
                color=color,
                linewidth=2,
                markersize=7,
                label=curve.name,
            )
        
        ax1.set_xlabel('Rate (bpp)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax1.set_title('Rate-Distortion Curves', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='lower right', fontsize=9)
        
        # Right: Comparison table
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        # Compute BD metrics
        table_data = []
        table_data.append(['Method', 'BD-Rate (%)', 'BD-PSNR (dB)'])
        
        for curve in curves:
            bd_rate = compute_bd_rate(
                curve.rates.tolist(),
                curve.psnrs.tolist(),
                reference_curve.rates.tolist(),
                reference_curve.psnrs.tolist(),
            )
            bd_psnr = compute_bd_psnr(
                curve.rates.tolist(),
                curve.psnrs.tolist(),
                reference_curve.rates.tolist(),
                reference_curve.psnrs.tolist(),
            )
            
            table_data.append([
                curve.name,
                f"{bd_rate:+.2f}",
                f"{bd_psnr:+.3f}",
            ])
        
        # Create table
        table = ax2.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.5, 0.25, 0.25],
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            cell = table[(0, i)]
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='white')
        
        # Color code BD-Rate cells
        for i in range(1, len(table_data)):
            bd_rate = float(table_data[i][1])
            cell = table[(i, 1)]
            
            if bd_rate < -5:
                cell.set_facecolor('#90ee90')  # Light green
            elif bd_rate < 0:
                cell.set_facecolor('#ffffb0')  # Light yellow
            else:
                cell.set_facecolor('#ffcccb')  # Light red
        
        ax2.set_title('BD-Rate Comparison\nvs ' + reference_curve.name,
                     fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def plot_multi_dataset(
        self,
        curves_by_dataset: Dict[str, List[RDCurve]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot RD curves for multiple datasets in subplots.
        
        Args:
            curves_by_dataset: Dict of {dataset_name: [curves]}
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        n_datasets = len(curves_by_dataset)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 5, n_rows * 4),
            dpi=self.dpi,
        )
        
        if n_datasets == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (dataset_name, curves) in enumerate(curves_by_dataset.items()):
            ax = axes[idx]
            
            for i, curve in enumerate(curves):
                color = self.colors[i % len(self.colors)]
                marker = self.markers[i % len(self.markers)]
                
                ax.plot(
                    curve.rates,
                    curve.psnrs,
                    marker=marker,
                    linestyle='-',
                    color=color,
                    linewidth=2,
                    markersize=6,
                    label=curve.name,
                )
            
            ax.set_xlabel('Rate (bpp)', fontsize=10, fontweight='bold')
            ax.set_ylabel('PSNR (dB)', fontsize=10, fontweight='bold')
            ax.set_title(dataset_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='lower right', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved multi-dataset plot to {save_path}")
        
        return fig


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create example curves
    curve1 = RDCurve(
        name='HPCM-Phase4',
        rates=[0.2, 0.4, 0.6, 0.8],
        psnrs=[32.5, 35.2, 37.1, 38.5],
        ms_ssims=[0.965, 0.978, 0.985, 0.990],
    )
    
    curve2 = RDCurve(
        name='BPG',
        rates=[0.25, 0.5, 0.75, 1.0],
        psnrs=[31.8, 34.5, 36.3, 37.8],
        ms_ssims=[0.960, 0.975, 0.982, 0.988],
    )
    
    curve3 = RDCurve(
        name='VTM',
        rates=[0.22, 0.45, 0.68, 0.9],
        psnrs=[32.0, 34.8, 36.7, 38.0],
    )
    
    # Plot
    plotter = RDCurvePlotter()
    
    # Simple plot
    fig1 = plotter.plot(
        curves=[curve1, curve2, curve3],
        title='Rate-Distortion Comparison',
        save_path='rd_curves.png',
    )
    
    # Plot with comparison table
    fig2 = plotter.plot_comparison_table(
        curves=[curve1, curve3],
        reference_curve=curve2,
        save_path='rd_comparison_table.png',
    )
    
    print("Plots created successfully!")
