"""
Model Comparison Module

Compare different compression methods and generate reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import logging

from .bd_rate import BDRateCalculator
from .rd_curve import RDCurve, RDCurvePlotter

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare compression models and generate reports.
    
    Supports multiple datasets, metrics, and export formats.
    """
    
    def __init__(self):
        """Initialize comparator."""
        self.bd_calc = BDRateCalculator()
        self.plotter = RDCurvePlotter()
    
    def compare(
        self,
        test_curves: Dict[str, RDCurve],
        reference_curve: RDCurve,
        dataset_name: str = 'Unknown',
    ) -> pd.DataFrame:
        """
        Compare multiple methods against a reference.
        
        Args:
            test_curves: Dict of {method_name: RDCurve}
            reference_curve: Reference RD curve
            dataset_name: Dataset name
        
        Returns:
            Comparison DataFrame
        """
        results = []
        
        for method_name, curve in test_curves.items():
            # BD-Rate
            bd_rate = self.bd_calc.bd_rate(
                curve.rates.tolist(),
                curve.psnrs.tolist(),
                reference_curve.rates.tolist(),
                reference_curve.psnrs.tolist(),
            )
            
            # BD-PSNR
            bd_psnr = self.bd_calc.bd_psnr(
                curve.rates.tolist(),
                curve.psnrs.tolist(),
                reference_curve.rates.tolist(),
                reference_curve.psnrs.tolist(),
            )
            
            # Rate range
            rate_min, rate_max = curve.get_rate_range()
            
            # PSNR range
            psnr_min, psnr_max = curve.get_psnr_range()
            
            results.append({
                'Dataset': dataset_name,
                'Method': method_name,
                'BD-Rate (%)': bd_rate,
                'BD-PSNR (dB)': bd_psnr,
                'Rate Min (bpp)': rate_min,
                'Rate Max (bpp)': rate_max,
                'PSNR Min (dB)': psnr_min,
                'PSNR Max (dB)': psnr_max,
                'Points': len(curve),
            })
        
        df = pd.DataFrame(results)
        return df
    
    def compare_multi_dataset(
        self,
        curves_by_dataset: Dict[str, Dict[str, RDCurve]],
        reference_curves: Dict[str, RDCurve],
    ) -> pd.DataFrame:
        """
        Compare across multiple datasets.
        
        Args:
            curves_by_dataset: Dict of {dataset: {method: curve}}
            reference_curves: Dict of {dataset: reference_curve}
        
        Returns:
            Combined comparison DataFrame
        """
        all_results = []
        
        for dataset_name, test_curves in curves_by_dataset.items():
            reference_curve = reference_curves[dataset_name]
            
            df = self.compare(test_curves, reference_curve, dataset_name)
            all_results.append(df)
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Add average row
        avg_df = combined_df.groupby('Method').agg({
            'BD-Rate (%)': 'mean',
            'BD-PSNR (dB)': 'mean',
        }).reset_index()
        avg_df['Dataset'] = 'Average'
        
        combined_df = pd.concat([combined_df, avg_df], ignore_index=True)
        
        return combined_df
    
    def generate_report(
        self,
        comparison_df: pd.DataFrame,
        output_dir: str,
        report_name: str = 'comparison_report',
    ):
        """
        Generate comprehensive comparison report.
        
        Args:
            comparison_df: Comparison DataFrame
            output_dir: Output directory
            report_name: Report name
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_path / f'{report_name}.csv'
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison CSV to {csv_path}")
        
        # Save formatted table
        txt_path = output_path / f'{report_name}.txt'
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPRESSION METHODS COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            f.write("=" * 80 + "\n")
        logger.info(f"Saved comparison report to {txt_path}")
        
        # Save JSON
        json_path = output_path / f'{report_name}.json'
        comparison_df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved comparison JSON to {json_path}")
        
        # Generate summary
        self._generate_summary(comparison_df, output_path, report_name)
    
    def _generate_summary(
        self,
        comparison_df: pd.DataFrame,
        output_path: Path,
        report_name: str,
    ):
        """Generate summary statistics."""
        summary_path = output_path / f'{report_name}_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 40 + "\n")
            
            # Average BD-Rate by method
            avg_by_method = comparison_df[comparison_df['Dataset'] != 'Average'].groupby('Method').agg({
                'BD-Rate (%)': ['mean', 'std', 'min', 'max'],
                'BD-PSNR (dB)': ['mean', 'std', 'min', 'max'],
            })
            
            f.write("\nAverage BD-Rate by Method:\n")
            f.write(avg_by_method.to_string())
            f.write("\n\n")
            
            # Best method per dataset
            f.write("Best Method per Dataset (lowest BD-Rate):\n")
            f.write("-" * 40 + "\n")
            
            for dataset in comparison_df['Dataset'].unique():
                if dataset == 'Average':
                    continue
                
                dataset_df = comparison_df[comparison_df['Dataset'] == dataset]
                best_idx = dataset_df['BD-Rate (%)'].idxmin()
                best_method = dataset_df.loc[best_idx, 'Method']
                best_bd_rate = dataset_df.loc[best_idx, 'BD-Rate (%)']
                
                f.write(f"{dataset}: {best_method} (BD-Rate: {best_bd_rate:.2f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Saved summary to {summary_path}")


class SOTAComparator:
    """
    Compare with state-of-the-art methods.
    
    Includes: VTM, VVC, BPG, JPEG2000, and learned methods.
    """
    
    # Pre-defined SOTA results (example values - replace with actual data)
    SOTA_RESULTS = {
        'Kodak': {
            'VTM': {
                'rates': [0.15, 0.25, 0.40, 0.65],
                'psnrs': [31.2, 33.8, 36.2, 38.5],
            },
            'BPG': {
                'rates': [0.20, 0.35, 0.55, 0.80],
                'psnrs': [30.5, 33.2, 35.8, 38.0],
            },
            'JPEG2000': {
                'rates': [0.25, 0.45, 0.70, 1.00],
                'psnrs': [29.8, 32.5, 35.0, 37.2],
            },
        },
        'CLIC': {
            'VTM': {
                'rates': [0.18, 0.30, 0.50, 0.75],
                'psnrs': [30.5, 33.0, 35.5, 37.8],
            },
            'BPG': {
                'rates': [0.22, 0.40, 0.60, 0.90],
                'psnrs': [29.8, 32.5, 35.0, 37.2],
            },
        },
    }
    
    def __init__(self):
        """Initialize SOTA comparator."""
        self.comparator = ModelComparator()
    
    def compare_with_sota(
        self,
        method_curves: Dict[str, RDCurve],
        dataset: str = 'Kodak',
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare with SOTA methods.
        
        Args:
            method_curves: Dict of {method_name: RDCurve}
            dataset: Dataset name
            output_dir: Output directory for plots
        
        Returns:
            Comparison DataFrame
        """
        if dataset not in self.SOTA_RESULTS:
            logger.warning(f"No SOTA results for dataset '{dataset}'")
            return pd.DataFrame()
        
        # Load SOTA curves
        sota_curves = {}
        for sota_name, data in self.SOTA_RESULTS[dataset].items():
            sota_curves[sota_name] = RDCurve(
                name=sota_name,
                rates=data['rates'],
                psnrs=data['psnrs'],
            )
        
        # Combine with test curves
        all_curves = {**sota_curves, **method_curves}
        
        # Use BPG as reference (common baseline)
        reference_curve = sota_curves.get('BPG', list(sota_curves.values())[0])
        
        # Compare
        comparison_df = self.comparator.compare(all_curves, reference_curve, dataset)
        
        # Generate plot
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            plotter = RDCurvePlotter()
            
            # Plot all curves
            plotter.plot(
                curves=list(all_curves.values()),
                title=f'Rate-Distortion Comparison on {dataset}',
                save_path=output_path / f'rd_comparison_{dataset}.png',
            )
            
            # Plot with comparison table
            test_curves_list = list(method_curves.values())
            if test_curves_list:
                plotter.plot_comparison_table(
                    curves=test_curves_list,
                    reference_curve=reference_curve,
                    save_path=output_path / f'rd_table_{dataset}.png',
                )
        
        return comparison_df
    
    def generate_sota_report(
        self,
        method_curves_by_dataset: Dict[str, Dict[str, RDCurve]],
        output_dir: str,
    ):
        """
        Generate comprehensive SOTA comparison report.
        
        Args:
            method_curves_by_dataset: Dict of {dataset: {method: curve}}
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_comparisons = []
        
        for dataset, method_curves in method_curves_by_dataset.items():
            comparison_df = self.compare_with_sota(
                method_curves,
                dataset=dataset,
                output_dir=output_dir,
            )
            all_comparisons.append(comparison_df)
        
        # Combine all datasets
        combined_df = pd.concat(all_comparisons, ignore_index=True)
        
        # Generate report
        self.comparator.generate_report(
            combined_df,
            output_dir,
            report_name='sota_comparison',
        )
        
        logger.info(f"Generated SOTA comparison report in {output_dir}")


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create example curves
    test_curve1 = RDCurve(
        name='HPCM-Phase4',
        rates=[0.2, 0.4, 0.6, 0.8],
        psnrs=[32.5, 35.2, 37.1, 38.5],
    )
    
    test_curve2 = RDCurve(
        name='HPCM-Phase3',
        rates=[0.25, 0.45, 0.65, 0.85],
        psnrs=[32.0, 34.8, 36.8, 38.2],
    )
    
    reference_curve = RDCurve(
        name='BPG',
        rates=[0.20, 0.35, 0.55, 0.80],
        psnrs=[30.5, 33.2, 35.8, 38.0],
    )
    
    # Compare
    comparator = ModelComparator()
    
    test_curves = {
        'HPCM-Phase4': test_curve1,
        'HPCM-Phase3': test_curve2,
    }
    
    comparison_df = comparator.compare(test_curves, reference_curve, 'Kodak')
    
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))
    
    # Generate report
    comparator.generate_report(comparison_df, './outputs/comparison', 'test_comparison')
    
    print("\nReport generated in ./outputs/comparison/")
