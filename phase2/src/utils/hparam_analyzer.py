"""
Hyperparameter Analysis Tool for Phase 2
Analyzes training results and recommends optimal parameters.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class HyperparameterAnalyzer:
    """Analyze hyperparameter search results and recommend optimal settings."""
    
    def __init__(self, results_dir: str):
        """
        Initialize analyzer with results directory.
        
        Args:
            results_dir: Directory containing hyperparameter search results
        """
        self.results_dir = Path(results_dir)
        self.results = []
        self.best_config = None
        
    def load_results(self, pattern: str = "lambda_*_gamma_*_wlr_*") -> None:
        """Load all experiment results from subdirectories."""
        print(f"Loading results from {self.results_dir}...")
        
        for exp_dir in self.results_dir.glob(pattern):
            if not exp_dir.is_dir():
                continue
                
            # Parse parameters from directory name
            parts = exp_dir.name.split('_')
            try:
                params = {
                    'lambda': float(parts[1]),
                    'gamma': float(parts[3]),
                    'w_lr': float(parts[5])
                }
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse {exp_dir.name}: {e}")
                continue
            
            # Load metrics from logs or checkpoint
            metrics = self._load_metrics(exp_dir)
            if metrics:
                self.results.append({
                    'exp_dir': str(exp_dir),
                    'params': params,
                    'metrics': metrics
                })
        
        print(f"Loaded {len(self.results)} experiment results")
        
    def _load_metrics(self, exp_dir: Path) -> Optional[Dict]:
        """Load metrics from experiment directory."""
        # Try to load from checkpoint
        checkpoint_path = exp_dir / 'epoch_best.pth.tar'
        if checkpoint_path.exists():
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return {
                'best_loss': checkpoint.get('best_loss', float('inf')),
                'final_epoch': checkpoint.get('epoch', 0)
            }
        
        # Try to load from metrics.json if exists
        metrics_path = exp_dir / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def analyze(self) -> pd.DataFrame:
        """Analyze results and create summary dataframe."""
        if not self.results:
            raise ValueError("No results loaded. Call load_results() first.")
        
        data = []
        for result in self.results:
            row = {
                'lambda': result['params']['lambda'],
                'gamma': result['params']['gamma'],
                'w_lr': result['params']['w_lr'],
                'best_loss': result['metrics'].get('best_loss', float('inf')),
                'final_epoch': result['metrics'].get('final_epoch', 0),
                'exp_dir': result['exp_dir']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('best_loss')
        
        # Find best configuration
        best_idx = df['best_loss'].idxmin()
        self.best_config = {
            'lambda': df.loc[best_idx, 'lambda'],
            'gamma': df.loc[best_idx, 'gamma'],
            'w_lr': df.loc[best_idx, 'w_lr'],
            'best_loss': df.loc[best_idx, 'best_loss']
        }
        
        return df
    
    def recommend_parameters(self) -> Dict:
        """Recommend optimal hyperparameters based on analysis."""
        if self.best_config is None:
            self.analyze()
        
        print("\n" + "="*60)
        print("RECOMMENDED HYPERPARAMETERS")
        print("="*60)
        print(f"Lambda:  {self.best_config['lambda']:.4f}")
        print(f"Gamma:   {self.best_config['gamma']:.4f}")
        print(f"W_LR:    {self.best_config['w_lr']:.4f}")
        print(f"Best Loss: {self.best_config['best_loss']:.4f}")
        print("="*60)
        
        return self.best_config
    
    def plot_heatmap(self, lambda_val: float, save_path: Optional[str] = None):
        """
        Plot heatmap of loss vs gamma and w_lr for a specific lambda.
        
        Args:
            lambda_val: Lambda value to plot
            save_path: Path to save figure (optional)
        """
        df = self.analyze()
        df_lambda = df[df['lambda'] == lambda_val]
        
        if len(df_lambda) == 0:
            print(f"No results found for lambda={lambda_val}")
            return
        
        # Pivot for heatmap
        pivot = df_lambda.pivot(index='gamma', columns='w_lr', values='best_loss')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis_r', 
                    cbar_kws={'label': 'Best Loss'})
        plt.title(f'Hyperparameter Search Results (λ={lambda_val})')
        plt.xlabel('Task Weight Learning Rate (w_lr)')
        plt.ylabel('Regularization Coefficient (gamma)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Plot comparison of all configurations.
        
        Args:
            save_path: Path to save figure (optional)
        """
        df = self.analyze()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Loss vs Gamma (grouped by w_lr)
        for w_lr in sorted(df['w_lr'].unique()):
            df_wlr = df[df['w_lr'] == w_lr]
            axes[0].plot(df_wlr['gamma'], df_wlr['best_loss'], 
                        marker='o', label=f'w_lr={w_lr}')
        axes[0].set_xlabel('Gamma')
        axes[0].set_ylabel('Best Loss')
        axes[0].set_title('Loss vs Gamma')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Loss vs W_LR (grouped by gamma)
        for gamma in sorted(df['gamma'].unique()):
            df_gamma = df[df['gamma'] == gamma]
            axes[1].plot(df_gamma['w_lr'], df_gamma['best_loss'], 
                        marker='s', label=f'gamma={gamma}')
        axes[1].set_xlabel('W_LR')
        axes[1].set_ylabel('Best Loss')
        axes[1].set_title('Loss vs Task Weight LR')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_report(self, output_path: str):
        """Save analysis report to file."""
        df = self.analyze()
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HYPERPARAMETER SEARCH ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("RECOMMENDED PARAMETERS:\n")
            f.write("-"*70 + "\n")
            f.write(f"Lambda:    {self.best_config['lambda']:.4f}\n")
            f.write(f"Gamma:     {self.best_config['gamma']:.4f}\n")
            f.write(f"W_LR:      {self.best_config['w_lr']:.4f}\n")
            f.write(f"Best Loss: {self.best_config['best_loss']:.4f}\n")
            f.write("\n")
            
            f.write("TOP 5 CONFIGURATIONS:\n")
            f.write("-"*70 + "\n")
            top5 = df.head(5)
            f.write(top5.to_string(index=False))
            f.write("\n\n")
            
            f.write("ALL RESULTS:\n")
            f.write("-"*70 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n")
        
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze hyperparameter search results")
    parser.add_argument("results_dir", type=str, help="Directory with search results")
    parser.add_argument("--output", type=str, default="hparam_analysis.txt",
                       help="Output report path")
    parser.add_argument("--plot-heatmap", action="store_true",
                       help="Generate heatmap")
    parser.add_argument("--lambda", type=float, default=0.013,
                       help="Lambda value for heatmap")
    args = parser.parse_args()
    
    analyzer = HyperparameterAnalyzer(args.results_dir)
    analyzer.load_results()
    analyzer.recommend_parameters()
    analyzer.save_report(args.output)
    
    if args.plot_heatmap:
        analyzer.plot_heatmap(args.lambda, 
                             save_path=f"heatmap_lambda_{args.lambda}.png")
    
    analyzer.plot_comparison(save_path="comparison.png")
