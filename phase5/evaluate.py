#!/usr/bin/env python3
"""
Evaluate Model on Multiple Datasets

Comprehensive evaluation script for learned image compression models.
Supports: Kodak, CLIC, Tecnick, and custom datasets.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import MetricsCalculator, RDCurve, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageDataset(torch.utils.data.Dataset):
    """Simple image dataset."""
    
    def __init__(self, image_dir: str, extensions: List[str] = ['.png', '.jpg', '.jpeg']):
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        self.image_paths = sorted(self.image_paths)
        
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        return img_tensor, str(img_path.name)


def evaluate_model(
    model: nn.Module,
    dataset_path: str,
    dataset_name: str,
    device: str = 'cuda',
    quality_levels: List[int] = [1, 2, 3, 4, 5],
) -> Dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Compression model
        dataset_path: Path to dataset
        dataset_name: Dataset name
        device: Computation device
        quality_levels: Quality levels to evaluate
    
    Returns:
        Evaluation results dict
    """
    model = model.to(device)
    model.eval()
    
    # Load dataset
    dataset = ImageDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Metrics calculator
    metrics_calc = MetricsCalculator(device=device)
    
    # Results storage
    results = {
        'dataset': dataset_name,
        'num_images': len(dataset),
        'quality_levels': {},
    }
    
    for quality in quality_levels:
        logger.info(f"Evaluating quality level {quality}...")
        
        # Set model quality
        if hasattr(model, 'set_quality'):
            model.set_quality(quality)
        
        # Evaluate all images
        all_psnrs = []
        all_ms_ssims = []
        all_bpps = []
        
        for img, img_name in tqdm(dataloader, desc=f'Quality {quality}'):
            img = img.to(device)
            
            with torch.no_grad():
                # Compress and decompress
                output = model(img)
                
                # Extract reconstructed image and bpp
                if isinstance(output, dict):
                    reconstructed = output['x_hat']
                    bpp = output['bpp'].mean().item()
                else:
                    reconstructed, bpp_dict = output
                    bpp = bpp_dict['bpp'].mean().item()
                
                # Calculate metrics
                metrics = metrics_calc.compute_all(img, reconstructed, bpp)
                
                all_psnrs.append(metrics['psnr'])
                if metrics['ms_ssim'] is not None:
                    all_ms_ssims.append(metrics['ms_ssim'])
                all_bpps.append(bpp)
        
        # Aggregate results
        results['quality_levels'][quality] = {
            'psnr_mean': float(np.mean(all_psnrs)),
            'psnr_std': float(np.std(all_psnrs)),
            'bpp_mean': float(np.mean(all_bpps)),
            'bpp_std': float(np.std(all_bpps)),
        }
        
        if all_ms_ssims:
            results['quality_levels'][quality]['ms_ssim_mean'] = float(np.mean(all_ms_ssims))
            results['quality_levels'][quality]['ms_ssim_std'] = float(np.std(all_ms_ssims))
        
        logger.info(f"Quality {quality}: "
                   f"PSNR={results['quality_levels'][quality]['psnr_mean']:.2f}±{results['quality_levels'][quality]['psnr_std']:.2f} dB, "
                   f"BPP={results['quality_levels'][quality]['bpp_mean']:.4f}±{results['quality_levels'][quality]['bpp_std']:.4f}")
    
    return results


def create_rd_curve(results: Dict, method_name: str) -> RDCurve:
    """Create RD curve from evaluation results."""
    rates = []
    psnrs = []
    ms_ssims = []
    
    for quality, metrics in results['quality_levels'].items():
        rates.append(metrics['bpp_mean'])
        psnrs.append(metrics['psnr_mean'])
        if 'ms_ssim_mean' in metrics:
            ms_ssims.append(metrics['ms_ssim_mean'])
    
    return RDCurve(
        name=method_name,
        rates=rates,
        psnrs=psnrs,
        ms_ssims=ms_ssims if ms_ssims else None,
        metadata=results,
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate compression model on multiple datasets')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='hpcm_base',
                       help='Model architecture')
    parser.add_argument('--method_name', type=str, default='HPCM-Phase4',
                       help='Method name for results')
    
    # Dataset arguments
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['kodak', 'clic', 'tecnick'],
                       help='Datasets to evaluate')
    parser.add_argument('--data_root', type=str, default='./datasets',
                       help='Root directory for datasets')
    
    # Evaluation arguments
    parser.add_argument('--quality_levels', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5],
                       help='Quality levels to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Computation device')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    
    # Import model (replace with actual import)
    from src.models.HPCM_Base import HPCM_Base
    model = HPCM_Base()
    
    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    
    logger.info("Model loaded successfully")
    
    # Evaluate on each dataset
    all_results = {}
    all_curves = {}
    
    dataset_paths = {
        'kodak': Path(args.data_root) / 'kodak',
        'clic': Path(args.data_root) / 'clic',
        'tecnick': Path(args.data_root) / 'tecnick',
    }
    
    for dataset in args.datasets:
        if dataset not in dataset_paths:
            logger.warning(f"Unknown dataset '{dataset}', skipping")
            continue
        
        dataset_path = dataset_paths[dataset]
        if not dataset_path.exists():
            logger.warning(f"Dataset path {dataset_path} does not exist, skipping")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating on {dataset.upper()}")
        logger.info(f"{'='*80}")
        
        # Evaluate
        results = evaluate_model(
            model=model,
            dataset_path=str(dataset_path),
            dataset_name=dataset,
            device=args.device,
            quality_levels=args.quality_levels,
        )
        
        all_results[dataset] = results
        
        # Create RD curve
        rd_curve = create_rd_curve(results, args.method_name)
        all_curves[dataset] = rd_curve
        
        # Save individual results
        results_path = output_dir / f'results_{dataset}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_path}")
        
        # Save RD curve
        curve_path = output_dir / f'rd_curve_{dataset}.json'
        rd_curve.save(str(curve_path))
    
    # Save combined results
    combined_path = output_dir / 'results_all.json'
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved combined results to {combined_path}")
    
    # Generate summary
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    
    for dataset, results in all_results.items():
        logger.info(f"\n{dataset.upper()}:")
        for quality, metrics in results['quality_levels'].items():
            logger.info(f"  Quality {quality}: "
                       f"PSNR={metrics['psnr_mean']:.2f} dB, "
                       f"BPP={metrics['bpp_mean']:.4f}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation complete! Results saved to {output_dir}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
