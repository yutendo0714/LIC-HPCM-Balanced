"""
HPCM Training Script with Hierarchical Balanced Optimization (Phase 3)
- 5-task decomposition: s1_distortion, s1_bpp, s2_distortion, s2_bpp, s3_bpp
- Scale-specific gamma values
- Hierarchical task weighting
"""
import argparse
import math
import random
import sys
import os
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import Phase 3 components
from src.optimizers.hierarchical_balanced import HierarchicalBalanced
from src.utils.scale_gamma_manager import ScaleGammaManager
from src.utils.hierarchical_loss import HierarchicalLoss

# Import HPCM models
from src.models import HPCM_Base, HPCM_Large


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform):
        self.data_dir = data_path
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.dataset_list[idx])
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(model, args, device):
    """Configure optimizer (standard Adam or Hierarchical Balanced)."""
    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    if args.use_hierarchical:
        # Phase 3: Hierarchical Balanced optimizer
        optimizer = HierarchicalBalanced(
            (params_dict[n] for n in sorted(parameters)),
            lr=args.learning_rate,
            gamma_s1=args.gamma_s1,
            gamma_s2=args.gamma_s2,
            gamma_s3=args.gamma_s3,
            w_lr=args.w_lr,
            max_norm=args.clip_max_norm,
            scale_weights=args.scale_weights,
            device=device,
        )
    else:
        # Standard Adam
        optimizer = optim.Adam(
            (params_dict[n] for n in sorted(parameters)),
            lr=args.learning_rate,
        )

    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )

    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args, device):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    s1_dist_meter = AverageMeter()
    s1_bpp_meter = AverageMeter()
    s2_dist_meter = AverageMeter()
    s2_bpp_meter = AverageMeter()
    s3_bpp_meter = AverageMeter()
    psnr_meter = AverageMeter()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # Forward pass
        out_net = model(d)
        
        if args.use_hierarchical:
            # Phase 3: Hierarchical loss (5 tasks)
            task_losses, loss_dict = criterion(out_net, d)
            
            # Balanced optimization
            if epoch == 0 and i == 0:
                # Initialize min losses
                optimizer.set_min_losses(task_losses.detach())
            
            # Compute weighted loss
            loss = optimizer.get_weighted_loss(task_losses)
            
        else:
            # Standard R-D loss
            out_criterion = criterion(out_net, d)
            loss = out_criterion['loss']
            loss_dict = out_criterion

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)

        optimizer.step()

        # Auxiliary loss
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        # Update task weights (for Hierarchical Balanced)
        if args.use_hierarchical:
            optimizer.update_task_weights(task_losses.detach())

        # Logging
        batch_size = d.size(0)
        loss_meter.update(loss.item(), batch_size)
        
        if args.use_hierarchical:
            s1_dist_meter.update(loss_dict['s1_distortion'].item(), batch_size)
            s1_bpp_meter.update(loss_dict['s1_bpp'].item(), batch_size)
            s2_dist_meter.update(loss_dict['s2_distortion'].item(), batch_size)
            s2_bpp_meter.update(loss_dict['s2_bpp'].item(), batch_size)
            s3_bpp_meter.update(loss_dict['s3_bpp'].item(), batch_size)
        
        # Compute PSNR
        mse = nn.functional.mse_loss(out_net['x_hat'], d)
        psnr = 10 * (torch.log(1.0 / mse) / np.log(10))
        psnr_meter.update(psnr.item(), batch_size)

        if i % 100 == 0:
            if args.use_hierarchical:
                print(f'[{epoch}/{args.epochs}][{i}/{len(train_dataloader)}] '
                      f'Loss: {loss_meter.avg:.4f} | PSNR: {psnr_meter.avg:.2f} | '
                      f's1_d: {s1_dist_meter.avg:.4f} | s2_d: {s2_dist_meter.avg:.4f}')
            else:
                print(f'[{epoch}/{args.epochs}][{i}/{len(train_dataloader)}] '
                      f'Loss: {loss_meter.avg:.4f} | PSNR: {psnr_meter.avg:.2f}')

    # Epoch statistics
    stats = {
        'train/loss': loss_meter.avg,
        'train/psnr': psnr_meter.avg,
    }
    
    if args.use_hierarchical:
        stats.update({
            'train/s1_distortion': s1_dist_meter.avg,
            'train/s1_bpp': s1_bpp_meter.avg,
            'train/s2_distortion': s2_dist_meter.avg,
            'train/s2_bpp': s2_bpp_meter.avg,
            'train/s3_bpp': s3_bpp_meter.avg,
        })
        
        # Task weights
        task_weights = optimizer.get_task_weights()
        stats.update({f'weights/{k}': v for k, v in task_weights.items()})
        
        # Scale contributions
        scale_contributions = optimizer.get_scale_contributions()
        stats.update({f'scale_contrib/{k}': v for k, v in scale_contributions.items()})

    return stats


def test_epoch(model, criterion, test_dataloader, epoch, args, device):
    """Test for one epoch."""
    model.eval()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            
            if args.use_hierarchical:
                task_losses, loss_dict = criterion(out_net, d)
                loss = task_losses.sum()  # Simple sum for evaluation
            else:
                out_criterion = criterion(out_net, d)
                loss = out_criterion['loss']
                loss_dict = out_criterion

            batch_size = d.size(0)
            loss_meter.update(loss.item(), batch_size)
            
            # Compute metrics
            mse = nn.functional.mse_loss(out_net['x_hat'], d)
            psnr = 10 * (torch.log(1.0 / mse) / np.log(10))
            psnr_meter.update(psnr.item(), batch_size)
            
            # BPP
            N, _, H, W = d.size()
            num_pixels = N * H * W
            bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )
            bpp_meter.update(bpp.item(), batch_size)

    stats = {
        'test/loss': loss_meter.avg,
        'test/psnr': psnr_meter.avg,
        'test/bpp': bpp_meter.avg,
    }

    return stats


def save_checkpoint(state, filename):
    """Save checkpoint to disk."""
    torch.save(state, filename)
    print(f'Checkpoint saved to {filename}')


def load_checkpoint(filepath, model, optimizer=None, aux_optimizer=None):
    """Load checkpoint from disk."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if aux_optimizer is not None and 'aux_optimizer' in checkpoint:
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f'Loaded checkpoint from {filepath} (epoch {epoch})')
    return epoch


def parse_args():
    parser = argparse.ArgumentParser(description='HPCM Training with Hierarchical Balanced (Phase 3)')
    
    # Model
    parser.add_argument('--model_name', type=str, default='HPCM_Base', choices=['HPCM_Base', 'HPCM_Large'])
    parser.add_argument('--lambda', dest='lmbda', type=float, default=0.013, help='Lambda for R-D tradeoff')
    
    # Data
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--aux-learning-rate', type=float, default=1e-3)
    parser.add_argument('--clip_max_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Phase 3: Hierarchical Balanced
    parser.add_argument('--use_hierarchical', action='store_true', help='Use hierarchical balanced optimizer')
    parser.add_argument('--gamma_s1', type=float, default=0.008, help='Gamma for scale 1')
    parser.add_argument('--gamma_s2', type=float, default=0.006, help='Gamma for scale 2')
    parser.add_argument('--gamma_s3', type=float, default=0.004, help='Gamma for scale 3')
    parser.add_argument('--w_lr', type=float, default=0.025, help='Learning rate for task weights')
    parser.add_argument('--scale_weights', type=float, nargs=3, default=[0.3, 0.4, 0.3],
                        help='Importance weights for scales [s1, s2, s3]')
    
    # Adaptive gamma scheduling
    parser.add_argument('--adaptive_gamma', action='store_true', help='Enable adaptive gamma scheduling')
    parser.add_argument('--gamma_strategy', type=str, default='hierarchical',
                        choices=['fixed', 'linear', 'cosine', 'adaptive', 'hierarchical'])
    
    # Checkpointing
    parser.add_argument('--save_path', type=str, default='./outputs/phase3')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--save_freq', type=int, default=50, help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs/phase3')
    parser.add_argument('--wandb_project', type=str, default='HPCM-Hierarchical')
    parser.add_argument('--wandb_name', type=str, default=None)
    
    # Device
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize WandB
    wandb_name = args.wandb_name or f"phase3_{args.model_name}_lambda{args.lmbda}"
    wandb.init(project=args.wandb_project, name=wandb_name, config=vars(args))
    
    # Load model
    if args.model_name == 'HPCM_Base':
        model = HPCM_Base.HPCM(N=192, M=320)
    elif args.model_name == 'HPCM_Large':
        model = HPCM_Large.HPCM(N=192, M=320)
    else:
        raise ValueError(f'Unknown model: {args.model_name}')
    
    model = model.to(device)
    print(f'Loaded model: {args.model_name}')
    
    # Create criterion
    if args.use_hierarchical:
        criterion = HierarchicalLoss(lmbda=args.lmbda, return_details=True)
    else:
        # Standard R-D loss (placeholder for compatibility)
        from phase2.train import RateDistortionLoss
        criterion = RateDistortionLoss(lmbda=args.lmbda)
    
    # Create optimizers
    optimizer, aux_optimizer = configure_optimizers(model, args, device)
    
    # Setup adaptive gamma manager (if enabled)
    gamma_manager = None
    if args.use_hierarchical and args.adaptive_gamma:
        initial_gammas = {
            's1': args.gamma_s1,
            's2': args.gamma_s2,
            's3': args.gamma_s3,
        }
        gamma_manager = ScaleGammaManager(
            initial_gammas=initial_gammas,
            strategy=args.gamma_strategy,
            total_epochs=args.epochs,
            device=device,
        )
    
    # Load checkpoint (if resuming)
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(args.checkpoint, model, optimizer, aux_optimizer)
        start_epoch += 1
    
    # Create datasets
    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(args.patch_size),
        transforms.ToTensor(),
    ])
    
    train_dataset = Dataset(args.train_dataset, train_transforms)
    test_dataset = Dataset(args.test_dataset, test_transforms)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
    )
    
    print(f'Train dataset: {len(train_dataset)} images')
    print(f'Test dataset: {len(test_dataset)} images')
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n=== Epoch {epoch}/{args.epochs} ===')
        
        # Update gamma (if adaptive)
        if gamma_manager is not None:
            gammas = gamma_manager.step(epoch)
            optimizer.set_scale_gammas(gammas['s1'], gammas['s2'], gammas['s3'])
            print(f'Adaptive gammas: s1={gammas["s1"]:.4f}, s2={gammas["s2"]:.4f}, s3={gammas["s3"]:.4f}')
        
        # Train
        train_stats = train_one_epoch(
            model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args, device
        )
        
        # Test
        test_stats = test_epoch(model, criterion, test_dataloader, epoch, args, device)
        
        # Log to WandB
        wandb.log({**train_stats, **test_stats, 'epoch': epoch})
        
        # Log gammas
        if gamma_manager is not None:
            current_gammas = gamma_manager.get_current_gammas()
            wandb.log({f'gamma/{k}': v for k, v in current_gammas.items()})
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_path, f'checkpoint_epoch{epoch}.pth')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'aux_optimizer': aux_optimizer.state_dict(),
                'loss': test_stats['test/loss'],
            }, checkpoint_path)
        
        # Save best model
        if test_stats['test/loss'] < best_loss:
            best_loss = test_stats['test/loss']
            best_path = os.path.join(args.save_path, 'best_model.pth')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'aux_optimizer': aux_optimizer.state_dict(),
                'loss': best_loss,
            }, best_path)
            print(f'✓ New best model saved (loss={best_loss:.4f})')
    
    print('\n=== Training Complete ===')
    wandb.finish()


if __name__ == '__main__':
    main()
