"""
HPCM Training Script with Context-Aware Fine-tuning (Phase 4)
- Layer-wise learning rates
- Freeze entropy model
- Scale-specific early stopping
- Progressive unfreezing
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

# Import Phase 4 components
from src.utils.layer_lr_manager import LayerLRManager, DiscriminativeLRScheduler
from src.utils.scale_early_stopping import ScaleEarlyStopping, AdaptivePatienceEarlyStopping

# Import Phase 3 components (for hierarchical balanced)
phase3_path = os.path.join(os.path.dirname(__file__), '..', 'phase3')
sys.path.insert(0, phase3_path)
from src.optimizers.hierarchical_balanced import HierarchicalBalanced
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


def configure_optimizers_phase4(model, args, device):
    """
    Configure optimizer with layer-wise learning rates (Phase 4).
    """
    # Create layer LR manager
    layer_lr_multipliers, freeze_patterns = LayerLRManager.create_hpcm_fine_tuning_config(
        base_lr=args.learning_rate,
        freeze_entropy=args.freeze_entropy,
        context_lr_ratio=args.context_lr_ratio,
    )
    
    lr_manager = LayerLRManager(
        model=model,
        base_lr=args.learning_rate,
        layer_lr_multipliers=layer_lr_multipliers,
        freeze_patterns=freeze_patterns,
    )
    
    # Apply freeze
    if args.freeze_entropy:
        lr_manager.apply_freeze()
    
    # Print status
    lr_manager.print_freeze_status()
    
    # Get parameter groups
    param_groups = lr_manager.get_parameter_groups()
    
    # Create optimizer
    if args.use_hierarchical:
        # Use hierarchical balanced with layer-wise LRs
        # Note: HierarchicalBalanced expects single params iterable
        # We'll use standard Adam with param groups, then apply balanced logic separately
        optimizer = optim.Adam(param_groups)
        print('Using Adam with layer-wise LRs (hierarchical balanced in loss)')
    else:
        optimizer = optim.Adam(param_groups)
        print('Using Adam with layer-wise LRs')
    
    # Auxiliary optimizer (for quantiles)
    aux_parameters = {
        n for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    
    if aux_parameters:
        aux_optimizer = optim.Adam(
            (model.get_parameter(n) for n in sorted(aux_parameters)),
            lr=args.aux_learning_rate,
        )
    else:
        aux_optimizer = None
    
    return optimizer, aux_optimizer, lr_manager


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args, device, lr_manager=None):
    """Train for one epoch with Phase 4 features."""
    model.train()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    
    # Track scale losses (if hierarchical)
    scale_meters = {
        's1': AverageMeter(),
        's2': AverageMeter(),
        's3': AverageMeter(),
    }

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        if aux_optimizer:
            aux_optimizer.zero_grad()

        # Forward pass
        out_net = model(d)
        
        if args.use_hierarchical:
            # Phase 3: Hierarchical loss
            task_losses, loss_dict = criterion(out_net, d)
            
            # Simple weighted sum (no optimizer-based balancing in Phase 4)
            loss = task_losses.sum()
            
            # Track scale losses
            scale_meters['s1'].update((loss_dict['s1_distortion'] + loss_dict['s1_bpp']).item())
            scale_meters['s2'].update((loss_dict['s2_distortion'] + loss_dict['s2_bpp']).item())
            scale_meters['s3'].update(loss_dict['s3_bpp'].item())
        else:
            # Standard R-D loss
            N, _, H, W = d.size()
            num_pixels = N * H * W
            
            out_criterion = {}
            out_criterion["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )
            out_criterion["mse_loss"] = nn.functional.mse_loss(out_net["x_hat"], d)
            out_criterion["loss"] = args.lmbda * 255**2 * out_criterion["mse_loss"] + out_criterion["bpp_loss"]
            
            loss = out_criterion['loss']
            loss_dict = out_criterion

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)

        optimizer.step()

        # Auxiliary loss
        if aux_optimizer:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        # Logging
        batch_size = d.size(0)
        loss_meter.update(loss.item(), batch_size)
        
        # Compute PSNR
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

        if i % 100 == 0:
            print(f'[{epoch}/{args.epochs}][{i}/{len(train_dataloader)}] '
                  f'Loss: {loss_meter.avg:.4f} | PSNR: {psnr_meter.avg:.2f} dB | BPP: {bpp_meter.avg:.4f}')

    # Epoch statistics
    stats = {
        'train/loss': loss_meter.avg,
        'train/psnr': psnr_meter.avg,
        'train/bpp': bpp_meter.avg,
    }
    
    if args.use_hierarchical:
        stats.update({
            'train/scale_s1': scale_meters['s1'].avg,
            'train/scale_s2': scale_meters['s2'].avg,
            'train/scale_s3': scale_meters['s3'].avg,
        })
    
    # Log layer-wise learning rates
    if lr_manager:
        for i, group in enumerate(optimizer.param_groups):
            name = group.get('name', f'group_{i}')
            stats[f'lr/{name}'] = group['lr']

    return stats


def test_epoch(model, criterion, test_dataloader, epoch, args, device):
    """Test for one epoch."""
    model.eval()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    
    # Scale losses
    scale_meters = {
        's1': AverageMeter(),
        's2': AverageMeter(),
        's3': AverageMeter(),
    }

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            
            if args.use_hierarchical:
                task_losses, loss_dict = criterion(out_net, d)
                loss = task_losses.sum()
                
                # Track scale losses
                scale_meters['s1'].update((loss_dict['s1_distortion'] + loss_dict['s1_bpp']).item())
                scale_meters['s2'].update((loss_dict['s2_distortion'] + loss_dict['s2_bpp']).item())
                scale_meters['s3'].update(loss_dict['s3_bpp'].item())
            else:
                N, _, H, W = d.size()
                num_pixels = N * H * W
                
                mse = nn.functional.mse_loss(out_net['x_hat'], d)
                bpp = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in out_net["likelihoods"].values()
                )
                loss = args.lmbda * 255**2 * mse + bpp

            batch_size = d.size(0)
            loss_meter.update(loss.item(), batch_size)
            
            # Compute metrics
            mse = nn.functional.mse_loss(out_net['x_hat'], d)
            psnr = 10 * (torch.log(1.0 / mse) / np.log(10))
            psnr_meter.update(psnr.item(), batch_size)
            
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
    
    if args.use_hierarchical:
        stats.update({
            'test/scale_s1': scale_meters['s1'].avg,
            'test/scale_s2': scale_meters['s2'].avg,
            'test/scale_s3': scale_meters['s3'].avg,
        })

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
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('Warning: Could not load optimizer state (layer groups may differ)')
    
    if aux_optimizer is not None and 'aux_optimizer' in checkpoint:
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f'Loaded checkpoint from {filepath} (epoch {epoch})')
    return epoch


def parse_args():
    parser = argparse.ArgumentParser(description='HPCM Training with Context-Aware Fine-tuning (Phase 4)')
    
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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--aux-learning-rate', type=float, default=1e-4)
    parser.add_argument('--clip_max_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Phase 3: Hierarchical
    parser.add_argument('--use_hierarchical', action='store_true', help='Use hierarchical loss')
    
    # Phase 4: Context-Aware Fine-tuning
    parser.add_argument('--freeze_entropy', action='store_true', help='Freeze entropy model')
    parser.add_argument('--context_lr_ratio', type=float, default=0.1, help='LR ratio for context layers')
    parser.add_argument('--progressive_unfreeze', action='store_true', help='Enable progressive unfreezing')
    parser.add_argument('--scale_early_stopping', action='store_true', help='Enable scale-specific early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=50, help='Patience for early stopping')
    
    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume/finetune')
    parser.add_argument('--save_path', type=str, default='./outputs/phase4')
    parser.add_argument('--save_freq', type=int, default=50, help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs/phase4')
    parser.add_argument('--wandb_project', type=str, default='HPCM-Phase4')
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
    wandb_name = args.wandb_name or f"phase4_{args.model_name}_lambda{args.lmbda}"
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
        criterion = None  # Loss computed inline
    
    # Create optimizers with Phase 4 features
    optimizer, aux_optimizer, lr_manager = configure_optimizers_phase4(model, args, device)
    
    # Setup early stopping
    early_stopping = None
    if args.scale_early_stopping:
        early_stopping = AdaptivePatienceEarlyStopping(
            scales=['s1', 's2', 's3'],
            initial_patience=args.early_stopping_patience,
            max_patience=200,
            mode='min',
            save_dir=args.save_path,
        )
    
    # Load checkpoint (if provided)
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
        
        # Progressive unfreezing (if enabled)
        if args.progressive_unfreeze:
            unfreeze_schedule = {
                100: ['h_a', 'h_s'],
                200: ['context'],
            }
            lr_manager.progressive_unfreeze(epoch, unfreeze_schedule)
        
        # Train
        train_stats = train_one_epoch(
            model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args, device, lr_manager
        )
        
        # Test
        test_stats = test_epoch(model, criterion, test_dataloader, epoch, args, device)
        
        # Log to WandB
        wandb.log({**train_stats, **test_stats, 'epoch': epoch})
        
        # Early stopping check (if enabled)
        if early_stopping and args.use_hierarchical:
            scale_metrics = {
                's1': test_stats['test/scale_s1'],
                's2': test_stats['test/scale_s2'],
                's3': test_stats['test/scale_s3'],
            }
            stop_signals = early_stopping.step(epoch, scale_metrics, model.state_dict())
            
            if early_stopping.should_stop_training():
                print('\n=== All scales converged, stopping training ===')
                break
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_path, f'checkpoint_epoch{epoch}.pth')
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'aux_optimizer': aux_optimizer.state_dict() if aux_optimizer else None,
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
                'aux_optimizer': aux_optimizer.state_dict() if aux_optimizer else None,
                'loss': best_loss,
            }, best_path)
            print(f'✓ New best model saved (loss={best_loss:.4f})')
    
    # Save early stopping history
    if early_stopping:
        early_stopping.save_history(os.path.join(args.save_path, 'early_stopping_history.json'))
        early_stopping.plot_history(os.path.join(args.save_path, 'early_stopping_plot.png'))
    
    print('\n=== Training Complete ===')
    wandb.finish()


if __name__ == '__main__':
    main()
