"""
HPCM Training Script with Balanced R-D Optimization (Phase 1)
Supports both standard Adam and Balanced optimizer.
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

# Add parent directory to path for src imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out['y_bpp'] = torch.log(output['likelihoods']['y']).sum() / (-math.log(2) * num_pixels)
        out['z_bpp'] = torch.log(output['likelihoods']['z']).sum() / (-math.log(2) * num_pixels)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        
        # ===== Phase 1: Separate distortion for Balanced R-D =====
        out["distortion"] = self.lmbda * 255 ** 2 * out["mse_loss"]
        out["loss"] = out["distortion"] + out["bpp_loss"]
        out["psnr"] = 10 * (torch.log(1 * 1 / out["mse_loss"]) / np.log(10))

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # [FIX] Get Python float to prevent graph leak
        if isinstance(val, torch.Tensor):
            val = val.detach().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, epoch, global_step, 
    clip_max_norm, use_balanced=False
):
    model.train()
    print(model.training)
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    distortion_loss = AverageMeter()  # Phase 1: New metric
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    t_start = time.time()
    for i, d in enumerate(train_dataloader):

        global_step += 1
        d = d.to(device)
        optimizer.zero_grad()
        out_net = model(d)

        out_criterion = criterion(out_net, d)
        
        # ===== Phase 1: Balanced R-D Integration =====
        if use_balanced:
            # Build 2-task losses
            task_losses = torch.stack([
                out_criterion["distortion"],
                out_criterion["bpp_loss"]
            ])
            # Backward with task balancing
            weighted_loss = optimizer.backward_with_task_balancing(
                task_losses, 
                shared_parameters=model.parameters()
            )
        else:
            # Standard training
            out_criterion["loss"].backward()
        
        # ===== Gradient clipping =====
        if clip_max_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if total_norm.isnan() or total_norm.isinf():
                print("non-finite norm, skip this batch")
                continue
        
        # ===== Optimizer step =====
        if use_balanced:
            optimizer.step(task_losses=task_losses)
            # Update task weights
            with torch.no_grad():
                out_net_new = model(d)
                out_criterion_new = criterion(out_net_new, d)
                task_losses_new = torch.stack([
                    out_criterion_new["distortion"],
                    out_criterion_new["bpp_loss"]
                ])
                optimizer.update_task_weights(task_losses_new.detach())
        else:
            optimizer.step()

        # ===== Update metrics =====
        bpp_loss.update(out_criterion["bpp_loss"])
        loss.update(out_criterion["loss"])
        mse_loss.update(out_criterion["mse_loss"])
        distortion_loss.update(out_criterion["distortion"])
        psnr.update(out_criterion["psnr"])
        y_bpp.update(out_criterion["y_bpp"])
        z_bpp.update(out_criterion["z_bpp"])
        
        # ===== Logging =====
        log_dict = {
            "train/loss": out_criterion["loss"].item(),
            "train/distortion": out_criterion["distortion"].item(),
            "train/bpp_loss": out_criterion["bpp_loss"].item(),
            "train/mse_loss": out_criterion["mse_loss"].item(),
            "train/psnr": out_criterion["psnr"].item(),
            "train/y_bpp": out_criterion["y_bpp"].item(),
            "train/z_bpp": out_criterion["z_bpp"].item(),
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }
        
        if use_balanced:
            # Log task weights
            task_weights = torch.softmax(optimizer.w, -1)
            log_dict["train/task_weight_distortion"] = task_weights[0].item()
            log_dict["train/task_weight_bpp"] = task_weights[1].item()
        
        wandb.log(log_dict, step=global_step)

        if i % 100 == 0:
            t_end = time.time() - t_start
            t_start = time.time()
            mode_str = "Balanced" if use_balanced else "Standard"
            print(
                f"Train epoch {epoch} [{mode_str}]: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tLoss: {loss.avg:.4f} |"
                f"\tDist: {distortion_loss.avg:.4f} |"
                f"\tMSE loss: {mse_loss.avg:.6f} |"
                f"\tPSNR: {psnr.avg:.3f} |"
                f"\tBpp loss: {bpp_loss.avg:.4f} |"
                f"\ty bpp: {y_bpp.avg:.4f} |"
                f"\tz bpp: {z_bpp.avg:.4f} |"
                f'\ttime: {t_end:.2f}s |'
            )
            if use_balanced:
                task_weights = torch.softmax(optimizer.w, -1)
                print(f"\tTask weights - Dist: {task_weights[0].item():.3f}, BPP: {task_weights[1].item():.3f}")
            torch.cuda.empty_cache()
        
    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, global_step):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    distortion_loss = AverageMeter()
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            distortion_loss.update(out_criterion["distortion"])
            psnr.update(out_criterion["psnr"])
            y_bpp.update(out_criterion["y_bpp"])
            z_bpp.update(out_criterion["z_bpp"])
            
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.4f} |"
        f"\tDist: {distortion_loss.avg:.4f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tPSNR: {psnr.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\ty bpp: {y_bpp.avg:.4f} |"
        f"\tz bpp: {z_bpp.avg:.4f} |"
    )
    wandb.log(
        {
            "test/loss": loss.avg,
            "test/distortion": distortion_loss.avg,
            "test/mse_loss": mse_loss.avg,
            "test/bpp_loss": bpp_loss.avg,
            "test/psnr": psnr.avg,
            "test/y_bpp": y_bpp.avg,
            "test/z_bpp": z_bpp.avg,
            "epoch": epoch,
        },
        step=global_step,
    )

    return loss.avg

def parse_args(argv):
    parser = argparse.ArgumentParser(description="HPCM training with Balanced R-D (Phase 1)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., HPCM_Base)")
    parser.add_argument("--model_class", type=str, default="hypers")
    parser.add_argument(
        "-tr_d", "--train_dataset", type=str, required=True, help="Training dataset path"
    )
    parser.add_argument(
        "-te_d", "--test_dataset", type=str, required=True, help="Testing dataset path"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=3001,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.013,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="./outputs/", help="Where to save model"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/", help="Where to save logs"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s)",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    
    # ===== Phase 1: Balanced R-D arguments =====
    parser.add_argument(
        "--use_balanced",
        action="store_true",
        help="Use Balanced R-D optimizer (default: False)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.003,
        help="Balanced optimizer gamma (regularization) (default: 0.003)"
    )
    parser.add_argument(
        "--w_lr",
        type=float,
        default=0.025,
        help="Balanced optimizer task weight learning rate (default: 0.025)"
    )
    
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(args)
    
    # Setup paths
    mode_suffix = "_balanced" if args.use_balanced else "_standard"
    args.log_dir = os.path.join(args.log_dir, args.model_name + '_lmbda' + str(args.lmbda) + mode_suffix)
    args.save_path = os.path.join(args.save_path, args.model_name + '_lmbda' + str(args.lmbda) + mode_suffix)
    if not os.path.exists(args.log_dir): 
        os.makedirs(args.log_dir)
    if not os.path.exists(args.save_path): 
        os.makedirs(args.save_path)
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    # Initialize wandb
    wandb_project = os.environ.get("WANDB_PROJECT", "LIC_HPCM_Phase1")
    wandb_run = wandb.init(
        project=wandb_project,
        config=vars(args),
        name=f"HPCM_{args.model_name}_lambda_{args.lmbda}{'_balanced' if args.use_balanced else '_standard'}",
    )

    # Setup datasets
    test_dataset = Dataset(
        args.test_dataset,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    train_dataset = Dataset(
        args.train_dataset,
        transform=transforms.Compose([
            transforms.RandomCrop(args.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # Load model
    import importlib
    net = importlib.import_module(f'.{args.model_name}', f'src.models').HPCM()
    print(net)
    net = net.to(device)
    wandb.watch(net, log="all", log_freq=100)

    # Learning rate scheduler
    lr_scheduler = lambda x : \
    5e-5 if x < 2750 else (
        1.5e-5 if x < 2850 else (
            5e-6 if x < 2950 else 5e-7
        )
    )

    last_epoch = 0

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    # ===== Phase 1: Optimizer setup =====
    if args.use_balanced:
        from src.optimizers.balanced import Balanced
        print(f"Using Balanced optimizer (gamma={args.gamma}, w_lr={args.w_lr})")
        optimizer = Balanced(
            net.parameters(),
            lr=5e-5,  # Base learning rate
            n_tasks=2,  # distortion + bpp
            gamma=args.gamma,
            w_lr=args.w_lr,
            max_norm=1.0,
            device=device
        )
        # Initialize minimum losses (negative values to start)
        optimizer.set_min_losses(torch.tensor([-1.0, -1.0], device=device))
    else:
        print("Using standard Adam optimizer")
        optimizer = optim.Adam(net.parameters(), lr=5e-5)
    
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    best_loss = float("inf")
    global_step = 0
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Load model state
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
            if 'epoch' in checkpoint:
                last_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {last_epoch}")
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
                print(f"Best loss so far: {best_loss:.4f}")
            if 'optimizer' in checkpoint and not args.use_balanced:
                # Only load optimizer state for standard Adam
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Optimizer state loaded")
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
                print(f"Resuming from global step {global_step}")
        else:
            # Just model weights
            net.load_state_dict(checkpoint)
            print("Model weights loaded (no training state)")
    
    # Training loop
    for epoch in range(last_epoch, args.epochs):

        lr = lr_scheduler(epoch)
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr
        
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        global_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            global_step,
            args.clip_max_norm,
            use_balanced=args.use_balanced
        )

        loss = test_epoch(epoch, test_dataloader, net, criterion, global_step)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best:
            print(f"epoch {epoch} is best now!")
            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
            }
            torch.save(checkpoint_dict, os.path.join(args.save_path, 'epoch_best.pth.tar'))
            
            # Save Balanced optimizer state if using it
            if args.use_balanced:
                optimizer.save_state(os.path.join(args.save_path, 'balanced_state_best.pth'))

        if epoch % 500 == 0:
            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
            }
            torch.save(checkpoint_dict, os.path.join(args.save_path, 'epoch_' + str(epoch) + '.pth.tar'))
            
            # Save Balanced optimizer state if using it
            if args.use_balanced:
                optimizer.save_state(os.path.join(args.save_path, f'balanced_state_{epoch}.pth'))


if __name__ == "__main__":
    main(sys.argv[1:])
