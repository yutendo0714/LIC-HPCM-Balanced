import argparse
import random
import shutil
import sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
import time
from compressai.datasets import ImageFolder
from compressai.optimizers import net_aux_optimizer
from compressai.models import Elic2022Official as train_model
from torch_ema import ExponentialMovingAverage
from timm.utils import unwrap_model
import copy
from pytorch_msssim import ms_ssim
from tqdm import tqdm
from typing import Any, Dict, List
import math
import os
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch.nn.functional as F

logger = logging.getLogger(__name__)  # Global logger

# set deterministic bahavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  

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


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"]


def log_info(msg):
    logger.info(msg)


def train_one_epoch(
    model, train_dataloader, optimizer, epoch, clip_max_norm, accelerator, ema_net
):
    model.train()
    # Update the quantiles for medians
    accelerator.unwrap_model(model).latent_codec.hyper.entropy_bottleneck._update_quantiles()  # for elic

    with tqdm(
        total=len(train_dataloader),
        desc=f"Train epoch {epoch}",
        unit="batch",
        disable=not accelerator.is_main_process,
    ) as pbar:
        for i, d in enumerate(train_dataloader):
            optimizer.zero_grad()
            out_net = model(d)
            if torch.isnan(out_net["rdloss"]):
                raise ValueError("Loss is NaN. Terminating training.")
            accelerator.backward(out_net["rdloss"])
            if clip_max_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
            ema_net.update(model.parameters())
            if accelerator.is_main_process:
                pbar.set_postfix(
                    {
                        "Iteration": i,
                        "Loss": f"{out_net['rdloss'].item():.4f}",
                        "MSE loss": f"{out_net['mse_loss'].item():.4f}",
                        "Bpp loss": f"{out_net['bpp_loss'].item():.4f}",
                    }
                )
                pbar.update(1)
    # Update the quantiles for tails
    log_info("Updating quantiles for tails")
    accelerator.unwrap_model(model).latent_codec.hyper.entropy_bottleneck._update_quantiles()  # for elic


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr-rgb"] = psnr(org, rec).item()
    metrics["ms-ssim-rgb"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def compress_decompress(epoch, test_dataloader, model, args, accelerator, dataset_name=None):
    dataset_name = dataset_name or "kodak"
    log_info(f"Evaluating on dataset: {dataset_name}")
    log_info(f"Len of test_dataloader: {len(test_dataloader)}")
    
    device = accelerator.device
    model_t = copy.deepcopy(model)
    model_t.eval()  
    model_t = unwrap_model(model_t)
    model_t.update(force=True)
    model_t = model_t.to(device)

    rd_losses = []
    mse_losses = []
    bpp_losses = []
    psnr_values = []
    ms_ssim_values = []

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)

            # Pad the image to multiple of 64
            _, _, h, w = d.size()
            pad_h = (64 - (h % 64)) % 64
            pad_w = (64 - (w % 64)) % 64
            padded_d = F.pad(d, (0, pad_w, 0, pad_h), value=0)

            out_enc = model_t.compress(padded_d)
            out_dec = model_t.decompress(out_enc["strings"], out_enc["shape"])

            # Remove padding
            out_dec["x_hat"] = out_dec["x_hat"][:, :, :h, :w]

            num_pixels = d.size(0) * d.size(2) * d.size(3)
            # Calculate the mse
            MSE = torch.nn.functional.mse_loss(out_dec["x_hat"], d)
            bpp_ = 0
            for s in out_enc["strings"]:
                for j in s:
                    if isinstance(j, list):
                        for i in j:
                            if isinstance(i, list):
                                for k in i:
                                    bpp_ += len(k)
                            else:
                                bpp_ += len(i)
                    else:
                        bpp_ += len(j)
            bpp_ *= 8.0 / num_pixels
            metrics = compute_metrics(d, out_dec["x_hat"], 255)
            rd_losses.append(args.lmbda * 255 ** 2 * MSE + bpp_)
            mse_losses.append(MSE.item())
            bpp_losses.append(bpp_)
            psnr_values.append(metrics["psnr-rgb"])
            ms_ssim_values.append(metrics["ms-ssim-rgb"])

    rd_losses = torch.tensor(rd_losses, device=device).to(accelerator.device)
    mse_losses = torch.tensor(mse_losses, device=device).to(accelerator.device)
    bpp_losses = torch.tensor(bpp_losses, device=device).to(accelerator.device)
    psnr_values = torch.tensor(psnr_values, device=device).to(accelerator.device)
    ms_ssim_values = torch.tensor(ms_ssim_values, device=device).to(accelerator.device)

    rd_losses = accelerator.gather(rd_losses)
    mse_losses = accelerator.gather(mse_losses)
    bpp_losses = accelerator.gather(bpp_losses)
    psnr_values = accelerator.gather(psnr_values)
    ms_ssim_values = accelerator.gather(ms_ssim_values)

    RD_avg = rd_losses.mean().item()
    mse_avg = mse_losses.mean().item()
    bpp_avg = bpp_losses.mean().item()
    psnr_avg = psnr_values.mean().item()
    ms_ssim_avg = ms_ssim_values.mean().item()

    if accelerator.is_main_process:
        log_info(
            f"Test epoch {epoch} Dataset=[{dataset_name}]:"
            f"\tRD: {RD_avg:.4f} |"
            f"\tMSE: {mse_avg:.4f} |"
            f"\tBpp: {bpp_avg:.4f} |"
            f"\tPSNR: {psnr_avg:.4f} |"
            f"\tMS-SSIM: {ms_ssim_avg:.4f}"
        )

    return RD_avg, {
        "dataset": dataset_name,
        "rd_loss": RD_avg,
        "mse_loss": mse_avg,
        "bpp": bpp_avg,
        "psnr": psnr_avg,
        "ms_ssim": ms_ssim_avg
    }


def evaluate_multiple_datasets(epoch, model, args, accelerator):
    """Evaluate the performance of multiple datasets and return the performance of the kodak dataset as a benchmark"""
    test_transforms = transforms.Compose([transforms.ToTensor()])
    
    # Define the list of datasets to test
    test_datasets = [
        {"name": "kodak", "split": "kodak"},
        {"name": "clic", "split": "clic2022-test"},
        {"name": "tecnick", "split": "RGB_OR_1200x1200"},
    ]
    
    results = {}
    kodak_loss = None
    
    # Evaluate each dataset individually
    for dataset_info in test_datasets:
        dataset_name = dataset_info["name"]
        dataset_split = dataset_info["split"]
        
        try:
            # Load current dataset
            current_dataset = ImageFolder(
                args.dataset, 
                split=dataset_split, 
                transform=test_transforms
            )
            
            # Creating a Data Loader
            current_dataloader = DataLoader(
                current_dataset,
                batch_size=args.test_batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=(accelerator.device == "cuda"),
            )
            
            # Preparing the data loader
            current_dataloader = accelerator.prepare(current_dataloader)
            
            # Evaluating the current dataset
            loss, metrics = compress_decompress(
                epoch, 
                current_dataloader, 
                model, 
                args, 
                accelerator, 
                dataset_name
            )
            
            # Save results
            results[dataset_name] = metrics
            
            # If it is a kodak dataset, save its loss value as the return value
            if dataset_name == "kodak":
                kodak_loss = loss
            
        except Exception as e:
            log_info(f"Error evaluating dataset {dataset_name}: {str(e)}")
    
    # Only print summary information in the main process
    if accelerator.is_main_process:
        # Printing the summary table
        log_info("\n" + "="*80)
        log_info(f"Summary of test dataset performance (Epoch {epoch}):")
        log_info("-"*80)
        log_info(f"{'Dataset':<12} | {'RD Loss':<10} | {'BPP':<10} | {'PSNR (dB)':<12} | {'MS-SSIM':<10}")
        log_info("-"*80)
        
        for dataset_name in ["kodak", "clic", "tecnick"]:
            if dataset_name in results:
                metrics = results[dataset_name]
                log_info(
                    f"{dataset_name:<12} | "
                    f"{metrics['rd_loss']:<10.4f} | "
                    f"{metrics['bpp']:<10.4f} | "
                    f"{metrics['psnr']:<12.4f} | "
                    f"{metrics['ms_ssim']:<10.4f}"
                )
        
        log_info("="*80)
    
    # Returns an infinite value if the result of kodak cannot be obtained
    return kodak_loss if kodak_loss is not None else float('inf')


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace("checkpoint", "checkpoint_best_loss"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/home/xxx/Project/dataset",
        help="Training dataset",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=2e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=14,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=2.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--pretrained", type=str, help="Path to a pretrained model")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results dir")
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    accelerator = Accelerator()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        set_seed(args.seed)

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(
                args.patch_size, pad_if_needed=True, padding_mode="reflect"
            ),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder(
        args.dataset, split="train2017", transform=train_transforms
    )
    test_dataset = ImageFolder(
        args.dataset, split="kodak", transform=test_transforms
    )

    device = accelerator.device

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = train_model(lmbda=args.lmbda)
    net = net.to(device)
    
    if args.compile:
        log_info("Compiling model")
        net = torch.compile(net)
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    dir_name = (
        results_dir + "/" + train_model.__name__ + "_" + str(args.lmbda).split(".")[1]
    )
    dir_path = dir_name
    if accelerator.is_main_process:
        if not os.path.exists(dir_path):
            os.mkdir(dir_name)
        log_info(f"Results will be saved in {dir_path}")
        # Copy the training script to the results directory
        script_path = __file__
        shutil.copy2(script_path, dir_path)
        # save model architecture
        with open(os.path.join(dir_name, "model.txt"), "w") as f:
            f.write(str(net))

    # Set up logging
    log_file = os.path.join(dir_name, "training.log")
    if accelerator.is_main_process:
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
        logging_level = logging.INFO
    else:
        handlers = [logging.StreamHandler(sys.stdout)]
        logging_level = logging.WARNING  # Suppress INFO logs for non-main processes

    logging.basicConfig(
        level=logging_level,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        handlers=handlers,
    )

    log_info(f"{args}")
    cmd_line_args =  sys.argv
    print("Command line arguments:", cmd_line_args)
    log_info(f"{cmd_line_args}")

    args.learning_rate *= accelerator.num_processes

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=10
    )

    net, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    ema_net = ExponentialMovingAverage(net.parameters(), decay=0.999)
    ema_net.to(device)

    start_epoch = 0
    best_loss = float("inf")

    # Handle resume functionality
    if args.resume:
        checkpoint_path = os.path.join(dir_name, "checkpoint.pth.tar")
        if os.path.exists(checkpoint_path):
            log_info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            unwrapped_model = unwrap_model(accelerator.unwrap_model(net))
            unwrapped_model.load_state_dict(checkpoint["state_dict"])
            ema_net.load_state_dict(checkpoint["ema_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["loss"]
            log_info(f"Checkpoint loaded. Last completed epoch: {checkpoint['epoch']}")
            log_info(f"Resuming training from epoch {start_epoch}")
            log_info(f"Best loss so far: {best_loss:.4f}")
        else:
            log_info(f"Warning: Resume requested but checkpoint not found at {checkpoint_path}")
            log_info("Starting training from scratch")
    
    # Handle pretrained model loading
    if args.pretrained:
        log_info(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        unwrapped_model = unwrap_model(accelerator.unwrap_model(net))
        unwrapped_model.load_state_dict(checkpoint["state_dict"])
        log_info("Pretrained model loaded successfully")

    # Training loop
    log_info(f"Starting training from epoch {start_epoch} to {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        log_info(f"Epoch {epoch}/{args.epochs-1}")
        
        train_one_epoch(
            net,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            accelerator,
            ema_net,
        )
        log_info(f"Local current time : {time.asctime(time.localtime(time.time()))}")

        # Evaluate the kodak dataset once using regular weights
        loss, _ = compress_decompress(epoch, test_dataloader, net, args, accelerator, "kodak")
        log_info(f'Regular weights loss: {loss:.4f}')

        # Apply EMA weights
        ema_net.store(net.parameters())
        ema_net.copy_to(net.parameters())
        
        # Evaluate multiple datasets using EMA weights, returning kodak performance as a primary metric
        kodak_loss = evaluate_multiple_datasets(epoch, net, args, accelerator)
        
        # Restore original weights
        ema_net.restore(net.parameters())
        
        # Use results from kodak dataset to update learning rate
        lr_scheduler.step(kodak_loss)

        # Use the results of the kodak dataset to determine if it is the best model to use
        is_best = kodak_loss < best_loss
        if is_best:
            log_info(f"New best model found! Kodak EMA loss: {kodak_loss:.4f} at epoch {epoch}")
        best_loss = min(kodak_loss, best_loss)
        
        if accelerator.is_main_process:
            if args.save:
                unwrapped_model = accelerator.unwrap_model(net)
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": unwrap_model(unwrapped_model).state_dict(),
                        "ema_state_dict": ema_net.state_dict(),
                        "loss": kodak_loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    os.path.join(dir_name, "checkpoint.pth.tar"),
                )

    accelerator.end_training()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main(sys.argv[1:])