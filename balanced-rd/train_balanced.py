import argparse
import random
import shutil
import sys
import logging
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple, Optional

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
from typing import Dict, List, Union, Any
import os
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch._dynamo
torch._dynamo.config.suppress_errors = True

#set determinism for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)  # Global logger


class Balanced(Optimizer):
    """
    Balanced optimizer.
    Parameters:
        params: Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr: The learning rate to use (default: 1e-3).
        betas: Adam's betas parameters (b1, b2) (default: (0.9, 0.999)).
        eps: Adam's epsilon for numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) (default: 0.0).
        amsgrad: Whether to use the AMSGrad variant (default: False).
        n_tasks: Number of tasks for balancing (default: 2).
        gamma: Regularization coefficient (default: 0.001).
        w_lr: Learning rate for task weights (default: 0.025).
        max_norm: Maximum gradient norm for clipping (default: 1.0).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        n_tasks: int = 2,
        gamma: float = 0.001,
        w_lr: float = 0.025,
        max_norm: float = 1.0,
        device: torch.device = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        
        # components
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_tasks = n_tasks
        self.gamma = gamma
        self.max_norm = max_norm
        
        # Initialize state
        self.min_losses = torch.zeros(n_tasks).to(self.device)
        self.w = torch.tensor([0.0] * n_tasks, device=self.device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr)
        self.prev_loss = None
        
        print(f'Balanced optimizer initialized with {n_tasks} tasks, gamma={gamma}, w_lr={w_lr}, max_norm={max_norm}')

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def set_min_losses(self, losses):
        """Set minimum losses for normalization."""
        self.min_losses = losses.to(self.device)

    def get_weighted_loss(self, losses):
        """Compute weighted loss."""
        self.prev_loss = losses.detach().clone()
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss

    def update_task_weights(self, curr_loss):
        """Update task weights based on current losses."""
        if self.prev_loss is None:
            return
            
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        # weight decay
        if self.gamma > 0:
            d += self.gamma * self.w
        self.w.grad = d
        self.w_opt.step()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None, task_losses: Optional[torch.Tensor] = None):
        """
        Performs a single optimization step.
        
        Args:
            closure: Optional closure function
            task_losses: Tensor of individual task losses for balancing
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # If task_losses are provided, compute weighted loss for backward pass
        if task_losses is not None:
            weighted_loss = self.get_weighted_loss(task_losses)
            # The backward pass should have been called outside this function
            # We just store the loss for potential task weight updates

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.dtype in {torch.float16, torch.bfloat16}:
                        grads.append(p.grad.float())
                    else:
                        grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp_avg_sq values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            self._balanced_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
            )

        return loss

    def _balanced_update(
        self,
        params: list,
        grads: list,
        exp_avgs: list,
        exp_avg_sqs: list,
        max_exp_avg_sqs: list,
        state_steps: list,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
    ):
        """Functional API for balanced algorithm computation."""
        
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            
            # Perform weight decay
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)

            step_size = lr / (1 - beta1 ** step)

            # Compute mask based on gradient-momentum alignment
            mask = (exp_avg * grad > 0).to(grad.dtype)
            
            # Normalize mask to maintain update scale
            # limit the scaling factor to avoid too large updates
            scaler = (1 / mask.mean().clamp_(min=1e-3)).clamp_(max=10.0) 
            mask = mask * scaler
            
            # Apply update
            cautious_update = (exp_avg * mask) / denom
            param.add_(cautious_update, alpha=-step_size)

    def backward_with_task_balancing(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted loss and perform backward pass with gradient clipping.
        
        Parameters:
            losses: Tensor of individual task losses
            shared_parameters: Model parameters for gradient clipping
            
        Returns:
            Weighted loss
        """
        loss = self.get_weighted_loss(losses=losses)
        loss.backward()
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss

    def save_state(self, path: str) -> None:
        """Save the complete state of the Balanced optimizer."""
        state_dict = {
            'w': self.w.detach().cpu(),
            'min_losses': self.min_losses.detach().cpu(),
            'prev_loss': self.prev_loss.detach().cpu() if self.prev_loss is not None else None,
            'w_optimizer_state': self.w_opt.state_dict(),
            'optimizer_state': self.state_dict(),
            'n_tasks': self.n_tasks,
            'gamma': self.gamma,
            'max_norm': self.max_norm,
            'w_learning_rate': self.w_opt.param_groups[0]['lr'],
            'device': str(self.device)
        }
        torch.save(state_dict, path)

    def load_state(self, path: str) -> None:
        """Load a previously saved state of the Balanced optimizer."""
        state_dict = torch.load(path, map_location=self.device)
            
        # Configuration validation
        assert self.n_tasks == state_dict['n_tasks'], \
            f"Mismatch in number of tasks. Current: {self.n_tasks}, Loaded: {state_dict['n_tasks']}"
        
        # Load FAMO weights and state
        self.w = state_dict['w'].to(self.device).requires_grad_(True)
        self.min_losses = state_dict['min_losses'].to(self.device)
        self.prev_loss = state_dict['prev_loss'].to(self.device) if state_dict['prev_loss'] is not None else None
        
        # Recreate task weight optimizer
        self.w_opt = torch.optim.Adam([self.w], lr=state_dict['w_learning_rate'])
        self.w_opt.load_state_dict(state_dict['w_optimizer_state'])
        
        # Load main optimizer state
        self.load_state_dict(state_dict['optimizer_state'])
        
        # Update other parameters
        self.gamma = state_dict['gamma']
        self.max_norm = state_dict['max_norm']
        
        print(f"Successfully loaded Balanced optimizer state from {path}")


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


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args, device):
    """Create Balanced optimizer with proper parameter separation."""
    
    # Use the original net_aux_optimizer to get proper parameter separation
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
    }
    optimizer_dict = net_aux_optimizer(net, conf)
    
    # Extract the parameters from the created Adam optimizer
    adam_optimizer = optimizer_dict["net"]
    net_params = adam_optimizer.param_groups[0]['params']
    
    # Create Balanced optimizer with the same parameters
    balanced_optimizer = Balanced(
        net_params, 
        lr=args.learning_rate,
        n_tasks=2,  # distortion and bpp
        gamma=args.gamma,
        w_lr=0.025,
        max_norm=1.0,
        device=device
    )
    
    log_info(f"Created Balanced optimizer with {len(net_params)} parameters")
    return balanced_optimizer


def log_info(msg):
    logger.info(msg)


def train_one_epoch(
    model, train_dataloader, optimizer, epoch, clip_max_norm, accelerator, ema_net, args
):
    model.train()
    # Update the quantiles for medians
    accelerator.unwrap_model(model).latent_codec.hyper.entropy_bottleneck._update_quantiles()#for elic

    with tqdm(
        total=len(train_dataloader),
        desc=f"Train epoch {epoch} (Balanced)",
        unit="batch",
        disable=not accelerator.is_main_process,
    ) as pbar:
        for i, d in enumerate(train_dataloader):
            optimizer.zero_grad()
            out_net = model(d)
            if torch.isnan(out_net["rdloss"]):
                raise ValueError("Loss is NaN. Terminating training.")
            
            distortion_loss = out_net["distortion"]
            bpp_loss = out_net["bpp_loss"]
            
            # Use Balanced optimizer with multi-task balancing
            task_losses = torch.stack([distortion_loss, bpp_loss])
            weighted_loss = optimizer.backward_with_task_balancing(
                task_losses, 
                shared_parameters=model.parameters()
            )
            
            # Apply additional gradient clipping if needed
            if clip_max_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)
            
            # Step with task losses for potential task weight updates
            optimizer.step(task_losses=task_losses)
            ema_net.update(model.parameters())
            
            # Update task weights based on new forward pass
            with torch.no_grad():
                out_net_new = model(d)
                if torch.isnan(out_net_new["rdloss"]):
                    raise ValueError("Loss is NaN. Terminating training.")
                distortion_loss_new = out_net_new["distortion"]
                bpp_loss_new = out_net_new["bpp_loss"]
                loss_new = torch.stack([distortion_loss_new, bpp_loss_new])
                optimizer.update_task_weights(loss_new.detach())
            
            if accelerator.is_main_process:
                pbar.set_postfix(
                    {
                        "Mode": "Balanced",
                        "Iteration": i,
                        "Loss": f"{out_net['rdloss'].item():.4f}",
                        "MSE loss": f"{out_net['mse_loss'].item():.4f}",
                        "Bpp loss": f"{out_net['bpp_loss'].item():.4f}",
                    }
                )
                pbar.update(1)
    
    # Update the quantiles for tails
    log_info("Updating quantiles for tails")
    accelerator.unwrap_model(model).latent_codec.hyper.entropy_bottleneck._update_quantiles()#for elic


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
    parser = argparse.ArgumentParser(description="Training script with Balanced optimizer (CautiousAdam + FAMO).")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/home/xxxx/Project/dataset",
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
    parser.add_argument("--gamma", type=float, default=0.003, help="Regularization coefficient")
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
    dir_name = (
        results_dir + "/" + train_model.__name__ + "_" + str(args.lmbda).split(".")[1] + '_balanced_' + str(args.gamma)
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
    cmd_line_args = sys.argv
    print("Command line arguments:", cmd_line_args)
    log_info(f"{cmd_line_args}")

    args.learning_rate *= accelerator.num_processes

    optimizer = configure_optimizers(net, args, device)
    # Initialize minimum losses for FAMO balancing
    optimizer.set_min_losses(torch.tensor([-1, -1], device=device))
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=10
    )

    net, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        net, train_dataloader, test_dataloader, lr_scheduler
    )

    ema_net = ExponentialMovingAverage(net.parameters(), decay=0.999)
    ema_net.to(device)

    start_epoch = 0
    best_loss = float("inf")
    
    # Handle resume functionality
    if args.resume:
        checkpoint_path = os.path.join(dir_name, "checkpoint.pth.tar")
        balanced_state_path = os.path.join(dir_name, "balanced_state.pth")
        balanced_fallback_path = os.path.join(dir_name, "balanced_state_best.pth")
        
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
            
            # Load Balanced optimizer state if available
            if os.path.exists(balanced_state_path):
                log_info(f"Loading Balanced optimizer state from {balanced_state_path}")
                optimizer.load_state(balanced_state_path)
            elif os.path.exists(balanced_fallback_path):
                log_info(f"Loading Balanced optimizer state from {balanced_fallback_path}")
                optimizer.load_state(balanced_fallback_path)
            else:
                log_info("Warning: No Balanced optimizer state file found, starting fresh")
            
            log_info(f"Resuming training from epoch {start_epoch}")
            log_info(f"Best loss so far: {best_loss:.4f}")
        else:
            log_info(f"Warning: Resume requested but checkpoint not found at {checkpoint_path}")
            log_info("Starting training from scratch")

    # Training loop with Balanced optimizer
    log_info(f"Starting training with Balanced optimizer from epoch {start_epoch} to {args.epochs}")
    
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
            args,
        )
        log_info(f"Local current time: {time.asctime(time.localtime(time.time()))}")

        # Evaluate the kodak dataset once using regular weights
        loss, _ = compress_decompress(epoch, test_dataloader, net, args, accelerator, "kodak")
        log_info(f'Regular weights loss: {loss:.4f}')

        # Apply EMA weights
        ema_net.store(net.parameters())
        ema_net.copy_to(net.parameters())
        
        # Evaluate multiple datasets using EMA weights
        kodak_loss = evaluate_multiple_datasets(epoch, net, args, accelerator)
        
        # Restore original weights
        ema_net.restore(net.parameters())
        
        if epoch > 50:
            lr_scheduler.step(kodak_loss)

        # Check for best model
        is_best = kodak_loss < best_loss
        if is_best:
            log_info(f"New best model found! Kodak EMA loss: {kodak_loss:.4f} at epoch {epoch}")
        best_loss = min(kodak_loss, best_loss)
        
        if accelerator.is_main_process:
            if args.save:
                # Save Balanced optimizer state
                optimizer.save_state(os.path.join(dir_name, "balanced_state.pth"))
                if is_best:
                    optimizer.save_state(os.path.join(dir_name, "balanced_state_best.pth"))
                
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