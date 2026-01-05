"""
Balanced optimizer for HPCM training.
Adapted from Balanced Rate-Distortion Optimization in Learned Image Compression (CVPR 2025).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple, Optional, List, Union


class Balanced(Optimizer):
    """
    Balanced optimizer combining CautiousAdam with FAMO (Fast Adaptive Multitask Optimization).
    
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
        
        # FAMO components
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
        """Compute weighted loss using FAMO."""
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
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state['step'] += 1
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
        """Functional API for balanced algorithm computation (CautiousAdam)."""
        
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
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)

            step_size = lr / (1 - beta1 ** step)

            # CautiousAdam: Compute mask based on gradient-momentum alignment
            mask = (exp_avg * grad > 0).to(grad.dtype)
            
            # Normalize mask to maintain update scale
            scaler = (1 / mask.mean().clamp_(min=1e-3)).clamp_(max=10.0) 
            mask = mask * scaler
            
            # Apply cautious update
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
