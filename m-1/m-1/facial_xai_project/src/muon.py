from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Newton-Schulz quintic iteration for polar decomposition (matrix orthogonalization).
    Approximates UV^T where G = U S V^T.

    Reference:
      Keller Jordan (2024), "Muon: An optimizer for hidden layers in neural networks"
      Bernstein & Newhouse (2024), Higham (2008), Björck and Bowie (1971).
    """
    assert G.ndim >= 2, f"Newton-Schulz requires at least 2D tensor, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Use bfloat16 on CUDA Ampere+ if supported for high throughput, float32 otherwise
    if G.is_cuda and torch.cuda.is_bf16_supported():
        X = G.bfloat16()
    else:
        X = G.float()

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    # Normalize spectral norm to <= 1
    norm = X.norm(dim=(-2, -1), keepdim=True) + eps
    X = X / norm

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(dtype=G.dtype)


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """
    Computes Muon update step:
      1. Nesterov momentum accumulation
      2. Flattening 4D/nD tensors to 2D
      3. Newton-Schulz matrix orthogonalization
      4. Aspect-ratio scaling factor
    """
    momentum.lerp_(grad, 1.0 - beta)
    update = torch.lerp(grad, momentum, beta) if nesterov else momentum

    orig_shape = update.shape
    if update.ndim >= 3:
        # Flatten conv filter (C_out, C_in, H, W) -> (C_out, C_in * H * W)
        update = update.view(update.size(0), -1)

    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # Aspect ratio scaling factor from Keller Jordan
    scale = max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    update = update * scale

    return update.view(orig_shape)


class Muon(Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton-Schulz) + Auxiliary AdamW Optimizer.

    - 2D / 4D hidden weights use Muon (Newton-Schulz orthogonalization + Nesterov momentum).
    - Biases, normalizations, and input/output layers use standard decoupled AdamW.
    - Fully compatible with PyTorch AMP GradScaler and CosineAnnealingLR schedulers.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 1e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            use_muon=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", True)
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)

            if use_muon:
                momentum = group.get("momentum", 0.95)
                nesterov = group.get("nesterov", True)
                ns_steps = group.get("ns_steps", 5)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]

                    update = muon_update(
                        g, buf, beta=momentum, ns_steps=ns_steps, nesterov=nesterov
                    )

                    if wd != 0:
                        p.mul_(1.0 - lr * wd)
                    p.sub_(update, alpha=lr)

            else:
                beta1, beta2 = group.get("betas", (0.9, 0.95))
                eps = group.get("eps", 1e-8)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    step_val = state["step"]

                    if wd != 0:
                        p.mul_(1.0 - lr * wd)

                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step_val
                    bias_correction2 = 1.0 - beta2 ** step_val
                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def build_muon_optimizer(
    model: nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.95),
) -> Muon:
    """
    Constructs Muon optimizer with automatic parameter partitioning:
      - Muon: hidden matrix/conv weights (ndim >= 2, non-singleton spatial/features).
      - AdamW: biases, LayerNorms, layer scales, input stem conv, and final classifier head.
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Stem / classifier head and 1D / channel-scale parameters are handled by AdamW
        is_stem_or_head = (
            "classifier" in name
            or "fc" in name
            or "features.0.0" in name
            or "stem.0" in name
            or "conv1" in name
        )
        non_singleton_dims = sum(1 for d in p.shape if d > 1)

        if p.ndim >= 2 and non_singleton_dims >= 2 and not is_stem_or_head:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    param_groups = [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": muon_lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        },
        {
            "params": adamw_params,
            "use_muon": False,
            "lr": adamw_lr,
            "weight_decay": weight_decay,
            "betas": betas,
        },
    ]

    return Muon(param_groups)

