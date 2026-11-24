"""
Implementation of SOAP: ShampoO with Adam in the Preconditioner's eigenbasis
Published as a conference paper at ICLR 2025.
Paper: "SOAP: Improving and Stabilizing Shampoo Using Adam for Language Modeling"
Authors: Nikhil Vyas, David Brandfonbrener, Depen Morwani, Sham Kakade, Rosie Zhao, Itai Shapira, Lucas Janson
Reference: https://github.com/nikhilvyas/SOAP
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class SOAP(Optimizer):
    """
    SOAP Optimizer (ICLR 2025)
    Runs AdamW in the eigenbasis provided by Shampoo's Kronecker-factored preconditioner.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 3e-4).
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.95)).
        eps: Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1e-4).
        precondition_frequency: Frequency (in steps) of updating the eigenbasis via power iteration + QR (default: 10).
        max_precond_dim: Maximum dimension size to apply preconditioning (default: 2048).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        precondition_frequency: int = 10,
        max_precond_dim: int = 2048,
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

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            freq = group["precondition_frequency"]
            max_dim = group["max_precond_dim"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Standard AdamW 1st moment in original coordinates
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Determine if parameter is 2D or 4D (conv) eligible for Shampoo eigenbasis
                    is_2d_eligible = (p.ndim == 2) or (p.ndim == 4)
                    if is_2d_eligible:
                        if p.ndim == 4:
                            m = p.shape[0]
                            n = p.shape[1:].numel()
                        else:
                            m, n = p.shape

                        can_precond = (m <= max_dim) and (n <= max_dim)
                        state["is_soap_layer"] = can_precond
                        if can_precond:
                            state["m"] = m
                            state["n"] = n
                            # 2nd moment in rotated eigenbasis (shape m x n)
                            state["exp_avg_sq_rotated"] = torch.zeros((m, n), dtype=p.dtype, device=p.device)
                            # Preconditioner matrices L (m x m) and R (n x n)
                            state["L"] = torch.zeros((m, m), dtype=torch.float32, device=p.device)
                            state["R"] = torch.zeros((n, n), dtype=torch.float32, device=p.device)
                            # Eigenvector matrices Q_L (m x m) and Q_R (n x n)
                            state["Q_L"] = torch.eye(m, dtype=torch.float32, device=p.device)
                            state["Q_R"] = torch.eye(n, dtype=torch.float32, device=p.device)
                    else:
                        state["is_soap_layer"] = False

                    if not state.get("is_soap_layer", False):
                        # Standard AdamW 2nd moment for 1D or unsupported layers
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]

                # Perform decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                exp_avg = state["exp_avg"]
                # 1. Update first moment in original space: M = beta1 * M + (1 - beta1) * G
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)

                if state.get("is_soap_layer", False):
                    m = state["m"]
                    n = state["n"]
                    L = state["L"]
                    R = state["R"]
                    Q_L = state["Q_L"]
                    Q_R = state["Q_R"]
                    V = state["exp_avg_sq_rotated"]

                    # Reshape gradient and momentum to 2D matrix (m x n)
                    G_2d = grad.view(m, n).to(torch.float32)
                    M_2d = exp_avg.view(m, n).to(torch.float32)

                    # 2. Project G and M into current eigenbasis:
                    # G' = Q_L^T G Q_R
                    # M' = Q_L^T M Q_R
                    G_proj = torch.matmul(Q_L.T, torch.matmul(G_2d, Q_R))
                    M_proj = torch.matmul(Q_L.T, torch.matmul(M_2d, Q_R))

                    # 3. Update second moment in rotated space:
                    # V = beta2 * V + (1 - beta2) * (G' ⊙ G')
                    V.mul_(beta2).addcmul_(G_proj, G_proj, value=1.0 - beta2)

                    # 4. Adam step in rotated eigenbasis:
                    # N' = (M' / bias_correction1) / (sqrt(V / bias_correction2) + eps)
                    V_hat = V / bias_correction2
                    denom = V_hat.sqrt().add_(eps)
                    step_size = lr / bias_correction1
                    N_proj = (M_proj / denom).mul_(-step_size)

                    # 5. Project preconditioned step back to original parameter space:
                    # N = Q_L N' Q_R^T
                    update_2d = torch.matmul(Q_L, torch.matmul(N_proj, Q_R.T))
                    p.add_(update_2d.view_as(p).to(p.dtype))

                    # 6. Update Shampoo preconditioner matrices:
                    # L = beta2 * L + (1 - beta2) * G G^T
                    # R = beta2 * R + (1 - beta2) * G^T G
                    L.mul_(beta2).addmm_(G_2d, G_2d.T, alpha=1.0 - beta2)
                    R.mul_(beta2).addmm_(G_2d.T, G_2d, alpha=1.0 - beta2)

                    # 7. Update eigenbasis via power iteration + QR decomposition every `freq` steps
                    if step % freq == 0:
                        # Algorithm 4: S = P Q; Q = QR(S)
                        S_L = torch.matmul(L, Q_L)
                        Q_L_new, _ = torch.linalg.qr(S_L)
                        state["Q_L"].copy_(Q_L_new)

                        S_R = torch.matmul(R, Q_R)
                        Q_R_new, _ = torch.linalg.qr(S_R)
                        state["Q_R"].copy_(Q_R_new)

                else:
                    # Fallback to standard AdamW update for 1D tensors (biases, normalization)
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

