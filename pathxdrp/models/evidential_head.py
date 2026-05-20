"""
Deep Evidential Regression head.
Reference: Amini et al., NeurIPS 2020.

Predicts four Normal-Inverse-Gamma parameters (γ, υ, α, β) from a latent vector.
Point estimate: μ = γ
Aleatoric variance: σ²_a = β / (α - 1)
Epistemic variance: σ²_e = β / (υ(α - 1))

Loss: NIG-NLL + λ * |y - γ| * (2υ + α)   [evidence regularisation]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialRegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4),  # → γ, log_υ, log_α, log_β
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: (B, in_dim) latent vector.
        Returns dict with keys: mu, nu, alpha, beta, pred, aleatoric, epistemic.

        Numerical note: the (nu * (alpha-1)) denominator underflows in fp16
        when both factors are small. We force this block to fp32 so the head
        is safe under torch.amp.autocast.
        """
        out = self.net(z)
        # Force the rest of the head into fp32 to keep evidential variances stable
        # under autocast (fp16 makes nu*(alpha-1) underflow → epistemic blows up).
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            out = out.float()
            gamma = out[:, 0]
            # Ensure υ > 0, α > 1, β > 0 via softplus.
            # Floors widened so the variance ratios stay finite even early in training.
            nu    = F.softplus(out[:, 1]) + 0.1   # floor raised: prevents nu→0 gaming the reg loss
            alpha = F.softplus(out[:, 2]) + 1.0 + 1e-2
            beta  = F.softplus(out[:, 3]) + 1e-2

            aleatoric = beta / (alpha - 1)
            epistemic = beta / (nu * (alpha - 1))

        return {
            "mu": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "pred": gamma,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
        }


def evidential_loss(
    pred: dict[str, torch.Tensor],
    y: torch.Tensor,
    lam: float = 0.1,
) -> torch.Tensor:
    """
    Combined NIG-NLL + evidence regularisation.
    Args:
        pred: output of EvidentialRegressionHead.forward()
        y:    (B,) ground-truth LN_IC50
        lam:  regularisation weight

    Numerical note: lgamma and log on fp16 inputs are unstable. We compute
    the loss in fp32 even when the rest of the model runs under autocast.
    """
    # Force fp32 for the loss math — lgamma/log on fp16 -> NaNs / ±inf.
    with torch.amp.autocast(device_type=y.device.type, enabled=False):
        gamma = pred["mu"].float()
        nu    = pred["nu"].float()
        alpha = pred["alpha"].float()
        beta  = pred["beta"].float()
        y_f   = y.float()

        # NIG negative log-likelihood
        twoBlambda = 2 * beta * (1 + nu)
        nll = (
            0.5 * torch.log(torch.tensor(torch.pi, device=y.device) / nu)
            - alpha * torch.log(twoBlambda)
            + (alpha + 0.5) * torch.log(nu * (y_f - gamma) ** 2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Evidence regularisation: penalise high evidence on wrong predictions
        error = torch.abs(y_f - gamma)
        reg = error * (2 * nu + alpha)

    return (nll + lam * reg).mean()
