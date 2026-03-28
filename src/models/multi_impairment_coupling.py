#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/05/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Multi-Impairment Coupling Constants (Theorem 2)

Three families of learnable positive constants for the multi-impairment
stability bound:

  rho(tau, p, sigma_j) >= |lambda_min(0)|
      - SUM_i (K_i * tau_i / tau_max_i)          [delay term]
      - SUM_i (R_i * p_i / (1 - p_i))            [packet loss term]
      - SUM_i (J_i * sigma_j_i^2 / sigma_max^2)  [jitter term]

All constants are parameterized as exp(log_param) to ensure positivity.

Budget initialization: K gets 50%, R gets 30%, J gets 20% of
  safety_factor * |lambda_min(0)| / n_generators
This reflects delay being the dominant impairment in typical conditions.
"""

import torch
import torch.nn as nn
from typing import Tuple


class MultiImpairmentCoupling(nn.Module):
    """
    Learnable coupling constants K_i, R_i, J_i for each generator.

    K_i: delay sensitivity (from base paper, retained)
    R_i: packet loss sensitivity (NEW)
    J_i: jitter sensitivity (NEW)
    """

    def __init__(
        self,
        n_generators: int,
        init_scale: float = 0.1,
        k_budget: float = 0.5,
        r_budget: float = 0.3,
        j_budget: float = 0.2,
        learnable: bool = True,
    ):
        """
        Args:
            n_generators: Number of generators
            init_scale: Base init scale (overridden by auto-scale if lambda_min provided)
            k_budget: Fraction of stability budget for K (delay)
            r_budget: Fraction of stability budget for R (loss)
            j_budget: Fraction of stability budget for J (jitter)
            learnable: Whether constants are learnable
        """
        super().__init__()
        self.n_generators = n_generators

        # Log-parameterization: param = exp(log_param) > 0
        k_init = torch.log(torch.ones(n_generators) * init_scale * k_budget)
        r_init = torch.log(torch.ones(n_generators) * init_scale * r_budget)
        j_init = torch.log(torch.ones(n_generators) * init_scale * j_budget)

        if learnable:
            self.log_K = nn.Parameter(k_init)
            self.log_R = nn.Parameter(r_init)
            self.log_J = nn.Parameter(j_init)
        else:
            self.register_buffer("log_K", k_init)
            self.register_buffer("log_R", r_init)
            self.register_buffer("log_J", j_init)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            K: Delay coupling constants [n_generators], all positive
            R: Loss resilience constants [n_generators], all positive
            J: Jitter sensitivity constants [n_generators], all positive
        """
        return torch.exp(self.log_K), torch.exp(self.log_R), torch.exp(self.log_J)

    def get_values(self) -> dict:
        """Get current values as detached dict."""
        K, R, J = self.forward()
        return {
            'K': K.detach(), 'R': R.detach(), 'J': J.detach(),
            'K_mean': K.mean().item(), 'R_mean': R.mean().item(), 'J_mean': J.mean().item(),
        }


def compute_multi_impairment_init_scale(
    n_generators: int,
    lambda_min_0: float,
    safety_factor: float = 0.9,
    target_erosion: float = 0.7,
) -> float:
    """
    Compute grid-size-aware initialization scale.

    Under moderate impairments (tau=100ms/500ms, p=0.15, sigma=50ms/200ms),
    the total erosion should consume target_erosion * |lambda_min| of the budget.
    This ensures that under severe impairments, rho can actually go negative.

    The init_scale per generator = target_erosion * |lambda_min| / n_gen.
    With the budget split (K:50%, R:30%, J:20%), moderate conditions give:
      erosion = n_gen * [K*0.2 + R*0.176 + J*0.0625] ~ n_gen * init_scale * 0.34
    We want this = target_erosion * |lambda_min|, so:
      init_scale = target_erosion * |lambda_min| / (n_gen * 0.34)
    """
    abs_lambda = abs(float(lambda_min_0))
    if abs_lambda < 1e-10 or n_generators <= 0:
        return 0.1

    # Effective weighted sum for moderate scenario:
    # K*tau/tau_max = K*0.2, R*p/(1-p) = R*0.176, J*sig²/sig_max² = J*0.0625
    # With budget split: K=0.5*s, R=0.3*s, J=0.2*s
    # Per-gen erosion = 0.5*s*0.2 + 0.3*s*0.176 + 0.2*s*0.0625 = s*0.1653
    effective_weight = 0.5 * 0.2 + 0.3 * 0.176 + 0.2 * 0.0625  # = 0.1653
    scale = target_erosion * abs_lambda / (n_generators * effective_weight)
    return max(min(scale, 10.0), 1e-6)


def compute_rho_multi_impairment(
    K: torch.Tensor,
    R: torch.Tensor,
    J: torch.Tensor,
    tau: torch.Tensor,
    tau_max: torch.Tensor,
    p: torch.Tensor,
    sigma_j: torch.Tensor,
    sigma_max: torch.Tensor,
    lambda_min_0: torch.Tensor,
) -> torch.Tensor:
    """
    Compute multi-impairment stability margin rho(tau, p, sigma_j).

    Args:
        K: Delay coupling [n_gen]
        R: Loss resilience [n_gen]
        J: Jitter sensitivity [n_gen]
        tau: Delays [batch, n_gen] in ms
        tau_max: Delay margins [n_gen] or [batch, n_gen] in ms
        p: Packet loss rates [batch, n_gen] in [0, p_max)
        sigma_j: Jitter std [batch, n_gen] in ms
        sigma_max: Max jitter for normalization [1] or scalar in ms
        lambda_min_0: Baseline eigenvalue [batch] or scalar

    Returns:
        rho: Stability margin [batch]
    """
    # Normalize tau
    if tau_max.dim() == 1:
        tau_max = tau_max.unsqueeze(0)
    tau_normalized = tau / tau_max  # [batch, n_gen]

    # Delay term: SUM K_i * tau_i / tau_max_i
    delay_term = (K.unsqueeze(0) * tau_normalized).sum(dim=-1)

    # Packet loss term: SUM R_i * p_i / (1 - p_i)
    # Clamp p to avoid division by zero at p=1
    p_safe = torch.clamp(p, max=0.999)
    loss_term = (R.unsqueeze(0) * p_safe / (1.0 - p_safe)).sum(dim=-1)

    # Jitter term: SUM J_i * sigma_j_i^2 / sigma_max^2
    if isinstance(sigma_max, torch.Tensor):
        sigma_max_val = sigma_max.float()
    else:
        sigma_max_val = float(sigma_max)
    sigma_max_sq = sigma_max_val ** 2
    jitter_term = (J.unsqueeze(0) * sigma_j ** 2 / sigma_max_sq).sum(dim=-1)

    # rho = |lambda_min(0)| - delay_term - loss_term - jitter_term
    rho = torch.abs(lambda_min_0) - delay_term - loss_term - jitter_term

    return rho
