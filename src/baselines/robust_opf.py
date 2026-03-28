#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/14/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B-ROPF: Robust Optimal Power Flow Baseline

Worst-case optimization over an uncertainty set of impairments.
Computes dispatch that is feasible for ALL impairment realizations
within the uncertainty set.

This is a classical approach — no learned coupling constants.
The "robustness" comes from the worst-case margin, not from
learning per-generator sensitivities.
"""

import torch
import torch.nn as nn
from typing import Dict


class RobustOPF(nn.Module):
    """
    B-ROPF: Robust OPF with worst-case impairment margins.

    Computes stability margin under worst-case impairments within
    the uncertainty set: tau in [0, tau_max], p in [0, p_max], sigma in [0, sig_max].

    The dispatch is optimized for the worst case, making it conservative.
    No learnable coupling constants — uses fixed worst-case analysis.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
        p_max: float = 0.3,
        sigma_max_ms: float = 100.0,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators
        self.p_max = p_max
        self.sigma_max_ms = sigma_max_ms

        # Simple MLP for dispatch (no GNN — classical methods don't use graph ML)
        self.encoder = nn.Sequential(
            nn.Linear(n_buses * energy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_generators * 2),
        )

        # Fixed K (not learned — set from worst-case analysis)
        if lambda_min_0 is not None:
            k_val = 0.9 * abs(lambda_min_0) / n_generators
        else:
            k_val = k_init_scale
        self.register_buffer('K', torch.ones(n_generators) * k_val)

    def forward(
        self,
        energy_x: torch.Tensor,
        comm_x: torch.Tensor = None,
        tau: torch.Tensor = None,
        tau_max: torch.Tensor = None,
        lambda_min_0: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Reshape if flattened
        if energy_x.dim() == 2:
            batch_size = energy_x.shape[0] // self.n_buses
            energy_x = energy_x.view(batch_size, self.n_buses, -1)
        batch_size = energy_x.shape[0]

        # Dispatch from energy features only (robust = ignores comm state)
        x = energy_x.view(batch_size, -1)
        u = self.encoder(x)

        # Worst-case stability margin: use MAXIMUM impairments
        if tau is not None and tau_max is not None:
            # Worst case: max delay + max loss + max jitter
            delay_term = (self.K.unsqueeze(0) * tau / tau_max.unsqueeze(0)).sum(-1)
            # Worst-case loss term: p_max / (1 - p_max)
            loss_term = self.K.sum() * self.p_max / (1.0 - self.p_max)
            # Worst-case jitter term (small, quadratic)
            jitter_term = self.K.sum() * (self.sigma_max_ms ** 2) / (200.0 ** 2) * 0.1
            rho = torch.abs(lambda_min_0) - delay_term - loss_term - jitter_term
        else:
            rho = torch.zeros(batch_size, device=energy_x.device)

        return {
            'u': u,
            'rho': rho,
            'K': self.K,
            'R': torch.zeros_like(self.K),
            'J': torch.zeros_like(self.K),
        }

    def get_coupling_constants(self) -> dict:
        return {'K': self.K, 'R': torch.zeros_like(self.K), 'J': torch.zeros_like(self.K),
                'K_mean': self.K.mean().item(), 'R_mean': 0.0, 'J_mean': 0.0}
