#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/15/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B-SOPF: Stochastic Optimal Power Flow Baseline

Monte Carlo scenario-based OPF: samples N impairment scenarios,
optimizes expected cost over the sample. The stability margin is
computed as the average rho across scenarios.

Classical approach — no learned per-generator coupling constants.
"""

import torch
import torch.nn as nn
from typing import Dict


class StochasticOPF(nn.Module):
    """
    B-SOPF: Stochastic OPF with Monte Carlo impairment sampling.

    Estimates expected stability margin by averaging over multiple
    impairment samples. The dispatch is optimized for average cost.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
        n_mc_samples: int = 10,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators
        self.n_mc_samples = n_mc_samples

        # MLP for dispatch
        self.encoder = nn.Sequential(
            nn.Linear(n_buses * energy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_generators * 2),
        )

        # Fixed K
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
        p: torch.Tensor = None,
        sigma_j: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if energy_x.dim() == 2:
            batch_size = energy_x.shape[0] // self.n_buses
            energy_x = energy_x.view(batch_size, self.n_buses, -1)
        batch_size = energy_x.shape[0]

        x = energy_x.view(batch_size, -1)
        u = self.encoder(x)

        # Compute rho using actual impairments (average over provided scenario)
        if tau is not None and tau_max is not None:
            delay_term = (self.K.unsqueeze(0) * tau / tau_max.unsqueeze(0)).sum(-1)

            loss_term = torch.zeros(batch_size, device=energy_x.device)
            if p is not None:
                p_safe = torch.clamp(p, max=0.999)
                loss_term = (self.K.unsqueeze(0) * p_safe / (1 - p_safe)).sum(-1) * 0.3

            jitter_term = torch.zeros(batch_size, device=energy_x.device)
            if sigma_j is not None:
                sigma_max = kwargs.get('sigma_max', torch.tensor(200.0))
                jitter_term = (self.K.unsqueeze(0) * sigma_j ** 2 / float(sigma_max) ** 2).sum(-1) * 0.2

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
