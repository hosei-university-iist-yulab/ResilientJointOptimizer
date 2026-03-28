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
B-Hinf: H-infinity Robust Controller Baseline

Classical robust control approach using H-infinity norm minimization.
Models delay, packet loss, and jitter as structured uncertainties
and designs a controller that minimizes the worst-case gain.

In the ML-comparable implementation: uses a neural network to approximate
the H-infinity controller, with fixed robustness margins (not learned).
"""

import torch
import torch.nn as nn
from typing import Dict


class HInfController(nn.Module):
    """
    B-Hinf: H-infinity inspired robust controller.

    Uses a conservatively designed controller with fixed robustness
    margins derived from H-infinity theory. The neural network
    approximates the controller mapping, but the stability margins
    are analytical (not learned).

    Key difference from Ours: margins are uniform across generators
    (same K for all), not per-generator learned values.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
        gamma_hinf: float = 2.0,  # H-inf gain bound
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators
        self.gamma_hinf = gamma_hinf

        # Controller network (takes both energy and comm features)
        total_input = n_buses * (energy_input_dim + comm_input_dim)
        self.controller = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_generators * 2),
        )

        # Fixed UNIFORM coupling constants (H-inf = same margin for all)
        if lambda_min_0 is not None:
            # Conservative: divide by gamma_hinf for robustness
            k_val = 0.9 * abs(lambda_min_0) / (n_generators * gamma_hinf)
        else:
            k_val = k_init_scale / gamma_hinf
        self.register_buffer('K', torch.ones(n_generators) * k_val)
        # H-inf treats all impairments uniformly
        self.register_buffer('R', torch.ones(n_generators) * k_val * 0.5)
        self.register_buffer('J', torch.ones(n_generators) * k_val * 0.3)

    def forward(
        self,
        energy_x: torch.Tensor,
        comm_x: torch.Tensor = None,
        tau: torch.Tensor = None,
        tau_max: torch.Tensor = None,
        lambda_min_0: torch.Tensor = None,
        p: torch.Tensor = None,
        sigma_j: torch.Tensor = None,
        sigma_max: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if energy_x.dim() == 2:
            batch_size = energy_x.shape[0] // self.n_buses
            energy_x = energy_x.view(batch_size, self.n_buses, -1)
            if comm_x is not None and comm_x.dim() == 2:
                comm_x = comm_x.view(batch_size, self.n_buses, -1)
        batch_size = energy_x.shape[0]

        # Concatenate energy + comm for controller input
        if comm_x is not None:
            x = torch.cat([energy_x, comm_x], dim=-1).view(batch_size, -1)
        else:
            x = energy_x.view(batch_size, -1)

        u = self.controller(x)

        # Compute rho with fixed uniform K, R, J
        rho = torch.zeros(batch_size, device=energy_x.device)
        if tau is not None and lambda_min_0 is not None:
            delay_term = (self.K.unsqueeze(0) * tau / tau_max.unsqueeze(0)).sum(-1)
            rho = torch.abs(lambda_min_0) - delay_term

            if p is not None:
                p_safe = torch.clamp(p, max=0.999)
                rho = rho - (self.R.unsqueeze(0) * p_safe / (1 - p_safe)).sum(-1)
            if sigma_j is not None:
                sig_max = float(sigma_max) if sigma_max is not None else 200.0
                rho = rho - (self.J.unsqueeze(0) * sigma_j ** 2 / sig_max ** 2).sum(-1)

        return {
            'u': u,
            'rho': rho,
            'K': self.K,
            'R': self.R,
            'J': self.J,
        }

    def get_coupling_constants(self) -> dict:
        return {'K': self.K, 'R': self.R, 'J': self.J,
                'K_mean': self.K.mean().item(), 'R_mean': self.R.mean().item(),
                'J_mean': self.J.mean().item()}
