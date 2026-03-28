#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/04/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B9: DeepOPF Baseline (V2)

Addresses Q5.2: Faithful reproduction of Pan et al. (2021),
"DeepOPF: A Feasibility-Optimized Deep Neural Network Approach for
AC Optimal Power Flow Problems."

Architecture: 3-layer MLP [400, 400, 400] with ReLU.
Input: Load vector. Output: Generator dispatch + stability margin.

The original DeepOPF solves OPF only; we extend it with coupling
constants for stability comparison.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..models.coupling import LearnableCouplingConstants


class DeepOPF(nn.Module):
    """
    B9: DeepOPF — MLP-based AC OPF solver.

    Faithful to Pan et al. (2021):
    - 3-layer MLP with hidden size 400
    - ReLU activations
    - Input: concatenated load and delay features
    - Output: generator setpoints

    Extended with learnable coupling constants for stability evaluation.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        hidden_dim: int = 400,
        num_layers: int = 3,
        k_init_scale: float = 0.1,
        dropout: float = 0.1,
        lambda_min_0: float = None,
    ):
        super().__init__()

        self.n_buses = n_buses
        self.n_generators = n_generators

        # Input: energy features [n_buses * 5] + comm features [n_buses * 3]
        input_dim = n_buses * 5 + n_buses * 3

        # Build MLP
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, n_generators * 2))
        self.mlp = nn.Sequential(*layers)

        # Auto-scale K initialization based on grid eigenvalue
        if lambda_min_0 is not None and k_init_scale == 0.1:
            from src.models.coupling import compute_k_init_scale
            k_init_scale = compute_k_init_scale(n_generators, lambda_min_0)

        # Coupling constants for stability
        self.coupling = LearnableCouplingConstants(n_generators, k_init_scale)

    def forward(
        self,
        energy_x: torch.Tensor,
        comm_x: torch.Tensor,
        tau: torch.Tensor,
        tau_max: torch.Tensor,
        lambda_min_0: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            energy_x: [batch, n_buses, 5]
            comm_x: [batch, n_buses, 3]
            tau: [batch, n_gen]
            tau_max: [n_gen]
            lambda_min_0: [batch]

        Returns:
            Dict with u, rho, K
        """
        if energy_x.dim() == 3:
            batch_size = energy_x.shape[0]
        else:
            batch_size = 1
            energy_x = energy_x.unsqueeze(0)
            comm_x = comm_x.unsqueeze(0)

        # Flatten features
        x = torch.cat([
            energy_x.reshape(batch_size, -1),
            comm_x.reshape(batch_size, -1),
        ], dim=-1)

        # MLP forward
        u = self.mlp(x)

        # Stability margin
        K = self.coupling()
        tau_normalized = tau / tau_max.unsqueeze(0) if tau_max.dim() == 1 else tau / tau_max
        delay_contribution = (K.unsqueeze(0) * tau_normalized).sum(dim=-1)
        rho = torch.abs(lambda_min_0) - delay_contribution

        return {
            'u': u,
            'rho': rho,
            'K': K,
        }

    def get_coupling_constants(self) -> torch.Tensor:
        return self.coupling()
