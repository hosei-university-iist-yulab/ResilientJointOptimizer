#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/07/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B2: MLP Joint Baseline

Simple feedforward network that jointly processes energy and communication.
No graph structure, no attention.
"""

import torch
import torch.nn as nn
from typing import Dict


class MLPJoint(nn.Module):
    """
    B2: MLP Joint Baseline

    Simple concatenation of energy and communication features
    followed by MLP layers.

    Key limitation: Ignores graph topology entirely.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators

        # Total input: energy + communication features flattened
        total_input = n_buses * (energy_input_dim + comm_input_dim)

        # Build MLP
        layers = []
        in_dim = total_input
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.control_head = nn.Linear(hidden_dim, n_generators * 2)  # P and Q

        # Auto-scale K initialization based on grid eigenvalue
        if lambda_min_0 is not None and k_init_scale == 0.1:
            from src.models.coupling import compute_k_init_scale
            k_init_scale = compute_k_init_scale(n_generators, lambda_min_0)

        # Learnable K
        self.log_K = nn.Parameter(torch.ones(n_generators) * torch.log(torch.tensor(k_init_scale)))

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
            energy_x: [batch, n_buses, 5] or [batch*n_buses, 5]
            comm_x: [batch, n_buses, 3] or [batch*n_buses, 3]
            tau: [batch, n_generators]
            tau_max: [n_generators]
            lambda_min_0: [batch] or scalar

        Returns:
            Dict with 'u', 'rho', 'K'
        """
        # Reshape if flattened
        if energy_x.dim() == 2:
            batch_size = energy_x.shape[0] // self.n_buses
            energy_x = energy_x.view(batch_size, self.n_buses, -1)
            comm_x = comm_x.view(batch_size, self.n_buses, -1)

        batch_size = energy_x.shape[0]

        # Concatenate and flatten
        x = torch.cat([energy_x, comm_x], dim=-1)  # [batch, n_buses, energy+comm]
        x = x.view(batch_size, -1)  # [batch, n_buses * (energy+comm)]

        # Encode
        h = self.encoder(x)

        # Control output
        u = self.control_head(h)

        # Coupling constants
        K = torch.exp(self.log_K)

        # Stability margin
        delay_contribution = (K.unsqueeze(0) * tau / tau_max.unsqueeze(0)).sum(dim=-1)
        rho = torch.abs(lambda_min_0.squeeze()) - delay_contribution

        return {
            'u': u,
            'rho': rho,
            'K': K,
        }

    def get_coupling_constants(self) -> torch.Tensor:
        return torch.exp(self.log_K)
