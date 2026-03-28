#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/03/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B5: CNN Joint Baseline

Convolutional approach treating bus features as 1D signal.
Joint processing of energy and communication via 1D CNN.
"""

import torch
import torch.nn as nn
from typing import Dict


class CNNJoint(nn.Module):
    """
    B5: CNN Joint Baseline

    Treats bus features as a 1D signal and applies convolutions.
    Concatenates energy and communication as channels.

    Key limitation: Assumes local patterns, no graph structure.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators

        input_channels = energy_input_dim + comm_input_dim

        # Build CNN layers
        layers = []
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = hidden_channels * (2 ** min(i, 2))  # Gradually increase channels
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.final_channels = in_ch

        # Global pooling + output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.control_head = nn.Sequential(
            nn.Linear(self.final_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, n_generators * 2),
        )

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

        # Concatenate and reshape for conv1d: [batch, channels, length]
        x = torch.cat([energy_x, comm_x], dim=-1)  # [batch, n_buses, channels]
        x = x.permute(0, 2, 1)  # [batch, channels, n_buses]

        # Apply convolutions
        h = self.conv_layers(x)  # [batch, final_channels, n_buses]

        # Global pooling
        h = self.global_pool(h).squeeze(-1)  # [batch, final_channels]

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
