#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/10/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B6: Vanilla Transformer Baseline

Standard transformer without physics mask or causal mask.
Uses standard self-attention on concatenated energy+communication features.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class VanillaTransformer(nn.Module):
    """
    B6: Vanilla Transformer Baseline

    Standard transformer encoder applied to bus features.
    No physics-informed attention mask, no graph structure.

    Key limitation: No physics mask, no causal constraints.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators
        self.embed_dim = embed_dim

        # Input projection
        input_dim = energy_input_dim + comm_input_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, n_buses, embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.control_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_generators * 2),
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

        # Concatenate and project
        x = torch.cat([energy_x, comm_x], dim=-1)  # [batch, n_buses, input_dim]
        x = self.input_proj(x)  # [batch, n_buses, embed_dim]

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer (no mask - vanilla attention)
        h = self.transformer(x)  # [batch, n_buses, embed_dim]

        # Global pooling (mean over buses)
        h_pooled = h.mean(dim=1)  # [batch, embed_dim]

        # Control output
        u = self.control_head(h_pooled)

        # Coupling constants
        K = torch.exp(self.log_K)

        # Stability margin
        delay_contribution = (K.unsqueeze(0) * tau / tau_max.unsqueeze(0)).sum(dim=-1)
        rho = torch.abs(lambda_min_0.squeeze()) - delay_contribution

        return {
            'u': u,
            'rho': rho,
            'K': K,
            'h': h_pooled,
        }

    def get_coupling_constants(self) -> torch.Tensor:
        return torch.exp(self.log_K)
