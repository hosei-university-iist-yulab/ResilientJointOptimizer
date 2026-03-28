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
B3: GNN-only Baseline

Uses graph neural networks but no attention mechanism.
Joint processing of energy and communication via message passing only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SimpleGNNLayer(nn.Module):
    """Simple message passing layer without attention."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] edge indices

        Returns:
            out: [N, out_dim]
        """
        row, col = edge_index
        N = x.size(0)

        # Self transformation
        out = self.lin_self(x)

        # Neighbor aggregation (mean)
        neigh_features = x[col]  # [E, in_dim]
        neigh_transformed = self.lin_neigh(neigh_features)  # [E, out_dim]

        # Aggregate by target node
        agg = torch.zeros(N, neigh_transformed.size(-1), device=x.device)
        agg.scatter_add_(0, row.unsqueeze(-1).expand(-1, neigh_transformed.size(-1)), neigh_transformed)

        # Count neighbors for mean
        count = torch.zeros(N, device=x.device)
        count.scatter_add_(0, row, torch.ones(edge_index.size(1), device=x.device))
        count = count.clamp(min=1).unsqueeze(-1)

        out = out + agg / count
        return out


class GNNOnly(nn.Module):
    """
    B3: GNN-only Baseline

    Uses graph structure but no attention mechanism.
    Simple message passing for both domains.

    Key limitation: No cross-domain attention, no physics mask.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators
        self.hidden_dim = hidden_dim

        # Input projections
        self.energy_proj = nn.Linear(energy_input_dim, hidden_dim)
        self.comm_proj = nn.Linear(comm_input_dim, hidden_dim)

        # GNN layers for energy
        self.energy_gnns = nn.ModuleList([
            SimpleGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.energy_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # GNN layers for communication
        self.comm_gnns = nn.ModuleList([
            SimpleGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.comm_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Fusion and output
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.control_head = nn.Linear(hidden_dim, n_generators * 2)

        self.dropout = nn.Dropout(dropout)

        # Auto-scale K initialization based on grid eigenvalue
        if lambda_min_0 is not None and k_init_scale == 0.1:
            from src.models.coupling import compute_k_init_scale
            k_init_scale = compute_k_init_scale(n_generators, lambda_min_0)

        # Learnable K
        self.log_K = nn.Parameter(torch.ones(n_generators) * torch.log(torch.tensor(k_init_scale)))

    def forward(
        self,
        energy_x: torch.Tensor,
        energy_edge_index: torch.Tensor,
        comm_x: torch.Tensor,
        comm_edge_index: torch.Tensor,
        tau: torch.Tensor,
        tau_max: torch.Tensor,
        lambda_min_0: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        """
        # Project inputs
        h_E = self.energy_proj(energy_x)
        h_I = self.comm_proj(comm_x)

        # Energy GNN
        for gnn, norm in zip(self.energy_gnns, self.energy_norms):
            h_E_new = gnn(h_E, energy_edge_index)
            h_E_new = norm(h_E_new)
            h_E_new = F.relu(h_E_new)
            h_E_new = self.dropout(h_E_new)
            h_E = h_E + h_E_new  # Residual

        # Communication GNN
        for gnn, norm in zip(self.comm_gnns, self.comm_norms):
            h_I_new = gnn(h_I, comm_edge_index)
            h_I_new = norm(h_I_new)
            h_I_new = F.relu(h_I_new)
            h_I_new = self.dropout(h_I_new)
            h_I = h_I + h_I_new  # Residual

        # Global pooling
        if batch is not None:
            # Batch-wise pooling
            batch_size = batch.max().item() + 1
            h_E_pooled = torch.zeros(batch_size, self.hidden_dim, device=h_E.device)
            h_I_pooled = torch.zeros(batch_size, self.hidden_dim, device=h_I.device)
            h_E_pooled.scatter_add_(0, batch.unsqueeze(-1).expand(-1, self.hidden_dim), h_E)
            h_I_pooled.scatter_add_(0, batch.unsqueeze(-1).expand(-1, self.hidden_dim), h_I)
            count = torch.bincount(batch).float().unsqueeze(-1)
            h_E_pooled = h_E_pooled / count
            h_I_pooled = h_I_pooled / count
        else:
            h_E_pooled = h_E.mean(dim=0, keepdim=True)
            h_I_pooled = h_I.mean(dim=0, keepdim=True)

        # Fusion (simple concatenation, no attention)
        h_joint = torch.cat([h_E_pooled, h_I_pooled], dim=-1)
        h_joint = F.relu(self.fusion(h_joint))

        # Control output
        u = self.control_head(h_joint)

        # Coupling constants
        K = torch.exp(self.log_K)

        # Stability margin
        batch_size = tau.shape[0]
        delay_contribution = (K.unsqueeze(0) * tau / tau_max.unsqueeze(0)).sum(dim=-1)
        rho = torch.abs(lambda_min_0.squeeze()) - delay_contribution

        return {
            'u': u,
            'rho': rho,
            'K': K,
            'h_E': h_E_pooled,
            'h_I': h_I_pooled,
        }

    def get_coupling_constants(self) -> torch.Tensor:
        return torch.exp(self.log_K)
