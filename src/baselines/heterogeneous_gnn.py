#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/05/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B8: Heterogeneous GNN Baseline (V2)

Addresses Q3.1: Single unified GNN over typed graph (energy + comm nodes),
as an alternative to the dual-domain approach.

Architecture: HeteroGNN with typed nodes (energy, comm) and typed edges
(power_line, comm_link, cross_domain). Uses message passing across all
node types simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..models.coupling import LearnableCouplingConstants


class HeteroGNNLayer(nn.Module):
    """Single layer of heterogeneous message passing."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # Separate message functions per edge type
        self.msg_power = nn.Linear(hidden_dim * 2, hidden_dim)
        self.msg_comm = nn.Linear(hidden_dim * 2, hidden_dim)
        self.msg_cross = nn.Linear(hidden_dim * 2, hidden_dim)

        # Update function (shared)
        self.update_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with typed edges.

        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edges [2, E]
            edge_type: Edge type labels [E] (0=power, 1=comm, 2=cross)

        Returns:
            Updated node features [N, hidden_dim]
        """
        src, dst = edge_index
        N = x.shape[0]

        # Compute messages per edge type
        msg_input = torch.cat([x[src], x[dst]], dim=-1)

        messages = torch.zeros_like(x[src])
        for etype, msg_fn in enumerate([self.msg_power, self.msg_comm, self.msg_cross]):
            mask = edge_type == etype
            if mask.any():
                messages[mask] = msg_fn(msg_input[mask])

        # Aggregate (mean)
        agg = torch.zeros(N, messages.shape[-1], device=x.device)
        count = torch.zeros(N, 1, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones_like(dst.unsqueeze(-1).float()))
        count = count.clamp(min=1)
        agg = agg / count

        # Update
        x_updated = self.update_fn(torch.cat([x, agg], dim=-1))
        return x + x_updated  # Residual


class HeterogeneousGNN(nn.Module):
    """
    B8: Heterogeneous GNN over unified typed graph.

    Combines energy and communication nodes in a single graph with
    typed edges for cross-domain message passing.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        k_init_scale: float = 0.1,
        dropout: float = 0.1,
        lambda_min_0: float = None,
    ):
        super().__init__()

        self.n_buses = n_buses
        self.n_generators = n_generators
        self.hidden_dim = hidden_dim

        # Input projections for each node type
        self.energy_embed = nn.Linear(5, hidden_dim)  # [P,Q,V,theta,omega]
        self.comm_embed = nn.Linear(3, hidden_dim)    # [tau,R,B]

        # GNN layers
        self.layers = nn.ModuleList([
            HeteroGNNLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])

        # Output: decode control actions from generator nodes
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_generators * 2),
        )

        # Auto-scale K initialization based on grid eigenvalue
        if lambda_min_0 is not None and k_init_scale == 0.1:
            from src.models.coupling import compute_k_init_scale
            k_init_scale = compute_k_init_scale(n_generators, lambda_min_0)

        # Coupling constants
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
            energy_x: [batch, n_buses, 5] or [N, 5]
            comm_x: [batch, n_buses, 3] or [N, 3]
            tau, tau_max, lambda_min_0: Stability parameters

        Returns:
            Dict with u, rho, K
        """
        if energy_x.dim() == 3:
            batch_size = energy_x.shape[0]
            n_nodes = energy_x.shape[1]
        else:
            batch_size = 1
            n_nodes = energy_x.shape[0]
            energy_x = energy_x.unsqueeze(0)
            comm_x = comm_x.unsqueeze(0)

        device = energy_x.device

        # Create unified graph: 2N nodes (N energy + N comm)
        # Embed both domains
        h_E = self.energy_embed(energy_x)  # [batch, N, hidden]
        h_I = self.comm_embed(comm_x)      # [batch, N, hidden]

        # Concatenate into unified node features
        h = torch.cat([h_E, h_I], dim=1)  # [batch, 2N, hidden]

        # Build unified edge index with types
        # Type 0: power edges (within energy nodes)
        # Type 1: comm edges (within comm nodes, offset by N)
        # Type 2: cross-domain edges (energy_i <-> comm_i)
        power_edges = torch.arange(n_nodes - 1, device=device)
        power_edge_index = torch.stack([power_edges, power_edges + 1])
        power_edge_index = torch.cat([
            power_edge_index,
            power_edge_index.flip(0),
        ], dim=1)

        comm_offset = n_nodes
        comm_edge_index = power_edge_index.clone()
        comm_edge_index = comm_edge_index + comm_offset

        cross_nodes = torch.arange(n_nodes, device=device)
        cross_edge_index = torch.stack([
            cross_nodes, cross_nodes + comm_offset,
        ])
        cross_edge_index = torch.cat([
            cross_edge_index,
            cross_edge_index.flip(0),
        ], dim=1)

        edge_index = torch.cat([
            power_edge_index, comm_edge_index, cross_edge_index,
        ], dim=1)

        edge_type = torch.cat([
            torch.zeros(power_edge_index.shape[1], dtype=torch.long, device=device),
            torch.ones(comm_edge_index.shape[1], dtype=torch.long, device=device),
            torch.full((cross_edge_index.shape[1],), 2, dtype=torch.long, device=device),
        ])

        # Process each sample in batch
        outputs_u = []
        for b in range(batch_size):
            h_b = h[b]  # [2N, hidden]

            for layer in self.layers:
                h_b = layer(h_b, edge_index, edge_type)

            # Pool energy nodes for control decoding
            h_energy = h_b[:n_nodes].mean(dim=0, keepdim=True)  # [1, hidden]
            u_b = self.decoder(h_energy)  # [1, n_gen*2]
            outputs_u.append(u_b)

        u = torch.cat(outputs_u, dim=0)  # [batch, n_gen*2]

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
