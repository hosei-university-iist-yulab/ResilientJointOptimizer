#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/31/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Graph Neural Network Encoders for Energy and Communication Domains

Implements physics-informed message passing for power system graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def _try_import_pyg():
    """Attempt to import torch_geometric with error handling."""
    import sys
    import io
    from contextlib import redirect_stderr

    # Suppress stderr during import attempt
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv, MessagePassing
        from torch_geometric.utils import add_self_loops, softmax
        sys.stderr = old_stderr
        return True, GATConv, GCNConv, SAGEConv, MessagePassing
    except Exception as e:
        sys.stderr = old_stderr
        return False, None, None, None, None


HAS_PYG, GATConv, GCNConv, SAGEConv, MessagePassing = _try_import_pyg()

if not HAS_PYG:
    import sys
    if not getattr(sys, '_torch_geometric_warned', False):
        sys._torch_geometric_warned = True
        print("Note: torch_geometric not available. Using fallback GNN implementation.")


class PhysicsMessagePassing(nn.Module):
    """
    Custom message passing that includes physics quantities.

    Messages include:
    - Node features h_j
    - Power flow P_ij on edge (i,j)
    - Communication delay τ_ij on edge (i,j)
    - Local stability margin ρ_local
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 3,  # [P_ij, tau_ij, rho_local]
        heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        # Node transformation
        self.lin_node = nn.Linear(in_channels, out_channels * heads, bias=False)

        # Edge transformation (physics features)
        self.lin_edge = nn.Linear(edge_dim, out_channels * heads, bias=False)

        # Attention
        self.att = nn.Parameter(torch.zeros(1, heads, out_channels))

        # Output
        self.lin_out = nn.Linear(out_channels * heads, out_channels)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.lin_out.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (P_ij, tau_ij, rho_local)

        Returns:
            out: Updated node features [N, out_channels]
        """
        N = x.size(0)
        H, C = self.heads, self.out_channels

        # Transform nodes
        x_transformed = self.lin_node(x).view(N, H, C)  # [N, H, C]

        # Get source and target node features
        row, col = edge_index  # row = source, col = target
        x_i = x_transformed[row]  # [E, H, C]
        x_j = x_transformed[col]  # [E, H, C]

        # Transform edge features if available
        if edge_attr is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, H, C)  # [E, H, C]
        else:
            edge_feat = torch.zeros(edge_index.size(1), H, C, device=x.device)

        # Message: combine source node and edge features
        msg = x_j + edge_feat  # [E, H, C]

        # Attention scores
        alpha = (msg * self.att).sum(dim=-1)  # [E, H]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Softmax over incoming edges
        alpha = self._softmax(alpha, col, N)
        alpha = self.dropout(alpha)

        # Aggregate messages
        out = torch.zeros(N, H, C, device=x.device)
        out.scatter_add_(0, col.view(-1, 1, 1).expand(-1, H, C), alpha.unsqueeze(-1) * msg)

        # Combine heads
        out = out.view(N, H * C)
        out = self.lin_out(out)

        return out

    def _softmax(self, alpha: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute softmax over incoming edges."""
        alpha_max = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_max.scatter_reduce_(0, index.view(-1, 1).expand(-1, alpha.size(1)), alpha, reduce='amax')
        alpha = alpha - alpha_max[index]
        alpha = torch.exp(alpha)

        alpha_sum = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_sum.scatter_add_(0, index.view(-1, 1).expand(-1, alpha.size(1)), alpha)
        alpha = alpha / (alpha_sum[index] + 1e-8)

        return alpha


class EnergyGNN(nn.Module):
    """
    GNN encoder for energy (power system) domain.

    Input features per bus: [P, Q, V, θ, ω]
    Output: Encoded node representations h_E
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        gnn_type: str = "GAT",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else output_dim

            if HAS_PYG:
                if gnn_type == "GAT":
                    # For GAT: always concat, then project if needed
                    conv = GATConv(
                        in_ch, out_ch // heads, heads=heads,
                        dropout=dropout, concat=True
                    )
                elif gnn_type == "GCN":
                    conv = GCNConv(in_ch, out_ch)
                elif gnn_type == "SAGE":
                    conv = SAGEConv(in_ch, out_ch)
                else:
                    conv = GCNConv(in_ch, out_ch)
            else:
                # Fallback: simple linear layer with neighborhood aggregation
                conv = nn.Linear(in_ch, out_ch)

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(out_ch if i < num_layers - 1 else output_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Optional edge features [E, edge_dim]

        Returns:
            h: Encoded node features [N, output_dim]
        """
        # Input projection
        h = self.input_proj(x)

        # GNN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if HAS_PYG:
                h_new = conv(h, edge_index)
            else:
                # Fallback: simple aggregation
                h_new = conv(h)

            h_new = norm(h_new)

            if i < self.num_layers - 1:
                h_new = F.relu(h_new)
                h_new = self.dropout(h_new)

            # Residual connection if dimensions match
            if h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

        return h


class CommunicationGNN(nn.Module):
    """
    GNN encoder for communication network domain.

    Input features per link: [τ, R, B] (delay, rate, bandwidth)
    Output: Encoded link/node representations h_I
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection for edge features
        self.edge_proj = nn.Linear(input_dim, hidden_dim)

        # Node initialization (from aggregated edge features)
        self.node_init = nn.Linear(hidden_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else output_dim

            if HAS_PYG:
                conv = GATConv(in_ch, out_ch // heads, heads=heads, dropout=dropout)
            else:
                conv = nn.Linear(in_ch, out_ch)

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(out_ch))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, input_dim] or edge features [E, input_dim]
            edge_index: Edge indices [2, E]
            num_nodes: Number of nodes (optional, inferred from x if node features)

        Returns:
            h: Encoded node features [N, output_dim]
        """
        # Determine if x is node features or edge features
        if num_nodes is None:
            num_nodes = x.size(0)

        # If x appears to be node features (N matches num_nodes)
        if x.size(0) == num_nodes:
            # Direct node feature projection
            h = self.edge_proj(x)  # Reuse edge_proj for input projection
            h = self.node_init(h)
        else:
            # x is edge features, aggregate to nodes
            edge_h = self.edge_proj(x)  # [E, hidden_dim]

            # Initialize node features by aggregating incident edge features
            row, col = edge_index
            h = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
            h.scatter_add_(0, col.view(-1, 1).expand(-1, self.hidden_dim), edge_h)

            # Count edges per node for mean aggregation
            edge_count = torch.zeros(num_nodes, device=x.device)
            edge_count.scatter_add_(0, col, torch.ones(edge_index.size(1), device=x.device))
            edge_count = edge_count.clamp(min=1).unsqueeze(-1)
            h = h / edge_count

            h = self.node_init(h)

        # GNN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if HAS_PYG:
                h_new = conv(h, edge_index)
            else:
                h_new = conv(h)

            h_new = norm(h_new)

            if i < self.num_layers - 1:
                h_new = F.relu(h_new)
                h_new = self.dropout(h_new)

            if h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

        return h


class DualDomainGNN(nn.Module):
    """
    Combined GNN that processes both energy and communication domains.

    Returns separate embeddings for each domain that can be used
    for cross-domain attention.
    """

    def __init__(
        self,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.energy_gnn = EnergyGNN(
            input_dim=energy_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

        self.comm_gnn = CommunicationGNN(
            input_dim=comm_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

    def forward(
        self,
        x_energy: torch.Tensor,
        edge_index_energy: torch.Tensor,
        x_comm: torch.Tensor,
        edge_index_comm: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both domains.

        Args:
            x_energy: Energy node features [N, 5]
            edge_index_energy: Power system topology [2, E_power]
            x_comm: Communication node features [N, 3]
            edge_index_comm: Communication topology [2, E_comm]
            batch: Batch assignment tensor [N] (optional)

        Returns:
            h_E: Energy embeddings [N, output_dim]
            h_I: Communication embeddings [N, output_dim]
        """
        # Process energy domain
        h_E = self.energy_gnn(x_energy, edge_index_energy)

        # Process communication domain
        # Use node features directly (not edge features)
        num_nodes = x_comm.size(0)
        h_I = self.comm_gnn(x_comm, edge_index_comm, num_nodes)

        return h_E, h_I
