#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/01/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Joint Energy-Information Co-Optimizer

The main model combining:
1. Dual-domain GNN encoders (Energy + Communication)
2. Hierarchical attention with causal and physics masks
3. Learnable coupling constants (K_i)
4. Control action decoder

Architecture:
    [Energy Graph] ─────► EnergyGNN ──────┐
                                          ├──► HierarchicalAttention ──► Decoder ──► u*
    [Comm Graph] ────────► CommGNN ───────┘                                     ↓
                                                                          [K_i, ρ(τ)]

Key innovation: First transformer that jointly optimizes energy dispatch
and communication scheduling with formal stability guarantees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .gnn import DualDomainGNN
from .attention import HierarchicalAttention
from .coupling import LearnableCouplingConstants


class ControlDecoder(nn.Module):
    """
    Decodes fused embeddings into control actions.

    Outputs:
        u: Control actions [P_gen, Q_gen] for controllable generators
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = embed_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

        # Output scaling for physical bounds
        self.register_buffer('output_scale', torch.ones(output_dim))
        self.register_buffer('output_bias', torch.zeros(output_dim))

    def set_bounds(self, p_min: torch.Tensor, p_max: torch.Tensor,
                   q_min: torch.Tensor, q_max: torch.Tensor):
        """Set physical bounds for control actions."""
        # Scale output to [p_min, p_max] and [q_min, q_max]
        n_gen = len(p_min)

        scale = torch.cat([
            (p_max - p_min) / 2,
            (q_max - q_min) / 2,
        ])
        bias = torch.cat([
            (p_max + p_min) / 2,
            (q_max + q_min) / 2,
        ])

        self.output_scale = scale
        self.output_bias = bias

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to control actions.

        Args:
            h: Fused embeddings [batch, N, embed_dim]

        Returns:
            u: Control actions [batch, output_dim]
        """
        # Global pooling over nodes
        h_global = h.mean(dim=1)  # [batch, embed_dim]

        # Decode
        u_raw = self.decoder(h_global)  # [batch, output_dim]

        # Apply tanh and scale to physical bounds
        u = torch.tanh(u_raw) * self.output_scale + self.output_bias

        return u


class DelayPredictor(nn.Module):
    """
    Predicts optimal communication delays for stability.

    Uses learned K_i to find τ* that maximizes stability margin.
    """

    def __init__(self, embed_dim: int, n_generators: int):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_generators),
            nn.Softplus(),  # Delays must be positive
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict optimal delays.

        Args:
            h: Fused embeddings [batch, N, embed_dim]

        Returns:
            tau: Predicted delays [batch, n_gen]
        """
        h_global = h.mean(dim=1)
        return self.predictor(h_global)


class JointOptimizer(nn.Module):
    """
    Main model for Energy-Information Co-Optimization.

    Combines all components into end-to-end trainable system.

    Key features:
    1. Dual-domain GNN encoders for power and communication graphs
    2. Hierarchical attention with physics and causal masks
    3. Learnable coupling constants K_i with positivity guarantee
    4. Control decoder with physical bound enforcement
    5. Stability margin computation for loss
    """

    def __init__(
        self,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        gnn_layers: int = 3,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        physics_gamma: float = 1.0,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
        learnable_k: bool = True,
        adaptive_gamma: bool = False,
        use_physics_mask: bool = True,
        use_causal_mask: bool = True,
        use_cross_attention: bool = True,
    ):
        """
        Initialize the Joint Optimizer.

        Args:
            n_generators: Number of generators (for K_i)
            energy_input_dim: Dimension of energy node features [P,Q,V,θ,ω]
            comm_input_dim: Dimension of comm node features [τ,R,B]
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension for decoder
            num_heads: Number of attention heads
            gnn_layers: Number of GNN layers
            decoder_layers: Number of decoder layers
            dropout: Dropout probability
            physics_gamma: Strength of physics mask
            k_init_scale: Initial scale for K_i
            learnable_k: Whether K_i are learnable
            adaptive_gamma: If True, physics mask gamma is learnable (V2, Q3.3)
            use_physics_mask: If False, disable physics mask (V2 ablation, Q3.2)
            use_causal_mask: If False, disable causal mask (V2 ablation, Q3.2)
            use_cross_attention: If False, disable cross-domain attention (V2 ablation, Q3.2)
        """
        super().__init__()

        self.n_generators = n_generators
        self.embed_dim = embed_dim
        self.use_cross_attention = use_cross_attention

        # Dual-domain GNN encoders
        self.gnn = DualDomainGNN(
            energy_input_dim=energy_input_dim,
            comm_input_dim=comm_input_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        # Hierarchical attention with masks (V2: adaptive gamma + ablation flags)
        effective_gamma = physics_gamma if use_physics_mask else 0.0
        self.attention = HierarchicalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            gamma=effective_gamma,
            adaptive_gamma=adaptive_gamma and use_physics_mask,
        )
        self.use_causal_mask = use_causal_mask

        # Auto-scale K init if lambda_min_0 is provided
        if lambda_min_0 is not None and k_init_scale == 0.1:
            from .coupling import compute_k_init_scale
            k_init_scale = compute_k_init_scale(n_generators, lambda_min_0)

        # Learnable coupling constants
        self.coupling = LearnableCouplingConstants(
            n_generators=n_generators,
            init_scale=k_init_scale,
            learnable=learnable_k,
        )

        # Default tau_max (can be updated)
        self.register_buffer('tau_max_default', torch.ones(n_generators) * 500.0)  # ms

        # Control decoder (output: P_gen and Q_gen)
        self.control_decoder = ControlDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=n_generators * 2,  # P and Q for each generator
            num_layers=decoder_layers,
            dropout=dropout,
        )

        # Delay predictor (optional)
        self.delay_predictor = DelayPredictor(embed_dim, n_generators)

        # Store config for serialization
        self.config = {
            'n_generators': n_generators,
            'energy_input_dim': energy_input_dim,
            'comm_input_dim': comm_input_dim,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'gnn_layers': gnn_layers,
            'decoder_layers': decoder_layers,
            'dropout': dropout,
            'physics_gamma': physics_gamma,
            'k_init_scale': k_init_scale,
            'learnable_k': learnable_k,
            'adaptive_gamma': adaptive_gamma,
            'use_physics_mask': use_physics_mask,
            'use_causal_mask': use_causal_mask,
            'use_cross_attention': use_cross_attention,
        }

    def forward(
        self,
        energy_x: torch.Tensor,
        energy_edge_index: torch.Tensor,
        comm_x: torch.Tensor,
        comm_edge_index: torch.Tensor,
        tau: torch.Tensor,
        tau_max: torch.Tensor,
        lambda_min_0: torch.Tensor,
        impedance_matrix: Optional[torch.Tensor] = None,
        dag_edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the joint optimizer.

        Args:
            energy_x: Energy node features [N, energy_input_dim]
            energy_edge_index: Power grid edges [2, E_power]
            comm_x: Communication node features [N, comm_input_dim]
            comm_edge_index: Communication network edges [2, E_comm]
            tau: Communication delays [batch, n_gen]
            tau_max: Maximum tolerable delays [n_gen]
            lambda_min_0: Baseline minimum eigenvalue [batch] or scalar
            impedance_matrix: Impedance matrix for physics mask [N, N]
            dag_edge_index: DAG edges for causal mask [2, E_dag]
            batch: Batch assignment for nodes [N]

        Returns:
            Dict containing:
                - u: Control actions [batch, n_gen * 2]
                - rho: Stability margin [batch]
                - K: Coupling constants [n_gen]
                - tau_pred: Predicted optimal delays [batch, n_gen]
                - h_E: Energy embeddings [batch, N, embed_dim]
                - h_I: Communication embeddings [batch, N, embed_dim]
                - attn_info: Attention weights
        """
        # Get coupling constants
        K = self.coupling()

        # Compute stability margin: ρ(τ) = |λ_min(0)| - Σ_i (K_i · τ_i / τ_max,i)
        tau_normalized = tau / tau_max.unsqueeze(0) if tau_max.dim() == 1 else tau / tau_max
        delay_contribution = (K.unsqueeze(0) * tau_normalized).sum(dim=-1)
        rho = torch.abs(lambda_min_0) - delay_contribution

        # Encode both domains with GNN
        h_E, h_I = self.gnn(
            energy_x, energy_edge_index,
            comm_x, comm_edge_index,
            batch=batch,
        )

        # Handle batching - reshape for attention
        if batch is not None:
            # Group by batch
            batch_size = batch.max().item() + 1
            nodes_per_graph = h_E.shape[0] // batch_size
            h_E = h_E.view(batch_size, nodes_per_graph, -1)
            h_I = h_I.view(batch_size, nodes_per_graph, -1)
        else:
            # Single graph - add batch dimension
            h_E = h_E.unsqueeze(0)
            h_I = h_I.unsqueeze(0)

        # Hierarchical attention with masks (V2: ablation flags)
        # Auto-disable attention for large sequences (>= 1000 nodes) to avoid OOM
        use_attn = self.use_cross_attention and h_E.shape[1] < 1000
        if not use_attn:
            # Skip cross-domain attention; concatenate and project
            h_fused = (h_E + h_I) / 2
            attn_info = None
        else:
            dag_edges = dag_edge_index if self.use_causal_mask else None
            h_fused, attn_info = self.attention(
                h_E, h_I,
                dag_edge_index=dag_edges,
                impedance_matrix=impedance_matrix,
            )

        # Decode control actions
        u = self.control_decoder(h_fused)

        # Predict optimal delays
        tau_pred = self.delay_predictor(h_fused)

        return {
            'u': u,
            'rho': rho,
            'K': K,
            'tau_pred': tau_pred,
            'h_E': h_E,
            'h_I': h_I,
            'h_fused': h_fused,
            'attn_info': attn_info,
        }

    def compute_control_only(
        self,
        energy_x: torch.Tensor,
        energy_edge_index: torch.Tensor,
        comm_x: torch.Tensor,
        comm_edge_index: torch.Tensor,
        impedance_matrix: Optional[torch.Tensor] = None,
        dag_edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fast forward pass for inference (control only).

        Returns only the control actions without stability computation.
        """
        # Encode both domains
        h_E, h_I = self.gnn(
            energy_x, energy_edge_index,
            comm_x, comm_edge_index,
            batch=batch,
        )

        # Handle batching
        if batch is not None:
            batch_size = batch.max().item() + 1
            nodes_per_graph = h_E.shape[0] // batch_size
            h_E = h_E.view(batch_size, nodes_per_graph, -1)
            h_I = h_I.view(batch_size, nodes_per_graph, -1)
        else:
            h_E = h_E.unsqueeze(0)
            h_I = h_I.unsqueeze(0)

        # Attention and decode
        h_fused, _ = self.attention(
            h_E, h_I,
            dag_edge_index=dag_edge_index,
            impedance_matrix=impedance_matrix,
        )

        return self.control_decoder(h_fused)

    def get_coupling_constants(self) -> torch.Tensor:
        """Return the current coupling constants K_i."""
        return self.coupling()

    def get_stability_margin(
        self,
        tau: torch.Tensor,
        tau_max: torch.Tensor,
        lambda_min_0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute stability margin for given delays."""
        K = self.coupling()
        tau_normalized = tau / tau_max.unsqueeze(0) if tau_max.dim() == 1 else tau / tau_max
        delay_contribution = (K.unsqueeze(0) * tau_normalized).sum(dim=-1)
        return torch.abs(lambda_min_0) - delay_contribution

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'JointOptimizer':
        """Create model from configuration dictionary."""
        return cls(**config)

    def extra_repr(self) -> str:
        return (
            f"n_generators={self.n_generators}, "
            f"embed_dim={self.embed_dim}"
        )


class JointOptimizerLite(nn.Module):
    """
    Lightweight version for inference/deployment.

    Removes training-specific components and uses fused operations.
    """

    def __init__(self, full_model: JointOptimizer):
        super().__init__()

        self.gnn = full_model.gnn
        self.attention = full_model.attention
        self.control_decoder = full_model.control_decoder

        # Freeze coupling constants
        self.register_buffer('K', full_model.coupling().detach())

        # Compile for faster inference
        self.gnn.eval()
        self.attention.eval()
        self.control_decoder.eval()

    @torch.no_grad()
    def forward(
        self,
        energy_x: torch.Tensor,
        energy_edge_index: torch.Tensor,
        comm_x: torch.Tensor,
        comm_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Fast inference forward pass."""
        h_E, h_I = self.gnn(
            energy_x, energy_edge_index,
            comm_x, comm_edge_index,
        )

        h_E = h_E.unsqueeze(0)
        h_I = h_I.unsqueeze(0)

        h_fused, _ = self.attention(h_E, h_I)

        return self.control_decoder(h_fused)
