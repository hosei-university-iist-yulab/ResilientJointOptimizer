#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/07/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Resilient Joint Optimizer (Topic 2: Our Model)

Extends the base JointOptimizer with multi-impairment stability bound:

  rho(tau, p, sigma_j) >= |lambda_min(0)|
      - SUM_i (K_i * tau_i / tau_max_i)
      - SUM_i (R_i * p_i / (1 - p_i))
      - SUM_i (J_i * sigma_j_i^2 / sigma_max^2)

Key differences from base JointOptimizer:
  - comm_input_dim=6 (was 3): [tau, R, B, p, sigma_j, s]
  - 3 families of learnable constants: K, R, J (was only K)
  - Channel state encoder for Markov model
  - Channel state predictor for L_channel loss

IMPORTANT: The base JointOptimizer in joint_optimizer.py is NOT modified.
It serves as baseline B10 (delay-only).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .gnn import DualDomainGNN
from .attention import HierarchicalAttention
from .joint_optimizer import ControlDecoder, DelayPredictor
from .multi_impairment_coupling import (
    MultiImpairmentCoupling,
    compute_multi_impairment_init_scale,
    compute_rho_multi_impairment,
)
from .channel_model import ChannelStateEncoder, ChannelStatePredictor


class ResilientJointOptimizer(nn.Module):
    """
    Multi-impairment resilient energy-information co-optimizer.

    Architecture:
        [Energy Graph] -----> EnergyGNN --------+
                                                 |-> HierarchicalAttention -> Decoder -> u*
        [Comm Graph 6-dim] -> CommGNN ----------+                                    |
                              + ChannelEncoder                               [K, R, J, rho]
    """

    def __init__(
        self,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 6,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        gnn_layers: int = 3,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        physics_gamma: float = 0.1,
        k_init_scale: float = 0.1,
        k_budget: float = 0.5,
        r_budget: float = 0.3,
        j_budget: float = 0.2,
        lambda_min_0: float = None,
        learnable_constants: bool = True,
        adaptive_gamma: bool = True,
        use_physics_mask: bool = True,
        use_causal_mask: bool = True,
        use_cross_attention: bool = True,
        channel_n_states: int = 3,
        channel_encoder_hidden: int = 64,
    ):
        super().__init__()

        self.n_generators = n_generators
        self.embed_dim = embed_dim
        self.use_cross_attention = use_cross_attention
        self.use_causal_mask = use_causal_mask

        # Dual-domain GNN encoders (reuses gnn.py classes, comm_input_dim=6)
        self.gnn = DualDomainGNN(
            energy_input_dim=energy_input_dim,
            comm_input_dim=comm_input_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        # Hierarchical attention with masks (reuses attention.py)
        effective_gamma = physics_gamma if use_physics_mask else 0.0
        self.attention = HierarchicalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            gamma=effective_gamma,
            adaptive_gamma=adaptive_gamma and use_physics_mask,
        )

        # Auto-scale init if lambda_min_0 provided
        if lambda_min_0 is not None and k_init_scale == 0.1:
            k_init_scale = compute_multi_impairment_init_scale(n_generators, lambda_min_0)

        # Multi-impairment coupling constants (K, R, J)
        self.coupling = MultiImpairmentCoupling(
            n_generators=n_generators,
            init_scale=k_init_scale,
            k_budget=k_budget,
            r_budget=r_budget,
            j_budget=j_budget,
            learnable=learnable_constants,
        )

        # Channel state encoder (NEW)
        self.channel_encoder = ChannelStateEncoder(
            n_states=channel_n_states,
            embed_dim=embed_dim,
            hidden_dim=channel_encoder_hidden,
        )

        # Channel state predictor (NEW, for L_channel loss)
        self.channel_predictor = ChannelStatePredictor(
            embed_dim=embed_dim,
            n_states=channel_n_states,
        )

        # Default tau_max
        self.register_buffer('tau_max_default', torch.ones(n_generators) * 500.0)

        # Control decoder (reuses joint_optimizer.py ControlDecoder)
        self.control_decoder = ControlDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=n_generators * 2,
            num_layers=decoder_layers,
            dropout=dropout,
        )

        # Delay predictor (reuses joint_optimizer.py DelayPredictor)
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
            'k_budget': k_budget,
            'r_budget': r_budget,
            'j_budget': j_budget,
            'learnable_constants': learnable_constants,
            'adaptive_gamma': adaptive_gamma,
            'use_physics_mask': use_physics_mask,
            'use_causal_mask': use_causal_mask,
            'use_cross_attention': use_cross_attention,
            'channel_n_states': channel_n_states,
            'channel_encoder_hidden': channel_encoder_hidden,
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
        p: torch.Tensor,
        sigma_j: torch.Tensor,
        sigma_max: torch.Tensor,
        channel_state: Optional[torch.Tensor] = None,
        impedance_matrix: Optional[torch.Tensor] = None,
        dag_edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the resilient joint optimizer.

        Args:
            energy_x: Energy node features [N, 5]
            energy_edge_index: Power grid edges [2, E_power]
            comm_x: Communication features [N, 6]: [tau, R, B, p, sigma_j, s]
            comm_edge_index: Communication edges [2, E_comm]
            tau: Delays [batch, n_gen] ms
            tau_max: Delay margins [n_gen] or [batch, n_gen] ms
            lambda_min_0: Baseline eigenvalue [batch] or scalar
            p: Packet loss rates [batch, n_gen] in [0, p_max)
            sigma_j: Jitter std [batch, n_gen] ms
            sigma_max: Max jitter for normalization, scalar or [1]
            channel_state: Markov state [batch, n_gen] (0/1/2), optional
            impedance_matrix: For physics mask [N, N], optional
            dag_edge_index: For causal mask [2, E_dag], optional
            batch: Batch assignment [N], optional

        Returns:
            Dict with: u, rho, K, R, J, tau_pred, h_E, h_I, h_fused,
                       attn_info, channel_pred
        """
        # Get coupling constants
        K, R, J = self.coupling()

        # Compute multi-impairment stability margin
        rho = compute_rho_multi_impairment(
            K, R, J, tau, tau_max, p, sigma_j, sigma_max, lambda_min_0,
        )

        # Encode both domains with GNN (comm_x is 6-dim)
        h_E, h_I = self.gnn(
            energy_x, energy_edge_index,
            comm_x, comm_edge_index,
            batch=batch,
        )

        # Add channel state encoding to communication embeddings
        if channel_state is not None:
            # channel_state is [batch, n_gen] but h_I is [N] (all nodes)
            # Use channel state from comm_x[:, 5] (the s feature, set by dataset)
            s_indices = comm_x[:, 5].clamp(0, 2).long()  # [N], clamp for safety
            channel_emb = self.channel_encoder(s_indices)  # [N, embed_dim]
            h_I = h_I + channel_emb

        # Handle batching
        if batch is not None:
            batch_size = batch.max().item() + 1
            nodes_per_graph = h_E.shape[0] // batch_size
            h_E = h_E.view(batch_size, nodes_per_graph, -1)
            h_I = h_I.view(batch_size, nodes_per_graph, -1)
        else:
            h_E = h_E.unsqueeze(0)
            h_I = h_I.unsqueeze(0)

        # Hierarchical attention with masks
        use_attn = self.use_cross_attention and h_E.shape[1] < 3000
        if not use_attn:
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

        # Predict channel states (for L_channel loss)
        channel_pred = self.channel_predictor(h_I)

        return {
            'u': u,
            'rho': rho,
            'K': K,
            'R': R,
            'J': J,
            'tau_pred': tau_pred,
            'h_E': h_E,
            'h_I': h_I,
            'h_fused': h_fused,
            'attn_info': attn_info,
            'channel_pred': channel_pred,
        }

    def get_stability_margin(
        self,
        tau: torch.Tensor,
        tau_max: torch.Tensor,
        lambda_min_0: torch.Tensor,
        p: torch.Tensor,
        sigma_j: torch.Tensor,
        sigma_max: torch.Tensor,
    ) -> torch.Tensor:
        """Compute stability margin for given impairments."""
        K, R, J = self.coupling()
        return compute_rho_multi_impairment(
            K, R, J, tau, tau_max, p, sigma_j, sigma_max, lambda_min_0,
        )

    def get_coupling_constants(self) -> dict:
        """Return current K, R, J values."""
        return self.coupling.get_values()

    @classmethod
    def from_config(cls, config: dict) -> 'ResilientJointOptimizer':
        """Create model from configuration dictionary."""
        return cls(**config)

    def extra_repr(self) -> str:
        return (
            f"n_generators={self.n_generators}, "
            f"embed_dim={self.embed_dim}, "
            f"comm_input_dim=6 (multi-impairment)"
        )
