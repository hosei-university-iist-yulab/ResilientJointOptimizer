#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/12/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B12: TCP-Retransmit Baseline

Models packet loss as additional delay rather than a separate impairment.
Uses the base JointOptimizer (delay-only) but converts loss to effective delay:

  tau_eff = tau + E[retransmissions] * RTT
          = tau + p/(1-p) * RTT

This tests whether packet loss can be adequately captured by just
increasing the delay, or whether the separate R_i term adds value.
"""

import torch
import torch.nn as nn
from typing import Dict

from src.models.joint_optimizer import JointOptimizer


class TCPRetransmitModel(nn.Module):
    """
    B12: Models packet loss as additional delay via TCP retransmission.

    Converts p to tau_eff, then uses delay-only JointOptimizer.
    Jitter sigma_j is ignored entirely.
    """

    def __init__(
        self,
        n_generators: int,
        n_buses: int = None,
        energy_input_dim: int = 5,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        gnn_layers: int = 3,
        dropout: float = 0.1,
        physics_gamma: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
        rtt_ms: float = 50.0,  # Round-trip time for retransmission
        use_physics_mask: bool = True,
        use_causal_mask: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.rtt_ms = rtt_ms

        self.base_model = JointOptimizer(
            n_generators=n_generators,
            energy_input_dim=energy_input_dim,
            comm_input_dim=3,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            gnn_layers=gnn_layers,
            dropout=dropout,
            physics_gamma=physics_gamma,
            k_init_scale=k_init_scale,
            lambda_min_0=lambda_min_0,
            learnable_k=True,
            use_physics_mask=use_physics_mask,
            use_causal_mask=use_causal_mask,
            use_cross_attention=use_cross_attention,
        )

    def forward(
        self,
        energy_x: torch.Tensor,
        comm_x: torch.Tensor,
        tau: torch.Tensor,
        tau_max: torch.Tensor,
        lambda_min_0: torch.Tensor,
        p: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Convert loss to delay, then use delay-only model."""
        # Strip comm to 3-dim
        if comm_x.shape[-1] > 3:
            comm_x_3 = comm_x[..., :3]
        else:
            comm_x_3 = comm_x

        # Convert packet loss to additional delay: tau_eff = tau + p/(1-p) * RTT
        if p is not None:
            p_safe = torch.clamp(p, max=0.999)
            retransmit_delay = p_safe / (1.0 - p_safe) * self.rtt_ms
            tau_eff = tau + retransmit_delay
        else:
            tau_eff = tau

        # Update comm_x delay column with effective delay
        if comm_x_3.dim() == 2:
            # Flattened [N, 3] — can't easily map tau_eff back. Use original.
            pass
        # Note: The main rho computation uses tau directly, not comm_x

        edge_index = kwargs.get('energy_edge_index', kwargs.get('comm_edge_index'))
        batch = kwargs.get('batch')
        impedance_matrix = kwargs.get('impedance_matrix')

        outputs = self.base_model(
            energy_x=energy_x,
            energy_edge_index=edge_index,
            comm_x=comm_x_3,
            comm_edge_index=edge_index,
            tau=tau_eff,  # Use effective delay (loss folded in)
            tau_max=tau_max,
            lambda_min_0=lambda_min_0,
            impedance_matrix=impedance_matrix,
            batch=batch,
        )

        outputs['R'] = torch.zeros_like(outputs['K'])
        outputs['J'] = torch.zeros_like(outputs['K'])

        return outputs

    def get_coupling_constants(self) -> dict:
        K = self.base_model.get_coupling_constants()
        return {'K': K, 'R': torch.zeros_like(K), 'J': torch.zeros_like(K),
                'K_mean': K.mean().item(), 'R_mean': 0.0, 'J_mean': 0.0}
