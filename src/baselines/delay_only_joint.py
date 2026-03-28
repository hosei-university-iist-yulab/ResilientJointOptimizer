#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/11/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B10: Delay-Only JointOptimizer Wrapper

Wraps the base paper's JointOptimizer to accept 6-dim comm_x
but only uses the first 3 dims [tau, R, B]. Ignores p, sigma_j, s.

Computes rho(tau) only — the delay-only bound from Theorem 1.
This is the BASE PAPER'S BEST MODEL used as a baseline to measure
how much safety margin is overestimated by ignoring packet loss and jitter.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from src.models.joint_optimizer import JointOptimizer


class DelayOnlyJointOptimizer(nn.Module):
    """
    B10: Base paper's JointOptimizer wrapped for 6-dim data.

    Strips 6-dim comm_x down to 3-dim [tau, R, B] before passing
    to the original JointOptimizer. Everything else is unchanged.
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
        decoder_layers: int = 2,
        dropout: float = 0.1,
        physics_gamma: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
        adaptive_gamma: bool = True,
        use_physics_mask: bool = True,
        use_causal_mask: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()

        self.base_model = JointOptimizer(
            n_generators=n_generators,
            energy_input_dim=energy_input_dim,
            comm_input_dim=3,  # ALWAYS 3 — delay-only
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            gnn_layers=gnn_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
            physics_gamma=physics_gamma,
            k_init_scale=k_init_scale,
            lambda_min_0=lambda_min_0,
            learnable_k=True,
            adaptive_gamma=adaptive_gamma,
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
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass — strips comm_x to 3-dim, delegates to base."""
        # Strip to [tau, R, B] only (first 3 features)
        if comm_x.shape[-1] > 3:
            comm_x_3 = comm_x[..., :3]
        else:
            comm_x_3 = comm_x

        edge_index = kwargs.get('energy_edge_index', kwargs.get('comm_edge_index'))
        batch = kwargs.get('batch')
        impedance_matrix = kwargs.get('impedance_matrix')

        outputs = self.base_model(
            energy_x=energy_x,
            energy_edge_index=edge_index,
            comm_x=comm_x_3,
            comm_edge_index=edge_index,
            tau=tau,
            tau_max=tau_max,
            lambda_min_0=lambda_min_0,
            impedance_matrix=impedance_matrix,
            batch=batch,
        )

        # Add dummy R, J for consistent interface
        outputs['R'] = torch.zeros_like(outputs['K'])
        outputs['J'] = torch.zeros_like(outputs['K'])

        return outputs

    def get_coupling_constants(self) -> dict:
        K = self.base_model.get_coupling_constants()
        return {'K': K, 'R': torch.zeros_like(K), 'J': torch.zeros_like(K),
                'K_mean': K.mean().item(), 'R_mean': 0.0, 'J_mean': 0.0}
