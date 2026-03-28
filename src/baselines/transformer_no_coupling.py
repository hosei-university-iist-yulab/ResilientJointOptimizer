#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/09/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B7: Transformer (no L_coupling) Baseline

Full architecture but trained WITHOUT the coupling loss.
This ablation tests the importance of the stability-aware loss term.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
from pathlib import Path

# Import the full model
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import JointOptimizer


class TransformerNoCoupling(nn.Module):
    """
    B7: Transformer without L_coupling Baseline

    Uses the SAME architecture as JointOptimizer but marks itself
    so training uses no coupling loss.

    This is an ablation to test: "Does L_coupling actually help?"

    Key difference: No log-barrier stability loss during training.
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
    ):
        super().__init__()

        # Auto-scale K initialization based on grid eigenvalue
        if lambda_min_0 is not None and k_init_scale == 0.1:
            from src.models.coupling import compute_k_init_scale
            k_init_scale = compute_k_init_scale(n_generators, lambda_min_0)

        # Use full JointOptimizer architecture
        self.model = JointOptimizer(
            n_generators=n_generators,
            energy_input_dim=energy_input_dim,
            comm_input_dim=comm_input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            gnn_layers=gnn_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
            physics_gamma=physics_gamma,
            k_init_scale=k_init_scale,
            learnable_k=True,
        )

        # Flag to indicate no coupling loss should be used
        self.use_coupling_loss = False
        self.n_generators = n_generators

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
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - delegates to JointOptimizer.

        Returns dict with 'u', 'rho', 'K', 'h_E', 'h_I'
        """
        outputs = self.model(
            energy_x=energy_x,
            energy_edge_index=energy_edge_index,
            comm_x=comm_x,
            comm_edge_index=comm_edge_index,
            tau=tau,
            tau_max=tau_max,
            lambda_min_0=lambda_min_0,
            impedance_matrix=impedance_matrix,
            batch=batch,
        )

        # Add flag for training loop
        outputs['use_coupling_loss'] = False

        return outputs

    def get_coupling_constants(self) -> torch.Tensor:
        return self.model.get_coupling_constants()

    @property
    def config(self):
        return self.model.config
