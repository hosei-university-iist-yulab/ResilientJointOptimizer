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
B11: Naive Multi-Impairment Baseline

Same architecture as ResilientJointOptimizer but K, R, J are FIXED
(not learned). Values set from theoretical formulas based on system
parameters. Tests the value of learning vs analytical constants.
"""

import torch
import torch.nn as nn
from typing import Dict

from src.models.resilient_optimizer import ResilientJointOptimizer


class NaiveMultiImpairment(nn.Module):
    """
    B11: Multi-impairment model with fixed (non-learnable) K, R, J.

    Uses the same architecture as ResilientJointOptimizer but freezes
    the coupling constants at their initialized values.
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
        dropout: float = 0.1,
        physics_gamma: float = 0.1,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
        use_physics_mask: bool = True,
        use_causal_mask: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()

        self.model = ResilientJointOptimizer(
            n_generators=n_generators,
            energy_input_dim=energy_input_dim,
            comm_input_dim=comm_input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            gnn_layers=gnn_layers,
            dropout=dropout,
            physics_gamma=physics_gamma,
            k_init_scale=k_init_scale,
            lambda_min_0=lambda_min_0,
            learnable_constants=False,  # FROZEN
            use_physics_mask=use_physics_mask,
            use_causal_mask=use_causal_mask,
            use_cross_attention=use_cross_attention,
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        return self.model(**kwargs)

    def get_coupling_constants(self) -> dict:
        return self.model.get_coupling_constants()
