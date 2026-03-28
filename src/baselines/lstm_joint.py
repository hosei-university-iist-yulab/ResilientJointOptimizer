#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/06/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B4: LSTM Joint Baseline

Recurrent approach treating buses as a sequence.
Joint processing of energy and communication via LSTM.
"""

import torch
import torch.nn as nn
from typing import Dict


class LSTMJoint(nn.Module):
    """
    B4: LSTM Joint Baseline

    Treats the power system as a sequence of buses.
    Processes concatenated energy+communication features with LSTM.

    Key limitation: No graph structure, arbitrary ordering of buses.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        k_init_scale: float = 0.1,
        lambda_min_0: float = None,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Input projection
        input_dim = energy_input_dim + comm_input_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output dimension depends on bidirectional
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Output heads
        self.control_head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_generators * 2),
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

        # Concatenate features
        x = torch.cat([energy_x, comm_x], dim=-1)  # [batch, n_buses, input_dim]

        # Project
        x = self.input_proj(x)  # [batch, n_buses, hidden_dim]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch, n_buses, hidden*2]

        # Use final hidden state (or mean pool)
        if self.bidirectional:
            # Concatenate forward and backward final states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [batch, hidden*2]
        else:
            h_final = h_n[-1]  # [batch, hidden]

        # Control output
        u = self.control_head(h_final)

        # Coupling constants
        K = torch.exp(self.log_K)

        # Stability margin
        delay_contribution = (K.unsqueeze(0) * tau / tau_max.unsqueeze(0)).sum(dim=-1)
        rho = torch.abs(lambda_min_0.squeeze()) - delay_contribution

        return {
            'u': u,
            'rho': rho,
            'K': K,
        }

    def get_coupling_constants(self) -> torch.Tensor:
        return torch.exp(self.log_K)
