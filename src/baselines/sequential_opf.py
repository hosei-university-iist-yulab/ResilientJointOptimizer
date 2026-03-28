#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/08/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
B1: Sequential OPF + QoS Baseline

Decoupled approach:
1. First solve OPF independently
2. Then optimize QoS given OPF solution

This is the traditional engineering approach without joint optimization.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class OPFSolver(nn.Module):
    """
    Simple neural OPF solver (energy domain only).
    Ignores communication delays entirely.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        input_dim: int = 5,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators

        # Simple MLP for OPF
        self.encoder = nn.Sequential(
            nn.Linear(n_buses * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output: P_gen for each generator
        self.decoder = nn.Linear(hidden_dim, n_generators)

    def forward(self, energy_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            energy_x: [batch, n_buses, input_dim]

        Returns:
            P_gen: [batch, n_generators]
        """
        batch_size = energy_x.shape[0]
        x_flat = energy_x.view(batch_size, -1)
        h = self.encoder(x_flat)
        P_gen = self.decoder(h)
        return P_gen


class QoSSolver(nn.Module):
    """
    Simple QoS optimizer (communication domain only).
    Takes OPF solution as fixed input.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        comm_input_dim: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators

        # Input: comm features + P_gen from OPF
        self.encoder = nn.Sequential(
            nn.Linear(n_buses * comm_input_dim + n_generators, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output: Q_gen (reactive power adjustment)
        self.decoder = nn.Linear(hidden_dim, n_generators)

    def forward(
        self,
        comm_x: torch.Tensor,
        P_gen: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            comm_x: [batch, n_buses, comm_input_dim]
            P_gen: [batch, n_generators] from OPF solver

        Returns:
            Q_gen: [batch, n_generators]
        """
        batch_size = comm_x.shape[0]
        comm_flat = comm_x.view(batch_size, -1)
        x = torch.cat([comm_flat, P_gen], dim=-1)
        h = self.encoder(x)
        Q_gen = self.decoder(h)
        return Q_gen


class SequentialOPFQoS(nn.Module):
    """
    B1: Sequential OPF + QoS Baseline

    Two-stage decoupled optimization:
    1. OPF solver produces P_gen (ignores communication)
    2. QoS solver adjusts Q_gen given P_gen and communication state

    Key limitation: No joint optimization, no coupling loss.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        hidden_dim: int = 128,
        lambda_min_0: float = None,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators

        self.opf_solver = OPFSolver(
            n_buses=n_buses,
            n_generators=n_generators,
            input_dim=energy_input_dim,
            hidden_dim=hidden_dim,
        )

        self.qos_solver = QoSSolver(
            n_buses=n_buses,
            n_generators=n_generators,
            comm_input_dim=comm_input_dim,
            hidden_dim=hidden_dim,
        )

        # Compute K init scale based on grid eigenvalue
        k_init_scale = 0.1
        if lambda_min_0 is not None:
            from src.models.coupling import compute_k_init_scale
            k_init_scale = compute_k_init_scale(n_generators, lambda_min_0)

        # Fixed K (not learned - baseline doesn't optimize coupling)
        self.register_buffer('K', torch.ones(n_generators) * k_init_scale)

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
        Sequential forward pass.

        Args:
            energy_x: [batch, n_buses, 5] energy features
            comm_x: [batch, n_buses, 3] communication features
            tau: [batch, n_generators] delays
            tau_max: [n_generators] max delays
            lambda_min_0: [batch] or scalar baseline eigenvalue

        Returns:
            Dict with 'u', 'rho', 'K'
        """
        # Reshape if flattened
        if energy_x.dim() == 2:
            batch_size = energy_x.shape[0] // self.n_buses
            energy_x = energy_x.view(batch_size, self.n_buses, -1)
            comm_x = comm_x.view(batch_size, self.n_buses, -1)

        batch_size = energy_x.shape[0]

        # Stage 1: OPF (ignores communication)
        P_gen = self.opf_solver(energy_x)

        # Stage 2: QoS adjustment
        Q_gen = self.qos_solver(comm_x, P_gen.detach())  # detach: no gradient flow

        # Combine outputs
        u = torch.cat([P_gen, Q_gen], dim=-1)

        # Compute stability margin (using fixed K)
        delay_contribution = (self.K.unsqueeze(0) * tau / tau_max.unsqueeze(0)).sum(dim=-1)
        rho = torch.abs(lambda_min_0.squeeze()) - delay_contribution

        return {
            'u': u,
            'rho': rho,
            'K': self.K,
            'P_gen': P_gen,
            'Q_gen': Q_gen,
        }

    def get_coupling_constants(self) -> torch.Tensor:
        """Return fixed K values."""
        return self.K
