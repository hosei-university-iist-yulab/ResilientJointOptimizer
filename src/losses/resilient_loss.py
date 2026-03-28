#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/09/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Resilient Joint Loss (Topic 2)

Wraps the base JointLoss (UNTOUCHED) and adds L_channel for
Markov channel state prediction.

L_total = JointLoss(u, rho, h_E, h_I, ...) + channel_weight * L_channel

The key insight: CouplingLoss takes rho as input. Since
ResilientJointOptimizer computes rho(tau, p, sigma_j) instead of
rho(tau), the same CouplingLoss automatically enforces the
multi-impairment bound. No modification needed.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .combined import JointLoss
from .channel_loss import ChannelStateLoss


class ResilientJointLoss(nn.Module):
    """
    Combined loss for multi-impairment resilient optimization.

    Reuses all base paper loss components (EnergyLoss, CommunicationLoss,
    CouplingLoss, PhysicsAwareContrastiveLoss) via JointLoss, and adds
    L_channel for Markov state prediction.
    """

    def __init__(
        self,
        # All base JointLoss params
        cost_weight: float = 1.0,
        voltage_weight: float = 10.0,
        frequency_weight: float = 100.0,
        latency_weight: float = 1.0,
        bandwidth_weight: float = 10.0,
        alpha: float = 1.0,
        beta: float = 0.1,
        rho_min: float = 0.01,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
        energy_weight: float = 1.0,
        communication_weight: float = 1.0,
        coupling_weight: float = 1.0,
        tau_budget: float = 100.0,
        tau_max: float = 500.0,
        # NEW: channel loss
        channel_weight: float = 0.5,
        channel_n_states: int = 3,
    ):
        super().__init__()

        self.channel_weight = channel_weight

        # Base paper loss (UNTOUCHED — instantiate, don't modify)
        self.base_loss = JointLoss(
            cost_weight=cost_weight,
            voltage_weight=voltage_weight,
            frequency_weight=frequency_weight,
            latency_weight=latency_weight,
            bandwidth_weight=bandwidth_weight,
            alpha=alpha,
            beta=beta,
            rho_min=rho_min,
            contrastive_weight=contrastive_weight,
            temperature=temperature,
            energy_weight=energy_weight,
            communication_weight=communication_weight,
            coupling_weight=coupling_weight,
            tau_budget=tau_budget,
            tau_max=tau_max,
        )

        # NEW: channel state prediction loss
        self.channel_loss = ChannelStateLoss(n_states=channel_n_states)

    def forward(
        self,
        # Model outputs
        u: torch.Tensor,
        rho: torch.Tensor,
        h_E: torch.Tensor,
        h_I: torch.Tensor,
        # Energy domain data
        P_gen: torch.Tensor,
        V: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
        P_load: Optional[torch.Tensor] = None,
        # Communication domain data
        tau: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        # Coupling data
        lambda_min_0: Optional[torch.Tensor] = None,
        u_prev: Optional[torch.Tensor] = None,
        # Physics data
        impedance_matrix: Optional[torch.Tensor] = None,
        # NEW: channel data
        channel_pred: Optional[torch.Tensor] = None,
        channel_state: Optional[torch.Tensor] = None,
        # Control flag
        use_coupling_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total resilient loss.

        Delegates to base JointLoss for L_E + L_I + L_coupling + L_contrastive,
        then adds L_channel.
        """
        # Base paper loss (unchanged computation)
        total_loss, components = self.base_loss(
            u=u, rho=rho, h_E=h_E, h_I=h_I,
            P_gen=P_gen, V=V, omega=omega, P_load=P_load,
            tau=tau, R=R,
            lambda_min_0=lambda_min_0, u_prev=u_prev,
            impedance_matrix=impedance_matrix,
            use_coupling_loss=use_coupling_loss,
        )

        # NEW: Channel state prediction loss
        if (self.channel_weight > 0
                and channel_pred is not None
                and channel_state is not None):
            L_channel, channel_components = self.channel_loss(
                channel_pred=channel_pred,
                channel_state=channel_state,
            )
            total_loss = total_loss + self.channel_weight * L_channel
            components.update(channel_components)

        # Update total
        components['L_total'] = total_loss.item()

        return total_loss, components
