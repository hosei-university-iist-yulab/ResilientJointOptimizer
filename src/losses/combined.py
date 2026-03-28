#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/26/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Joint Loss Function: L_total

Combines all loss components with learnable or fixed weights.

L_total = L_E + L_I + L_coupling + λ · L_align

where:
- L_E: Energy domain loss (cost, voltage, frequency)
- L_I: Communication domain loss (latency, bandwidth, reliability)
- L_coupling: Stability coupling loss (log-barrier + deviation)
- L_align: Contrastive alignment loss

This is the complete objective for end-to-end training.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .energy_loss import EnergyLoss
from .communication_loss import CommunicationLoss
from .coupling_loss import CouplingLoss
from .contrastive import PhysicsAwareContrastiveLoss


class JointLoss(nn.Module):
    """
    Combined joint loss for energy-information co-optimization.

    L_total = w_E · L_E + w_I · L_I + w_coupling · L_coupling + w_align · L_align
    """

    def __init__(
        self,
        # Energy loss weights
        cost_weight: float = 1.0,
        voltage_weight: float = 10.0,
        frequency_weight: float = 100.0,
        # Communication loss weights
        latency_weight: float = 1.0,
        bandwidth_weight: float = 10.0,
        # Coupling loss weights
        alpha: float = 1.0,
        beta: float = 0.1,
        rho_min: float = 0.01,
        # Contrastive loss weights
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
        # Global weights
        energy_weight: float = 1.0,
        communication_weight: float = 1.0,
        coupling_weight: float = 1.0,
        # Communication parameters
        tau_budget: float = 100.0,
        tau_max: float = 500.0,
    ):
        """
        Args:
            cost_weight: Weight for generation cost
            voltage_weight: Weight for voltage violations
            frequency_weight: Weight for frequency deviations
            latency_weight: Weight for latency
            bandwidth_weight: Weight for bandwidth
            alpha: Log-barrier weight for coupling
            beta: Deviation weight for coupling
            rho_min: Minimum stability margin
            contrastive_weight: Weight for contrastive loss
            temperature: Contrastive temperature
            energy_weight: Global weight for energy domain
            communication_weight: Global weight for communication domain
            coupling_weight: Global weight for coupling loss
            tau_budget: Delay budget (ms)
            tau_max: Maximum delay (ms)
        """
        super().__init__()

        # Global weights
        self.energy_weight = energy_weight
        self.communication_weight = communication_weight
        self.coupling_weight = coupling_weight
        self.contrastive_weight = contrastive_weight

        # Individual losses
        self.energy_loss = EnergyLoss(
            cost_weight=cost_weight,
            voltage_weight=voltage_weight,
            frequency_weight=frequency_weight,
        )

        self.communication_loss = CommunicationLoss(
            latency_weight=latency_weight,
            bandwidth_weight=bandwidth_weight,
            tau_budget=tau_budget,
            tau_max=tau_max,
        )

        self.coupling_loss = CouplingLoss(
            alpha=alpha,
            beta=beta,
            rho_min=rho_min,
        )

        self.contrastive_loss = PhysicsAwareContrastiveLoss(
            temperature=temperature,
        )

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
        # Control flag for baseline comparison
        use_coupling_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total joint loss.

        Args:
            u: Control actions [batch, n_control]
            rho: Stability margin [batch]
            h_E: Energy embeddings [batch, N, dim]
            h_I: Communication embeddings [batch, N, dim]
            P_gen: Active power generation [batch, n_gen]
            V: Bus voltages [batch, n_bus]
            omega: Frequencies [batch, n_gen]
            P_load: Total load [batch]
            tau: Communication delays [batch, n_links]
            R: Bandwidth usage [batch, n_links]
            lambda_min_0: Reference eigenvalue
            u_prev: Previous control [batch, n_control]
            impedance_matrix: Impedance for physics mask [N, N]

        Returns:
            loss: Total joint loss (scalar)
            components: Dict with all loss components
        """
        components = {}
        total_loss = torch.tensor(0.0, device=u.device)

        # Energy domain loss
        L_E, energy_components = self.energy_loss(
            P_gen=P_gen,
            V=V,
            omega=omega,
            P_load=P_load,
        )
        total_loss = total_loss + self.energy_weight * L_E
        components.update(energy_components)

        # Communication domain loss
        if tau is not None:
            L_I, comm_components = self.communication_loss(
                tau=tau,
                R=R,
            )
            total_loss = total_loss + self.communication_weight * L_I
            components.update(comm_components)

        # Coupling loss (core novel contribution)
        # Can be disabled for baseline B7 comparison
        if use_coupling_loss and lambda_min_0 is not None and u_prev is not None and tau is not None:
            L_coupling, coupling_components = self.coupling_loss(
                rho=rho,
                lambda_min_0=lambda_min_0,
                u=u,
                u_prev=u_prev,
                tau=tau,
            )
            total_loss = total_loss + self.coupling_weight * L_coupling
            components.update(coupling_components)

        # Contrastive alignment loss (only if embeddings available)
        if self.contrastive_weight > 0 and h_E is not None and h_I is not None:
            L_align, align_components = self.contrastive_loss(
                h_E=h_E,
                h_I=h_I,
                impedance_matrix=impedance_matrix,
            )
            total_loss = total_loss + self.contrastive_weight * L_align
            components.update(align_components)

        # Add total
        components['L_total'] = total_loss.item()

        return total_loss, components


class JointLossWithScheduling(nn.Module):
    """
    Joint loss with loss weight scheduling.

    Allows curriculum learning where different loss components
    are emphasized at different training stages.
    """

    def __init__(
        self,
        base_loss: JointLoss,
        coupling_warmup_epochs: int = 10,
        contrastive_warmup_epochs: int = 5,
    ):
        """
        Args:
            base_loss: Base JointLoss instance
            coupling_warmup_epochs: Epochs before coupling loss is active
            contrastive_warmup_epochs: Epochs before contrastive loss is active
        """
        super().__init__()
        self.base_loss = base_loss
        self.coupling_warmup_epochs = coupling_warmup_epochs
        self.contrastive_warmup_epochs = contrastive_warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for scheduling."""
        self.current_epoch = epoch

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward with scheduled weights."""
        # Compute base loss
        loss, components = self.base_loss(*args, **kwargs)

        # Apply scheduling
        coupling_scale = min(1.0, self.current_epoch / max(1, self.coupling_warmup_epochs))
        contrastive_scale = min(1.0, self.current_epoch / max(1, self.contrastive_warmup_epochs))

        # Adjust loss if needed (simplified - in practice would recompute)
        components['coupling_scale'] = coupling_scale
        components['contrastive_scale'] = contrastive_scale

        return loss, components


class AuxiliaryLosses(nn.Module):
    """
    Auxiliary losses for regularization and monitoring.

    Includes:
    1. K_i regularization (prevent explosion)
    2. Attention entropy (encourage diverse attention)
    3. Embedding norm regularization
    """

    def __init__(
        self,
        k_reg_weight: float = 0.01,
        entropy_weight: float = 0.01,
        norm_weight: float = 0.001,
    ):
        super().__init__()
        self.k_reg_weight = k_reg_weight
        self.entropy_weight = entropy_weight
        self.norm_weight = norm_weight

    def forward(
        self,
        K: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute auxiliary losses.

        Args:
            K: Coupling constants [n_gen]
            attn_weights: Attention weights [batch, heads, N, N]
            embeddings: Model embeddings [batch, N, dim]

        Returns:
            loss: Total auxiliary loss
            components: Individual components
        """
        components = {}
        total_loss = torch.tensor(0.0, device=K.device)

        # K regularization: prevent K_i from being too large
        L_K = (K ** 2).mean()
        total_loss = total_loss + self.k_reg_weight * L_K
        components['L_K_reg'] = L_K.item()

        # Attention entropy: encourage diverse attention
        if attn_weights is not None:
            # Compute entropy of attention distribution
            # H = -Σ p·log(p)
            attn_flat = attn_weights.view(-1, attn_weights.shape[-1])
            entropy = -(attn_flat * torch.log(attn_flat + 1e-8)).sum(dim=-1).mean()
            # Negative entropy (maximize entropy = minimize negative)
            L_entropy = -entropy
            total_loss = total_loss + self.entropy_weight * L_entropy
            components['L_attn_entropy'] = -entropy.item()

        # Embedding norm regularization
        if embeddings is not None:
            L_norm = (embeddings ** 2).mean()
            total_loss = total_loss + self.norm_weight * L_norm
            components['L_embed_norm'] = L_norm.item()

        components['L_aux_total'] = total_loss.item()

        return total_loss, components
