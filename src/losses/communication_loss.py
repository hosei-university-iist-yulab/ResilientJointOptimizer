#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/27/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Communication Domain Loss: L_I

Penalizes:
1. End-to-end latency
2. Bandwidth utilization (or violation)
3. Packet loss / reliability
4. Jitter (delay variance)

L_I = w_lat · L_latency + w_bw · L_bandwidth + w_rel · L_reliability
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class LatencyLoss(nn.Module):
    """
    End-to-end latency loss.

    L_latency = Σ_i τ_i / τ_budget

    Penalizes high delays, normalized by delay budget.
    """

    def __init__(
        self,
        tau_budget: float = 100.0,
        tau_max: Optional[float] = None,
    ):
        """
        Args:
            tau_budget: Target delay budget (ms)
            tau_max: Maximum tolerable delay (hard constraint)
        """
        super().__init__()
        self.tau_budget = tau_budget
        self.tau_max = tau_max

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute latency loss.

        Args:
            tau: Communication delays [batch, n_links] in ms

        Returns:
            loss: Latency loss [batch]
        """
        # Normalized latency
        normalized_tau = tau / self.tau_budget

        # If tau_max specified, add hard constraint penalty
        if self.tau_max is not None:
            violation = torch.relu(tau - self.tau_max)
            penalty = (violation ** 2).sum(dim=-1) * 10.0  # Large penalty
            return normalized_tau.mean(dim=-1) + penalty

        return normalized_tau.mean(dim=-1)


class BandwidthLoss(nn.Module):
    """
    Bandwidth utilization/violation loss.

    L_bandwidth = Σ_i max(0, R_i - R_max,i)² + w_util · (1 - R_i/R_max,i)

    Penalizes:
    1. Bandwidth violations (hard constraint)
    2. Under-utilization (soft penalty)
    """

    def __init__(
        self,
        R_max: Optional[torch.Tensor] = None,
        utilization_weight: float = 0.01,
    ):
        """
        Args:
            R_max: Maximum bandwidth per link [n_links]
            utilization_weight: Weight for utilization penalty
        """
        super().__init__()
        if R_max is not None:
            self.register_buffer('R_max', R_max)
        else:
            self.R_max = None
        self.utilization_weight = utilization_weight

    def set_capacity(self, R_max: torch.Tensor):
        """Set link capacities."""
        self.register_buffer('R_max', R_max)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        """
        Compute bandwidth loss.

        Args:
            R: Bandwidth usage [batch, n_links]

        Returns:
            loss: Bandwidth loss [batch]
        """
        if self.R_max is None:
            # No capacity constraints, just penalize total bandwidth
            return R.sum(dim=-1)

        # Violation penalty (exceeding capacity)
        violation = torch.relu(R - self.R_max)
        L_violation = (violation ** 2).sum(dim=-1)

        # Utilization penalty (encourage efficient use)
        utilization = R / (self.R_max + 1e-8)
        L_utilization = ((1 - utilization) ** 2).sum(dim=-1)

        return L_violation + self.utilization_weight * L_utilization


class ReliabilityLoss(nn.Module):
    """
    Reliability / packet loss penalty.

    L_reliability = -log(Π_i (1 - p_loss,i)) ≈ Σ_i p_loss,i

    For small loss probabilities, this simplifies to sum of loss rates.
    """

    def __init__(self, target_reliability: float = 0.999):
        """
        Args:
            target_reliability: Target end-to-end reliability
        """
        super().__init__()
        self.target_reliability = target_reliability

    def forward(self, p_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute reliability loss.

        Args:
            p_loss: Packet loss probability [batch, n_links]

        Returns:
            loss: Reliability loss [batch]
        """
        # Clamp loss probability to valid range
        p_loss = torch.clamp(p_loss, min=1e-8, max=1 - 1e-8)

        # End-to-end reliability
        reliability = torch.prod(1 - p_loss, dim=-1)

        # Penalty for falling below target
        shortfall = torch.relu(self.target_reliability - reliability)

        return shortfall ** 2


class JitterLoss(nn.Module):
    """
    Delay jitter (variance) loss.

    L_jitter = Var(τ) = E[(τ - E[τ])²]

    High jitter is problematic for control systems as it makes
    delay compensation difficult.
    """

    def __init__(self, max_jitter: Optional[float] = None):
        """
        Args:
            max_jitter: Maximum acceptable jitter (ms²)
        """
        super().__init__()
        self.max_jitter = max_jitter

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute jitter loss.

        Args:
            tau: Delays [batch, n_links]

        Returns:
            loss: Jitter loss [batch]
        """
        # Compute variance along link dimension
        jitter = tau.var(dim=-1)

        if self.max_jitter is not None:
            # Hard constraint on maximum jitter
            violation = torch.relu(jitter - self.max_jitter)
            return jitter + 10.0 * violation ** 2

        return jitter


class CommunicationLoss(nn.Module):
    """
    Combined Communication Domain Loss.

    L_I = w_lat · L_latency + w_bw · L_bandwidth + w_rel · L_reliability + w_jit · L_jitter

    Combines all communication quality metrics into single objective.
    """

    def __init__(
        self,
        latency_weight: float = 1.0,
        bandwidth_weight: float = 10.0,
        reliability_weight: float = 100.0,
        jitter_weight: float = 1.0,
        tau_budget: float = 100.0,
        tau_max: Optional[float] = 500.0,
        target_reliability: float = 0.999,
        max_jitter: Optional[float] = None,
    ):
        """
        Args:
            latency_weight: Weight for latency loss
            bandwidth_weight: Weight for bandwidth loss
            reliability_weight: Weight for reliability loss
            jitter_weight: Weight for jitter loss
            tau_budget: Target delay budget (ms)
            tau_max: Maximum tolerable delay (ms)
            target_reliability: Target end-to-end reliability
            max_jitter: Maximum acceptable jitter (ms²)
        """
        super().__init__()

        self.latency_weight = latency_weight
        self.bandwidth_weight = bandwidth_weight
        self.reliability_weight = reliability_weight
        self.jitter_weight = jitter_weight

        self.latency_loss = LatencyLoss(tau_budget, tau_max)
        self.bandwidth_loss = BandwidthLoss()
        self.reliability_loss = ReliabilityLoss(target_reliability)
        self.jitter_loss = JitterLoss(max_jitter)

    def set_bandwidth_capacity(self, R_max: torch.Tensor):
        """Set link bandwidth capacities."""
        self.bandwidth_loss.set_capacity(R_max)

    def forward(
        self,
        tau: torch.Tensor,
        R: Optional[torch.Tensor] = None,
        p_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total communication domain loss.

        Args:
            tau: Communication delays [batch, n_links] in ms
            R: Bandwidth usage [batch, n_links] (optional)
            p_loss: Packet loss probability [batch, n_links] (optional)

        Returns:
            loss: Total communication loss (scalar)
            components: Dict with individual loss terms
        """
        components = {}
        total_loss = torch.tensor(0.0, device=tau.device)

        # Latency loss (always computed)
        L_latency = self.latency_loss(tau)
        total_loss = total_loss + self.latency_weight * L_latency.mean()
        components['L_latency'] = L_latency.mean().item()

        # Jitter loss
        L_jitter = self.jitter_loss(tau)
        total_loss = total_loss + self.jitter_weight * L_jitter.mean()
        components['L_jitter'] = L_jitter.mean().item()

        # Bandwidth loss (if R provided)
        if R is not None:
            L_bandwidth = self.bandwidth_loss(R)
            total_loss = total_loss + self.bandwidth_weight * L_bandwidth.mean()
            components['L_bandwidth'] = L_bandwidth.mean().item()

        # Reliability loss (if p_loss provided)
        if p_loss is not None:
            L_reliability = self.reliability_loss(p_loss)
            total_loss = total_loss + self.reliability_weight * L_reliability.mean()
            components['L_reliability'] = L_reliability.mean().item()

        # Add delay statistics
        components['tau_mean'] = tau.mean().item()
        components['tau_max'] = tau.max().item()
        components['tau_std'] = tau.std().item()

        components['L_I_total'] = total_loss.item()

        return total_loss, components
