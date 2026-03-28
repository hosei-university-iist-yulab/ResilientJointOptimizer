#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/29/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Energy Domain Loss: L_E

Penalizes:
1. Generation cost (economic dispatch)
2. Voltage violations
3. Frequency deviations
4. Power imbalance

L_E = w_cost · C(P_gen) + w_V · L_voltage + w_f · L_frequency + L_balance
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class GenerationCostLoss(nn.Module):
    """
    Economic dispatch cost function.

    C(P_gen) = Σ_i (a_i · P_i² + b_i · P_i + c_i)

    Standard quadratic cost function for thermal generators.
    """

    def __init__(
        self,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            a: Quadratic coefficients [n_gen]
            b: Linear coefficients [n_gen]
            c: Constant coefficients [n_gen]
        """
        super().__init__()

        if a is not None:
            self.register_buffer('a', a)
            self.register_buffer('b', b)
            self.register_buffer('c', c)
            self.n_gen = len(a)
        else:
            self.a = None
            self.b = None
            self.c = None
            self.n_gen = None

    def set_cost_coefficients(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ):
        """Set cost coefficients from power system data."""
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.register_buffer('c', c)
        self.n_gen = len(a)

    def forward(self, P_gen: torch.Tensor) -> torch.Tensor:
        """
        Compute generation cost.

        Args:
            P_gen: Active power generation [batch, n_gen]

        Returns:
            cost: Total generation cost [batch]
        """
        if self.a is None:
            # Default: simple quadratic cost
            return (P_gen ** 2).sum(dim=-1)

        # Quadratic cost: C = a·P² + b·P + c
        cost = (
            (self.a * P_gen ** 2).sum(dim=-1) +
            (self.b * P_gen).sum(dim=-1) +
            self.c.sum()
        )
        return cost


class VoltageLoss(nn.Module):
    """
    Voltage violation penalty.

    L_voltage = Σ_i max(0, V_i - V_max)² + max(0, V_min - V_i)²

    Penalizes voltages outside [V_min, V_max] range.
    """

    def __init__(
        self,
        v_min: float = 0.95,
        v_max: float = 1.05,
    ):
        """
        Args:
            v_min: Minimum voltage (p.u.)
            v_max: Maximum voltage (p.u.)
        """
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute voltage violation loss.

        Args:
            V: Bus voltages [batch, n_bus] in p.u.

        Returns:
            loss: Voltage violation loss [batch]
        """
        # Upper bound violation
        upper_violation = torch.relu(V - self.v_max)

        # Lower bound violation
        lower_violation = torch.relu(self.v_min - V)

        # Squared penalty
        loss = (upper_violation ** 2 + lower_violation ** 2).sum(dim=-1)

        return loss


class FrequencyLoss(nn.Module):
    """
    Frequency deviation penalty.

    L_frequency = Σ_i (Δω_i)² = Σ_i (ω_i - ω_nom)²

    Penalizes deviation from nominal frequency.
    """

    def __init__(self, omega_nom: float = 1.0):
        """
        Args:
            omega_nom: Nominal frequency (p.u., typically 1.0)
        """
        super().__init__()
        self.omega_nom = omega_nom

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency deviation loss.

        Args:
            omega: Generator frequencies [batch, n_gen] in p.u.

        Returns:
            loss: Frequency deviation loss [batch]
        """
        delta_omega = omega - self.omega_nom
        loss = (delta_omega ** 2).sum(dim=-1)
        return loss


class PowerBalanceLoss(nn.Module):
    """
    Power balance constraint loss.

    L_balance = (Σ P_gen - Σ P_load - P_loss)²

    Enforces that generation equals load plus losses.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        P_gen: torch.Tensor,
        P_load: torch.Tensor,
        P_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute power balance loss.

        Args:
            P_gen: Generation [batch, n_gen]
            P_load: Load [batch, n_load] or [batch]
            P_loss: Network losses [batch] (optional)

        Returns:
            loss: Power balance violation [batch]
        """
        total_gen = P_gen.sum(dim=-1)

        if P_load.dim() > 1:
            total_load = P_load.sum(dim=-1)
        else:
            total_load = P_load

        if P_loss is not None:
            imbalance = total_gen - total_load - P_loss
        else:
            imbalance = total_gen - total_load

        loss = imbalance ** 2
        return loss


class EnergyLoss(nn.Module):
    """
    Combined Energy Domain Loss.

    L_E = w_cost · C(P_gen) + w_V · L_voltage + w_f · L_frequency + w_bal · L_balance

    Combines economic dispatch, voltage regulation, frequency control,
    and power balance into single differentiable objective.
    """

    def __init__(
        self,
        cost_weight: float = 1.0,
        voltage_weight: float = 10.0,
        frequency_weight: float = 100.0,
        balance_weight: float = 100.0,
        v_min: float = 0.95,
        v_max: float = 1.05,
        omega_nom: float = 1.0,
    ):
        """
        Args:
            cost_weight: Weight for generation cost
            voltage_weight: Weight for voltage violations
            frequency_weight: Weight for frequency deviations
            balance_weight: Weight for power balance
            v_min: Minimum voltage (p.u.)
            v_max: Maximum voltage (p.u.)
            omega_nom: Nominal frequency (p.u.)
        """
        super().__init__()

        self.cost_weight = cost_weight
        self.voltage_weight = voltage_weight
        self.frequency_weight = frequency_weight
        self.balance_weight = balance_weight

        self.cost_loss = GenerationCostLoss()
        self.voltage_loss = VoltageLoss(v_min, v_max)
        self.frequency_loss = FrequencyLoss(omega_nom)
        self.balance_loss = PowerBalanceLoss()

    def set_cost_coefficients(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ):
        """Set generation cost coefficients."""
        self.cost_loss.set_cost_coefficients(a, b, c)

    def forward(
        self,
        P_gen: torch.Tensor,
        Q_gen: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
        P_load: Optional[torch.Tensor] = None,
        P_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total energy domain loss.

        Args:
            P_gen: Active power generation [batch, n_gen]
            Q_gen: Reactive power generation [batch, n_gen] (optional)
            V: Bus voltages [batch, n_bus] (optional)
            omega: Frequencies [batch, n_gen] (optional)
            P_load: Total load [batch] (optional)
            P_loss: Network losses [batch] (optional)

        Returns:
            loss: Total energy loss (scalar)
            components: Dict with individual loss terms
        """
        components = {}
        total_loss = torch.tensor(0.0, device=P_gen.device)

        # Generation cost (always computed)
        L_cost = self.cost_loss(P_gen)
        total_loss = total_loss + self.cost_weight * L_cost.mean()
        components['L_cost'] = L_cost.mean().item()

        # Voltage loss (if V provided)
        if V is not None:
            L_voltage = self.voltage_loss(V)
            total_loss = total_loss + self.voltage_weight * L_voltage.mean()
            components['L_voltage'] = L_voltage.mean().item()

        # Frequency loss (if omega provided)
        if omega is not None:
            L_freq = self.frequency_loss(omega)
            total_loss = total_loss + self.frequency_weight * L_freq.mean()
            components['L_frequency'] = L_freq.mean().item()

        # Power balance loss (if P_load provided)
        if P_load is not None:
            L_balance = self.balance_loss(P_gen, P_load, P_loss)
            total_loss = total_loss + self.balance_weight * L_balance.mean()
            components['L_balance'] = L_balance.mean().item()

        components['L_E_total'] = total_loss.item()

        return total_loss, components
