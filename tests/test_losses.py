#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/16/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Unit Tests for Loss Functions

Tests:
- CouplingLoss
- EnergyLoss
- CommunicationLoss
- PhysicsAwareContrastiveLoss
- JointLoss
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.losses import (
    CouplingLoss,
    EnergyLoss,
    CommunicationLoss,
    PhysicsAwareContrastiveLoss,
    JointLoss,
)
from src.losses.coupling_loss import LogBarrierStabilityLoss, ControlDeviationLoss


class TestLogBarrierStabilityLoss:
    """Tests for log-barrier stability loss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = LogBarrierStabilityLoss(rho_min=0.01)

        rho = torch.tensor([0.5, 0.3, 0.1])
        lambda_min_0 = torch.tensor(-0.5)

        loss = loss_fn(rho, lambda_min_0)

        assert loss.shape == (3,)
        assert torch.all(loss >= 0), "Loss should be non-negative"

    def test_loss_increases_as_rho_decreases(self):
        """Test that loss increases as stability margin decreases."""
        loss_fn = LogBarrierStabilityLoss()

        lambda_min_0 = torch.tensor(-0.5)

        rho_high = torch.tensor([0.4])
        rho_low = torch.tensor([0.1])

        loss_high_rho = loss_fn(rho_high, lambda_min_0)
        loss_low_rho = loss_fn(rho_low, lambda_min_0)

        assert loss_low_rho > loss_high_rho, "Lower rho should have higher loss"

    def test_rho_clamping(self):
        """Test that negative rho is clamped."""
        loss_fn = LogBarrierStabilityLoss(rho_min=0.01)

        rho = torch.tensor([-0.1])  # Negative (unstable)
        lambda_min_0 = torch.tensor(-0.5)

        loss = loss_fn(rho, lambda_min_0)

        assert torch.isfinite(loss), "Loss should be finite even for negative rho"


class TestControlDeviationLoss:
    """Tests for control deviation loss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = ControlDeviationLoss()

        u = torch.randn(4, 10)
        u_prev = torch.randn(4, 10)
        tau = torch.rand(4, 5) * 100

        loss = loss_fn(u, u_prev, tau)

        assert loss.shape == (4,)
        assert torch.all(loss >= 0), "Loss should be non-negative"

    def test_zero_gradient_zero_loss(self):
        """Test that constant control gives zero loss."""
        loss_fn = ControlDeviationLoss()

        u = torch.ones(4, 10)
        u_prev = torch.ones(4, 10)  # Same as u
        tau = torch.rand(4, 5) * 100

        loss = loss_fn(u, u_prev, tau)

        assert torch.allclose(loss, torch.zeros(4), atol=1e-6)


class TestCouplingLoss:
    """Tests for combined coupling loss."""

    def test_basic_computation(self):
        """Test basic loss computation."""
        loss_fn = CouplingLoss(alpha=1.0, beta=0.1)

        rho = torch.tensor([0.3, 0.4])
        lambda_min_0 = torch.tensor(-0.5)
        u = torch.randn(2, 10)
        u_prev = torch.randn(2, 10)
        tau = torch.rand(2, 5) * 100

        loss, components = loss_fn(rho, lambda_min_0, u, u_prev, tau)

        assert torch.isfinite(loss)
        assert 'L_stability' in components
        assert 'L_deviation' in components
        assert 'rho_mean' in components

    def test_weight_scaling(self):
        """Test that alpha and beta scale loss correctly."""
        rho = torch.tensor([0.3])
        lambda_min_0 = torch.tensor(-0.5)
        u = torch.randn(1, 10)
        u_prev = torch.randn(1, 10)
        tau = torch.rand(1, 5) * 100

        loss_fn_1 = CouplingLoss(alpha=1.0, beta=0.1)
        loss_fn_2 = CouplingLoss(alpha=2.0, beta=0.1)

        loss1, comp1 = loss_fn_1(rho, lambda_min_0, u, u_prev, tau)
        loss2, comp2 = loss_fn_2(rho, lambda_min_0, u, u_prev, tau)

        # Higher alpha should increase stability component
        assert comp2['L_stability'] == comp1['L_stability']  # Raw value same
        assert loss2 > loss1  # But total loss higher due to weight


class TestEnergyLoss:
    """Tests for energy domain loss."""

    def test_generation_cost(self):
        """Test generation cost computation."""
        loss_fn = EnergyLoss(cost_weight=1.0, voltage_weight=0.0, frequency_weight=0.0)

        P_gen = torch.tensor([[100.0, 200.0, 150.0]])

        loss, components = loss_fn(P_gen=P_gen)

        assert torch.isfinite(loss)
        assert 'L_cost' in components

    def test_voltage_violation(self):
        """Test voltage violation penalty."""
        loss_fn = EnergyLoss(cost_weight=0.0, voltage_weight=1.0, v_min=0.95, v_max=1.05)

        P_gen = torch.tensor([[100.0]])

        # Normal voltage - no violation
        V_normal = torch.tensor([[1.0, 1.02, 0.98]])
        loss_normal, _ = loss_fn(P_gen=P_gen, V=V_normal)

        # Violated voltage
        V_violated = torch.tensor([[1.1, 0.9, 1.0]])  # Out of bounds
        loss_violated, _ = loss_fn(P_gen=P_gen, V=V_violated)

        assert loss_violated > loss_normal

    def test_frequency_deviation(self):
        """Test frequency deviation penalty."""
        loss_fn = EnergyLoss(cost_weight=0.0, frequency_weight=1.0)

        P_gen = torch.tensor([[100.0]])

        # Normal frequency
        omega_normal = torch.tensor([[1.0]])
        loss_normal, _ = loss_fn(P_gen=P_gen, omega=omega_normal)

        # Deviated frequency
        omega_deviated = torch.tensor([[1.1]])
        loss_deviated, _ = loss_fn(P_gen=P_gen, omega=omega_deviated)

        assert loss_deviated > loss_normal


class TestCommunicationLoss:
    """Tests for communication domain loss."""

    def test_latency_loss(self):
        """Test latency loss computation."""
        loss_fn = CommunicationLoss(
            latency_weight=1.0,
            bandwidth_weight=0.0,
            reliability_weight=0.0,
            jitter_weight=0.0,
        )

        tau_low = torch.tensor([[10.0, 20.0, 30.0]])
        tau_high = torch.tensor([[100.0, 200.0, 300.0]])

        loss_low, _ = loss_fn(tau=tau_low)
        loss_high, _ = loss_fn(tau=tau_high)

        assert loss_high > loss_low, "Higher delay should have higher loss"

    def test_jitter_loss(self):
        """Test jitter (variance) loss."""
        loss_fn = CommunicationLoss(
            latency_weight=0.0,
            jitter_weight=1.0,
        )

        # Low jitter (consistent delays)
        tau_consistent = torch.tensor([[50.0, 50.0, 50.0]])

        # High jitter (variable delays)
        tau_variable = torch.tensor([[10.0, 50.0, 90.0]])

        loss_consistent, _ = loss_fn(tau=tau_consistent)
        loss_variable, _ = loss_fn(tau=tau_variable)

        assert loss_variable > loss_consistent


class TestPhysicsAwareContrastiveLoss:
    """Tests for physics-aware contrastive loss."""

    def test_basic_computation(self):
        """Test basic contrastive loss computation."""
        loss_fn = PhysicsAwareContrastiveLoss(temperature=0.07)

        h_E = torch.randn(4, 10, 64)  # [batch, nodes, dim]
        h_I = torch.randn(4, 10, 64)

        loss, info = loss_fn(h_E, h_I)

        assert torch.isfinite(loss)
        assert 'contrastive_accuracy' in info
        assert 0 <= info['contrastive_accuracy'] <= 1

    def test_perfect_alignment(self):
        """Test loss with perfectly aligned embeddings."""
        loss_fn = PhysicsAwareContrastiveLoss(temperature=0.5)

        h_E = torch.randn(4, 10, 64)
        h_I = h_E.clone()  # Perfect alignment

        loss, info = loss_fn(h_E, h_I)

        # Accuracy should be high with perfect alignment
        assert info['contrastive_accuracy'] > 0.9

    def test_physics_weighting(self):
        """Test that impedance matrix affects loss."""
        loss_fn = PhysicsAwareContrastiveLoss(temperature=0.07, gamma=1.0)

        h_E = torch.randn(4, 10, 64)
        h_I = torch.randn(4, 10, 64)

        # Create impedance matrix
        impedance = torch.rand(10, 10) * 100

        loss_with_physics, _ = loss_fn(h_E, h_I, impedance_matrix=impedance)
        loss_without_physics, _ = loss_fn(h_E, h_I, impedance_matrix=None)

        # Losses should be different due to physics weighting
        assert not torch.allclose(loss_with_physics, loss_without_physics)


class TestJointLoss:
    """Tests for combined joint loss."""

    def test_all_components(self):
        """Test that all loss components are computed."""
        loss_fn = JointLoss()

        # Prepare all inputs
        batch_size = 2
        n_gen = 5
        n_nodes = 14

        u = torch.randn(batch_size, n_gen * 2)
        rho = torch.tensor([0.3, 0.4])
        h_E = torch.randn(batch_size, n_nodes, 64)
        h_I = torch.randn(batch_size, n_nodes, 64)
        P_gen = torch.randn(batch_size, n_gen) * 100
        tau = torch.rand(batch_size, n_gen) * 100
        lambda_min_0 = torch.tensor(-0.5)
        u_prev = torch.randn(batch_size, n_gen * 2)
        P_load = torch.ones(batch_size) * 100

        loss, components = loss_fn(
            u=u,
            rho=rho,
            h_E=h_E,
            h_I=h_I,
            P_gen=P_gen,
            tau=tau,
            lambda_min_0=lambda_min_0,
            u_prev=u_prev,
            P_load=P_load,
        )

        assert torch.isfinite(loss)
        assert 'L_total' in components
        assert 'L_E_total' in components
        assert 'L_I_total' in components

    def test_gradient_flow(self):
        """Test that gradients flow through joint loss."""
        loss_fn = JointLoss()

        batch_size = 2
        n_gen = 5
        n_nodes = 14

        # Make some inputs require grad
        u = torch.randn(batch_size, n_gen * 2, requires_grad=True)
        h_E = torch.randn(batch_size, n_nodes, 64, requires_grad=True)
        h_I = torch.randn(batch_size, n_nodes, 64, requires_grad=True)

        rho = torch.tensor([0.3, 0.4])
        P_gen = torch.randn(batch_size, n_gen) * 100
        tau = torch.rand(batch_size, n_gen) * 100
        lambda_min_0 = torch.tensor(-0.5)
        u_prev = torch.randn(batch_size, n_gen * 2)
        P_load = torch.ones(batch_size) * 100

        loss, _ = loss_fn(
            u=u,
            rho=rho,
            h_E=h_E,
            h_I=h_I,
            P_gen=P_gen,
            tau=tau,
            lambda_min_0=lambda_min_0,
            u_prev=u_prev,
            P_load=P_load,
        )

        loss.backward()

        assert u.grad is not None
        assert h_E.grad is not None
        assert h_I.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
