#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/17/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Unit Tests for Model Components

Tests:
- LearnableCouplingConstants
- GNN encoders
- Attention modules
- JointOptimizer
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    JointOptimizer,
    EnergyGNN,
    CommunicationGNN,
    DualDomainGNN,
    LearnableCouplingConstants,
    CausalAttention,
    CrossDomainAttention,
    HierarchicalAttention,
    PhysicsMask,
    CausalMask,
)


class TestLearnableCouplingConstants:
    """Tests for LearnableCouplingConstants."""

    def test_initialization(self):
        """Test proper initialization."""
        n_gen = 5
        coupling = LearnableCouplingConstants(n_gen, init_scale=0.1)
        K = coupling()

        assert K.shape == (n_gen,)
        assert torch.all(K > 0), "K_i must be positive"
        assert torch.allclose(K, torch.ones(n_gen) * 0.1, atol=1e-6)

    def test_positivity_guarantee(self):
        """Test that K_i remains positive after optimization."""
        n_gen = 5
        coupling = LearnableCouplingConstants(n_gen, init_scale=0.1)

        # Simulate gradient descent that would push K negative
        optimizer = torch.optim.SGD(coupling.parameters(), lr=10.0)

        for _ in range(100):
            K = coupling()
            loss = K.sum()  # Push K toward zero
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        K_final = coupling()
        assert torch.all(K_final > 0), "K_i must remain positive"

    def test_learnable_flag(self):
        """Test learnable vs fixed K_i."""
        n_gen = 5

        # Learnable
        coupling_learn = LearnableCouplingConstants(n_gen, learnable=True)
        assert hasattr(coupling_learn, 'log_K')
        assert isinstance(coupling_learn.log_K, nn.Parameter)

        # Fixed
        coupling_fixed = LearnableCouplingConstants(n_gen, learnable=False)
        assert not isinstance(coupling_fixed.log_K, nn.Parameter)


class TestGNNEncoders:
    """Tests for GNN encoder modules."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample graph data."""
        n_nodes = 14
        n_edges = 20
        x = torch.randn(n_nodes, 5)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        return x, edge_index, n_nodes

    def test_energy_gnn_shapes(self, sample_graph):
        """Test EnergyGNN output shapes."""
        x, edge_index, n_nodes = sample_graph

        gnn = EnergyGNN(input_dim=5, hidden_dim=64, output_dim=32, num_layers=2)
        h = gnn(x, edge_index)

        assert h.shape == (n_nodes, 32)

    def test_communication_gnn_shapes(self, sample_graph):
        """Test CommunicationGNN output shapes."""
        x, edge_index, n_nodes = sample_graph
        x_comm = torch.randn(n_nodes, 3)

        gnn = CommunicationGNN(input_dim=3, hidden_dim=64, output_dim=32, num_layers=2)
        h = gnn(x_comm, edge_index, n_nodes)

        assert h.shape == (n_nodes, 32)

    def test_dual_domain_gnn(self, sample_graph):
        """Test DualDomainGNN output shapes."""
        _, edge_index, n_nodes = sample_graph
        x_energy = torch.randn(n_nodes, 5)
        x_comm = torch.randn(n_nodes, 3)

        gnn = DualDomainGNN(
            energy_input_dim=5,
            comm_input_dim=3,
            hidden_dim=64,
            output_dim=32,
        )
        h_E, h_I = gnn(x_energy, edge_index, x_comm, edge_index)

        assert h_E.shape == (n_nodes, 32)
        assert h_I.shape == (n_nodes, 32)


class TestAttentionModules:
    """Tests for attention modules."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        batch_size = 2
        n_nodes = 14
        embed_dim = 64
        h_E = torch.randn(batch_size, n_nodes, embed_dim)
        h_I = torch.randn(batch_size, n_nodes, embed_dim)
        return h_E, h_I, batch_size, n_nodes, embed_dim

    def test_physics_mask(self):
        """Test PhysicsMask computation."""
        n_nodes = 5
        impedance = torch.rand(n_nodes, n_nodes) * 100

        mask = PhysicsMask(gamma=1.0)
        M = mask(impedance)

        assert M.shape == (n_nodes, n_nodes)
        assert torch.all(M <= 0), "Physics mask should be non-positive"
        # Diagonal should be zero (self-impedance normalized to 0)
        assert torch.allclose(M.diagonal(), torch.zeros(n_nodes), atol=0.1)

    def test_causal_mask(self):
        """Test CausalMask computation."""
        n_nodes = 5
        # Create simple DAG: 0 -> 1 -> 2 -> 3 -> 4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

        mask = CausalMask()
        M = mask.from_dag(edge_index, n_nodes)

        assert M.shape == (n_nodes, n_nodes)
        # Non-ancestors should be -inf
        assert torch.isinf(M[0, 4])  # 4 is not ancestor of 0

    def test_hierarchical_attention(self, sample_embeddings):
        """Test HierarchicalAttention output shapes."""
        h_E, h_I, batch_size, n_nodes, embed_dim = sample_embeddings

        attn = HierarchicalAttention(embed_dim=embed_dim, num_heads=4)
        h_fused, attn_info = attn(h_E, h_I)

        assert h_fused.shape == (batch_size, n_nodes, embed_dim)
        assert 'causal_attn' in attn_info
        assert 'cross_attn' in attn_info


class TestJointOptimizer:
    """Tests for the main JointOptimizer model."""

    @pytest.fixture
    def model(self):
        """Create JointOptimizer instance."""
        return JointOptimizer(
            n_generators=5,
            energy_input_dim=5,
            comm_input_dim=3,
            embed_dim=64,
            hidden_dim=128,
            num_heads=4,
            gnn_layers=2,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        batch_size = 2
        n_nodes = 14
        n_gen = 5

        return {
            'energy_x': torch.randn(batch_size * n_nodes, 5),
            'comm_x': torch.randn(batch_size * n_nodes, 3),
            'edge_index': torch.randint(0, n_nodes, (2, 20)),
            'tau': torch.rand(batch_size, n_gen) * 100,
            'tau_max': torch.ones(n_gen) * 500,
            'lambda_min_0': torch.tensor([-0.5]),
            'batch': torch.arange(batch_size).repeat_interleave(n_nodes),
        }

    def test_forward_shapes(self, model, sample_batch):
        """Test forward pass output shapes."""
        outputs = model(
            energy_x=sample_batch['energy_x'],
            energy_edge_index=sample_batch['edge_index'],
            comm_x=sample_batch['comm_x'],
            comm_edge_index=sample_batch['edge_index'],
            tau=sample_batch['tau'],
            tau_max=sample_batch['tau_max'],
            lambda_min_0=sample_batch['lambda_min_0'],
            batch=sample_batch['batch'],
        )

        batch_size = 2
        n_gen = 5
        n_nodes = 14

        assert outputs['u'].shape == (batch_size, n_gen * 2)
        assert outputs['rho'].shape == (batch_size,)
        assert outputs['K'].shape == (n_gen,)
        assert outputs['h_E'].shape == (batch_size, n_nodes, 64)

    def test_stability_margin_computation(self, model):
        """Test stability margin is computed correctly."""
        tau = torch.tensor([[50.0, 100.0, 150.0, 200.0, 250.0]])
        tau_max = torch.ones(5) * 500
        lambda_min_0 = torch.tensor([-0.5])

        rho = model.get_stability_margin(tau, tau_max, lambda_min_0)

        # rho should be less than |lambda_min_0| due to delay contribution
        assert rho.item() < abs(lambda_min_0.item())
        assert rho.item() > 0  # Still stable

    def test_gradient_flow(self, model, sample_batch):
        """Test gradients flow through the model."""
        outputs = model(
            energy_x=sample_batch['energy_x'],
            energy_edge_index=sample_batch['edge_index'],
            comm_x=sample_batch['comm_x'],
            comm_edge_index=sample_batch['edge_index'],
            tau=sample_batch['tau'],
            tau_max=sample_batch['tau_max'],
            lambda_min_0=sample_batch['lambda_min_0'],
            batch=sample_batch['batch'],
        )

        loss = outputs['u'].sum() + outputs['rho'].sum()
        loss.backward()

        # Check gradients exist for key parameters
        assert model.coupling.log_K.grad is not None
        assert torch.any(model.coupling.log_K.grad != 0)


class TestModelSerialization:
    """Tests for model saving/loading."""

    def test_save_load_state_dict(self, tmp_path):
        """Test saving and loading model state."""
        model = JointOptimizer(n_generators=5, embed_dim=64)

        # Save
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        # Load into new model
        model2 = JointOptimizer(n_generators=5, embed_dim=64)
        model2.load_state_dict(torch.load(save_path))

        # Check K values match
        K1 = model.get_coupling_constants()
        K2 = model2.get_coupling_constants()
        assert torch.allclose(K1, K2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
