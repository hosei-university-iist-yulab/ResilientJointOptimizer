#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/15/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Unit Tests for Data Pipeline

Tests:
- SyntheticDelayGenerator
- DelayConfig
- PowerGridDataset
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    SyntheticDelayGenerator,
    DelayConfig,
    CorrelatedDelayGenerator,
)


class TestDelayConfig:
    """Tests for DelayConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DelayConfig()

        assert config.distribution == "lognormal"
        assert config.mean_ms == 50.0
        assert config.std_ms == 20.0
        assert config.min_ms == 5.0
        assert config.max_ms == 500.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = DelayConfig(
            distribution="uniform",
            mean_ms=100.0,
            std_ms=30.0,
            min_ms=10.0,
            max_ms=1000.0,
        )

        assert config.distribution == "uniform"
        assert config.mean_ms == 100.0


class TestSyntheticDelayGenerator:
    """Tests for SyntheticDelayGenerator."""

    @pytest.fixture
    def generator(self):
        """Create default generator."""
        return SyntheticDelayGenerator(n_generators=5, seed=42)

    def test_generate_shapes(self, generator):
        """Test output shapes."""
        batch_size = 16
        tau = generator.generate(batch_size)

        assert tau.shape == (batch_size, 5)

    def test_generate_range(self, generator):
        """Test values are within bounds."""
        tau = generator.generate(1000)

        assert torch.all(tau >= generator.config.min_ms)
        assert torch.all(tau <= generator.config.max_ms)

    def test_lognormal_distribution(self):
        """Test lognormal distribution properties."""
        config = DelayConfig(distribution="lognormal", mean_ms=50.0, std_ms=20.0)
        generator = SyntheticDelayGenerator(n_generators=5, config=config, seed=42)

        # Generate many samples
        tau = generator.generate(10000)

        # Check approximate mean (accounting for clamping)
        mean = tau.mean().item()
        assert 30 < mean < 100, f"Mean {mean} not in expected range"

    def test_uniform_distribution(self):
        """Test uniform distribution properties."""
        config = DelayConfig(distribution="uniform", mean_ms=100.0, std_ms=50.0)
        generator = SyntheticDelayGenerator(n_generators=5, config=config, seed=42)

        tau = generator.generate(10000)

        # Uniform should have flatter histogram
        mean = tau.mean().item()
        assert 50 < mean < 150

    def test_exponential_distribution(self):
        """Test exponential distribution properties."""
        config = DelayConfig(distribution="exponential", mean_ms=50.0)
        generator = SyntheticDelayGenerator(n_generators=5, config=config, seed=42)

        tau = generator.generate(10000)

        # Check values are positive
        assert torch.all(tau > 0)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        gen1 = SyntheticDelayGenerator(n_generators=5, seed=42)
        gen2 = SyntheticDelayGenerator(n_generators=5, seed=42)

        tau1 = gen1.generate(100)
        tau2 = gen2.generate(100)

        assert torch.allclose(tau1, tau2)

    def test_device_placement(self, generator):
        """Test device placement."""
        tau_cpu = generator.generate(10, device='cpu')
        assert tau_cpu.device.type == 'cpu'

    def test_statistics(self, generator):
        """Test get_statistics method."""
        stats = generator.get_statistics(num_samples=10000)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'p50' in stats
        assert 'p95' in stats

        # Check min/max are within bounds
        assert stats['min'] >= generator.config.min_ms
        assert stats['max'] <= generator.config.max_ms

    def test_tau_max_computation(self, generator):
        """Test tau_max computation from stability."""
        lambda_min_0 = -0.5
        K = torch.ones(5) * 0.1

        tau_max = generator.compute_tau_max(lambda_min_0, K, safety_factor=0.9)

        assert tau_max.shape == (5,)
        assert torch.all(tau_max > 0)
        assert torch.all(tau_max <= 1000)  # Reasonable upper bound


class TestCorrelatedDelayGenerator:
    """Tests for CorrelatedDelayGenerator."""

    def test_basic_generation(self):
        """Test basic correlated delay generation."""
        n_gen = 5

        # Create correlation matrix
        correlation = torch.eye(n_gen)
        correlation[0, 1] = 0.8
        correlation[1, 0] = 0.8

        generator = CorrelatedDelayGenerator(
            n_generators=n_gen,
            correlation_matrix=correlation,
            seed=42,
        )

        tau = generator.generate(1000)

        assert tau.shape == (1000, n_gen)

    def test_correlation_preserved(self):
        """Test that correlations are approximately preserved."""
        n_gen = 3

        # Strong correlation between first two generators
        correlation = torch.tensor([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])

        generator = CorrelatedDelayGenerator(
            n_generators=n_gen,
            correlation_matrix=correlation,
            config=DelayConfig(distribution="lognormal"),
            seed=42,
        )

        tau = generator.generate(5000)

        # Compute empirical correlation
        tau_normalized = (tau - tau.mean(dim=0)) / tau.std(dim=0)
        empirical_corr = tau_normalized.T @ tau_normalized / tau.shape[0]

        # Check correlation between first two is high
        assert empirical_corr[0, 1].item() > 0.5, "Correlation not preserved"

    def test_from_impedance(self):
        """Test creating generator from impedance matrix."""
        n_gen = 5
        n_bus = 10

        # Create fake impedance matrix
        impedance = torch.rand(n_bus, n_bus) * 100
        impedance = (impedance + impedance.T) / 2  # Symmetric
        impedance.fill_diagonal_(0)

        generator = CorrelatedDelayGenerator.from_impedance(
            impedance_matrix=impedance,
            n_generators=n_gen,
            seed=42,
        )

        tau = generator.generate(100)

        assert tau.shape == (100, n_gen)


class TestDatasetIntegration:
    """Integration tests for dataset (requires pandapower)."""

    @pytest.mark.skipif(
        True,  # Skip by default as it requires pandapower
        reason="Requires pandapower installation"
    )
    def test_power_grid_dataset(self):
        """Test PowerGridDataset creation."""
        from src.data import PowerGridDataset

        dataset = PowerGridDataset(
            case_id=14,
            num_scenarios=100,
            seed=42,
        )

        assert len(dataset) == 100

        sample = dataset[0]
        assert 'energy_x' in sample
        assert 'comm_x' in sample
        assert 'tau' in sample

    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Requires pandapower installation"
    )
    def test_create_dataloaders(self):
        """Test dataloader creation."""
        from src.data import create_dataloaders

        train_loader, val_loader, test_loader = create_dataloaders(
            case_id=14,
            num_scenarios=100,
            batch_size=16,
            num_workers=0,
            seed=42,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
