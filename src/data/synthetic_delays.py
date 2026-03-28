#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/22/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Synthetic Communication Delay Generator

Generates realistic communication delays for power grid communication networks.

Delay models:
1. Log-normal: Models network queuing delays
2. Exponential: Models random access delays
3. Uniform: Baseline for comparison
4. Topology-aware: Delays based on network path length
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Literal
from dataclasses import dataclass


@dataclass
class DelayConfig:
    """Configuration for delay generation."""
    distribution: Literal["lognormal", "exponential", "uniform", "gamma", "pareto"] = "lognormal"
    mean_ms: float = 50.0  # Mean delay in milliseconds
    std_ms: float = 20.0   # Standard deviation
    min_ms: float = 5.0    # Minimum delay
    max_ms: float = 500.0  # Maximum delay
    pareto_alpha: float = 1.5  # Pareto shape parameter (V2, Q2.5)


class SyntheticDelayGenerator:
    """
    Generate synthetic communication delays for training.

    Supports multiple delay distributions that model different
    communication network characteristics.
    """

    def __init__(
        self,
        n_generators: int,
        config: Optional[DelayConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_generators: Number of generators in the system
            config: Delay configuration
            seed: Random seed for reproducibility
        """
        self.n_generators = n_generators
        self.config = config or DelayConfig()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Precompute distribution parameters
        self._setup_distribution()

    def _setup_distribution(self):
        """Setup distribution parameters from config."""
        mean = self.config.mean_ms
        std = self.config.std_ms

        if self.config.distribution == "lognormal":
            # Log-normal: X = exp(μ + σZ) where Z ~ N(0,1)
            # E[X] = exp(μ + σ²/2)
            # Var[X] = (exp(σ²) - 1) * exp(2μ + σ²)
            # Solve for μ, σ given mean, std
            variance = std ** 2
            self.mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
            self.sigma = np.sqrt(np.log(1 + variance / mean ** 2))

        elif self.config.distribution == "exponential":
            # Exponential: rate = 1/mean
            self.rate = 1.0 / mean

        elif self.config.distribution == "gamma":
            # Gamma: shape k, scale θ
            # mean = k*θ, var = k*θ²
            variance = std ** 2
            self.k = mean ** 2 / variance
            self.theta = variance / mean

        elif self.config.distribution == "uniform":
            # Uniform: [a, b]
            # mean = (a+b)/2, var = (b-a)²/12
            range_half = std * np.sqrt(3)
            self.a = max(self.config.min_ms, mean - range_half)
            self.b = min(self.config.max_ms, mean + range_half)

        elif self.config.distribution == "pareto":
            # Pareto: heavy-tail distribution (V2, Q2.5)
            self.pareto_alpha = self.config.pareto_alpha
            self.pareto_x_min = self.config.min_ms

    def generate(
        self,
        batch_size: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate batch of delay samples.

        Args:
            batch_size: Number of samples to generate
            device: Target device

        Returns:
            tau: Delays [batch_size, n_generators] in ms
        """
        if self.config.distribution == "lognormal":
            # Sample from log-normal
            z = torch.randn(batch_size, self.n_generators)
            tau = torch.exp(self.mu + self.sigma * z)

        elif self.config.distribution == "exponential":
            # Sample from exponential
            u = torch.rand(batch_size, self.n_generators)
            tau = -torch.log(u) / self.rate

        elif self.config.distribution == "gamma":
            # Sample from gamma
            tau = torch.distributions.Gamma(self.k, 1/self.theta).sample(
                (batch_size, self.n_generators)
            )

        elif self.config.distribution == "uniform":
            # Sample from uniform
            tau = torch.rand(batch_size, self.n_generators) * (self.b - self.a) + self.a

        elif self.config.distribution == "pareto":
            # Sample from Pareto (V2, Q2.5)
            u = torch.rand(batch_size, self.n_generators)
            tau = self.pareto_x_min / (u ** (1.0 / self.pareto_alpha))

        else:
            raise ValueError(f"Unknown distribution: {self.config.distribution}")

        # Clamp to valid range
        tau = torch.clamp(tau, min=self.config.min_ms, max=self.config.max_ms)

        return tau.to(device)

    def generate_with_topology(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        batch_size: int = 1,
        hop_delay_ms: float = 10.0,
        base_delay_ms: float = 5.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate delays based on network topology.

        Delays increase with hop count from a central node.

        Args:
            edge_index: Communication network edges [2, E]
            num_nodes: Number of nodes
            batch_size: Number of samples
            hop_delay_ms: Additional delay per hop
            base_delay_ms: Base processing delay
            device: Target device

        Returns:
            tau: Delays [batch_size, n_generators] in ms
        """
        # Build adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  # Symmetric

        # Compute shortest paths using BFS (approximate with matrix powers)
        dist = torch.full((num_nodes, num_nodes), float('inf'))
        dist.fill_diagonal_(0)

        power = adj.clone()
        for d in range(1, num_nodes):
            mask = (dist == float('inf')) & (power > 0)
            dist[mask] = d
            power = power @ adj

        # Use distance from node 0 (control center) as delay basis
        hop_counts = dist[0, :self.n_generators]
        hop_counts = torch.where(
            torch.isinf(hop_counts),
            torch.tensor(num_nodes, dtype=torch.float),
            hop_counts
        )

        # Base delay + hop delay + random jitter
        base = base_delay_ms + hop_delay_ms * hop_counts
        jitter = self.generate(batch_size, device='cpu')

        tau = base.unsqueeze(0) + 0.2 * jitter  # 20% jitter

        # Clamp and move to device
        tau = torch.clamp(tau, min=self.config.min_ms, max=self.config.max_ms)
        return tau.to(device)

    def compute_tau_max(
        self,
        lambda_min_0: float,
        K: torch.Tensor,
        safety_factor: float = 0.9,
    ) -> torch.Tensor:
        """
        Compute maximum tolerable delays for stability.

        From Theorem 1: ρ(τ) = |λ_min(0)| - Σ_i (K_i · τ_i / τ_max,i) > 0

        For stability, we need Σ_i (K_i · τ_i / τ_max,i) < |λ_min(0)|

        Args:
            lambda_min_0: Minimum eigenvalue at zero delay
            K: Coupling constants [n_gen]
            safety_factor: Safety margin (< 1 for conservative estimate)

        Returns:
            tau_max: Maximum tolerable delays [n_gen] in ms
        """
        # Assume equal distribution of stability budget
        stability_budget = safety_factor * abs(lambda_min_0)
        budget_per_gen = stability_budget / self.n_generators

        # τ_max,i = budget_per_gen / K_i (in appropriate units)
        tau_max = budget_per_gen / (K + 1e-8)

        # Convert to ms and clamp to reasonable range
        tau_max = torch.clamp(tau_max * 1000, min=10.0, max=1000.0)

        return tau_max

    def get_statistics(self, num_samples: int = 10000) -> Dict[str, float]:
        """
        Compute empirical statistics of the delay distribution.

        Args:
            num_samples: Number of samples for estimation

        Returns:
            Dict with mean, std, min, max, percentiles
        """
        samples = self.generate(num_samples)

        return {
            "mean": samples.mean().item(),
            "std": samples.std().item(),
            "min": samples.min().item(),
            "max": samples.max().item(),
            "p50": samples.median().item(),
            "p95": torch.quantile(samples, 0.95).item(),
            "p99": torch.quantile(samples, 0.99).item(),
        }


class CorrelatedDelayGenerator(SyntheticDelayGenerator):
    """
    Generate spatially correlated delays.

    Models the fact that nearby nodes in the communication network
    often experience similar delays due to shared infrastructure.
    """

    def __init__(
        self,
        n_generators: int,
        correlation_matrix: Optional[torch.Tensor] = None,
        config: Optional[DelayConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_generators: Number of generators
            correlation_matrix: Delay correlation [n_gen, n_gen]
            config: Delay configuration
            seed: Random seed
        """
        super().__init__(n_generators, config, seed)

        if correlation_matrix is not None:
            # Compute Cholesky decomposition for sampling
            self.L = torch.linalg.cholesky(correlation_matrix)
        else:
            # Default: identity (independent delays)
            self.L = torch.eye(n_generators)

    def generate(
        self,
        batch_size: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Generate correlated delays."""
        # Generate independent standard normal
        z = torch.randn(batch_size, self.n_generators)

        # Apply correlation via Cholesky
        z_correlated = z @ self.L.T

        # Transform to target distribution
        if self.config.distribution == "lognormal":
            tau = torch.exp(self.mu + self.sigma * z_correlated)
        else:
            # For other distributions, use copula approach (simplified)
            # Transform to uniform via standard normal CDF, then to target
            u = torch.distributions.Normal(0, 1).cdf(z_correlated)
            tau = self.config.min_ms + u * (self.config.max_ms - self.config.min_ms)

        tau = torch.clamp(tau, min=self.config.min_ms, max=self.config.max_ms)
        return tau.to(device)

    @classmethod
    def from_impedance(
        cls,
        impedance_matrix: torch.Tensor,
        n_generators: int,
        config: Optional[DelayConfig] = None,
        seed: Optional[int] = None,
    ) -> "CorrelatedDelayGenerator":
        """
        Create generator with correlations derived from impedance.

        Nodes with lower impedance (electrically close) have more
        correlated delays.

        Args:
            impedance_matrix: Impedance [n_gen, n_gen]
            n_generators: Number of generators
            config: Delay config
            seed: Random seed

        Returns:
            Generator instance
        """
        # Correlation inversely proportional to impedance
        Z = impedance_matrix[:n_generators, :n_generators]
        Z_max = Z.max() + 1e-8

        # Correlation = 1 - Z/Z_max (normalized)
        correlation = 1 - Z / Z_max

        # Ensure positive definiteness
        correlation = 0.5 * (correlation + correlation.T)  # Symmetrize
        correlation.fill_diagonal_(1.0)

        # Add small diagonal for numerical stability
        correlation = correlation + 0.01 * torch.eye(n_generators)
        correlation = correlation / correlation.diagonal().unsqueeze(1).sqrt()
        correlation = correlation / correlation.diagonal().unsqueeze(0).sqrt()

        return cls(n_generators, correlation, config, seed)
