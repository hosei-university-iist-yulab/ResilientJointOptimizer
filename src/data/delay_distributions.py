#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/21/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Delay Distribution Library (V2)

Addresses Q2.5: Train on lognormal, evaluate on all distributions.
Includes realistic 5G/6G heavy-tail models.
"""

import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from .synthetic_delays import DelayConfig, SyntheticDelayGenerator


@dataclass
class ParetoDelayConfig:
    """Pareto (heavy-tail) delay configuration."""
    alpha: float = 1.5  # Shape parameter (lower = heavier tail)
    x_min_ms: float = 10.0  # Scale parameter (minimum delay)
    max_ms: float = 1000.0


# Standard delay distributions for robustness testing
DELAY_DISTRIBUTIONS = {
    'lognormal': DelayConfig(
        distribution='lognormal', mean_ms=50.0, std_ms=20.0,
    ),
    'exponential': DelayConfig(
        distribution='exponential', mean_ms=50.0, std_ms=50.0,
    ),
    'gamma': DelayConfig(
        distribution='gamma', mean_ms=50.0, std_ms=25.0,
    ),
    'uniform': DelayConfig(
        distribution='uniform', mean_ms=50.0, std_ms=25.0,
    ),
    'pareto': DelayConfig(
        distribution='pareto', mean_ms=50.0, std_ms=40.0,
    ),
}


def generate_pareto_delays(
    n_generators: int,
    batch_size: int = 1,
    alpha: float = 1.5,
    x_min_ms: float = 10.0,
    max_ms: float = 1000.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate Pareto-distributed delays (5G/6G realistic heavy tails).

    P(X > x) = (x_min / x)^alpha for x >= x_min

    Args:
        n_generators: Number of generators
        batch_size: Number of samples
        alpha: Shape parameter (lower = heavier tails)
        x_min_ms: Scale parameter (minimum delay)
        max_ms: Maximum delay cutoff
        seed: Random seed

    Returns:
        tau: Delays [batch_size, n_generators] in ms
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    pareto_samples = (rng.pareto(alpha, size=(batch_size, n_generators)) + 1) * x_min_ms
    delays = np.clip(pareto_samples, x_min_ms, max_ms)
    return torch.tensor(delays, dtype=torch.float32)


def get_distribution_stats(
    distribution_name: str,
    n_generators: int = 10,
    n_samples: int = 10000,
) -> Dict:
    """Get empirical statistics for a distribution."""
    config = DELAY_DISTRIBUTIONS[distribution_name]

    if distribution_name == 'pareto':
        samples = generate_pareto_delays(n_generators, n_samples)
    else:
        gen = SyntheticDelayGenerator(n_generators, config)
        samples = gen.generate(n_samples)

    return {
        'mean': samples.mean().item(),
        'std': samples.std().item(),
        'median': samples.median().item(),
        'p95': torch.quantile(samples, 0.95).item(),
        'p99': torch.quantile(samples, 0.99).item(),
        'min': samples.min().item(),
        'max': samples.max().item(),
    }
