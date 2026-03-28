#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/01/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Communication Impairment Generator

Generates packet loss rates and jitter standard deviations for
multi-impairment stability analysis (Topic 2: Theorem 2).

Distributions:
  - Packet loss p_i ~ Beta(alpha, beta), clipped to [0, p_max]
  - Jitter sigma_j_i ~ Gamma(shape, scale), clipped to [0, sigma_max]

These are sampled independently per generator per scenario.
The Markov channel model (channel_simulator.py) can override
these with state-dependent values.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ImpairmentConfig:
    """Configuration for communication impairment generation."""

    # Packet loss: Beta distribution
    loss_alpha: float = 2.0       # Beta shape alpha
    loss_beta: float = 20.0       # Beta shape beta (mean ~ alpha/(alpha+beta) ~ 9%)
    loss_max: float = 0.5         # Maximum loss rate (p_crit safety)

    # Jitter: Gamma distribution
    jitter_shape: float = 2.0     # Gamma shape k
    jitter_scale: float = 15.0    # Gamma scale theta (mean = k*theta ~ 30ms)
    jitter_max_ms: float = 200.0  # Maximum jitter std in ms


class ImpairmentGenerator:
    """
    Generate packet loss rates and jitter for each generator.

    Usage:
        gen = ImpairmentGenerator(n_generators=10, config=ImpairmentConfig())
        p, sigma_j = gen.generate(batch_size=1000)
        # p: [1000, 10] in [0, 0.5]
        # sigma_j: [1000, 10] in [0, 200] ms
    """

    def __init__(
        self,
        n_generators: int,
        config: Optional[ImpairmentConfig] = None,
        seed: Optional[int] = None,
    ):
        self.n_generators = n_generators
        self.config = config or ImpairmentConfig()
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        batch_size: int = 1,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch of impairment samples.

        Args:
            batch_size: Number of scenarios
            device: Target device

        Returns:
            p: Packet loss rates [batch_size, n_generators] in [0, loss_max]
            sigma_j: Jitter std [batch_size, n_generators] in [0, jitter_max] ms
        """
        cfg = self.config

        # Packet loss: Beta distribution
        p_raw = self.rng.beta(
            cfg.loss_alpha, cfg.loss_beta,
            size=(batch_size, self.n_generators),
        )
        p = np.clip(p_raw, 0.0, cfg.loss_max)

        # Jitter: Gamma distribution
        sigma_raw = self.rng.gamma(
            cfg.jitter_shape, cfg.jitter_scale,
            size=(batch_size, self.n_generators),
        )
        sigma_j = np.clip(sigma_raw, 0.0, cfg.jitter_max_ms)

        return (
            torch.tensor(p, dtype=torch.float32, device=device),
            torch.tensor(sigma_j, dtype=torch.float32, device=device),
        )

    def generate_from_channel_state(
        self,
        states: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate impairments conditioned on Markov channel states.

        Each state has characteristic impairment levels:
          GOOD (0):     p ~ Beta(2, 200) mean ~1%,  sigma ~ Gamma(2, 2.5) mean ~5ms
          DEGRADED (1): p ~ Beta(2, 18)  mean ~10%, sigma ~ Gamma(2, 25)  mean ~50ms
          FAILED (2):   p = 1.0 (total loss),       sigma = 0 (irrelevant)

        Args:
            states: Channel state indices [batch_size, n_generators] (0/1/2)
            device: Target device

        Returns:
            p: Packet loss rates [batch_size, n_generators]
            sigma_j: Jitter std [batch_size, n_generators] ms
        """
        batch_size, n_gen = states.shape
        p = np.zeros((batch_size, n_gen), dtype=np.float64)
        sigma_j = np.zeros((batch_size, n_gen), dtype=np.float64)

        # State-dependent parameters
        state_params = {
            0: {"loss_a": 2, "loss_b": 200, "jit_k": 2, "jit_s": 2.5},   # GOOD
            1: {"loss_a": 2, "loss_b": 18, "jit_k": 2, "jit_s": 25.0},    # DEGRADED
            # FAILED: deterministic p=1, sigma=0
        }

        for s_val, params in state_params.items():
            mask = states == s_val
            n_samples = mask.sum()
            if n_samples == 0:
                continue
            p[mask] = self.rng.beta(params["loss_a"], params["loss_b"], size=n_samples)
            sigma_j[mask] = self.rng.gamma(params["jit_k"], params["jit_s"], size=n_samples)

        # FAILED state: total packet loss
        failed_mask = states == 2
        p[failed_mask] = 1.0
        sigma_j[failed_mask] = 0.0

        # Clip
        p = np.clip(p, 0.0, self.config.loss_max)
        sigma_j = np.clip(sigma_j, 0.0, self.config.jitter_max_ms)

        return (
            torch.tensor(p, dtype=torch.float32, device=device),
            torch.tensor(sigma_j, dtype=torch.float32, device=device),
        )
