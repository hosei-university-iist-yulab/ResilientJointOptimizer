#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/02/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Markov Channel Simulator

Simulates a 3-state Markov chain for each generator's communication link:
  State 0 (GOOD):     Normal operation — low delay, low loss, low jitter
  State 1 (DEGRADED): Congestion/interference — moderate impairments
  State 2 (FAILED):   Link failure — total packet loss, no communication

The transition matrix is configurable. Default values represent
a typical SCADA communication link with infrequent failures.

Reference: Extended Gilbert-Elliott model with 3 states.
  Gilbert (1960), Elliott (1963), Bell System Technical Journal.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


STATE_GOOD = 0
STATE_DEGRADED = 1
STATE_FAILED = 2
STATE_NAMES = ["GOOD", "DEGRADED", "FAILED"]


@dataclass
class ChannelConfig:
    """Configuration for Markov channel model."""

    # Transition matrix P[i,j] = P(next=j | current=i)
    # Default: GOOD is sticky (0.95), DEGRADED moderately sticky (0.85),
    # FAILED has 20% recovery chance per step
    transition_matrix: List[List[float]] = field(default_factory=lambda: [
        [0.95, 0.04, 0.01],  # From GOOD
        [0.05, 0.85, 0.10],  # From DEGRADED
        [0.20, 0.10, 0.70],  # From FAILED
    ])

    # Initial state distribution (start in GOOD)
    initial_distribution: List[float] = field(default_factory=lambda: [0.9, 0.08, 0.02])


class ChannelSimulator:
    """
    Simulate Markov channel states for each generator's communication link.

    Usage:
        sim = ChannelSimulator(n_generators=10, config=ChannelConfig())
        states = sim.simulate(batch_size=1000)
        # states: [1000, 10] with values in {0, 1, 2}
    """

    def __init__(
        self,
        n_generators: int,
        config: Optional[ChannelConfig] = None,
        seed: Optional[int] = None,
    ):
        self.n_generators = n_generators
        self.config = config or ChannelConfig()
        self.rng = np.random.RandomState(seed)

        # Validate and store transition matrix
        P = np.array(self.config.transition_matrix, dtype=np.float64)
        assert P.shape == (3, 3), f"Transition matrix must be 3x3, got {P.shape}"
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"Rows must sum to 1, got {row_sums}"
        self.P = P

        # Initial distribution
        pi0 = np.array(self.config.initial_distribution, dtype=np.float64)
        assert len(pi0) == 3
        pi0 = pi0 / pi0.sum()  # Normalize
        self.pi0 = pi0

    def simulate(
        self,
        batch_size: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Sample channel states for each generator (single time step).

        Each generator's state is sampled independently from the
        stationary distribution of the Markov chain.

        Args:
            batch_size: Number of scenarios
            device: Target device

        Returns:
            states: Channel state indices [batch_size, n_generators] (0/1/2)
        """
        # Compute stationary distribution: pi @ P = pi
        # Solve (P^T - I) pi = 0 with sum(pi) = 1
        stationary = self._compute_stationary()

        states = self.rng.choice(
            3, size=(batch_size, self.n_generators), p=stationary,
        )
        return torch.tensor(states, dtype=torch.long, device=device)

    def simulate_trajectory(
        self,
        n_steps: int,
        batch_size: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Simulate multi-step trajectories for each generator.

        Useful for evaluating channel state prediction and
        proactive stability management.

        Args:
            n_steps: Number of time steps
            batch_size: Number of independent trajectories
            device: Target device

        Returns:
            trajectories: [batch_size, n_generators, n_steps] (0/1/2)
        """
        trajs = np.zeros((batch_size, self.n_generators, n_steps), dtype=np.int64)

        # Sample initial states
        for b in range(batch_size):
            for g in range(self.n_generators):
                state = self.rng.choice(3, p=self.pi0)
                trajs[b, g, 0] = state
                for t in range(1, n_steps):
                    state = self.rng.choice(3, p=self.P[state])
                    trajs[b, g, t] = state

        return torch.tensor(trajs, dtype=torch.long, device=device)

    def _compute_stationary(self) -> np.ndarray:
        """Compute stationary distribution of the Markov chain."""
        # Solve pi @ P = pi => pi @ (P - I) = 0 with sum(pi) = 1
        # Equivalent: (P^T - I) @ pi = 0
        A = self.P.T - np.eye(3)
        # Replace last equation with normalization constraint
        A[-1, :] = 1.0
        b = np.zeros(3)
        b[-1] = 1.0
        pi = np.linalg.solve(A, b)
        pi = np.clip(pi, 0, 1)
        pi /= pi.sum()
        return pi

    def get_stationary_distribution(self) -> np.ndarray:
        """Return the stationary distribution."""
        return self._compute_stationary()

    def get_state_name(self, state_idx: int) -> str:
        """Return human-readable state name."""
        return STATE_NAMES[state_idx]
