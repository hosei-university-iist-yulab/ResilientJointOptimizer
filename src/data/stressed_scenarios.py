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
Stressed Scenario Generator for Stability Testing

Addresses Q1.6 and Q2.4: All models achieve 100% stability on normal conditions.
Need scenarios that push the system toward instability to differentiate models.

Stress types:
- High load: Scale loads beyond nominal (110%, 120%)
- N-1 contingency: Remove critical transmission lines
- Extreme delay: Push communication delays to 500-1000ms
- Heavy-tail delays: Pareto-distributed delays (5G/6G realistic)
- Combined: Multiple stressors simultaneously
"""

import torch
import numpy as np
from copy import deepcopy
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class StressConfig:
    """Configuration for a stress scenario."""
    name: str
    load_factor: float = 1.0
    remove_line_idx: Optional[int] = None
    remove_n_lines: int = 0
    tau_ms_override: Optional[float] = None
    delay_distribution: str = "lognormal"  # lognormal, pareto, uniform
    delay_alpha: float = 1.5  # Pareto shape parameter


# Predefined stress scenarios
STRESS_SCENARIOS = {
    'normal': StressConfig(name='normal'),
    'load_105': StressConfig(name='load_105', load_factor=1.05),
    'load_110': StressConfig(name='load_110', load_factor=1.10),
    'load_115': StressConfig(name='load_115', load_factor=1.15),
    'load_120': StressConfig(name='load_120', load_factor=1.20),
    'n1': StressConfig(name='n1', remove_n_lines=1),
    'n2': StressConfig(name='n2', remove_n_lines=2),
    'delay_300': StressConfig(name='delay_300', tau_ms_override=300.0),
    'delay_500': StressConfig(name='delay_500', tau_ms_override=500.0),
    'delay_800': StressConfig(name='delay_800', tau_ms_override=800.0),
    'delay_1000': StressConfig(name='delay_1000', tau_ms_override=1000.0),
    'pareto_delays': StressConfig(name='pareto_delays', delay_distribution='pareto'),
    'combined_moderate': StressConfig(
        name='combined_moderate', load_factor=1.10,
        tau_ms_override=300.0, remove_n_lines=1
    ),
    'combined_severe': StressConfig(
        name='combined_severe', load_factor=1.20,
        tau_ms_override=500.0, remove_n_lines=2
    ),
}


class StressedScenarioGenerator:
    """Generate stressed power system scenarios."""

    def __init__(self, base_case: Dict):
        """
        Args:
            base_case: Base IEEE case dict from IEEECaseLoader.load()
        """
        self.base_case = base_case

    def high_load(self, case: Dict, factor: float = 1.1) -> Dict:
        """
        Scale all loads by factor.

        At 1.2x, stability should degrade significantly.

        Args:
            case: Power system case dict
            factor: Load scaling factor

        Returns:
            Modified case dict
        """
        case = deepcopy(case)
        case['P_load'] = case['P_load'] * factor
        case['Q_load'] = case['Q_load'] * factor
        return case

    def n1_contingency(
        self,
        case: Dict,
        line_idx: Optional[int] = None,
        n_lines: int = 1,
        seed: int = 42,
    ) -> Dict:
        """
        Remove transmission lines (N-1 or N-k contingency).

        Args:
            case: Power system case dict
            line_idx: Specific line to remove. If None, remove most loaded.
            n_lines: Number of lines to remove
            seed: Random seed for line selection

        Returns:
            Modified case dict with updated edge_index and impedance_matrix
        """
        case = deepcopy(case)
        edge_index = case['edge_index']
        n_lines_total = edge_index.shape[1]

        if n_lines >= n_lines_total:
            return case  # Can't remove all lines

        if line_idx is not None:
            lines_to_remove = [line_idx]
        else:
            # Remove lines with highest impedance (most critical for stability)
            impedances = case.get('line_impedance', torch.ones(n_lines_total))
            rng = np.random.RandomState(seed)

            # Preferentially remove high-impedance lines (critical paths)
            probs = impedances.numpy()
            probs = probs / probs.sum()
            lines_to_remove = rng.choice(
                n_lines_total, size=min(n_lines, n_lines_total),
                replace=False, p=probs
            ).tolist()

        # Remove selected lines
        mask = torch.ones(n_lines_total, dtype=torch.bool)
        for idx in lines_to_remove:
            mask[idx] = False

        case['edge_index'] = edge_index[:, mask]
        if 'line_impedance' in case:
            case['line_impedance'] = case['line_impedance'][mask]

        # Rebuild impedance matrix
        n_buses = case['n_buses']
        impedance_matrix = torch.ones(n_buses, n_buses) * 1e6
        row, col = case['edge_index']
        for i, (r, c) in enumerate(zip(row.tolist(), col.tolist())):
            z = case['line_impedance'][i].item()
            impedance_matrix[r, c] = z
            impedance_matrix[c, r] = z
        impedance_matrix.fill_diagonal_(0)
        case['impedance_matrix'] = impedance_matrix

        return case

    def extreme_delay(self, tau: torch.Tensor, tau_ms: float = 500.0) -> torch.Tensor:
        """Set all delays to an extreme value."""
        return torch.full_like(tau, tau_ms / 1000.0)

    def heavy_tail_delays(
        self,
        n_gen: int,
        alpha: float = 1.5,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Generate Pareto-distributed delays (5G/6G realistic heavy tails).

        Args:
            n_gen: Number of generators
            alpha: Pareto shape parameter (lower = heavier tails)
            seed: Random seed

        Returns:
            tau: Delays [n_gen] in seconds
        """
        rng = np.random.RandomState(seed)
        pareto_delays = (rng.pareto(alpha, size=n_gen) + 1) * 0.05  # Scale
        delays_s = np.clip(pareto_delays, 0.005, 1.0)  # 5ms to 1000ms
        return torch.tensor(delays_s, dtype=torch.float32)

    def apply_stress(
        self,
        config: StressConfig,
        tau: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> Tuple[Dict, torch.Tensor]:
        """
        Apply a stress configuration to the base case.

        Args:
            config: Stress scenario configuration
            tau: Current delay tensor [batch, n_gen]. If None, generates new.
            seed: Random seed

        Returns:
            Tuple of (modified_case, modified_tau)
        """
        case = deepcopy(self.base_case)
        n_gen = case['n_generators']

        # Apply load stress
        if config.load_factor != 1.0:
            case = self.high_load(case, config.load_factor)

        # Apply N-k contingency
        if config.remove_n_lines > 0:
            case = self.n1_contingency(
                case, n_lines=config.remove_n_lines, seed=seed
            )
        elif config.remove_line_idx is not None:
            case = self.n1_contingency(
                case, line_idx=config.remove_line_idx
            )

        # Apply delay stress
        if tau is None:
            tau = torch.ones(n_gen) * 0.05  # 50ms default

        if config.tau_ms_override is not None:
            tau = self.extreme_delay(tau, config.tau_ms_override)
        elif config.delay_distribution == 'pareto':
            tau = self.heavy_tail_delays(n_gen, config.delay_alpha, seed)

        return case, tau

    def generate_stressed_batch(
        self,
        config: StressConfig,
        batch_size: int = 32,
        seed: int = 42,
    ) -> Dict:
        """
        Generate a batch of stressed scenarios.

        Args:
            config: Stress configuration
            batch_size: Number of scenarios in batch
            seed: Random seed

        Returns:
            Batch dict compatible with model forward pass
        """
        n_gen = self.base_case['n_generators']
        rng = np.random.RandomState(seed)

        taus = []
        for i in range(batch_size):
            _, tau = self.apply_stress(config, seed=seed + i)
            taus.append(tau)

        tau_batch = torch.stack(taus)

        # Use the stressed case topology
        stressed_case, _ = self.apply_stress(config, seed=seed)

        return {
            'tau': tau_batch,
            'case': stressed_case,
            'config_name': config.name,
        }
