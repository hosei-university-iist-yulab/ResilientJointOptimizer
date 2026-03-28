#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/03/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Resilient Power Grid Dataset

Extends the base PowerGridDataset pattern with 6-dim communication features:
  [tau, R, B, p, sigma_j, s]

where:
  tau     = communication delay (ms)         — from SyntheticDelayGenerator (base)
  R       = bandwidth (Mbps)                 — uniform (base)
  B       = buffer size (packets)            — uniform (base)
  p       = packet loss rate [0, p_max]      — from ImpairmentGenerator (NEW)
  sigma_j = jitter std (ms) [0, sigma_max]   — from ImpairmentGenerator (NEW)
  s       = Markov channel state (0/1/2)     — from ChannelSimulator (NEW)

The base PowerGridDataset (3-dim) is kept UNTOUCHED for B10 baseline.
This class is used ONLY by ResilientJointOptimizer (our model) and
baselines that accept 6-dim comm input.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List

from .ieee_cases import IEEECaseLoader
from .synthetic_delays import SyntheticDelayGenerator, DelayConfig
from .impairment_generator import ImpairmentGenerator, ImpairmentConfig
from .channel_simulator import ChannelSimulator, ChannelConfig


class ResilientPowerGridDataset(Dataset):
    """
    Dataset for multi-impairment energy-information co-optimization.

    Each sample contains:
    - Energy features: [P, Q, V, theta, omega] per bus (same as base)
    - Communication features: [tau, R, B, p, sigma_j, s] per node (EXTENDED)
    - Graph structure: edge_index for both domains (same as base)
    - Stability data: lambda_min_0, tau_max (same as base)
    - Impairment data: packet_loss, jitter, channel_state (NEW)
    """

    def __init__(
        self,
        case_id: int = 39,
        num_scenarios: int = 1000,
        delay_config: Optional[DelayConfig] = None,
        impairment_config: Optional[ImpairmentConfig] = None,
        channel_config: Optional[ChannelConfig] = None,
        load_variation: float = 0.2,
        sigma_max_ms: float = 200.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            case_id: Real power system case (39, 57, 118, 145, 300, 1354, 1888, 2869)
            num_scenarios: Number of load scenarios to generate
            delay_config: Communication delay configuration
            impairment_config: Packet loss and jitter configuration
            channel_config: Markov channel model configuration
            load_variation: Load variation factor (+/- percentage)
            sigma_max_ms: Maximum jitter for normalization in rho computation
            seed: Random seed
        """
        super().__init__()

        self.case_id = case_id
        self.num_scenarios = num_scenarios
        self.load_variation = load_variation
        self.sigma_max_ms = sigma_max_ms

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Load real power system case
        self.case_loader = IEEECaseLoader(case_id)
        self.base_case = self.case_loader.load()

        # Setup generators
        n_gen = self.base_case['n_generators']

        self.delay_generator = SyntheticDelayGenerator(
            n_generators=n_gen,
            config=delay_config or DelayConfig(),
            seed=seed,
        )

        self.impairment_generator = ImpairmentGenerator(
            n_generators=n_gen,
            config=impairment_config or ImpairmentConfig(),
            seed=seed + 1000 if seed is not None else None,
        )

        self.channel_simulator = ChannelSimulator(
            n_generators=n_gen,
            config=channel_config or ChannelConfig(),
            seed=seed + 2000 if seed is not None else None,
        )

        # Generate all scenarios
        self._generate_scenarios()

    def _generate_scenarios(self):
        """Pre-generate all load scenarios, delays, and impairments."""
        n_bus = self.base_case['n_buses']
        n_gen = self.base_case['n_generators']

        # --- Load variations (same as base) ---
        P_load_base = self.base_case['P_load']
        Q_load_base = self.base_case['Q_load']
        variation = torch.randn(self.num_scenarios, n_bus) * self.load_variation
        self.P_load = torch.clamp(P_load_base.unsqueeze(0) * (1 + variation), min=0)
        self.Q_load = torch.clamp(Q_load_base.unsqueeze(0) * (1 + variation), min=0)

        # --- Communication delays (same as base) ---
        self.tau = self.delay_generator.generate(self.num_scenarios)

        # --- Eigenvalues (same as base) ---
        self.lambda_min_0 = torch.full(
            (self.num_scenarios,),
            self.base_case['lambda_min'],
        )

        # --- Tau max (same as base) ---
        from src.models.coupling import compute_k_init_scale
        k_val = compute_k_init_scale(n_gen, self.base_case['lambda_min'])
        K_init = torch.ones(n_gen) * k_val
        self.tau_max = self.delay_generator.compute_tau_max(
            self.base_case['lambda_min'], K_init,
        ).unsqueeze(0).expand(self.num_scenarios, -1)

        # --- Initial control (same as base) ---
        self.u_init = torch.zeros(self.num_scenarios, n_gen * 2)
        if 'P_gen' in self.base_case:
            P_gen_base = self.base_case['P_gen']
            Q_gen_base = self.base_case.get('Q_gen', torch.zeros_like(P_gen_base))
            self.u_init = torch.cat([
                P_gen_base.unsqueeze(0).expand(self.num_scenarios, -1),
                Q_gen_base.unsqueeze(0).expand(self.num_scenarios, -1),
            ], dim=-1)

        # --- NEW: Markov channel states ---
        self.channel_state = self.channel_simulator.simulate(
            batch_size=self.num_scenarios,
        )  # [num_scenarios, n_gen] long tensor (0/1/2)

        # --- NEW: Packet loss and jitter (state-dependent) ---
        self.packet_loss, self.jitter = self.impairment_generator.generate_from_channel_state(
            states=self.channel_state.numpy(),
        )  # [num_scenarios, n_gen] each

    def __len__(self) -> int:
        return self.num_scenarios

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with 6-dim communication features.

        Returns dict compatible with ResilientJointOptimizer.forward().
        """
        n_bus = self.base_case['n_buses']
        n_gen = self.base_case['n_generators']

        V = self.base_case['V']
        theta = self.base_case['theta']

        # Energy features [n_bus, 5] — same as base
        energy_x = torch.stack([
            self.P_load[idx],
            self.Q_load[idx],
            V,
            theta,
            torch.zeros(n_bus),
        ], dim=-1)

        # Communication features [n_bus, 6] — EXTENDED
        gen_buses = self.base_case.get('gen_buses', torch.arange(n_gen))

        # Per-bus delay
        tau_full = torch.zeros(n_bus)
        tau_full[gen_buses[:n_gen]] = self.tau[idx]

        # Per-bus bandwidth and buffer (uniform)
        R = torch.ones(n_bus) * 100.0
        B = torch.ones(n_bus) * 1000.0

        # Per-bus packet loss
        p_full = torch.zeros(n_bus)
        p_full[gen_buses[:n_gen]] = self.packet_loss[idx]

        # Per-bus jitter
        sigma_full = torch.zeros(n_bus)
        sigma_full[gen_buses[:n_gen]] = self.jitter[idx]

        # Per-bus channel state (as float for GNN input)
        s_full = torch.zeros(n_bus)
        s_full[gen_buses[:n_gen]] = self.channel_state[idx].float()

        comm_x = torch.stack([tau_full, R, B, p_full, sigma_full, s_full], dim=-1)

        return {
            # Node features
            'energy_x': energy_x,              # [n_bus, 5]
            'comm_x': comm_x,                  # [n_bus, 6]
            # Graph structure
            'energy_edge_index': self.base_case['edge_index'],
            'comm_edge_index': self.base_case['edge_index'],
            # Stability data (same as base)
            'tau': self.tau[idx],              # [n_gen]
            'tau_max': self.tau_max[idx],      # [n_gen]
            'lambda_min_0': self.lambda_min_0[idx],  # scalar
            # NEW: impairment data
            'packet_loss': self.packet_loss[idx],     # [n_gen]
            'jitter': self.jitter[idx],               # [n_gen]
            'channel_state': self.channel_state[idx], # [n_gen] long
            'sigma_max': torch.tensor(self.sigma_max_ms),  # scalar
            # Load and control (same as base)
            'P_load': self.P_load[idx].sum(),
            'u_prev': self.u_init[idx],
            # Metadata
            'idx': idx,
            'case_id': self.case_id,
            'n_generators': n_gen,
        }

    def get_impedance_matrix(self) -> Optional[torch.Tensor]:
        """Get impedance matrix for physics mask. None for large grids."""
        z = self.base_case.get('impedance_matrix')
        if z is None:
            return None
        return z

    def get_base_case(self) -> Dict[str, torch.Tensor]:
        """Get base case data."""
        return self.base_case


def resilient_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate for ResilientPowerGridDataset."""
    result = {}
    for key in batch[0].keys():
        if key in ['energy_edge_index', 'comm_edge_index']:
            result[key] = batch[0][key]
        elif key in ['case_id', 'n_generators', 'idx']:
            result[key] = torch.tensor([b[key] for b in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result


def create_resilient_dataloaders(
    case_id: int = 39,
    num_scenarios: int = 1000,
    train_split: float = 0.7,
    val_split: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    delay_config: Optional[DelayConfig] = None,
    impairment_config: Optional[ImpairmentConfig] = None,
    channel_config: Optional[ChannelConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for resilient dataset."""
    dataset = ResilientPowerGridDataset(
        case_id=case_id,
        num_scenarios=num_scenarios,
        delay_config=delay_config,
        impairment_config=impairment_config,
        channel_config=channel_config,
        seed=seed,
    )

    n = len(dataset)
    n_train = int(train_split * n)
    n_val = int(val_split * n)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_ds = torch.utils.data.Subset(dataset, indices[:n_train])
    val_ds = torch.utils.data.Subset(dataset, indices[n_train:n_train + n_val])
    test_ds = torch.utils.data.Subset(dataset, indices[n_train + n_val:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=resilient_collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=resilient_collate_fn,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=resilient_collate_fn,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
