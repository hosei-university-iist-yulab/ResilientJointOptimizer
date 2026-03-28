#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/20/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
PyTorch Dataset for Power Grid Co-Optimization

Combines:
1. IEEE test cases (power grid structure)
2. Load scenarios (varying demand)
3. Communication delays (synthetic)
4. Stability data (eigenvalues)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
import numpy as np

from .ieee_cases import IEEECaseLoader
from .synthetic_delays import SyntheticDelayGenerator, DelayConfig


class PowerGridDataset(Dataset):
    """
    Dataset for energy-information co-optimization.

    Each sample contains:
    - Energy features: [P, Q, V, theta, omega] per bus
    - Communication features: [tau, R, B] per node
    - Graph structure: edge_index for both domains
    - Stability data: lambda_min_0, tau_max
    - Load scenario: P_load, Q_load
    """

    def __init__(
        self,
        case_id: int = 14,
        num_scenarios: int = 1000,
        delay_config: Optional[DelayConfig] = None,
        load_variation: float = 0.2,
        seed: Optional[int] = None,
        stress_config: Optional['StressConfig'] = None,
    ):
        """
        Args:
            case_id: IEEE case number (14, 39, 118, 300)
            num_scenarios: Number of load scenarios to generate
            delay_config: Communication delay configuration
            load_variation: Load variation factor (±percentage)
            seed: Random seed
            stress_config: Optional stress scenario config (V2, Q1.6/Q2.4).
                          If provided, applies stress to generated scenarios.
        """
        super().__init__()

        self.case_id = case_id
        self.num_scenarios = num_scenarios
        self.load_variation = load_variation
        self.stress_config = stress_config

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Load base case (real power system models only)
        self.case_loader = IEEECaseLoader(case_id)
        self.base_case = self.case_loader.load()

        # Apply stress to base case topology if configured
        if stress_config is not None:
            self._apply_stress_to_base_case(stress_config, seed or 42)

        # Setup delay generator
        n_gen = self.base_case['n_generators']
        self.delay_generator = SyntheticDelayGenerator(
            n_generators=n_gen,
            config=delay_config or DelayConfig(),
            seed=seed,
        )

        # Generate scenarios
        self._generate_scenarios()

    def _apply_stress_to_base_case(self, stress_config, seed: int):
        """Apply stress configuration to modify the base case (V2)."""
        from .stressed_scenarios import StressedScenarioGenerator
        generator = StressedScenarioGenerator(self.base_case)
        stressed_case, _ = generator.apply_stress(stress_config, seed=seed)
        self.base_case = stressed_case

    def _generate_scenarios(self):
        """Pre-generate all load scenarios and delays."""
        n_bus = self.base_case['n_buses']
        n_gen = self.base_case['n_generators']

        # Base load from IEEE case
        P_load_base = self.base_case['P_load']
        Q_load_base = self.base_case['Q_load']

        # Generate load variations
        variation = torch.randn(self.num_scenarios, n_bus) * self.load_variation
        self.P_load = P_load_base.unsqueeze(0) * (1 + variation)
        self.Q_load = Q_load_base.unsqueeze(0) * (1 + variation)

        # Ensure non-negative loads
        self.P_load = torch.clamp(self.P_load, min=0)
        self.Q_load = torch.clamp(self.Q_load, min=0)

        # Generate communication delays
        self.tau = self.delay_generator.generate(self.num_scenarios)

        # Compute eigenvalues for each scenario (simplified: use base case)
        self.lambda_min_0 = torch.full(
            (self.num_scenarios,),
            self.base_case['lambda_min']
        )

        # Compute tau_max based on stability (scale K_init with grid size)
        from src.models.coupling import compute_k_init_scale
        k_val = compute_k_init_scale(n_gen, self.base_case['lambda_min'])
        K_init = torch.ones(n_gen) * k_val
        self.tau_max = self.delay_generator.compute_tau_max(
            self.base_case['lambda_min'],
            K_init,
        ).unsqueeze(0).expand(self.num_scenarios, -1)

        # Generate initial control actions (from OPF or zero)
        self.u_init = torch.zeros(self.num_scenarios, n_gen * 2)
        if 'P_gen' in self.base_case:
            P_gen_base = self.base_case['P_gen']
            Q_gen_base = self.base_case.get('Q_gen', torch.zeros_like(P_gen_base))
            self.u_init = torch.cat([
                P_gen_base.unsqueeze(0).expand(self.num_scenarios, -1),
                Q_gen_base.unsqueeze(0).expand(self.num_scenarios, -1),
            ], dim=-1)

    def __len__(self) -> int:
        return self.num_scenarios

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns dict with all data needed for training.
        """
        n_bus = self.base_case['n_buses']
        n_gen = self.base_case['n_generators']

        # Energy features: [P, Q, V, theta, omega]
        # For now, use base case values with load variation
        V = self.base_case['V']  # Voltage magnitudes [n_bus]
        theta = self.base_case['theta']  # Voltage angles [n_bus]

        # Generator states
        omega = torch.ones(n_gen)  # Nominal frequency

        # Build energy node features
        energy_x = torch.stack([
            self.P_load[idx],  # Active power (load)
            self.Q_load[idx],  # Reactive power (load)
            V,                  # Voltage magnitude
            theta,              # Voltage angle
            torch.zeros(n_bus),  # Frequency deviation (bus level)
        ], dim=-1)  # [n_bus, 5]

        # Communication features: [tau, R, B]
        # tau: delays for generator nodes, 0 for others
        tau_full = torch.zeros(n_bus)
        gen_buses = self.base_case.get('gen_buses', torch.arange(n_gen))
        tau_full[gen_buses[:n_gen]] = self.tau[idx]

        # R: bandwidth (uniform for now)
        R = torch.ones(n_bus) * 100  # 100 Mbps

        # B: buffer size
        B = torch.ones(n_bus) * 1000  # 1000 packets

        comm_x = torch.stack([tau_full, R, B], dim=-1)  # [n_bus, 3]

        return {
            # Node features
            'energy_x': energy_x,
            'comm_x': comm_x,
            # Graph structure
            'energy_edge_index': self.base_case['edge_index'],
            'comm_edge_index': self.base_case['edge_index'],  # Same topology
            # Stability data
            'tau': self.tau[idx],
            'tau_max': self.tau_max[idx],
            'lambda_min_0': self.lambda_min_0[idx],
            # Load data
            'P_load': self.P_load[idx].sum(),
            # Control data
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
        """Get base IEEE case data."""
        return self.base_case


class MultiCaseDataset(Dataset):
    """
    Dataset combining multiple IEEE test cases.

    Allows training on diverse grid topologies.
    """

    def __init__(
        self,
        case_ids: List[int] = [14, 39, 118],
        scenarios_per_case: int = 500,
        delay_config: Optional[DelayConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            case_ids: List of IEEE case numbers
            scenarios_per_case: Scenarios per case
            delay_config: Delay configuration
            seed: Random seed
        """
        super().__init__()

        self.datasets = []
        self.case_indices = []
        self.cumulative_sizes = [0]

        for i, case_id in enumerate(case_ids):
            try:
                ds = PowerGridDataset(
                    case_id=case_id,
                    num_scenarios=scenarios_per_case,
                    delay_config=delay_config,
                    seed=seed + i if seed else None,
                )
                self.datasets.append(ds)
                self.case_indices.extend([i] * len(ds))
                self.cumulative_sizes.append(
                    self.cumulative_sizes[-1] + len(ds)
                )
            except Exception as e:
                print(f"Warning: Could not load IEEE case {case_id}: {e}")

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample, handling multi-case indexing."""
        # Find which dataset this index belongs to
        for i, (start, end) in enumerate(zip(
            self.cumulative_sizes[:-1],
            self.cumulative_sizes[1:]
        )):
            if start <= idx < end:
                local_idx = idx - start
                return self.datasets[i][local_idx]

        raise IndexError(f"Index {idx} out of range")


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-size graphs.

    Handles batching of graphs with different numbers of nodes.
    """
    # Separate items that need special handling
    batch_size = len(batch)

    # For now, assume all samples have same graph structure (same case)
    # Stack simple tensors
    result = {}

    for key in batch[0].keys():
        if key in ['energy_edge_index', 'comm_edge_index']:
            # Edge indices: keep as list or stack with offset
            result[key] = batch[0][key]  # Assume same structure
        elif key in ['case_id', 'n_generators', 'idx']:
            # Metadata: keep as list or tensor
            result[key] = torch.tensor([b[key] for b in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            # Stack tensors
            result[key] = torch.stack([b[key] for b in batch])
        else:
            # Keep as is
            result[key] = [b[key] for b in batch]

    return result


def create_dataloaders(
    case_id: int = 14,
    num_scenarios: int = 1000,
    train_split: float = 0.7,
    val_split: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    delay_config: Optional[DelayConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        case_id: IEEE case number
        num_scenarios: Total number of scenarios
        train_split: Training set fraction
        val_split: Validation set fraction
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed
        delay_config: Delay configuration

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    dataset = PowerGridDataset(
        case_id=case_id,
        num_scenarios=num_scenarios,
        delay_config=delay_config,
        seed=seed,
    )

    # Split indices
    n = len(dataset)
    n_train = int(train_split * n)
    n_val = int(val_split * n)
    n_test = n - n_train - n_val

    # Random split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
