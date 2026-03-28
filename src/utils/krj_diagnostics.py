#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/18/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
K/R/J Diagnostics for Multi-Impairment Coupling Constants

Tracks learning dynamics of all three constant families during training.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class KRJSnapshot:
    """Snapshot of K, R, J values at a training epoch."""
    epoch: int
    K: np.ndarray
    R: np.ndarray
    J: np.ndarray
    rho_mean: float = 0.0
    stability_rate: float = 0.0


class KRJDiagnostics:
    """
    Track and analyze K_i, R_i, J_i learning dynamics.

    Usage:
        diag = KRJDiagnostics(n_generators=10)
        for epoch in range(epochs):
            # ... train ...
            diag.record(epoch, model, rho_mean, stability_rate)
        summary = diag.summary()
    """

    def __init__(self, n_generators: int):
        self.n_generators = n_generators
        self.history: List[KRJSnapshot] = []

    def record(
        self,
        epoch: int,
        model,
        rho_mean: float = 0.0,
        stability_rate: float = 0.0,
    ):
        """Record current K, R, J values from model."""
        vals = model.get_coupling_constants()
        self.history.append(KRJSnapshot(
            epoch=epoch,
            K=vals['K'].cpu().numpy(),
            R=vals['R'].cpu().numpy(),
            J=vals['J'].cpu().numpy(),
            rho_mean=rho_mean,
            stability_rate=stability_rate,
        ))

    def summary(self) -> Dict:
        """Compute summary statistics of learning dynamics."""
        if not self.history:
            return {}

        first = self.history[0]
        last = self.history[-1]

        return {
            'n_epochs': len(self.history),
            # Initial values
            'K_init_mean': float(first.K.mean()),
            'R_init_mean': float(first.R.mean()),
            'J_init_mean': float(first.J.mean()),
            # Final values
            'K_final_mean': float(last.K.mean()),
            'R_final_mean': float(last.R.mean()),
            'J_final_mean': float(last.J.mean()),
            'K_final_std': float(last.K.std()),
            'R_final_std': float(last.R.std()),
            'J_final_std': float(last.J.std()),
            # Change from init
            'K_change_pct': float(abs(last.K.mean() - first.K.mean()) / (first.K.mean() + 1e-10) * 100),
            'R_change_pct': float(abs(last.R.mean() - first.R.mean()) / (first.R.mean() + 1e-10) * 100),
            'J_change_pct': float(abs(last.J.mean() - first.J.mean()) / (first.J.mean() + 1e-10) * 100),
            # Dominance ratio (which impairment matters most)
            'K_R_J_ratio': f"{last.K.mean():.4f} : {last.R.mean():.4f} : {last.J.mean():.4f}",
            # Per-generator final values
            'per_generator_K': last.K.tolist(),
            'per_generator_R': last.R.tolist(),
            'per_generator_J': last.J.tolist(),
            # Stability
            'final_rho_mean': last.rho_mean,
            'final_stability_rate': last.stability_rate,
        }

    def get_learning_curves(self) -> Dict[str, np.ndarray]:
        """Get arrays for plotting learning curves."""
        epochs = np.array([s.epoch for s in self.history])
        K_means = np.array([s.K.mean() for s in self.history])
        R_means = np.array([s.R.mean() for s in self.history])
        J_means = np.array([s.J.mean() for s in self.history])
        rho_means = np.array([s.rho_mean for s in self.history])

        return {
            'epochs': epochs,
            'K_mean': K_means,
            'R_mean': R_means,
            'J_mean': J_means,
            'rho_mean': rho_means,
        }
