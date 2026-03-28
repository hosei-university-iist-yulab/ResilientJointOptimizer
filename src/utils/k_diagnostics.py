#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/12/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
K_i Learning Diagnostics

Tracks per-generator coupling constant K_i evolution during training.
Addresses Q1.3: K barely changes from initialization on IEEE 14.

Provides:
- KLearningTracker: Log and visualize K_i trajectories
- Convergence analysis: Check if K converges regardless of initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class KLearningTracker:
    """Track per-generator K_i evolution during training."""

    def __init__(self, n_generators: int):
        """
        Args:
            n_generators: Number of generators (determines number of K_i to track)
        """
        self.n_generators = n_generators
        self.history = {f'K_{i}': [] for i in range(n_generators)}
        self.epoch_history = []
        self.k_init = None

    def log(self, epoch: int, K_values):
        """
        Record K_i values at a given epoch.

        Args:
            epoch: Current training epoch
            K_values: K_i values (numpy array or list of length n_generators)
        """
        if hasattr(K_values, 'numpy'):
            K_values = K_values.detach().cpu().numpy()
        K_values = np.atleast_1d(K_values)

        if self.k_init is None:
            self.k_init = K_values.copy()

        self.epoch_history.append(epoch)
        for i in range(min(len(K_values), self.n_generators)):
            self.history[f'K_{i}'].append(float(K_values[i]))

    def get_summary(self) -> Dict:
        """Get summary statistics of K learning."""
        if not self.epoch_history:
            return {}

        final_K = np.array([self.history[f'K_{i}'][-1]
                            for i in range(self.n_generators)])

        summary = {
            'k_init_mean': float(self.k_init.mean()) if self.k_init is not None else None,
            'k_final_mean': float(final_K.mean()),
            'k_final_std': float(final_K.std()),
            'k_final_min': float(final_K.min()),
            'k_final_max': float(final_K.max()),
            'k_change_pct': float(
                abs(final_K.mean() - self.k_init.mean()) / self.k_init.mean() * 100
            ) if self.k_init is not None else None,
            'n_epochs': len(self.epoch_history),
            'per_generator_final': final_K.tolist(),
        }
        return summary

    def plot_trajectories(self, save_path: str, title_suffix: str = ""):
        """
        Plot per-generator K_i over training epochs.

        Args:
            save_path: Path to save the figure
            title_suffix: Optional suffix for plot title
        """
        if not self.epoch_history:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Individual K_i trajectories
        ax = axes[0]
        for i in range(self.n_generators):
            name = f'K_{i}'
            values = self.history[name]
            ax.plot(self.epoch_history[:len(values)], values,
                    label=name, alpha=0.7, linewidth=1.5)

        if self.k_init is not None:
            ax.axhline(y=float(self.k_init.mean()), color='gray',
                       linestyle='--', alpha=0.5, label=f'Init ({self.k_init.mean():.3f})')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$K_i$')
        ax.set_title(f'Per-Generator Coupling Constants{title_suffix}')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Right: K_i distribution evolution at epoch checkpoints
        ax = axes[1]
        n_epochs = len(self.epoch_history)
        if n_epochs > 4:
            epochs_to_show = [0, n_epochs // 4, n_epochs // 2, n_epochs - 1]
        else:
            epochs_to_show = list(range(n_epochs))

        for idx in epochs_to_show:
            k_vals = [self.history[f'K_{i}'][idx]
                      for i in range(self.n_generators)
                      if idx < len(self.history[f'K_{i}'])]
            if k_vals:
                ax.hist(k_vals, alpha=0.4, bins=max(5, self.n_generators // 2),
                        label=f'Epoch {self.epoch_history[idx]}')

        ax.set_xlabel(r'$K_i$ value')
        ax.set_ylabel('Count')
        ax.set_title(f'K Distribution Over Training{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def to_dict(self) -> Dict:
        """Export tracker data to a serializable dict."""
        return {
            'n_generators': self.n_generators,
            'epoch_history': self.epoch_history,
            'k_history': {
                name: values for name, values in self.history.items()
            },
            'k_init': self.k_init.tolist() if self.k_init is not None else None,
            'summary': self.get_summary(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'KLearningTracker':
        """Create tracker from serialized dict."""
        tracker = cls(data['n_generators'])
        tracker.epoch_history = data['epoch_history']
        tracker.history = data['k_history']
        if data['k_init'] is not None:
            tracker.k_init = np.array(data['k_init'])
        return tracker
