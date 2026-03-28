#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/19/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Visualization for Multi-Impairment Results (Topic 2)

Adds only what the existing visualization.py cannot cover:
  1. Impairment sweep curves (rho vs p, rho vs sigma_j)
  2. Per-impairment stacked bar (K/R/J contribution per generator)
  3. KRJ learning curves (3 families over epochs)

Style matches existing: seaborn-v0_8-whitegrid, dpi=150, fontsize=12/14.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')


def plot_impairment_sweep(
    sweep_data: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
):
    """
    Plot rho vs packet loss and rho vs jitter from sweep results.

    Args:
        sweep_data: Dict with 'rho_vs_p' and 'rho_vs_sigma_j' lists
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # rho vs packet loss
    p_vals = [d['p'] for d in sweep_data['rho_vs_p']]
    rho_p = [d['rho'] for d in sweep_data['rho_vs_p']]
    ax1.plot(np.array(p_vals) * 100, rho_p, 'b-o', linewidth=2, markersize=3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Instability threshold')
    ax1.set_xlabel('Packet Loss Rate (%)', fontsize=12)
    ax1.set_ylabel('Stability Margin (ρ)', fontsize=12)
    ax1.set_title('(a) Margin vs Packet Loss', fontsize=14)
    ax1.legend(fontsize=10)

    # rho vs jitter
    sig_vals = [d['sigma_j'] for d in sweep_data['rho_vs_sigma_j']]
    rho_sig = [d['rho'] for d in sweep_data['rho_vs_sigma_j']]
    ax2.plot(sig_vals, rho_sig, 'g-o', linewidth=2, markersize=3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Instability threshold')
    ax2.set_xlabel('Jitter Std (ms)', fontsize=12)
    ax2.set_ylabel('Stability Margin (ρ)', fontsize=12)
    ax2.set_title('(b) Margin vs Jitter', fontsize=14)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved impairment sweep plot to {save_path}")

    return fig


def plot_impairment_contribution(
    K: np.ndarray,
    R: np.ndarray,
    J: np.ndarray,
    tau: float = 100.0,
    tau_max: float = 500.0,
    p: float = 0.1,
    sigma_j: float = 30.0,
    sigma_max: float = 200.0,
    lambda_min_0: float = 0.04,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Stacked bar: per-generator breakdown of stability margin erosion
    into delay, loss, and jitter contributions.

    Args:
        K, R, J: Learned constants [n_gen] as numpy arrays
        tau, tau_max, p, sigma_j, sigma_max: Impairment values
        lambda_min_0: Baseline eigenvalue
        save_path: Path to save figure
        figsize: Figure size
    """
    n_gen = len(K)
    gen_labels = [f'G{i+1}' for i in range(n_gen)]

    # Per-generator contributions
    delay_contrib = K * tau / tau_max
    loss_contrib = R * p / (1 - p)
    jitter_contrib = J * sigma_j**2 / sigma_max**2

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_gen)
    width = 0.6

    ax.bar(x, delay_contrib, width, label='Delay (K·τ/τ_max)', color='#2196F3')
    ax.bar(x, loss_contrib, width, bottom=delay_contrib,
           label='Loss (R·p/(1-p))', color='#FF9800')
    ax.bar(x, jitter_contrib, width, bottom=delay_contrib + loss_contrib,
           label='Jitter (J·σ²/σ²_max)', color='#4CAF50')

    # Reference line: total budget
    ax.axhline(y=abs(lambda_min_0), color='r', linestyle='--', linewidth=1.5,
               label=f'|λ_min(0)| = {abs(lambda_min_0):.4f}')

    ax.set_xlabel('Generator', fontsize=12)
    ax.set_ylabel('Margin Erosion', fontsize=12)
    ax.set_title('Per-Generator Impairment Contribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(gen_labels[:min(n_gen, 20)], fontsize=9)
    ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved impairment contribution plot to {save_path}")

    return fig


def plot_krj_evolution(
    curves: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot K, R, J mean values over training epochs.

    Args:
        curves: Dict with 'epochs', 'K_mean', 'R_mean', 'J_mean' arrays
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = curves['epochs']
    ax.plot(epochs, curves['K_mean'], 'b-o', linewidth=2, markersize=4, label='K (delay)')
    ax.plot(epochs, curves['R_mean'], 'r-s', linewidth=2, markersize=4, label='R (loss)')
    ax.plot(epochs, curves['J_mean'], 'g-^', linewidth=2, markersize=4, label='J (jitter)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Coupling Constant', fontsize=12)
    ax.set_title('K, R, J Learning Dynamics', fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved KRJ evolution plot to {save_path}")

    return fig
