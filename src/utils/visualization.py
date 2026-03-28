#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/14/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Visualization Tools for Energy-Information Co-Optimization

Provides:
1. Attention map visualization
2. Stability margin plots
3. K_i evolution during training
4. Domain embedding visualization
5. Power grid topology with attention
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_attention_maps(
    attn_weights: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    cmap: str = 'viridis',
):
    """
    Visualize attention weights.

    Args:
        attn_weights: Dict with 'causal_attn' and 'cross_attn' tensors
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Causal attention
    if 'causal_attn' in attn_weights:
        causal = attn_weights['causal_attn']
        if causal.dim() == 4:
            causal = causal[0, 0].detach().cpu().numpy()  # [batch, heads, N, N] -> [N, N]
        elif causal.dim() == 3:
            causal = causal[0].detach().cpu().numpy()

        im1 = axes[0].imshow(causal, cmap=cmap, aspect='auto')
        axes[0].set_title('Causal Self-Attention (Energy)', fontsize=12)
        axes[0].set_xlabel('Key (Source)')
        axes[0].set_ylabel('Query (Target)')
        plt.colorbar(im1, ax=axes[0], label='Attention Weight')

    # Cross-domain attention
    if 'cross_attn' in attn_weights:
        cross = attn_weights['cross_attn']
        if cross.dim() == 4:
            cross = cross[0, 0].detach().cpu().numpy()
        elif cross.dim() == 3:
            cross = cross[0].detach().cpu().numpy()

        im2 = axes[1].imshow(cross, cmap=cmap, aspect='auto')
        axes[1].set_title('Cross-Domain Attention (Energy → Comm)', fontsize=12)
        axes[1].set_xlabel('Communication Key')
        axes[1].set_ylabel('Energy Query')
        plt.colorbar(im2, ax=axes[1], label='Attention Weight')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention map to {save_path}")

    return fig


def plot_stability_margin(
    rho_history: List[float],
    tau_history: Optional[List[float]] = None,
    threshold: float = 0.0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot stability margin over training or evaluation.

    Args:
        rho_history: List of stability margin values
        tau_history: Optional list of delay values
        threshold: Stability threshold (ρ > threshold is stable)
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot stability margin
    steps = range(len(rho_history))
    ax1.plot(steps, rho_history, 'b-', linewidth=2, label='Stability Margin (ρ)')
    ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=1.5, label='Stability Threshold')

    # Fill stable/unstable regions
    rho_array = np.array(rho_history)
    ax1.fill_between(steps, threshold, rho_array,
                     where=(rho_array > threshold), alpha=0.3, color='green', label='Stable')
    ax1.fill_between(steps, threshold, rho_array,
                     where=(rho_array <= threshold), alpha=0.3, color='red', label='Unstable')

    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Stability Margin (ρ)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot delay on secondary axis
    if tau_history:
        ax2 = ax1.twinx()
        ax2.plot(steps, tau_history, 'g--', linewidth=1.5, alpha=0.7, label='Mean Delay (τ)')
        ax2.set_ylabel('Mean Delay (ms)', fontsize=12, color='g')
        ax2.tick_params(axis='y', labelcolor='g')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if tau_history:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')

    ax1.set_title('Stability Margin Analysis', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved stability plot to {save_path}")

    return fig


def plot_k_evolution(
    K_history: List[torch.Tensor],
    epochs: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot evolution of K_i values during training.

    Args:
        K_history: List of K tensors at different epochs
        epochs: Optional list of epoch numbers
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy
    K_array = np.array([K.detach().cpu().numpy() for K in K_history])
    n_epochs, n_gen = K_array.shape

    if epochs is None:
        epochs = list(range(n_epochs))

    # Plot each K_i
    colors = plt.cm.tab10(np.linspace(0, 1, n_gen))
    for i in range(n_gen):
        ax.plot(epochs, K_array[:, i], '-o', color=colors[i],
                linewidth=2, markersize=4, label=f'K_{i+1}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Coupling Constant (K_i)', fontsize=12)
    ax.set_title('Evolution of Learned Coupling Constants', fontsize=14)
    ax.legend(loc='best', ncol=min(n_gen, 5))
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved K evolution plot to {save_path}")

    return fig


def plot_embedding_space(
    h_E: torch.Tensor,
    h_I: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    method: str = 'pca',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Visualize energy and communication embeddings in 2D.

    Args:
        h_E: Energy embeddings [N, dim]
        h_I: Communication embeddings [N, dim]
        labels: Optional node labels
        method: Dimensionality reduction method ('pca' or 'tsne')
        save_path: Path to save figure
        figsize: Figure size
    """
    from sklearn.decomposition import PCA

    fig, ax = plt.subplots(figsize=figsize)

    # Detach and move to CPU
    h_E = h_E.detach().cpu().numpy()
    h_I = h_I.detach().cpu().numpy()

    # Combine embeddings
    h_all = np.vstack([h_E, h_I])
    n = len(h_E)

    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
        h_2d = reducer.fit_transform(h_all)
    else:
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=min(30, n - 1))
            h_2d = reducer.fit_transform(h_all)
        except ImportError:
            reducer = PCA(n_components=2)
            h_2d = reducer.fit_transform(h_all)

    h_E_2d = h_2d[:n]
    h_I_2d = h_2d[n:]

    # Plot
    ax.scatter(h_E_2d[:, 0], h_E_2d[:, 1], c='blue', alpha=0.7,
               s=100, label='Energy Domain', marker='o')
    ax.scatter(h_I_2d[:, 0], h_I_2d[:, 1], c='orange', alpha=0.7,
               s=100, label='Communication Domain', marker='^')

    # Draw lines connecting corresponding nodes
    for i in range(n):
        ax.plot([h_E_2d[i, 0], h_I_2d[i, 0]],
                [h_E_2d[i, 1], h_I_2d[i, 1]],
                'k-', alpha=0.2, linewidth=0.5)

    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_title('Energy-Communication Embedding Space', fontsize=14)
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved embedding plot to {save_path}")

    return fig


def plot_physics_mask(
    impedance_matrix: torch.Tensor,
    physics_mask: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Visualize impedance matrix and resulting physics mask.

    Args:
        impedance_matrix: Impedance [N, N]
        physics_mask: Physics mask M [N, N]
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Impedance matrix
    Z = impedance_matrix.detach().cpu().numpy()
    Z = np.clip(Z, 0, np.percentile(Z, 95))  # Clip outliers
    im1 = axes[0].imshow(Z, cmap='hot', aspect='auto')
    axes[0].set_title('Impedance Matrix (Z)', fontsize=12)
    axes[0].set_xlabel('Bus j')
    axes[0].set_ylabel('Bus i')
    plt.colorbar(im1, ax=axes[0], label='Impedance (Ω)')

    # Physics mask
    M = physics_mask.detach().cpu().numpy()
    im2 = axes[1].imshow(M, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Physics Mask (M = -γ·Z/Z_max)', fontsize=12)
    axes[1].set_xlabel('Bus j')
    axes[1].set_ylabel('Bus i')
    plt.colorbar(im2, ax=axes[1], label='Mask Value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved physics mask plot to {save_path}")

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot training and validation curves.

    Args:
        train_losses: Training loss history
        val_losses: Validation loss history
        metrics: Optional dict of additional metrics
        save_path: Path to save figure
        figsize: Figure size
    """
    n_plots = 1 + (len(metrics) if metrics else 0)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r--', linewidth=2, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].set_yscale('log')

    # Additional metrics
    if metrics:
        for i, (name, values) in enumerate(metrics.items()):
            ax = axes[i + 1]
            ax.plot(epochs[:len(values)], values, 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(name)
            ax.set_title(name)

    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig


def plot_delay_distribution(
    tau: torch.Tensor,
    tau_max: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot distribution of communication delays.

    Args:
        tau: Delays [batch, n_gen] or [n_samples]
        tau_max: Maximum tolerable delays
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    tau_np = tau.detach().cpu().numpy().flatten()

    # Histogram
    ax.hist(tau_np, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Delay Distribution')

    # Add tau_max lines
    if tau_max is not None:
        tau_max_np = tau_max.detach().cpu().numpy()
        for i, tm in enumerate(tau_max_np):
            ax.axvline(x=tm, color='red', linestyle='--', alpha=0.7,
                       label=f'τ_max (Gen {i+1})' if i < 3 else '')

    ax.set_xlabel('Delay (ms)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Communication Delay Distribution', fontsize=14)
    ax.legend(loc='upper right')

    # Add statistics
    stats_text = f'Mean: {tau_np.mean():.1f} ms\nStd: {tau_np.std():.1f} ms\nMax: {tau_np.max():.1f} ms'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved delay distribution to {save_path}")

    return fig


def create_visualization_report(
    model_outputs: Dict[str, torch.Tensor],
    training_history: Dict[str, List[float]],
    output_dir: str,
):
    """
    Create comprehensive visualization report.

    Args:
        model_outputs: Dict with model outputs
        training_history: Dict with training metrics
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating visualization report...")

    # 1. Attention maps
    if 'attn_info' in model_outputs:
        plot_attention_maps(
            model_outputs['attn_info'],
            save_path=str(output_dir / 'attention_maps.png')
        )

    # 2. Embedding space
    if 'h_E' in model_outputs and 'h_I' in model_outputs:
        h_E = model_outputs['h_E']
        h_I = model_outputs['h_I']
        if h_E.dim() == 3:
            h_E = h_E[0]  # Take first batch
            h_I = h_I[0]
        plot_embedding_space(
            h_E, h_I,
            save_path=str(output_dir / 'embedding_space.png')
        )

    # 3. Training curves
    if 'train_loss' in training_history and 'val_loss' in training_history:
        other_metrics = {k: v for k, v in training_history.items()
                        if k not in ['train_loss', 'val_loss']}
        plot_training_curves(
            training_history['train_loss'],
            training_history['val_loss'],
            metrics=other_metrics if other_metrics else None,
            save_path=str(output_dir / 'training_curves.png')
        )

    # 4. K evolution
    if 'K_history' in training_history:
        plot_k_evolution(
            training_history['K_history'],
            save_path=str(output_dir / 'k_evolution.png')
        )

    # 5. Stability margin
    if 'rho_history' in training_history:
        tau_history = training_history.get('tau_history', None)
        plot_stability_margin(
            training_history['rho_history'],
            tau_history=tau_history,
            save_path=str(output_dir / 'stability_margin.png')
        )

    print(f"Visualizations saved to {output_dir}")
    plt.close('all')
