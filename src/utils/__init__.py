#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/11/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""Utility functions for energy-info co-optimization."""

from .visualization import (
    plot_attention_maps,
    plot_stability_margin,
    plot_k_evolution,
    plot_embedding_space,
    plot_physics_mask,
    plot_training_curves,
    plot_delay_distribution,
    create_visualization_report,
)
from .statistical_tests import (
    set_all_seeds,
    compute_statistics,
    pairwise_wilcoxon,
    cohens_d,
    friedman_nemenyi,
    format_results_table,
)
from .k_diagnostics import KLearningTracker
from .time_domain_simulation import (
    DelayedSwingEquationSimulator,
    compute_empirical_margin_independent,
    build_delay_coupling_matrix,
)
from .economic_analysis import (
    k_to_capacity_recovery,
    compute_annual_savings,
    full_economic_analysis,
    format_economic_summary,
)

__all__ = [
    # Visualization
    "plot_attention_maps",
    "plot_stability_margin",
    "plot_k_evolution",
    "plot_embedding_space",
    "plot_physics_mask",
    "plot_training_curves",
    "plot_delay_distribution",
    "create_visualization_report",
    # Statistical tests (V2)
    "set_all_seeds",
    "compute_statistics",
    "pairwise_wilcoxon",
    "cohens_d",
    "friedman_nemenyi",
    "format_results_table",
    # K diagnostics (V2)
    "KLearningTracker",
    # Time-domain simulation (V2)
    "DelayedSwingEquationSimulator",
    "compute_empirical_margin_independent",
    "build_delay_coupling_matrix",
    # Economic analysis (V2)
    "k_to_capacity_recovery",
    "compute_annual_savings",
    "full_economic_analysis",
    "format_economic_summary",
]
