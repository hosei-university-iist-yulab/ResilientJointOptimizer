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
Statistical Testing Framework for Journal-Quality Results

Addresses Q2.2: No confidence intervals, no standard deviations, no p-values.

Provides:
- set_all_seeds: Reproducible random state
- compute_statistics: Mean, std, CI_95 for all metrics
- pairwise_wilcoxon: Non-parametric paired test with Holm-Sidak correction
- cohens_d: Effect size computation
- friedman_nemenyi: Overall model comparison
"""

import os
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Optional
from scipy import stats

# Project-wide GPU restriction: always use CUDA 4-7
CUDA_VISIBLE_DEVICES = "4,5,6,7"


def restrict_gpus(devices: str = CUDA_VISIBLE_DEVICES):
    """Restrict PyTorch to specific CUDA devices. Must be called before any CUDA op."""
    # If the orchestrator already pinned us to a specific GPU, respect it
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = devices


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_statistics(results_list: List[Dict]) -> Dict:
    """
    Compute mean, std, CI, min, max for all metric keys.

    Args:
        results_list: List of result dicts (one per seed), each with same keys

    Returns:
        Dict mapping metric names to {mean, std, ci_lower, ci_upper, min, max, values}
    """
    if not results_list:
        return {}

    # Handle list of scalars (e.g., [0.5, 0.6, 0.4])
    if isinstance(results_list[0], (int, float)):
        values = [float(v) for v in results_list]
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        ci_95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        return {
            'mean': mean, 'std': std,
            'ci_lower': mean - ci_95, 'ci_upper': mean + ci_95,
            'min': float(np.min(values)), 'max': float(np.max(values)),
            'values': values,
        }

    metrics = {}
    for key in results_list[0].keys():
        values = []
        for r in results_list:
            v = r.get(key)
            if isinstance(v, (int, float)):
                values.append(float(v))

        if not values:
            continue

        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0.0
        ci_95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0

        metrics[key] = {
            'mean': float(mean),
            'std': float(std),
            'ci_lower': float(mean - ci_95),
            'ci_upper': float(mean + ci_95),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values,
        }

    return metrics


def pairwise_wilcoxon(
    proposed_values: List[float],
    baseline_values_dict: Dict[str, List[float]],
    alternative: str = 'greater',
) -> Dict:
    """
    Wilcoxon signed-rank test with Holm-Sidak correction.

    Non-parametric (no normality assumption). Paired (same scenarios).

    Args:
        proposed_values: Metric values for the proposed model [n_seeds]
        baseline_values_dict: {baseline_name: [values]} for each baseline
        alternative: 'greater' (proposed > baseline), 'two-sided', or 'less'

    Returns:
        Dict mapping baseline name to {W, p_raw, p_corrected, cohens_d, significant}
    """
    results = {}
    p_values = []

    for name, baseline_values in baseline_values_dict.items():
        n = min(len(proposed_values), len(baseline_values))
        if n < 5:
            # Too few samples for Wilcoxon; use simple t-test instead
            stat, p = stats.ttest_rel(proposed_values[:n], baseline_values[:n])
            if alternative == 'greater':
                p = p / 2 if stat > 0 else 1 - p / 2
        else:
            try:
                stat, p = stats.wilcoxon(
                    proposed_values[:n],
                    baseline_values[:n],
                    alternative=alternative,
                )
            except ValueError:
                # All differences are zero
                stat, p = 0.0, 1.0

        d = cohens_d(proposed_values[:n], baseline_values[:n])
        p_values.append((name, p))
        results[name] = {
            'W': float(stat),
            'p_raw': float(p),
            'cohens_d': float(d),
            'n_paired': n,
        }

    # Holm-Sidak correction for multiple comparisons
    p_values.sort(key=lambda x: x[1])
    m = len(p_values)
    for rank, (name, p) in enumerate(p_values):
        p_corrected = 1 - (1 - p) ** (m - rank)
        results[name]['p_corrected'] = float(min(p_corrected, 1.0))
        results[name]['significant_005'] = p_corrected < 0.05
        results[name]['significant_001'] = p_corrected < 0.01

    return results


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Cohen's d effect size.

    Interpretation: small=0.2, medium=0.5, large=0.8

    Args:
        group1: Values from group 1
        group2: Values from group 2

    Returns:
        d: Effect size (positive means group1 > group2)
    """
    g1 = np.array(group1)
    g2 = np.array(group2)
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def friedman_nemenyi(
    all_model_results: Dict[str, List[float]],
) -> Dict:
    """
    Friedman test (non-parametric ANOVA) with Nemenyi post-hoc.

    For comparing all models simultaneously.

    Args:
        all_model_results: {model_name: [values_per_scenario]}

    Returns:
        Dict with friedman_p, significant, mean_ranks, critical_difference
    """
    model_names = list(all_model_results.keys())
    k = len(model_names)

    if k < 3:
        return {'friedman_p': 1.0, 'significant': False,
                'note': 'Need >= 3 models for Friedman test'}

    # Ensure all lists have the same length
    min_len = min(len(v) for v in all_model_results.values())
    data = [all_model_results[name][:min_len] for name in model_names]

    try:
        stat, p = stats.friedmanchisquare(*data)
    except ValueError:
        return {'friedman_p': 1.0, 'significant': False,
                'note': 'Friedman test failed (insufficient variance)'}

    # Compute mean ranks
    data_array = np.array(data).T  # [n_scenarios, k_models]
    ranks = np.zeros_like(data_array, dtype=float)
    for i in range(data_array.shape[0]):
        ranks[i] = stats.rankdata(-data_array[i])  # Negative for descending

    mean_ranks = {name: float(ranks[:, j].mean())
                  for j, name in enumerate(model_names)}

    result = {
        'friedman_stat': float(stat),
        'friedman_p': float(p),
        'significant': p < 0.05,
        'mean_ranks': mean_ranks,
    }

    if p < 0.05:
        # Nemenyi critical difference
        q_alpha = 2.569  # q for alpha=0.05, k groups (approximate)
        n = min_len
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
        result['critical_difference'] = float(cd)
        result['nemenyi_note'] = ('Models with rank difference > CD are '
                                  'significantly different')

    return result


def format_result_cell(mean: float, std: float, p: Optional[float] = None) -> str:
    """Format a result cell for LaTeX table: mean ± std with significance markers."""
    sig = ""
    if p is not None:
        if p < 0.001:
            sig = " ***"
        elif p < 0.01:
            sig = " **"
        elif p < 0.05:
            sig = " *"

    return f"{mean:.4f} $\\pm$ {std:.4f}{sig}"


def format_results_table(
    statistics_dict: Dict[str, Dict],
    significance_dict: Optional[Dict] = None,
    metric_key: str = 'mean_stability_margin',
) -> str:
    """
    Generate a formatted results table string.

    Args:
        statistics_dict: {model_name: {metric: {mean, std, ...}}}
        significance_dict: {model_name: {p_corrected, ...}} (vs proposed)
        metric_key: Which metric to format

    Returns:
        Formatted table string
    """
    lines = []
    lines.append(f"{'Model':<30} {metric_key:<30}")
    lines.append("-" * 60)

    for model_name, stats_data in statistics_dict.items():
        if metric_key in stats_data:
            s = stats_data[metric_key]
            p = None
            if significance_dict and model_name in significance_dict:
                p = significance_dict[model_name].get('p_corrected')
            cell = format_result_cell(s['mean'], s['std'], p)
            lines.append(f"{model_name:<30} {cell:<30}")

    return "\n".join(lines)
