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

"""
Economic Analysis Utilities (V2)

Addresses Q7.4: Convert coupling constant K reduction
to real-world economic metrics.

Provides:
- MW capacity recovery from K reduction
- $/hour savings from deferred generation
- Annual savings projection
"""

import numpy as np
from typing import Dict, Optional


# Default economic parameters (US grid, 2024 estimates)
DEFAULT_PARAMS = {
    'electricity_price_per_mwh': 50.0,  # $/MWh (LMP average)
    'reserve_margin_cost_per_mw_year': 40000.0,  # $/MW-year
    'spinning_reserve_cost_per_mwh': 10.0,  # $/MWh
    'hours_per_year': 8760,
    'capacity_factor': 0.85,
}


def k_to_capacity_recovery(
    K_reduction: float,
    n_generators: int,
    tau_max_s: float = 0.5,
    lambda_min_0: float = 0.4,
    generator_capacity_mw: float = 100.0,
) -> Dict:
    """
    Convert K reduction to recovered MW capacity.

    Theory: Reducing K_i means the stability margin ρ(τ) increases
    for the same delay. This allows operating closer to capacity limits
    or tolerating larger delays without reducing generation.

    Args:
        K_reduction: Absolute reduction in K (e.g., 0.1 → 0.08 = 0.02)
        n_generators: Number of generators
        tau_max_s: Maximum delay margin (seconds)
        lambda_min_0: Baseline minimum eigenvalue
        generator_capacity_mw: Average generator capacity (MW)

    Returns:
        Dict with MW recovered, stability improvement, etc.
    """
    # Stability margin improvement
    # Δρ = Σ_i (ΔK_i · τ_i / τ_max,i) ≈ n_gen · ΔK · (τ_avg / τ_max)
    tau_avg_ratio = 0.5  # Average delay is typically 50% of max
    delta_rho = n_generators * K_reduction * tau_avg_ratio

    # Relative stability improvement
    rho_baseline = lambda_min_0 - n_generators * 0.1 * tau_avg_ratio
    relative_improvement = delta_rho / max(rho_baseline, 1e-6) * 100

    # Capacity recovery: stability headroom can be converted to
    # additional loading (approximately proportional)
    loading_increase_pct = relative_improvement * 0.5  # Conservative factor
    mw_recovered = n_generators * generator_capacity_mw * loading_increase_pct / 100

    return {
        'K_reduction': K_reduction,
        'delta_rho': delta_rho,
        'relative_stability_improvement_pct': relative_improvement,
        'loading_increase_pct': loading_increase_pct,
        'mw_recovered': mw_recovered,
    }


def compute_annual_savings(
    mw_recovered: float,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Compute annual economic savings from recovered capacity.

    Args:
        mw_recovered: Additional MW capacity available
        params: Economic parameters (uses defaults if None)

    Returns:
        Dict with savings breakdown
    """
    p = params or DEFAULT_PARAMS

    # 1. Energy savings: avoided generation curtailment
    energy_mwh = mw_recovered * p['hours_per_year'] * p['capacity_factor']
    energy_savings = energy_mwh * p['electricity_price_per_mwh']

    # 2. Reserve margin savings: deferred capacity investment
    reserve_savings = mw_recovered * p['reserve_margin_cost_per_mw_year']

    # 3. Spinning reserve savings: reduced need for fast-start reserves
    spinning_savings = mw_recovered * p['hours_per_year'] * p['spinning_reserve_cost_per_mwh'] * 0.1

    total_savings = energy_savings + reserve_savings + spinning_savings

    return {
        'mw_recovered': mw_recovered,
        'energy_savings_per_year': energy_savings,
        'reserve_savings_per_year': reserve_savings,
        'spinning_reserve_savings_per_year': spinning_savings,
        'total_savings_per_year': total_savings,
        'savings_per_hour': total_savings / p['hours_per_year'],
    }


def full_economic_analysis(
    K_baseline: float = 0.10,
    K_optimized: float = 0.08,
    n_generators: int = 10,
    generator_capacity_mw: float = 100.0,
    lambda_min_0: float = 0.4,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Complete economic analysis comparing baseline and optimized K.

    Args:
        K_baseline: Baseline coupling constant (before optimization)
        K_optimized: Optimized coupling constant (after training)
        n_generators: Number of generators
        generator_capacity_mw: Average generator capacity
        lambda_min_0: Baseline minimum eigenvalue
        params: Economic parameters

    Returns:
        Complete economic analysis dict
    """
    K_reduction = K_baseline - K_optimized

    capacity = k_to_capacity_recovery(
        K_reduction, n_generators,
        lambda_min_0=lambda_min_0,
        generator_capacity_mw=generator_capacity_mw,
    )

    savings = compute_annual_savings(capacity['mw_recovered'], params)

    return {
        'K_baseline': K_baseline,
        'K_optimized': K_optimized,
        'K_reduction': K_reduction,
        'K_reduction_pct': K_reduction / K_baseline * 100,
        **capacity,
        **savings,
    }


def format_economic_summary(analysis: Dict) -> str:
    """Format economic analysis as readable summary."""
    lines = [
        "=" * 60,
        "ECONOMIC ANALYSIS",
        "=" * 60,
        f"K reduction: {analysis['K_baseline']:.4f} → {analysis['K_optimized']:.4f} "
        f"({analysis['K_reduction_pct']:.1f}%)",
        f"Stability improvement: {analysis['relative_stability_improvement_pct']:.1f}%",
        f"MW capacity recovered: {analysis['mw_recovered']:.1f} MW",
        "",
        "Annual savings breakdown:",
        f"  Energy (avoided curtailment): ${analysis['energy_savings_per_year']:,.0f}",
        f"  Reserve margin (deferred):    ${analysis['reserve_savings_per_year']:,.0f}",
        f"  Spinning reserves:            ${analysis['spinning_reserve_savings_per_year']:,.0f}",
        f"  TOTAL:                        ${analysis['total_savings_per_year']:,.0f}/year",
        f"  Per hour:                     ${analysis['savings_per_hour']:,.2f}/hour",
        "=" * 60,
    ]
    return "\n".join(lines)
