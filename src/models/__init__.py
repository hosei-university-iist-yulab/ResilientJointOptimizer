#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/30/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""Model components for Energy-Information Co-Optimization."""

from .gnn import EnergyGNN, CommunicationGNN, DualDomainGNN
from .attention import (
    CausalAttention,
    CrossDomainAttention,
    HierarchicalAttention,
    PhysicsMask,
    CausalMask,
)
from .coupling import LearnableCouplingConstants, StabilityMarginComputer
from .joint_optimizer import JointOptimizer, JointOptimizerLite

# Topic 2: Multi-impairment resilient model (NEW — base classes untouched)
from .multi_impairment_coupling import MultiImpairmentCoupling
from .channel_model import ChannelStateEncoder, ChannelStatePredictor
from .resilient_optimizer import ResilientJointOptimizer

__all__ = [
    # GNN encoders
    "EnergyGNN",
    "CommunicationGNN",
    "DualDomainGNN",
    # Attention
    "CausalAttention",
    "CrossDomainAttention",
    "HierarchicalAttention",
    "PhysicsMask",
    "CausalMask",
    # Coupling (base paper)
    "LearnableCouplingConstants",
    "StabilityMarginComputer",
    # Joint model (base paper = B10)
    "JointOptimizer",
    "JointOptimizerLite",
    # Topic 2: Multi-impairment (NEW)
    "MultiImpairmentCoupling",
    "ChannelStateEncoder",
    "ChannelStatePredictor",
    "ResilientJointOptimizer",
]
